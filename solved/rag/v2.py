"""
v2:

Changes:

- We create `text_splitter` to divide documents into chunks (slightly different
behavior from `RecursiveCharacterTextSplitter`, IMO better). This is the most
extensive part of this version and I don't recommend doing it as a first step
when migrating from langchain. But the code can be reused and is very easy to
understand
- We use ChromaDB directly without langchain
- We create `scrape_web` to get the web content


Extra:
- We persist the database to avoid collection creation costs in each execution
"""

import os
import re
import sys
from typing import Generator
from uuid import uuid4

import bs4

# Note: We don't use `langchain_chroma` but `chromadb`
import chromadb
import chromadb.utils.embedding_functions as ef
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
db = chromadb.PersistentClient(path="./ragdatabase")


def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def prompt(docs: list[str], question: str) -> str:
    return f"""HUMAN

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {'\n\n'.join(docs)}

Answer:"""


def scrape_web(url: str) -> str:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    elements = [soup.find(class_="post-title"), soup.find(class_="post-content")]
    return " ".join([element.get_text() for element in elements])


def text_splitter(doc: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Simple splitter similar to `RecursiveCharacterTextSplitter` from langchain.
    With slight differences: the division characters are more consistent.

    Despite its apparent "complexity", this is code that I normally
    reuse or if necessary for the project, I keep the langchain text splitter.
    """
    splits = re.split(r"[\s\.,;:]+", doc)

    # We accumulate the splits until reaching the chunk size
    # adding an overlap of `chunk_overlap` characters from the previous chunk
    prev_chunk = ""
    chunk = ""
    for subchunk in splits:
        length = len(chunk) + len(subchunk)
        if length > chunk_size:
            yield prev_chunk[-chunk_overlap:] + chunk
            prev_chunk = chunk
            chunk = ""
        else:
            chunk += " " + subchunk
    if chunk:
        yield chunk


def fill_db(docs: Generator[str, None, None]):
    """
    Function to fill the database with documents and populate it with
    chunks of the documents.

    For me, handling the database directly is something that helps me a lot to
    debug and control the pipeline: caching, reuse, use of various embeddings, etc.
    """
    collection = db.get_or_create_collection(
        "rag",
        embedding_function=ef.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY")
        ),
    )

    if collection.count() > 0:
        # If the database already exists, creation is skipped to save
        # time (and money).
        return collection

    for doc in docs:
        for chunk in text_splitter(doc):
            collection.add(documents=chunk, ids=[uuid4().hex])

    return collection


def chatbot(question: str):
    """
    Central function of our chatbot. Equivalent to the LangChain pipeline.
    """
    db = fill_db([scrape_web("https://lilianweng.github.io/posts/2023-06-23-agent/")])
    context = db.query(query_texts=[question], n_results=5)["documents"][0]
    return llm(prompt(context, question))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What is Task Decomposition?"

    print(f"Human: {question}")
    print(f"Chatbot: {chatbot(question)}")
