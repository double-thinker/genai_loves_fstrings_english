"""
v1:

In general, there are two elements that are reasonable to keep from the original
langchain pipeline because migrating them is more complicated and/or we want to
ensure the same functionality:

1. Scraping / Doc loader
2. The chunking / text splitting strategy.
   Some splitters are easier than others. Depending on the situation, I usually
   replace them or not.

You can see an example where I replace the scraping, doc loader, and splitter
in `v2.py`.
"""

import sys
import warnings

import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

warnings.filterwarnings("ignore", message="API key must be provided")
load_dotenv()

client = OpenAI()


def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Basic function to call the API. Define it in a way that it can be
    replaceable by other providers in the future.

    There are two libraries that already package a common interface for LLMs:
    - [`llm` by Simon Willison](https://pypi.org/project/llm/)
    - [`litellm`](https://pypi.org/project/litellm/) if you don't mind using a proxy

    But my recommendation is to use the API directly from the provider. Each one
    implements slight differences that are relevant to control. Also, a third-party
    library is always behind API changes.
    """
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def prompt(docs: list[Document], question: str) -> str:
    """
    Prompt to use with RAG.

    **It is HIGHLY recommended** to download the prompt from the hub and write it
    in the code to have it versioned and be able to read it (see slides). For example,
    here we download it from https://smith.langchain.com/hub/rlm/rag-prompt
    """
    # Before:
    # prompt = hub.pull("rlm/rag-prompt")

    return f"""HUMAN

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {'\n\n'.join((doc.page_content for doc in docs))}

Answer:"""


# We keep the RAG, scraping, and embedding part. See `v2.py`.

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


def chatbot(question: str):
    """
    Central function of our chatbot. Equivalent to the LangChain pipeline.
    """
    # Get relevant documents
    docs = retriever.invoke(question)

    # Generate a response
    return llm(prompt(docs, question))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What is Task Decomposition?"

    print(f"Human: {question}")
    print(f"Chatbot: {chatbot(question)}")
