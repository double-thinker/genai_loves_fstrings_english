"""
Exercise 1:
Example taken from https://python.langchain.com/docs/tutorials/rag/ with some
minor changes (dotenv vs hardcoded API key, CLI and ignoring LangSmith warnings)

We will reimplement this pipeline using "pure python". We'll use chroma as
the vector store. The text splitter and web base loader are allowed.
You can use other libraries like beautifulsoup, requests, etc.

Suggestions:
- Start with a simple solution: fictional rag context, without using a
  vector store
- Make a simple function for OpenAI API calls
- Make another function to create the prompt

The solution is in the `solved` module
"""

import sys
import warnings

import bs4
import dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", message="API key must be provided")

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
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

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What is Task Decomposition?"

    print(f"Human: {question}")
    print(f"Chatbot: {rag_chain.invoke(question)}")
