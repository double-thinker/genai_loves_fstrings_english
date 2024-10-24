"""
Exercise 2:
Reimplement the smartllm algorithm using langchain. To do this, we will intercept
the calls to the OpenAI API using the `observability` module that is
in this repository.
"""

import sys

from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI

DEFAULT = """What is the meaning of life?"""


def smartllm(question: str = DEFAULT):
    prompt = PromptTemplate.from_template(question)
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")

    chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=2, verbose=False)
    return chain.invoke({})["resolution"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(smartllm(question))
