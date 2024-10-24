"""
v1: we add observability without modifying the original code
"""

import sys

from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI

DEFAULT = """What is the meaning of life?"""


def smartllm(question: str = DEFAULT):
    prompt = PromptTemplate.from_template(question)
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")

    chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=2)
    return chain.invoke({})["resolution"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(smartllm(question))
