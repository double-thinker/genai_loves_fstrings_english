"""
v2: we reimplement the smartllm algorithm without using langchain in its three
components:

- idea generation ('idea_prompt')
- idea critique ('critique_prompt')
- final combination ('merge_prompt')
"""

import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

DEFAULT = """What is the meaning of life?"""


def llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def idea_prompt(question: str) -> str:
    return f"""Question: {question}
Answer: Let's work this out in a step by step way to be sure we have the right answer:
"""


def critique_prompt(question: str, ideas: list[str]) -> str:
    return f"""{idea_prompt(question)}
{'\n'.join(f"> Idea {i+1}: {idea}" for i, idea in enumerate(ideas))}
You are a researcher tasked with investigating the 2 response options provided.
List the flaws and faulty logic of each answer option. Let's work this out in a
step by step way to be sure we have all the errors:
"""


def merge_prompt(question: str, ideas: list[str], critique: str) -> str:
    return f"""{critique_prompt(question, ideas)}
{critique}
You are a resolver tasked with 1) finding which of the 2 answer
options the researcher thought was best, 2) improving that answer and
3) printing the answer in full. Don't output anything for step 1 or 2,
only the full answer in 3. Let's work this out in a step by step way to be
sure we have the right answer:
"""


def smartllm(question: str = DEFAULT, n_ideas: int = 2):
    ideas = [llm(idea_prompt(question)) for _ in range(n_ideas)]
    critique = llm(critique_prompt(question, ideas))
    return llm(merge_prompt(question, ideas, critique))


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(smartllm(question))
