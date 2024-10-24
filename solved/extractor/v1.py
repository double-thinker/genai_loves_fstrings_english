"""
v1: we create a data model for the talks and the prompts to extract
fields from a document.

We could use pydantic to define the data model. To make visible
the entire process, we will manually create a basic model manager (`Model` and
`Field`).
"""

from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@dataclass
class Field:
    """
    A field to extract from a document.

    `validator` is a function that validates the field and raises an exception if it's not
    valid.
    """

    name: str
    description: str
    validator: Callable[[any], None] = lambda _: None

    def __str__(self) -> str:
        return f"'{self.name}': '{self.description}'"


@dataclass
class Model:
    fields: list[Field]


talk = Model(
    fields=[
        Field(name="title", description="The title of the talk"),
        Field(name="speaker", description="The name of the speaker"),
        Field(name="links", description="The links mentioned in the talk"),
        Field(
            name="technologies", description="The technologies mentioned in the talk"
        ),
    ]
)


def extract_fields_prompt(model: Model, doc: str) -> dict[str, any]:
    """
    Generates a prompt (perhaps excessively ironic) to extract fields from
    a document.
    """
    return f"""You are an expert information extractor. As a child, you dreamed of
this job. Now you can make it a reality. The future of humanity depends on it.
Plus, if you do it well, you'll get a tip of 100k€.

# Document

{doc}


# Fields

Fields to extract: {model.fields}.

Use a "```json" block to return the fields.
"""
