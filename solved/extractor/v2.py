"""
v2: We add the JSON block parser and process a first document
(this talk): https://pretalx.com/pycones-2024/talk/SKZFHY.ics
"""

import json
import re
from dataclasses import dataclass
from pprint import pprint
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from requests import get

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


JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*)\s*```")


def parse_json_block(output: str) -> dict[str, any]:
    """
    We process the LLM output by looking for a JSON markdown block.

    Trick: using a format like markdown makes parsing the response easier.
    LLMs are trained with many MD documents and work well with that
    structure and syntax (document mimicking).
    """
    match = JSON_BLOCK_RE.search(output)
    if not match:
        raise ValueError("No JSON block found")
    return json.loads(match.group(1))


def extractor(model: Model, doc: str) -> dict[str, any] | None:
    output = llm(extract_fields_prompt(model, doc))
    return parse_json_block(output)


if __name__ == "__main__":
    doc = get("https://pretalx.com/pycones-2024/talk/SKZFHY.ics").text
    pprint(extractor(talk, doc))
