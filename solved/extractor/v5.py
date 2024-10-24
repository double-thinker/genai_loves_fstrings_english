"""
v5: Validation of the `technologies` field.

Now we'll use an LLM to validate the `technologies` field with criteria
that require processing natural language and reasoning about the content (see
`validate_techs`).
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


def fix_fields_prompt(
    model: Model,
    doc: str,
    parsed: dict[str, any],
    validation_errors: list[tuple[str, Exception]],
) -> str:
    return f"""You are an expert information extractor. You need to correct the
extraction errors that occurred in the following document:

# Document

{doc}

# Previous extraction

{parsed}

# Extraction errors

{'\n'.join(f"- {field}: {e}" for field, e in validation_errors)}

# Corrected extraction

```json
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


def validate_output(
    output: str, model: Model
) -> tuple[dict[str, any], list[tuple[str, Exception]]]:
    """
    We parse and validate the LLM output.

    Returns a tuple with the parsed dictionary and a list of validation
    errors if any.
    """
    parsed = parse_json_block(output)
    validation_errors = []

    for field in model.fields:
        try:
            field.validator(parsed[field.name])
        except Exception as e:
            validation_errors.append((field.name, e.args[0]))

    return parsed, validation_errors


def extractor(model: Model, doc: str, max_retries: int = 3) -> dict[str, any] | None:
    parsed, validation_errors = None, []
    for _ in range(max_retries):
        if validation_errors:
            prompt = fix_fields_prompt(model, doc, parsed, validation_errors)
        else:
            prompt = extract_fields_prompt(model, doc)

        output = llm(prompt)
        parsed, validation_errors = validate_output(output, model)
        if not validation_errors:
            return parsed
    return None


def validate_links(links: list[dict[str, str]]) -> None:
    for link in links:
        if not isinstance(link, dict):
            raise ValueError(
                "Links must be a dictionary with the following structure: {{'url': str, 'description': str}}"
            )
        if not link.get("url"):
            raise ValueError(f"No link (`url`) provided in {link}")
        if not link.get("description"):
            raise ValueError(f"No description (`description`) provided in {link}")


def validate_techs(techs: list[str]) -> None:
    output = parse_json_block(
        llm(f"""From the following list of tags: {techs} verify that the following conditions are met:
- Acronyms are described in parentheses, for example: `IPv6 (Internet Protocol Version 6)`
- They are written in English

Show the errors in a list formatted as a JSON array with the following format:
                     
```json
[
    "Not in English: Programación en Python, ...",
    ...
]
```

If there are no errors, return an empty list.

# Errors

```json
[""")
    )

    if output:
        raise ValueError(output)


talk = Model(
    fields=[
        Field(name="title", description="The title of the talk"),
        Field(name="speaker", description="The name of the speaker"),
        Field(
            name="links",
            description="The links mentioned in the talk",
            validator=validate_links,
        ),
        Field(
            name="technologies",
            description="The technologies mentioned in the talk",
            validator=validate_techs,
        ),
    ]
)

if __name__ == "__main__":
    doc = get("https://pretalx.com/pycones-2024/talk/SKZFHY.ics").text
    pprint(extractor(talk, doc))
