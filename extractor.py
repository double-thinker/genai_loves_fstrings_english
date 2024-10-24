"""
We will implement an information extractor from a document (talks from
the PyConES 2024 event) with validators. Similar to `instructor`.

We won't use `pydantic` models to simplify the code and to show
the functioning of the entire process.

For this, we will extract the fields of each talk:

- `title`: title of the talk
- `speaker`: name of the speaker
- `links`: links mentioned in the talk
- `technologies`: technologies mentioned in the talk

There must be two validations of this data model:

- `links` must contain 'url' and 'description' for each link
- `technologies` must be in English and if there are acronyms, they should be in
   parentheses, for example: `RAG (Retrieval Augmented Generation)`

The first validation can be done programmatically. The second requires the
use of an LLM for semantic validation.
"""
