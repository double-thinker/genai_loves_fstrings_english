# GenAI ❤️ f-string: Developing with Generative AI without black boxes

[Spanish version here](https://github.com/double-thinker/genai_loves_fstrings)

[Slides](https://raw.githubusercontent.com/double-thinker/genai_loves_fstrings_english/main/slides/talk.pdf)

## TLDR

Generative AI frameworks generate "framework-lock" very quickly, and sometimes it's not worth it. It can be reimplemented in Python in a more readable, debuggable, and maintainable way. In my consulting experiences, I see how technical debt and "lock-in" outweigh the advantages of frameworks.

In this workshop, we'll see through examples how to implement moderately complex pipelines without using frameworks, understanding their advantages and disadvantages.

[Workshop given at PyConES 2024](https://pretalx.com/pycones-2024/talk/SKZFHY/).

## Description

Generative AI frameworks are too "magical": they hide details in seemingly complex pipelines. Simple operations like text interpolations (an `f-string`, that is) are called `StringConcatenationManagerDeluxe` (dramatization).

These tools accelerate development in the initial phases but at a price: maintenance, observability, and independence suffer. There are increasingly more voices in the community ([[1]](https://hamel.dev/blog/posts/prompt/), [[2]](https://github.com/langchain-ai/langchain/discussions/18876)) that are beginning to make this problem explicit.

This workshop aims to summarize my experience developing products with LLMs to avoid "framework-lock" and make developments easier to extend, migrate, and debug:

- Part 1: "Framework-lock": Is it worth it? How to migrate?
- Part 2: Show me the prompt: dissecting current frameworks
- Part 3: From scratch: best practices for developing complex applications

If you're developing applications with LLMs and notice that you're "fighting" with your framework, or if you're about to start developing with Generative AI and want to save yourself quite a few headaches, this workshop might interest you.

## Instructions

### Development Environment

To carry out the workshop, you can use the already configured devcontainer in GitHub Codespaces without installing anything else (recommended):

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/double-thinker/genai_loves_fstrings_english/?quickstart=1)

To activate the virtual environment and run the examples:

```bash
source .venv/bin/activate
# python -m ...
```

If you prefer to use your favorite editor, you can clone the repository and use `uv` to install the environment:

```bash
git clone https://github.com/double-thinker/genai_loves_fstrings
cd genai_loves_fstrings
uv sync
source .venv/bin/activate
```

### OpenAI Account

Add your API key to the `.env` file. You have an example in `.env.example`.
You'll need between 1 and 3€ in credits to complete the workshop.

## Activities

Although you can see the result of the activities in `solved/`, I recommend that you read the [talk presentation](slides/talk.pdf) in parallel and do the activities step by step.

The solutions have iterations that you can see in the files `v1.py`, `v2.py`, etc.

### 01. RAG Example

RAG is considered the "Hello world" of Generative AI. Taking a RAG example from the LangChain documentation, we'll see the limitations and reimplement it in a way that is more readable and easy to maintain.

In `rag.py` you'll find the unmodified RAG example and in `solved/rag` the solution.

```bash
python -m rag

# Solutions
python -m solved.rag.v1
python -m solved.rag.v2
```

Example:

```
Human: What is Task Decomposition?
Chatbot: Task Decomposition is the process of breaking down complex tasks into smaller, more manageable steps. This can be achieved through techniques like Chain of Thought (CoT) prompting, which encourages the model to think step by step, and Tree of Thoughts, which explores multiple reasoning possibilities at each step. The decomposition can be guided by simple prompts, task-specific instructions, or human input.
```

You can try with different questions:

```bash
python -m rag "What is RAG?"
python -m rag "What is a chatbot?"
```

### 02. `SmartLLMChain`

Now we'll analyze `SmartLLMChain` by intercepting calls to the OpenAI API as suggested in [Hamel's article](https://hamel.dev/blog/posts/prompt/) but without needing a proxy that intercepts calls like `mitmproxy`.

For this, the `observability` module is provided, which shows the content of calls to the OpenAI API and their result in the terminal.

Knowing how it works, we'll reimplement `SmartLLMChain` without using `langchain` to compare the advantages and disadvantages of both approaches.

Solution:

```bash
python -m solved.smartllm.v2 "What is the meaning of life?"
```

```
The researcher thought that the flaws in Idea 1 were less significant compared to Idea 2.

Improved Answer:
The meaning of life is a complex and deeply philosophical question that has been debated for centuries. While different individuals and cultures may have unique perspectives on the purpose of life, it is essential to consider both subjective beliefs and objective truths in exploring this topic. One possible perspective is that the meaning of life is to seek happiness, fulfillment, and personal growth, while others may find purpose in serving others, making a positive impact on society, or pursuing spiritual enlightenment. It is crucial to reflect on one's values, goals, and aspirations to find a sense of purpose and direction in life. Ultimately, living a meaningful and intentional life can lead to a sense of fulfillment and satisfaction, regardless of external circumstances.
```

### 03. Information Extraction with Validators

Finally, we'll perform a case of information extraction from documents. There are several ways to do this: using the API of a provider that supports function calls (for example, Anthropic or OpenAI), using [constrained search](https://huggingface.co/blog/constrained-beam-search) which requires control over how inference is performed (some providers don't support it), or using prompt engineering and validation logic.

We'll use this last strategy to show how more complex flows can be implemented directly with Python without the need for frameworks.

```bash
python -m solved.extractor.v5
```

It generates a dictionary with the data extracted from the document:

```python
{'links': [{'description': 'Discussion about the problem of AI frameworks',
            'url': 'https://hamel.dev/blog/posts/prompt/'},
           {'description': 'Discussion about framework-lock',
            'url': 'https://github.com/langchain-ai/langchain/discussions/18876'},
           {'description': 'Introductory tutorial on LLM using langchain',
            'url': 'https://python.langchain.com/v0.1/docs/use_cases/question_answering/'}],
 'speaker': 'Alejandro Vidal',
 'technologies': ['OpenAI', 'LLMs', 'langchain'],
 'title': 'GenAI❤️f-string. Developing with Generative AI without black boxes.'}
```

## Important

- We're not going to use best practices to build the prompts. The goal is to compare implementations from scratch vs frameworks.
- Similarly, the code structure is designed for teaching during the workshop and is not the most suitable for productive development.
