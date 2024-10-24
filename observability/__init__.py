"""
This is a proof of concept for intercepting calls to a function and modifying
the arguments and return value. Used to intercept calls to OpenAI's API and add
debugging information.

It's not perfect, but it works for this session.

If you are interested in using this, please contact me. I will share a more
mature version as a library for LLM observability.
"""

import observability.openai  # noqa
