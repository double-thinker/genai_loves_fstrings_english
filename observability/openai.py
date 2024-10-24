from functools import wraps

from termcolor import colored

from .patch import patch


def logged_competion(fn, _):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for msg in kwargs.get("messages", []):
            lines = msg["content"].split("\n")
            for line in lines:
                print(colored(f"> {line}", "green"))

        result = fn(*args, **kwargs)
        lines = result.choices[0].message.content.split("\n")
        for line in lines:
            print(colored(f"< {line}", "blue", attrs=["bold"]))

        print("\n\n\n")

        return result

    return wrapper


patch({"openai:OpenAI().chat.completions.create": logged_competion})
