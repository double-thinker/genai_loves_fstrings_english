import sys
from functools import wraps
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
from typing import Any, Callable


class InterceptLoader(Loader):
    def __init__(
        self,
        fullname,
        real_spec,
        path: str,
        interceptions: dict[str, Callable[[Any], Any]],
    ):
        self.fullname = fullname
        self.real_spec = real_spec
        self.path = path
        self.interceptions = interceptions

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        real_module = self.real_spec.loader.load_module(self.fullname)

        def wrap_attr(attr, path):
            if callable(attr):
                this_path = path + "()"

                if this_path in self.interceptions:
                    return self.interceptions[this_path](attr, real_module)

                if not any(
                    key.startswith(this_path) for key in self.interceptions.keys()
                ):
                    return attr

                @wraps(attr)
                def wrapped(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return wrap_attr(result, this_path)

                return wrapped

            else:
                if not any(key.startswith(path) for key in self.interceptions.keys()):
                    return attr

                return AttrWrapper(attr, path, self.interceptions)

        class AttrWrapper:
            def __init__(self, attr, path, intercepting):
                self.attr = attr
                self.path = path
                self.intercepting = intercepting

            def __getattr__(self, name):
                attr = getattr(self.attr, name)
                path = f"{self.path}.{name}"
                if path in self.intercepting:
                    return self.intercepting[path](attr, self.attr)
                return wrap_attr(attr, path)

        for attr_name in dir(real_module):
            attr = getattr(real_module, attr_name)
            setattr(module, attr_name, wrap_attr(attr, f"{self.path}{attr_name}"))


class InterceptFinder(MetaPathFinder):
    def __init__(self, interceptions: dict[str, Callable[[Any], Any]]):
        super().__init__()
        self.interceptions = interceptions
        self.intercepting = set()
        self.fullname_to_intercept = set(
            key.split(":")[0] for key in interceptions.keys()
        )

    def find_spec(self, fullname, path, target=None):
        if fullname in self.fullname_to_intercept and fullname not in self.intercepting:
            self.intercepting.add(fullname)
            spec = find_spec(fullname)
            self.intercepting.remove(fullname)
            return spec_from_loader(
                fullname,
                InterceptLoader(
                    fullname,
                    spec,
                    path=f"{fullname}:",
                    interceptions=self.interceptions,
                ),
            )
        return None


def patch(patches: dict[str, Callable[[Any], Any]]):
    """
    Patch the import system to intercept calls to the functions specified in
    `patches` in this format:

    ```python
    {
        "openai:OpenAI().chat.completions.create": patch_openai_create
    }
    ```

    The signature of `patch_openai_create` is `(original_fn, module)` where
    `original_fn` is the function that was intercepted and `module` is the
    module that contains the function.

    You can use `()` to intercept the return value of a callable.
    """
    sys.meta_path.insert(
        0,
        InterceptFinder(patches),
    )
