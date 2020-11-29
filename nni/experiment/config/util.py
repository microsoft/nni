# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Miscellaneous utility functions.
"""

from pathlib import Path

from .base import ConfigBase

PathLike = Union[Path, str]

def case_insensitive(key: str) -> str:
    return key.lower().replace('_', '')

def camel_case(key: str) -> str:
    words = key.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def absolute_path(config: ConfigBase, field: str, create=False) -> None:
    path = Path(getattr(config, field))
    if not path.is_absolute():
        if config._file is None:
            path = path.resolve()
        else:
            path = (config._file.parent / path).resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
        path = path.resolve()  # Path.resolve() does not work for non-exist path on windows
    setattr(config, field, str(path))
