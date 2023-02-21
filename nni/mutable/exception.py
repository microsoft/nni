# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['ConstraintViolation', 'SampleValidationError', 'SampleMissingError']

from typing import overload


class SampleValidationError(ValueError):
    """Exception raised when a sample is invalid."""

    def __init__(self, msg: str, paths: list[str] | None = None):
        super().__init__(msg)
        self.msg = msg
        self.paths: list[str] = paths or []

    def __str__(self) -> str:
        if self.paths:
            return self.msg + ' (path:' + ' -> '.join(map(str, self.paths)) + ')'
        else:
            return self.msg


class SampleMissingError(SampleValidationError):
    """Raised when a required sample with a particular label is missing."""

    @overload
    def __init__(self, label_or_msg: str, keys: list[str]) -> None: ...

    @overload
    def __init__(self, label_or_msg: str) -> None: ...

    def __init__(self, label_or_msg: str, keys: list[str] | None = None) -> None:
        if keys is None:
            super().__init__(label_or_msg)
        else:
            super().__init__(f'Label {label_or_msg} is missing from sample. Existing keys are: {keys}')


class ConstraintViolation(SampleValidationError):
    """Exception raised when constraint is violated."""
