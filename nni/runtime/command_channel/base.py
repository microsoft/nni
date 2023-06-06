# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any

Command = Any  # TODO

class CommandChannel:
    def send(self, command: Command) -> None:
        raise NotImplementedError()

    def receive(self) -> Command | None:
        raise NotImplementedError()
