# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os

class Recoverable:

    def load_checkpoint(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def get_checkpoint_path(self) -> str | None:
        ckp_path = os.getenv('NNI_CHECKPOINT_DIRECTORY')
        if ckp_path is not None and os.path.isdir(ckp_path):
            return ckp_path
        return None
