# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from random import Random
import string
import re


def generate_experiment_id() -> str:
    return ''.join(Random().sample(string.ascii_lowercase + string.digits, 8))


def is_valid_experiment_id(experiment_id: str) -> bool:
    return re.match(r'^[A-Za-z0-9_\-]{1,32}$', experiment_id) is not None


def create_experiment_directory(experiment_id: str) -> Path:
    path = Path.home() / 'nni-experiments' / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# TODO: port shangning's work here, and use it in Experiment.start()/.stop()
