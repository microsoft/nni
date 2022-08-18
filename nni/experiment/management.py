# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import random
import string


def generate_experiment_id() -> str:
    try:
        # This try..catch is for backward-compatibility,
        # in case shortuuid is not installed for some reason.
        import shortuuid
        return shortuuid.ShortUUID(alphabet=string.ascii_lowercase + string.digits).random(length=8)
    except ImportError:
        # shortuuid is not installed, use legacy random string instead
        return ''.join(random.sample(string.ascii_lowercase + string.digits, 8))


def create_experiment_directory(experiment_id: str) -> Path:
    path = Path.home() / 'nni-experiments' / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# TODO: port shangning's work here, and use it in Experiment.start()/.stop()
