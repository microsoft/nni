from pathlib import Path
import random
import string


def generate_experiment_id() -> str:
    return ''.join(random.sample(string.ascii_lowercase + string.digits, 8))


def create_experiment_directory(experiment_id: str) -> Path:
    path = Path.home() / 'nni-experiments' / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# TODO: port shangning's work here, and use it in Experiment.start()/.stop()
