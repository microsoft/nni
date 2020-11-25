from pathlib import Path
import random
import string


def _generate_experiment_id() -> str:
    return ''.join(random.sample(string.ascii_lowercase + string.digits, 8))


def _create_experiment_path(experiment_id: str) -> Path:
    path = Path.home() / 'nni-experiments' / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


class _SimpleFileLock:
    ...

