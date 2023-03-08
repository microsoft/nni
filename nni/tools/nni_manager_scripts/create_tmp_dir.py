import os
from pathlib import Path
import sys
import tempfile

def main() -> None:
    exp_id = sys.argv[1]
    env_id = sys.argv[2]

    if sys.platform != 'win32':
        os.umask(0)

    dir_ = Path(tempfile.gettempdir(), 'nni', exp_id, env_id)
    dir_.mkdir(parents=True, exist_ok=True)

    print(dir_)

if __name__ == '__main__':
    main()
