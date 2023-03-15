# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Create the working directory for trial keeper and print its absolute path.

By default the directory is "~/nni-experiments/EXP_ID/environments/ENV_ID/trial_keeper".
If the directory already exists (e.g. resume), try "trial_keeper_1", then "trial_keeper_2", etc.

Usage:

    python -m nni.tools.nni_manager_scripts.create_trial_keeper_dir EXP_ID ENV_ID
"""

from pathlib import Path
import sys

def main() -> None:
    exp_id = sys.argv[1]
    env_id = sys.argv[2]

    exps_dir = Path.home() / 'nni-experiments'  # TODO: aware of config file and env var
    env_dir = exps_dir / exp_id / 'environments' / env_id

    trial_keeper_dir = env_dir / 'trial_keeper'
    i = 0
    while trial_keeper_dir.exists():
        i += 1
        trial_keeper_dir = env_dir / f'trial_keeper_{i}'

    trial_keeper_dir.mkdir(parents=True, exist_ok=True)
    print(trial_keeper_dir)

if __name__ == '__main__':
    main()
