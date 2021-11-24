# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

from colorama import Fore
import yaml

from nni.experiment import Experiment, RunMode
from nni.experiment.config import ExperimentConfig, convert

def create_experiment(args):
    config_file = Path(args.config)
    port = args.port
    debug = args.debug
    url_prefix = args.url_prefix
    foreground = args.foreground

    if not config_file.is_file():
        print(Fore.RED + 'ERROR: "{config_file}" is not a valid file.' + Fore.RESET)
        exit(1)

    with config_file.open() as config:
        config_content = yaml.safe_load(config)

    v1_platform = config_content.get('trainingServicePlatform')
    if v1_platform == 'adl':
        from . import legacy_launcher
        legacy_launcher.create_experiment(args)
        exit()

    if v1_platform:
        try:
            v2_config = convert.to_v2(config_content)
        except Exception:
            print(Fore.RED + 'ERROR: You are using legacy config file, please update it to latest format.' + Fore.RESET)
            print(Fore.RED + 'Reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html' + Fore.RESET)
            exit(1)
        print(Fore.YELLOW + f'WARNING: You are using legacy config file, please update it to latest format:' + Fore.RESET)
        print(Fore.YELLOW + '=' * 80 + Fore.RESET)
        print(yaml.dump(v2_config).strip())
        print(Fore.YELLOW + '=' * 80 + Fore.RESET)
        print(Fore.YELLOW + 'Reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html' + Fore.RESET)
        config = ExperimentConfig(**v2_config)
        print(config)
        print(config.json())
    else:
        config = ExperimentConfig.load(config_file)

    exp = Experiment(config)
    exp.url_prefix = url_prefix
    run_mode = RunMode.Foreground if foreground else RunMode.Detach
    exp.start(port, debug, run_mode)

def resume_experiment(args):
    exp_id = args.id
    port = args.port
    debug = args.debug
    foreground = args.foreground
    exp_dir = args.experiment_dir

    exp = Experiment._resume(exp_id, exp_dir)
    run_mode = RunMode.Foreground if foreground else RunMode.Detach
    exp.start(port, debug, run_mode)

def view_experiment(args):
    exp_id = args.id
    port = args.port
    exp_dir = args.experiment_dir

    exp = Experiment._view(exp_id, exp_dir)
    exp.start(port, run_mode=RunMode.Detach)
