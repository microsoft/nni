# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from getpass import getuser
from pathlib import Path
import tempfile

from colorama import Fore
import yaml

from nni.experiment import Experiment, RunMode
from nni.experiment.config import ExperimentConfig, convert, utils
from nni.tools.annotation import expand_annotations, generate_search_space

def create_experiment(args):
    # to make it clear what are inside args
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
        utils.set_base_path(config_file.parent)
        config = ExperimentConfig(**v2_config)
        utils.unset_base_path()
    else:
        config = ExperimentConfig.load(config_file)

    if config.use_annotation:
        path = Path(tempfile.gettempdir(), getuser(), 'nni', 'annotation')
        path.mkdir(parents=True, exist_ok=True)
        path = tempfile.mkdtemp(dir=path)
        code_dir = expand_annotations(config.trial_code_directory, path)
        config.trial_code_directory = code_dir
        config.search_space = generate_search_space(code_dir)
        assert config.search_space, 'ERROR: Generated search space is empty'
        config.use_annotation = False

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
