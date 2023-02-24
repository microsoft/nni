# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path

from colorama import Fore
import yaml

from nni.experiment import Experiment, RunMode
from nni.experiment.config import ExperimentConfig, convert, utils

# used for v1-only legacy setup, remove them later
from nni.experiment.launcher import get_stopped_experiment_config_json
from . import legacy_launcher

_logger = logging.getLogger(__name__)

def create_experiment(args):
    # to make it clear what are inside args
    config_file = Path(args.config)
    port = args.port
    debug = args.debug
    url_prefix = args.url_prefix
    foreground = args.foreground

    if not config_file.is_file():
        _logger.error(f'"{config_file}" is not a valid file.')
        exit(1)

    with config_file.open(encoding='utf_8') as config:
        config_content = yaml.safe_load(config)

    v1_platform = config_content.get('trainingServicePlatform')
    if v1_platform:
        can_convert = True
        if v1_platform == 'adl':
            can_convert = False
        if v1_platform in ['kubeflow', 'frameworkcontroller']:
            reuse = config_content.get(v1_platform + 'Config', {}).get('reuse')
            can_convert = (reuse != False)  # if user does not explicitly specify it, convert to reuse mode

        if not can_convert:
            legacy_launcher.create_experiment(args)
            exit()

        try:
            v2_config = convert.to_v2(config_content)
        except Exception:
            _logger.error(
                'You are using legacy config format with incorrect fields or values, '
                'to get more accurate error message please update it to the new format.'
            )
            _logger.error('Reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html')
            exit(1)
        _logger.warning(f'You are using legacy config file, please update it to latest format:')
        # use `print` here because logging will add timestamp and make it hard to copy paste
        print(Fore.YELLOW + '=' * 80 + Fore.RESET)
        print(yaml.dump(v2_config, sort_keys=False).strip())
        print(Fore.YELLOW + '=' * 80 + Fore.RESET)
        print(Fore.YELLOW + 'Reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html' + Fore.RESET)

        utils.set_base_path(config_file.parent)
        config = ExperimentConfig(**v2_config)
        utils.unset_base_path()

    else:
        config = ExperimentConfig.load(config_file)

    if config.use_annotation:
        _logger.error('You are using annotation to specify search space. This is not supported since NNI v3.0.')
        exit(1)

    exp = Experiment(config)
    exp.url_prefix = url_prefix

    if foreground:
        exp.start(port, debug, RunMode.Foreground)
        exp._wait_completion()

    else:
        exp.start(port, debug, RunMode.Detach)
        _logger.info(f'To stop experiment run "nnictl stop {exp.id}" or "nnictl stop --all"')
        _logger.info('Reference: https://nni.readthedocs.io/en/stable/reference/nnictl.html')

def resume_experiment(args):
    exp_id = args.id
    port = args.port
    debug = args.debug
    foreground = args.foreground
    exp_dir = args.experiment_dir

    # NOTE: Backward compatibility
    config_json = get_stopped_experiment_config_json(exp_id, exp_dir)
    if config_json.get('trainingServicePlatform'):
        legacy_launcher.resume_experiment(args)
        exit()

    config = ExperimentConfig(**config_json)
    if type(config) != ExperimentConfig:
        _logger.error('Non-HPO experiment cannot be resumed with nnictl. Please use experiment.resume() in Python API.')
        exit(1)

    experiment = Experiment(config, id=exp_id)
    # Do not need to call `load_checkpoint()` here as there is nothing to load.
    experiment._action = 'resume'
    # Can't use experiment.resume() here because resume() will automatically run in RunMode.Background,
    # and thus the NNI manager process will be killed once the main process exits.
    experiment.start(port, debug, RunMode.Foreground if foreground else RunMode.Detach)

def view_experiment(args):
    exp_id = args.id
    port = args.port
    exp_dir = args.experiment_dir

    # NOTE: Backward compatibility
    config_json = get_stopped_experiment_config_json(exp_id, exp_dir)
    if config_json.get('trainingServicePlatform'):
        legacy_launcher.view_experiment(args)
        exit()

    config = ExperimentConfig(**config_json)
    if type(config) != ExperimentConfig:
        _logger.warning(
            'Non-HPO experiment detected. '
            'Though `nnictl view` is designed to be agnostic to experiment types, it is only tested to view HPO experiments. '
            'Report an issue if you encounter any problem.'
        )

    Experiment(config, id=exp_id).view(port, non_blocking=True)  # non-blocking is in detach mode.
