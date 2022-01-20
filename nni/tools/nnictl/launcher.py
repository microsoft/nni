# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from getpass import getuser
import logging
from pathlib import Path
import tempfile

from colorama import Fore
import yaml

from nni.experiment import Experiment, RunMode
from nni.experiment.config import ExperimentConfig, convert, utils
from nni.runtime.log import init_logger_for_command_line
from nni.tools.annotation import expand_annotations, generate_search_space

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

    # it should finally be done in nnictl main function
    # but for now don't break routines without logging support
    init_logger_for_command_line()
    logging.getLogger('nni').setLevel(logging.INFO)

    if not config_file.is_file():
        _logger.error(f'"{config_file}" is not a valid file.')
        exit(1)

    with config_file.open() as config:
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

    _logger.info(f'To stop experiment run "nnictl stop {exp.id}" or "nnictl stop --all"')
    _logger.info('Reference: https://nni.readthedocs.io/en/stable/Tutorial/Nnictl.html')

def resume_experiment(args):
    exp_id = args.id
    port = args.port
    debug = args.debug
    foreground = args.foreground
    exp_dir = args.experiment_dir

    init_logger_for_command_line()
    logging.getLogger('nni').setLevel(logging.INFO)

    config_json = get_stopped_experiment_config_json(exp_id, exp_dir)
    if config_json.get('trainingServicePlatform'):
        legacy_launcher.resume_experiment(args)
        exit()

    exp = Experiment._resume(exp_id, exp_dir)
    run_mode = RunMode.Foreground if foreground else RunMode.Detach
    exp.start(port, debug, run_mode)

def view_experiment(args):
    exp_id = args.id
    port = args.port
    exp_dir = args.experiment_dir

    init_logger_for_command_line()
    logging.getLogger('nni').setLevel(logging.INFO)

    config_json = get_stopped_experiment_config_json(exp_id, exp_dir)
    if config_json.get('trainingServicePlatform'):
        legacy_launcher.view_experiment(args)
        exit()

    exp = Experiment._view(exp_id, exp_dir)
    exp.start(port, run_mode=RunMode.Detach)
