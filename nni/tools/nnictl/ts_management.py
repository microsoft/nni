import importlib
import json

from nni.runtime.config import get_config_file
from .common_utils import print_error, print_green

def register(args):
    try:
        module = importlib.import_module(args.package)
    except Exception:
        print_error(f'Cannot import package {args.package}')
        return

    try:
        info = module.nni_training_service_info
    except Exception:
        print_error(f'Cannot read nni_training_service_info from {args.package}')
        return

    try:
        info.config_class()
    except Exception:
        print_error('Bad experiment config class')
        return

    try:
        service_config = {
            'node_module_path': info.node_module_path,
            'node_class_name': info.node_class_name,
        }
        json.dumps(service_config)
    except Exception:
        print_error('Bad node_module_path or bad node_class_name')
        return

    config = _load()
    update = args.package in config

    config[args.package] = service_config
    _save(config)

    if update:
        print_green(f'Sucessfully updated {arg.package}')
    else:
        print_green(f'Sucessfully registered {arg.package}')

def unregister(args):
    config = _load()
    if args.package not in config:
        print_error(f'{args.package} is not a registered training service')
        return
    config.pop(args.package, None)
    _save(config)
    print_green(f'Sucessfully unregistered {args.package}')

def list_services(_):
    print('\n'.join(_load().keys()))

def _load():
    return json.load(get_config_file.open())

def _save(config):
    json.dump(config, get_config_file.open('w'), indent=4)
