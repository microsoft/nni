from collections import defaultdict

#import ruamel.yaml as yaml
import json

_global_counters: 'Dict[str, int]' = defaultdict(int)

def global_counter(name: str) -> int:
    _global_counters[name] += 1
    return _global_counters[name]

def uuid() -> int:
    return global_counter('uuid')


def import_(target: str, allow_none: bool = False) -> 'Any':
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


_experiment_config = None


def register_experiment_config(exp) -> None:
    global _experiment_config
    assert _experiment_config is None
    _experiment_config = exp

def experiment_config() -> 'Any':
    global _experiment_config
    return _experiment_config


def snake_to_camel(string: str) -> str:
    if string.islower():
        return ''.join(word.title() for word in string.split('_'))
    else:
        return string
