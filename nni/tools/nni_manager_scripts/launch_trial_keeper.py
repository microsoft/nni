import json
import os
from pathlib import Path
from subprocess import Popen
import sys
from typing import Dict

import nni_node

Config = Dict[str, str]

def main() -> None:
    print('## stdout debug', flush=True)
    print('## stderr debug', file=sys.stderr, flush=True)
    init_dir = Path(sys.argv[1])
    config = load_config(init_dir)
    output_dir = create_output_dir(config)
    proc = launch_trial_keeper(config, output_dir)
    save_result(init_dir, output_dir, proc)

def load_config(init_dir: Path) -> Config:
    config_json = (init_dir / 'config.json').read_text('utf_8')
    config = json.loads(config_json)
    config['experimentsDirectory'] = str(Path.home() / 'nni-experiments')  # TODO: use global config
    config['pythonInterpreter'] = sys.executable
    return config

def create_output_dir(config: Config) -> Path:
    env_dir = Path(config['experimentsDirectory'], config['experimentId'], 'environments', config['environmentId'])
    output_dir = env_dir / 'trial_keeper'
    i = 0
    while output_dir.exists():
        i += 1
        output_dir = env_dir / f'trial_keeper_{i}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def launch_trial_keeper(config: Config, output_dir: Path) -> Popen:
    config_path = output_dir / 'config.json'
    config_json = json.dumps(config, ensure_ascii=False, indent=4)
    config_path.write_text(config_json)

    stdout = (output_dir / 'stdout.txt').open('a')
    stderr = (output_dir / 'stderr.txt').open('a')

    node_dir = Path(nni_node.__path__[0])  # type: ignore
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'common/trial_keeper/main.js')
    cmd = [node, '--max-old-space-size=4096', '--trace-uncaught', main_js, str(config_path)]

    # TODO: cwd must be node_dir, or trial_keeper will not work (because of app-module-path/cwd)
    if sys.platform == 'win32':
        from subprocess import CREATE_NEW_PROCESS_GROUP
        return Popen(cmd, stdout=stdout, stderr=stderr, cwd=node_dir, creationflags=CREATE_NEW_PROCESS_GROUP)
    else:
        return Popen(cmd, stdout=stdout, stderr=stderr, cwd=node_dir, preexec_fn=os.setpgrp)  # type: ignore

def save_result(init_dir: Path, output_dir: Path, proc: Popen) -> None:
    result = {'ouputDir': str(output_dir), 'pid': proc.pid}
    result_json = json.dumps(result, ensure_ascii=False, indent=4)
    (init_dir / 'launch.json').write_text(result_json, 'utf_8')
    (output_dir / 'launch.json').write_text(result_json, 'utf_8')

if __name__ == '__main__':
    main()
