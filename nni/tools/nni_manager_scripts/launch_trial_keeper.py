import os
from pathlib import Path
from subprocess import Popen
import sys

def debug() -> Popen:
    import nni_node
    node_dir = Path(nni_node.__path__[0])  # type: ignore
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'common/trial_keeper/main.js')
    cmd = [node, '--max-old-space-size=4096', '--trace-uncaught', main_js]
    #cmd += nni_manager_args.to_command_line_args()
    cmd += [
        '--experiment-id', 'tk_debug',
        '--experiments-directory', '/home/lz/nni-experiments',
        '--log-level', 'debug',
        '--python-interpreter', sys.executable,
        '--platform', 'remote',
        '--environment-id', 'testenv',
        '--manager-command-channel', 'ws://localhost:8080/manager',
    ]

    return Popen(cmd, cwd=node_dir)  # type: ignore

    if run_mode.value == 'detach':
        log = Path(nni_manager_args.experiments_directory, nni_manager_args.experiment_id, 'log')
        out = (log / 'nnictl_stdout.log').open('a')
        err = (log / 'nnictl_stderr.log').open('a')
        header = f'Experiment {nni_manager_args.experiment_id} start: {datetime.now()}'
        header = '-' * 80 + '\n' + header + '\n' + '-' * 80 + '\n'
        out.write(header)
        err.write(header)

    else:
        out = None
        err = None

    if sys.platform == 'win32':
        from subprocess import CREATE_NEW_PROCESS_GROUP
        return Popen(cmd, stdout=out, stderr=err, cwd=node_dir, creationflags=CREATE_NEW_PROCESS_GROUP)
    else:
        return Popen(cmd, stdout=out, stderr=err, cwd=node_dir, preexec_fn=os.setpgrp)  # type: ignore

debug()
input()
