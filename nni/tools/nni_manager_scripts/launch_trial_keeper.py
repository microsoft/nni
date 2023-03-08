import os
from pathlib import Path
from subprocess import Popen
import sys

def debug() -> Popen:
    import nni_node
    node_dir = Path(nni_node.__path__[0])  # type: ignore
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'common/trial_keeper/main.js')
    cmd = [node, '--max-old-space-size=4096', '--trace-uncaught', main_js, sys.argv[1]]

    return Popen(cmd, cwd=node_dir)  # type: ignore

debug()
