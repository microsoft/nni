import json
from pathlib import Path
import shutil

from jupyter_core.paths import jupyter_config_dir, jupyter_data_dir

import nni_node

_backend_config_file = Path(jupyter_config_dir(), 'jupyter_server_config.d', 'nni.json')
_backend_config_content = {
    'ServerApp': {
        'jpserver_extensions': {
            'nni.tools.jupyter_extension': True
        }
    }
}

_frontend_src = Path(nni_node.__path__[0], 'jupyter-extension')
_frontend_dst = Path(jupyter_data_dir(), 'labextensions', 'nni-jupyter-extension')

def install():
    _backend_config_file.parent.mkdir(parents=True, exist_ok=True)
    _backend_config_file.write_text(json.dumps(_backend_config_content))

    _frontend_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_frontend_src, _frontend_dst)

def uninstall():
    _backend_config_file.unlink()
    shutil.rmtree(_frontend_dst)
