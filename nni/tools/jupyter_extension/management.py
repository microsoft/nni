import json
from pathlib import Path
import shutil
import os

from jupyter_core.paths import jupyter_config_dir, jupyter_data_dir

import nni_node

def _get_jupyter_lab_version():
    try:
        import jupyterlab
        return jupyterlab.__version__
    except ImportError:
        return '3.x'

jupyter_lab_major_version = _get_jupyter_lab_version().split('.')[0]

_backend_config_file = Path(jupyter_config_dir(), 'jupyter_server_config.d', 'nni.json')
_backend_config_content = {
    'ServerApp': {
        'jpserver_extensions': {
            'nni.tools.jupyter_extension': True
        }
    }
}
_v2_backend_config_file = Path(jupyter_config_dir(), 'jupyter_notebook_config.d', 'nni.json')
_v2_backend_config_content = {
    "NotebookApp": {
        "nbserver_extensions": {
            "nni.tools.jupyter_extension": True
        }
    }
}

_frontend_src = Path(nni_node.__path__[0], 'jupyter-extension')
_frontend_dst = Path(jupyter_data_dir(), 'labextensions', 'nni-jupyter-extension')

def install():
    _backend_config_file.parent.mkdir(parents=True, exist_ok=True)
    _backend_config_file.write_text(json.dumps(_backend_config_content))

    _frontend_dst.parent.mkdir(parents=True, exist_ok=True)

    if jupyter_lab_major_version == '2':
        _v2_backend_config_file.parent.mkdir(parents=True, exist_ok=True)
        _v2_backend_config_file.write_text(json.dumps(_v2_backend_config_content))

        if (_frontend_src.is_symlink()):
            linkto = os.path.realpath(_frontend_src)
            os.symlink(linkto, _frontend_dst)
        else:
            shutil.copytree(_frontend_src, _frontend_dst)
    else:
        shutil.copytree(_frontend_src, _frontend_dst)

def uninstall():
    _backend_config_file.unlink()
    if jupyter_lab_major_version == '2':
        _v2_backend_config_file.unlink()
    shutil.rmtree(_frontend_dst)
