"""
Extract an archive created by pack_dependencies.py.
"""

import json
from pathlib import Path
import shutil
import site
from zipfile import ZipFile

def main():
    ZipFile('cache.zip').extractall()
    empty_dirs = json.loads(Path('directories.json').read_text())
    symlinks = json.loads(Path('symlinks.json').read_text())
    for dir_ in empty_dirs:
        Path(dir_).mkdir(parents=True, exist_ok=True)
    for link, target in symlinks.items():
        Path(link).symlink_to(target)  # hopefully nobody uses symlink on windows

    site_packages = Path(site.getusersitepackages())
    assert not site_packages.exists()
    site_packages.parent.mkdir(parents=True, exist_ok=True)
    shutil.move('cache/python-dependencies', site_packages)
    shutil.move('cache/nni-manager-dependencies', 'ts/nni_manager/node_modules')
    shutil.move('cache/webui-dependencies', 'ts/webui/node_modules')

if __name__ == '__main__':
    main()
