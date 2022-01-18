"""
Extract an archive created by pack_dependencies.py.
"""

import json
import os
from pathlib import Path
import shutil
import site
from zipfile import ZipFile

def main() -> None:
    extract_all(ZipFile('cache.zip'))
    empty_dirs = json.loads(Path('directories.json').read_text())
    symlinks = json.loads(Path('symlinks.json').read_text())
    for dir_ in empty_dirs:
        Path(dir_).mkdir(parents=True, exist_ok=True)
    for link, target in symlinks.items():
        Path(link).symlink_to(target)  # hopefully nobody uses symlink on windows

    site_packages = Path(site.getusersitepackages())
    assert not site_packages.exists()
    site_packages.parent.mkdir(parents=True, exist_ok=True)

    script_dir = get_user_script_directory()
    assert not script_dir.exists()
    script_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.move('cache/python-dependencies', site_packages)
    shutil.move('cache/python-scripts', script_dir)
    shutil.move('cache/nni-manager-dependencies', 'ts/nni_manager/node_modules')
    shutil.move('cache/webui-dependencies', 'ts/webui/node_modules')

def extract_all(zf: ZipFile) -> None:
    # fix a bug in ZipFile.extractall()
    # see https://stackoverflow.com/questions/39296101
    for info in zf.infolist():
        path = zf.extract(info)
        if info.external_attr > 0xffff:
            os.chmod(path, info.external_attr >> 16)

def get_user_script_directory() -> Path:
    # strangely this is not a site api
    if sys.platform == 'win32':
        return Path(site.getuserbase(), f'Python{sys.version_info.major}{sys.version_info.minor}', 'Scripts')
    else:
        return Path(site.getuserbase(), 'bin')

if __name__ == '__main__':
    main()
