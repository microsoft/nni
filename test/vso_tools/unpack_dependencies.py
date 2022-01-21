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

    move_or_merge(Path('cache/python-dependencies'), Path(site.getuserbase()))
    shutil.move('cache/nni-manager-dependencies', 'ts/nni_manager/node_modules')
    shutil.move('cache/webui-dependencies', 'ts/webui/node_modules')

def extract_all(zf: ZipFile) -> None:
    # fix a bug in ZipFile.extractall()
    # see https://stackoverflow.com/questions/39296101
    for info in zf.infolist():
        path = zf.extract(info)
        if info.external_attr > 0xffff:
            os.chmod(path, info.external_attr >> 16)

def move_or_merge(src: Path | str, dst: Path | str) -> None:
    if dst.exists():
        assert dst.is_dir()
        for file in src.iterdir():
            move_or_merge(file, dst / file.name)
    else:
        shutil.move(src, dst)

if __name__ == '__main__':
    main()
