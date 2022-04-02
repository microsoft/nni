"""
Create a zip archive named sys.argv[1], containing python-packages and node_modules.
Use unpack_dependencies.py to extract the archive.

Example usage:

    python test/vso_tools/pack_dependencies.py dependencies.zip
"""

import json
import os
from pathlib import Path
import shutil
import site
import sys
from zipfile import ZIP_DEFLATED, ZipFile

def main() -> None:
    cache = Path('cache')
    cache.mkdir()
    shutil.move('python-packages', 'cache/python-dependencies')
    shutil.move('ts/nni_manager/node_modules', 'cache/nni-manager-dependencies')
    shutil.move('ts/webui/node_modules', 'cache/webui-dependencies')

    archive = ZipFile(sys.argv[1], 'w', ZIP_DEFLATED, compresslevel=9)
    symlinks = {}
    empty_dirs = set()
    for file in sorted(cache.rglob('*')):
        if file.parent.parent == cache or file.parent.name == 'site-packages':
            print('Compress', file, flush=True)
        if file.is_symlink():
            symlinks[str(file)] = os.readlink(file)  # file.readlink() was added in Python 3.9
            continue
        if file.is_dir():
            empty_dirs.add(str(file))
        else:
            archive.write(file)
        empty_dirs.discard(str(file.parent))
    archive.writestr('symlinks.json', json.dumps(symlinks, indent=4))
    archive.writestr('directories.json', json.dumps(list(empty_dirs), indent=4))
    archive.close()

if __name__ == '__main__':
    main()
