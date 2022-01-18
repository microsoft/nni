"""
Create an archive containing user site-packages and node_modules, in directory `sys.argv[1]`.
This should only be used in a disposable environment.
Use unpack_dependencies.py to extract the archive.
"""

import json
from pathlib import Path
import shutil
import site
import sys
from zipfile import ZIP_DEFLATED, ZipFile

def main():
    cache = Path('cache')
    cache.mkdir()
    #shutil.move(site.getusersitepackages(), 'cache/python-dependencies')
    shutil.move('ts/nni_manager/node_modules', 'cache/nni-manager-dependencies')
    #shutil.move('ts/webui/node_modules', 'cache/webui-dependencies')

    archive = ZipFile('cache.zip', 'w', ZIP_DEFLATED, compresslevel=9)
    symlinks = {}
    empty_dirs = set()
    for file in sorted(cache.rglob('*')):
        if file.parent == cache:
            print('Compress', file)
        empty_dirs.discard(str(file.parent))
        if file.is_symlink():
            symlinks[str(file)] = str(file.readlink())
        elif file.is_dir():
            empty_dirs.add(str(file))
        else:
            archive.write(file)
    archive.writestr('symlinks.json', json.dumps(symlinks, indent=4))
    archive.writestr('directories.json', json.dumps(list(empty_dirs), indent=4))

    assert Path(sys.argv[1]).is_dir()
    shutil.move('cache.zip', sys.argv[1])

if __name__ == '__main__':
    main()
