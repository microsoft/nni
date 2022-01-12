"""
This is to keep Chinese doc update to English doc. Should be run regularly.
The files in whitelist will be kept unchanged, as they will be translated manually.

Under docs, run

    python tools/chineselink.py
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path


def walk(path):
    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p


# Keeps files as discussed in
# https://github.com/microsoft/nni/issues/4298
# Not the recommended way of sphinx though: https://docs.readthedocs.io/en/stable/guides/manage-translations-sphinx.html

whitelist = [
    '_templates/index.html',  # I think no one ever remembers to update this file. Might need to rethink about this.
    'Overview.rst',
    'installation.rst',
    'Tutorial/InstallationLinux.rst',
    'Tutorial/InstallationWin.rst',
    'Tutorial/QuickStart.rst',
    'TrialExample/Trials.rst',
    'Tutorial/WebUI.rst',
    'NAS/QuickStart.rst',
    'Compression/Overview.rst',
    'Compression/QuickStart.rst',
]

suffix_list = [
    '.html',
    '.md',
    '.rst',
    '.ipynb',
]

for path in whitelist:
    assert (Path('zh_CN') / path).exists(), path

content_tables = []
for path in walk(Path('en_US')):
    if path.suffix == '.rst':
        is_content_table = False
        for line in path.open('r').readlines():
            if is_content_table:
                if not line.startswith('  ') and line.strip():
                    is_content_table = False
            if 'toctree::' in line:
                is_content_table = True
        if is_content_table:
            content_tables.append(path.relative_to('en_US').as_posix())

print('Whitelist:' ,content_tables)
whitelist += content_tables

pipeline_mode = len(sys.argv) > 1 and sys.argv[1] == 'check'
failed_files = []


def need_to_translate(source, target):
    if not target.exists():
        failed_files.append('(missing) ' + target.as_posix())
        if pipeline_mode:
            return
        shutil.copyfile(source, target)
    if target.suffix == '.html':
        return  # FIXME I don't know how to process html
    target_checksum = hashlib.sha256(path.open('rb').read()).hexdigest()[:32]
    checksum = target.open('r').readline().strip()[3:]
    if checksum != target_checksum:
        failed_files.append('(out-of-date) ' + target.as_posix())
        if pipeline_mode:
            return
    contents = target.open('r').readlines()
    firstline = '.. ' + target_checksum + '\n'
    if contents[0].startswith('.. '):
        contents = [firstline] + contents[1:]
    else:
        contents = [firstline, '\n'] + contents
    target.open('w').writelines(contents)


for path in walk(Path('en_US')):
    relative_path = path.relative_to('en_US')
    if relative_path.as_posix().startswith('_build'):
        continue
    if path.suffix in suffix_list:
        target_path = (Path('zh_CN') / relative_path)
        if relative_path.as_posix() in whitelist:
            # whitelist files. should be translated
            need_to_translate(path, target_path)
            print(f'Skipped linking for {path} as it is in whitelist.')
        else:
            target_path.parent.mkdir(exist_ok=True)
            link_path = path
            for _ in range(len(list(Path(relative_path).parents))):
                link_path = Path('..') / link_path
            if not target_path.is_symlink() or os.readlink(target_path) != link_path.as_posix():
                failed_files.append('(invalid link) ' + target_path.as_posix())
                if not pipeline_mode:
                    target_path.unlink(missing_ok=True)
                    target_path.symlink_to(link_path)

# delete redundant files
for path in walk(Path('zh_CN')):
    if path.suffix in suffix_list:
        relative_path = path.relative_to('zh_CN')
        if not (Path('en_US') / relative_path).exists():
            failed_files.append('(redundant) ' + path.as_posix())
            if not pipeline_mode:
                print(f'Deleting {path}')
                path.unlink()


if pipeline_mode and failed_files:
    raise ValueError(
        'The following files are not up-to-date. Please run "python3 tools/chineselink.py" under docs folder '
        'to refresh them and update their corresponding translation.\n' + '\n'.join(['  ' + line for line in failed_files]))
if failed_files:
    print('Updated files:', failed_files)
