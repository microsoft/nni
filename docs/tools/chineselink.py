"""
This is to keep Chinese doc update to English doc. Should be run regularly.
There is no sane way to check the contents though. PR review should enforce contributors to update the corresponding translation.
See https://github.com/microsoft/nni/issues/4298 for discussion.

Under docs, run

    python tools/chineselink.py
"""

import hashlib
import shutil
import sys
from pathlib import Path


def iterate_dir(path):
    for p in Path(path).iterdir():
        if p.is_dir():
            yield from iterate_dir(p)
            continue
        yield p

suffix_list = [
    '.html',
    '.md',
    '.rst',
    '.ipynb',
]

pipeline_mode = len(sys.argv) > 1 and sys.argv[1] == 'check'
failed_files = []

# in case I need to change `_zh` to something else
# files = list(filter(lambda d: d.name.endswith('zh_CN.rst'), iterate_dir('source')))
# for file in files:
#     os.rename(file, file.parent / (file.name[:-7] + file.name[-4:]))


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


for path in iterate_dir(Path('source')):
    relative_path = path.relative_to('source')
    if relative_path.as_posix().startswith('_build'):
        continue
    if path.suffix in suffix_list:
        if '_zh.' not in path.name:
            target_path = path.parent / (path.stem + '_zh' + path.suffix)
            if target_path.exists():
                # whitelist files. should be translated
                need_to_translate(path, target_path)
                print(f'Skipped linking for {path} as it is in whitelist.')
        else:
            source_path = path.parent / (path.stem[:-3] + path.suffix)
            if not source_path.exists():
                # delete redundant files
                failed_files.append('(redundant) ' + source_path.as_posix())
                if not pipeline_mode:
                    print(f'Deleting {source_path}')
                    source_path.unlink()


if pipeline_mode and failed_files:
    raise ValueError(
        'The following files are not up-to-date. Please run "python3 tools/chineselink.py" under docs folder '
        'to refresh them and update their corresponding translation.\n' + '\n'.join(['  ' + line for line in failed_files]))
if failed_files:
    print('Updated files:', failed_files)
