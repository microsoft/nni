"""
This is to keep Chinese doc update to English doc. Should be run regularly.
The files in whitelist will be kept unchanged, as they will be translated manually.

Under docs, run

    python tools/chineselink.py
"""

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

print(content_tables)
whitelist += content_tables

for path in walk(Path('en_US')):
    relative_path = path.relative_to('en_US')
    if relative_path.as_posix().startswith('_build'):
        continue
    if path.suffix in ('.html', '.md', '.rst'):
        target_path = (Path('zh_CN') / relative_path)
        if relative_path.as_posix() in whitelist:
            print(f'Skipped linking for {path}')
        else:
            target_path.unlink(missing_ok=True)
            target_path.parent.mkdir(exist_ok=True)
            link_path = path
            for _ in range(len(list(Path(relative_path).parents))):
                link_path = Path('..') / link_path
            target_path.symlink_to(link_path)

# delete redundant files
for path in walk(Path('zh_CN')):
    if path.suffix in ('.html', '.md', '.rst'):
        relative_path = path.relative_to('zh_CN')
        if not (Path('en_US') / relative_path).exists():
            print(f'Deleting {path}')
            path.unlink()
