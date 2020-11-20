import argparse
import m2r
import os
import re
import pathlib


def single_line_process(line):
    # https://github.com/sphinx-doc/sphinx/issues/3921
    return re.sub(r'(`.*? <.*?>`)_', r'\1__', line)

for root, dirs, files in os.walk('en_US'):
    for file in files:
        if not file.endswith('.md'):
            continue
        out = m2r.parse_from_file((pathlib.Path(root) / file).as_posix())
        lines = out.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == '.. code-block:: eval_rst':
                space_count = line.index('.')
                lines[i] = lines[i + 1] = None
                i += 2
                while i < len(lines) and lines[i].startswith(' ' * (space_count + 2)):
                    lines[i] = lines[i][space_count:]
                    i += 1
            elif line.strip() == '.. code-block' or line.strip() == '.. code-block::':
                lines[i] += ':: bash'
                i += 1
            else:
                i += 1

        lines = [l for l in lines if l is not None]

        lines = list(map(single_line_process, lines))

        out = '\n'.join(lines)

        if 'eval_rst' in out:
            import pdb; pdb.set_trace()
        with open(pathlib.Path(root) / (pathlib.Path(file).stem + '.rst'), 'w') as f:
            f.write(out)
        os.remove(pathlib.Path(root) / file)
