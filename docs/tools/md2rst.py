import argparse
import m2r
import os
import re
import shutil
from pathlib import Path

# FIXME:
# /home/v-yugzh/nnidev/docs/en_US/Assessor/BuiltinAssessor.rst:84: WARNING: Inline emphasis start-string without end-string.
# /home/v-yugzh/nnidev/docs/en_US/Compression/Overview.rst:5: WARNING: The "contents" directive may not be used within topics or body elements.
# /home/v-yugzh/nnidev/docs/en_US/Compression/Pruner.rst:223: WARNING: Block quote ends without a blank line; unexpected unindent.
# /home/v-yugzh/nnidev/docs/en_US/Compression/Pruner.rst:734: WARNING: Title level inconsistent:
# /home/v-yugzh/nnidev/docs/en_US/SupportedFramework_Library.rst:76: WARNING: Definition list ends without a blank line; unexpected unindent.
# /home/v-yugzh/nnidev/docs/en_US/Tuner/HyperbandAdvisor.rst:65: WARNING: Error parsing content block for the "list-table" directive: uniform two-level bullet list expected, but row 7 does not contain the same number of items as row 1 (5 vs 6).
# /home/v-yugzh/nnidev/docs/en_US/Tutorial/HowToUseDocker.rst:83: WARNING: Inline strong start-string without end-string.
# /home/v-yugzh/nnidev/docs/en_US/NAS/NasGuide.rst:52: WARNING: undefined label: /nas/advanced.md#extend-the-ability-of-one-shot-trainers (if the link has no caption the label must precede a section header)

def single_line_process(line):
    # https://github.com/sphinx-doc/sphinx/issues/3921
    return re.sub(r'(`.*? <.*?>`)_', r'\1__', line)


for root, dirs, files in os.walk('en_US'):
    root = Path(root)
    for file in files:
        if not file.endswith('.md'):
            continue

        out = m2r.parse_from_file((root / file).as_posix())
        lines = out.split('\n')
        if lines[0] == '\n':
            lines = lines[1:]

        # remove code-block eval_rst
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

        with open(root / (Path(file).stem + '.rst'), 'w') as f:
            f.write(out)

        # back it up and remove
        moved_root = Path('archive_en_US') / root.relative_to('en_US')
        moved_root.mkdir(exist_ok=True)
        shutil.move(root / file, moved_root / file)
