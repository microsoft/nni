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
    if line == ' .. contents::':
        return '.. contents::'
    # https://github.com/sphinx-doc/sphinx/issues/3921
    line = re.sub(r'(`.*? <.*?>`)_', r'\1__', line)
    # inline emphasis
    line = re.sub(r'\*\*\\ (.*?)\\ \*\*', r' **\1** ', line)
    line = re.sub(r'\*(.*?)\\ \*', r'*\1*', line)
    line = re.sub(r'\*\*(.*?) \*\*', r'**\1** ', line)
    line = re.sub(r'\\\*\\\*(.*?)\*\*', r'**\1**', line)
    line = re.sub(r'\\\*\\\*(.*?)\*\*\\ ', r'**\1**', line)
    line = line.replace(r'\* - `\**', r'* - `**')
    line = re.sub(r'\\\* \*\*(.*?)\*\* \(\\\*\s*(.*?)\s*\*\\ \)', r'* \1 (\2)', line)
    line = re.sub(r'\<(.*)\.md(\>|#)', r'<\1\2', line)

    # special case (per line handling)
    line = line.replace('Nb = |Db|', r'Nb = \|Db\|')
    line = line.replace('  Here is just a small list of libraries ', '\nHere is just a small list of libraries ')
    line = line.replace('  Find the data management region in job submission page.', 'Find the data management region in job submission page.')
    line = line.replace('Tuner/InstallCustomizedTuner.md', 'Tuner/InstallCustomizedTuner')
    line = line.replace('&#10003;', ':raw-html:`&#10003;`')
    # line = line.replace('\* **optimize_mode** ', '* **optimize_mode** ')
    if line == '~' * len(line):
        line = '^' * len(line)
    return line


def special_case_replace(full_text):
    replace_pairs = {}
    replace_pairs['PyTorch\n"""""""'] = '**PyTorch**'
    replace_pairs['Search Space\n============'] = '.. role:: raw-html(raw)\n   :format: html\n\nSearch Space\n============'
    for file in os.listdir(Path(__file__).parent / 'patches'):
        with open(Path(__file__).parent / 'patches' / file) as f:
            r, s = f.read().split('%%%%%%\n')
        replace_pairs[r] = s
    for r, s in replace_pairs.items():
        full_text = full_text.replace(r, s)
    return full_text


def process_table(content):
    content = content.replace('------ |', '------|')
    lines = []
    for line in content.split('\n'):
        if line.startswith('  |'):
            line = line[2:]
        lines.append(line)
    return '\n'.join(lines)


for root, dirs, files in os.walk('en_US'):
    root = Path(root)
    for file in files:
        if not file.endswith('.md') or file == 'Release_v1.0.md':
            continue

        with open(root / file) as f:
            md_content = f.read()

        if file == 'Nnictl.md':
            md_content = process_table(md_content)

        out = m2r.convert(md_content)
        lines = out.split('\n')
        if lines[0] == '':
            lines = lines[1:]

        # remove code-block eval_rst
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == '.. code-block:: eval_rst':
                space_count = line.index('.')
                lines[i] = lines[i + 1] = None
                if i > 0 and lines[i - 1]:
                    lines[i] = ''  # blank line
                i += 2
                while i < len(lines) and (lines[i].startswith(' ' * (space_count + 3)) or lines[i] == ''):
                    lines[i] = lines[i][space_count + 3:]
                    i += 1
            elif line.strip() == '.. code-block' or line.strip() == '.. code-block::':
                lines[i] += ':: bash'
                i += 1
            else:
                i += 1

        lines = [l for l in lines if l is not None]

        lines = list(map(single_line_process, lines))

        out = '\n'.join(lines)
        out = special_case_replace(out)

        with open(root / (Path(file).stem + '.rst'), 'w') as f:
            f.write(out)

        # back it up and remove
        moved_root = Path('archive_en_US') / root.relative_to('en_US')
        moved_root.mkdir(exist_ok=True)
        shutil.move(root / file, moved_root / file)
