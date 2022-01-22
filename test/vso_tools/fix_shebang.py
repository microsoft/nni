"""
Change "#!" line to "#!/usr/bin/env python" for all files in directory sys.argv[1].
"""

from pathlib import Path
import sys

for file in Path(sys.argv[1]).iterdir():
    if file.name == '__pycache__':
        continue
    try:
        text = file.read_text()
        assert text.startswith('#!'), 'no shebang'
        shebang, content = text.split('\n', 1)
        assert 'python' in shebang, 'not python script'
        file.write_text('#!/usr/bin/env python\n' + content)
    except Exception as e:
        print(f'Skip {file}: {repr(e)}')
