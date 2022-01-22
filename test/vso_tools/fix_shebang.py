"""
Change "#!" line to "#!/usr/bin/env python" for all files in directory sys.argv[1].
"""

from pathlib import Path
import sys

if sys.platform == 'win32':
    exit()

for file in Path(sys.argv[1]).iterdir():
    try:
        text = file.read_text()
        assert text.startswith('#!'), 'no shebang'
        shebang, content = text.split('\n', 1)
        assert 'python' in shebang, 'not python script'
        file.write_text('#!/usr/bin/env python\n' + content)
    except Exception as e:
        print(f'Skip {file}: {repr(e)}')
