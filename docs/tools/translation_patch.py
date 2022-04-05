"""
Fix a troublsome translation in sphinx.
Related PR: https://github.com/sphinx-doc/sphinx/pull/10303
"""

import subprocess
from pathlib import Path

import sphinx

sphinx_path = Path(sphinx.__path__[0]) / 'locale/zh_CN/LC_MESSAGES'
po_content = (sphinx_path / 'sphinx.po').read_text()
po_content = po_content.replace('%s的别名', '%s 的别名')
(sphinx_path / 'sphinx.po').write_text(po_content)

# build po -> mo
subprocess.run(['msgfmt', '-c', str(sphinx_path / 'sphinx.po'), '-o', str(sphinx_path / 'sphinx.mo')], check=True)
