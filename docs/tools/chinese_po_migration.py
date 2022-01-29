"""
To migrate the old chinese translation into new ones.
"""

from pathlib import Path

whitelist = [
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
    english_content = (Path('en_US') / path).read_text().strip().splitlines()
    chinese_content = (Path('zh_CN') / path).read_text().strip().splitlines()
    print(path, len(english_content), len(chinese_content))
