import os
import shutil
from pathlib import Path


for root, dirs, files in os.walk('archive_en_US'):
    root = Path(root)
    for file in files:
        moved_root = Path('en_US') / root.relative_to('archive_en_US')
        shutil.move(root / file, moved_root / file)
        os.remove(moved_root / (Path(file).stem + '.rst'))
