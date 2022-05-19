"""
Unpacked the downloaded zipped datasets.
Opposite to ``pack_testdata.py``.
"""

import argparse
import glob
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data', type=str)

    args = parser.parse_args()

    for zipfile in glob.glob(os.path.join(args.data_dir, '*.zip')):
        print(f'Unpacking {zipfile}')
        shutil.unpack_archive(zipfile, extract_dir=args.data_dir)


if __name__ == '__main__':
    main()
