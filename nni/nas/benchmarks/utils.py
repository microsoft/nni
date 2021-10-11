import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import requests
import tqdm
from playhouse.sqlite_ext import SqliteExtDatabase

from .constants import DB_URLS, DATABASE_DIR


json_dumps = functools.partial(json.dumps, sort_keys=True)

# to prevent repetitive loading of benchmarks
_loaded_benchmarks = {}


def load_or_download_file(local_path: str, download_url: str, download: bool = False, progress: bool = True):
    f = None
    hash_prefix = Path(local_path).stem.split('-')[-1]

    _logger = logging.getLogger(__name__)

    try:
        sha256 = hashlib.sha256()

        if Path(local_path).exists():
            _logger.info('"%s" already exists. Checking hash.', local_path)
            with Path(local_path).open('rb') as fr:
                while True:
                    chunk = fr.read(8192)
                    if len(chunk) == 0:
                        break
                    sha256.update(chunk)
        elif download:
            _logger.info('"%s" does not exist. Downloading "%s"', local_path, download_url)

            # Follow download implementation in torchvision:
            # We deliberately save it in a temp file and move it after
            # download is complete. This prevents a local working checkpoint
            # being overridden by a broken download.
            dst_dir = Path(local_path).parent
            dst_dir.mkdir(exist_ok=True, parents=True)

            f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
            r = requests.get(download_url, stream=True)
            total_length = int(r.headers.get('content-length'))
            with tqdm.tqdm(total=total_length, disable=not progress,
                           unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    sha256.update(chunk)
                    pbar.update(len(chunk))
                    f.flush()
        else:
            raise FileNotFoundError('Download is not enabled, but file still does not exist: {}'.format(local_path))

        digest = sha256.hexdigest()
        if not digest.startswith(hash_prefix):
            raise RuntimeError('Invalid hash value (expected "{}", got "{}")'.format(hash_prefix, digest))

        if f is not None:
            shutil.move(f.name, local_path)
    finally:
        if f is not None:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)


def load_benchmark(benchmark: str) -> SqliteExtDatabase:
    """
    Load a benchmark as a database.

    Parmaeters
    ----------
    benchmark : str
        Benchmark name like nasbench201.
    """
    if benchmark in _loaded_benchmarks:
        return _loaded_benchmarks[benchmark]
    url = DB_URLS[benchmark]
    local_path = os.path.join(DATABASE_DIR, os.path.basename(url))
    load_or_download_file(local_path, url)
    _loaded_benchmarks[benchmark] = SqliteExtDatabase(local_path, autoconnect=True)
    return _loaded_benchmarks[benchmark]


def download_benchmark(benchmark: str, progress: bool = True):
    """
    Download a converted benchmark.

    Parameters
    ----------
    benchmark : str
        Benchmark name like nasbench201.
    """
    url = DB_URLS[benchmark]
    local_path = os.path.join(DATABASE_DIR, os.path.basename(url))
    load_or_download_file(local_path, url, True, progress)
