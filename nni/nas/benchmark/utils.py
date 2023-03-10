# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import json
import os

from playhouse.sqlite_ext import SqliteExtDatabase

from nni.common.blob_utils import load_or_download_file

from .constants import DB_URLS, DATABASE_DIR


json_dumps = functools.partial(json.dumps, sort_keys=True)

# to prevent repetitive loading of benchmarks
_loaded_benchmarks = {}


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
    try:
        load_or_download_file(local_path, url)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Please use `nni.nas.benchmark.download_benchmark("{benchmark}")` to setup the benchmark first before using it.'
        )
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
