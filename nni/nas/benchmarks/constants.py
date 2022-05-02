# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from nni.common.blob_utils import NNI_BLOB, nni_cache_home


ENV_NASBENCHMARK_DIR = 'NASBENCHMARK_DIR'


def _get_nasbenchmark_dir():
    nni_home = nni_cache_home()
    return os.getenv(ENV_NASBENCHMARK_DIR, os.path.join(nni_home, 'nasbenchmark'))


DATABASE_DIR = _get_nasbenchmark_dir()

DB_URLS = {
    'nasbench101': f'{NNI_BLOB}/nasbenchmark/nasbench101-209f5694.db',
    'nasbench201': f'{NNI_BLOB}/nasbenchmark/nasbench201-b2b60732.db',
    'nds': f'{NNI_BLOB}/nasbenchmark/nds-5745c235.db'
}
