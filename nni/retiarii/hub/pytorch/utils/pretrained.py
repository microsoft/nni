# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from nni.common.blob_utils import NNI_BLOB, nni_cache_home, load_or_download_file


PRETRAINED_WEIGHT_URLS = {
    # proxylessnas
    'acenas-m1': ''

}


def load_pretrained_weight(name: str) -> str:
    if name not in PRETRAINED_WEIGHT_URLS:
        raise ValueError(f'"{name}" is not a valid pretrained weight file.')
    url = PRETRAINED_WEIGHT_URLS[name]

    local_path = os.path.join(nni_cache_home(), 'nashub')
    load_or_download_file(local_path, url)
    return local_path
