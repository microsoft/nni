# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from nni.common.blob_utils import NNI_BLOB, nni_cache_home, load_or_download_file


PRETRAINED_WEIGHT_URLS = {
    # proxylessnas
    'acenas-m1': f'{NNI_BLOB}/nashub/acenas-m1-e215f1b8.pth',
    'acenas-m2': f'{NNI_BLOB}/nashub/acenas-m2-a8ee9e8f.pth',
    'acenas-m3': f'{NNI_BLOB}/nashub/acenas-m3-66a5ed7b.pth',

}


def load_pretrained_weight(name: str, **kwargs) -> str:
    if name not in PRETRAINED_WEIGHT_URLS:
        raise ValueError(f'"{name}" is not a valid pretrained weight file.')
    url = PRETRAINED_WEIGHT_URLS[name]

    local_path = os.path.join(nni_cache_home(), 'nashub', url.split('/')[-1])
    load_or_download_file(local_path, url, **kwargs)
    return local_path
