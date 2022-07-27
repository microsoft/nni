# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Weights available in this file are processed with scripts in https://github.com/ultmaster/spacehub-conversion,
and uploaded with :func:`nni.common.blob_utils.upload_file`.
"""

import os

from nni.common.blob_utils import NNI_BLOB, nni_cache_home, load_or_download_file


PRETRAINED_WEIGHT_URLS = {
    # proxylessnas
    'acenas-m1': f'{NNI_BLOB}/nashub/acenas-m1-e215f1b8.pth',
    'acenas-m2': f'{NNI_BLOB}/nashub/acenas-m2-a8ee9e8f.pth',
    'acenas-m3': f'{NNI_BLOB}/nashub/acenas-m3-66a5ed7b.pth',
    'proxyless-cpu': f'{NNI_BLOB}/nashub/proxyless-cpu-2df03430.pth',
    'proxyless-gpu': f'{NNI_BLOB}/nashub/proxyless-gpu-dbe6dd15.pth',
    'proxyless-mobile': f'{NNI_BLOB}/nashub/proxyless-mobile-8668a978.pth',

    # mobilenetv3
    'mobilenetv3-large-100': f'{NNI_BLOB}/nashub/mobilenetv3-large-100-420e040a.pth',
    'mobilenetv3-small-050': f'{NNI_BLOB}/nashub/mobilenetv3-small-050-05cb7a80.pth',
    'mobilenetv3-small-075': f'{NNI_BLOB}/nashub/mobilenetv3-small-075-c87d8acb.pth',
    'mobilenetv3-small-100': f'{NNI_BLOB}/nashub/mobilenetv3-small-100-8332faac.pth',
    'cream-014': f'{NNI_BLOB}/nashub/cream-014-060aea24.pth',
    'cream-043': f'{NNI_BLOB}/nashub/cream-043-bec949e1.pth',
    'cream-114': f'{NNI_BLOB}/nashub/cream-114-fc272590.pth',
    'cream-287': f'{NNI_BLOB}/nashub/cream-287-a0fcba33.pth',
    'cream-481': f'{NNI_BLOB}/nashub/cream-481-d85779b6.pth',
    'cream-604': f'{NNI_BLOB}/nashub/cream-604-9ee425f7.pth',

    # nasnet
    'darts-v2': f'{NNI_BLOB}/nashub/darts-v2-5465b0d2.pth',

    # spos
    'spos': f'{NNI_BLOB}/nashub/spos-0b17f6fc.pth',

    # autoformer
    'autoformer-tiny': f'{NNI_BLOB}/nashub/autoformer-searched-tiny-1e90ebc1.pth',
    'autoformer-small': f'{NNI_BLOB}/nashub/autoformer-searched-small-4bc5d4e5.pth',
    'autoformer-base': f'{NNI_BLOB}/nashub/autoformer-searched-base-c417590a.pth'
}


def load_pretrained_weight(name: str, **kwargs) -> str:
    if name not in PRETRAINED_WEIGHT_URLS:
        raise ValueError(f'"{name}" do not have a valid pretrained weight file.')
    url = PRETRAINED_WEIGHT_URLS[name]

    local_path = os.path.join(nni_cache_home(), 'nashub', url.split('/')[-1])
    load_or_download_file(local_path, url, **kwargs)
    return local_path
