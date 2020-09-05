#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import errno
import hashlib
import logging
import os

import torch

logger = logging.getLogger(__name__)


def download_file(url, model_dir=None, progress=True):
    """ Download url to `model_dir`
        Append hash of the url to the filename to make it unique.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        logger.warning(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        torch_home = torch.hub._get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + "_" + filename
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        torch.hub.download_url_to_file(
            url, cached_file, None, progress=progress
        )

    return cached_file
