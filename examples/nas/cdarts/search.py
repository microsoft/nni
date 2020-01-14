# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import utils
from config import SearchConfig
from datasets.cifar import get_search_datasets
from model import Model
from nni.nas.pytorch.cdarts import CdartsTrainer

if __name__ == "__main__":
    config = SearchConfig()
    main_proc = not config.distributed or config.local_rank == 0
    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=config.dist_url,
                                             rank=config.local_rank, world_size=config.world_size)
    if main_proc:
        os.makedirs(config.output_path, exist_ok=True)
    if config.distributed:
        torch.distributed.barrier()
    logger = utils.get_logger(os.path.join(config.output_path, 'search.log'))
    if main_proc:
        config.print_params(logger.info)
    utils.reset_seed(config.seed)

    loaders, samplers = get_search_datasets(config)
    model_small = Model(config.dataset, 8).cuda()
    if config.share_module:
        model_large = Model(config.dataset, 20, shared_modules=model_small.shared_modules).cuda()
    else:
        model_large = Model(config.dataset, 20).cuda()

    criterion = nn.CrossEntropyLoss()
    trainer = CdartsTrainer(model_small, model_large, criterion, loaders, samplers, logger, config)
    trainer.train()
