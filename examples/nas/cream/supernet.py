import os
import argparse
import time
import numpy as np
import logging
import torch.nn as nn
from datetime import datetime
from copy import deepcopy

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

from dataset import Dataset, create_loader, resolve_data_config
from models.hypernet import _gen_supernet
from utils.flops_table import LatencyEst
from utils.helpers import *
from utils.EMA import ModelEma
from utils.saver import CheckpointSaver
from utils.loss import LabelSmoothingCrossEntropy
from utils.scheduler import create_scheduler
from torch.utils.tensorboard import SummaryWriter

from nni.nas.pytorch.cream import CreamSupernetTrainer
from nni.nas.pytorch.random import RandomMutator

logger = logging.getLogger("nni.cream.supernet")


def add_weight_decay_supernet(model, args, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    meta_layer_no_decay = []
    meta_layer_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'meta_layer' in name:
                meta_layer_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if 'meta_layer' in name:
                meta_layer_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0., 'lr': args.lr},
        {'params': decay, 'weight_decay': weight_decay, 'lr': args.lr},
        {'params': meta_layer_no_decay, 'weight_decay': 0., 'lr': args.meta_lr},
        {'params': meta_layer_decay, 'weight_decay': 0, 'lr': args.meta_lr},
    ]

def create_optimizer_supernet(args, model, filter_bias_and_bn=True):
    from torch import optim as optim
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        weight_decay /= args.lr
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay_supernet(model, args, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, weight_decay=weight_decay, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, weight_decay=weight_decay, eps=args.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Training')
    # Dataset / Model parameters
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--model', default='hypernet', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                        help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                        help='Dropout rate (default: 0.)')
    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='spos_linear', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--grad', type=int, default=1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const',
                        help='Random erase mode (default: "const")')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='label smoothing (default: 0.1)')
    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--update_iter", default=1, type=int)
    parser.add_argument("--slice", default=4, type=int)
    parser.add_argument("--pool_size", default=10, type=int)
    parser.add_argument('--resunit', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--dil_conv', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--tiny', action='store_true', default=False)
    parser.add_argument('--flops_maximum', default=600, type=int)
    parser.add_argument('--flops_minimum', default=0, type=int)
    parser.add_argument('--pick_method', default='meta', type=str)
    parser.add_argument('--meta_lr', default=1e-2, type=float)
    parser.add_argument('--meta_sta_epoch', default=-1, type=int)
    parser.add_argument('--how_to_prob', default='pre_prob', type=str)
    parser.add_argument('--pre_prob', default=(0.05, 0.2, 0.05, 0.5, 0.05, 0.15), type=tuple)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed. Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        import random
        port = random.randint(0, 50000)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # tcp://127.0.0.1:{}'.format(port), rank=args.local_rank, world_size=8)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    model, sta_num, size_factor = _gen_supernet(
        flops_minimum=args.flops_minimum,
        flops_maximum=args.flops_maximum,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        global_pool=args.gp,
        resunit=args.resunit,
        dil_conv=args.dil_conv,
        slice=args.slice)

    if args.local_rank == 0:
        print("Model Searched Using FLOPs {}".format(size_factor * 32))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    if args.local_rank == 0:
        logger.info(args)

    choice_num = 6
    if args.resunit:
        choice_num += 1
    if args.dil_conv:
        choice_num += 2

    if args.local_rank == 0:
        logger.info("Choice_num: {}".format(choice_num))

    model_est = LatencyEst(model)

    if args.local_rank == 0:
        logger.info('Model %s created, param count: %d' %
                    (args.model, sum([m.numel() for m in model.parameters()])))

    # data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # optionally resume from a checkpoint
    optimizer_state = None
    resume_epoch = None
    if args.resume:
        optimizer_state, resume_epoch = resume_checkpoint(model, args.resume)

    if args.num_gpu > 1:
        if args.amp:
            logging.warning(
                'AMP does not work well with nn.DataParallel, disabling. Use distributed mode for multi-GPU AMP.')
            args.amp = False
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    optimizer = create_optimizer_supernet(args, model)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state['optimizer'])

    if args.distributed:
        if has_apex:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logger.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            model = DDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(num_epochs))

    if args.tiny:
        from dataset.tiny_imagenet import get_newimagenet
        [loader_train, loader_eval], [train_sampler, test_sampler] = get_newimagenet(args.data, args.batch_size)
    else:
        train_dir = os.path.join(args.data, 'train')
        if not os.path.exists(train_dir):
            logger.error('Training folder does not exist at: {}'.format(train_dir))
            exit(1)
        dataset_train = Dataset(train_dir)

        collate_fn = None

        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            re_prob=args.reprob,
            re_mode=args.remode,
            color_jitter=args.color_jitter,
            interpolation='random',  # FIXME cleanly resolve this? data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
        )

        eval_dir = os.path.join(args.data, 'val')
        if not os.path.isdir(eval_dir):
            logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
        dataset_eval = Dataset(eval_dir)

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=4 * args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
        )

    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    val_loss = nn.CrossEntropyLoss().cuda()

    mutator = RandomMutator(model)

    trainer = CreamSupernetTrainer(model, criterion, optimizer, args.epochs,
                                   train_loader=loader_train, valid_loader=loader_eval,
                                   mutator=mutator, batch_size=args.batch_size,
                                   log_frequency=args.log_interval, est=model_est, meta_sta_epoch=args.meta_sta_epoch,
                                   update_iter=args.update_iter, slices=args.slice, pool_size=args.pool_size,
                                   pick_method=args.pick_method, lr_scheduler=lr_scheduler, distributed=args.distributed,
                                   local_rank=args.local_rank, val_loss=val_loss)
    trainer.train()


if __name__ == '__main__':
    main()

