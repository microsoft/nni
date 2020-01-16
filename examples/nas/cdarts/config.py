# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from functools import partial


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        ########### basic settings ############
        parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
        parser.add_argument('--n_classes', type=int, default=10)
        parser.add_argument('--stem_multiplier', type=int, default=3)
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--data_dir', type=str, default='data/cifar', help='cifar dataset')
        parser.add_argument('--output_path', type=str, default='./outputs', help='')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--log_frequency', type=int, default=10, help='print frequency')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--steps_per_epoch', type=int, default=None, help='how many steps per epoch, use None for one pass of dataset')

        ########### learning rate ############
        parser.add_argument('--w_lr', type=float, default=0.05, help='lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4, help='weight decay for weights')
        parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--alpha_lr', type=float, default=6e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')
        parser.add_argument('--nasnet_lr', type=float, default=0.1, help='lr of nasnet')

        ########### alternate training ############
        parser.add_argument('--epochs', type=int, default=32, help='# of search epochs')
        parser.add_argument('--warmup_epochs', type=int, default=2, help='# warmup epochs of super model')
        parser.add_argument('--loss_alpha', type=float, default=1, help='loss alpha')
        parser.add_argument('--loss_T', type=float, default=2, help='loss temperature')
        parser.add_argument('--interactive_type', type=str, default='kl', choices=['kl', 'smoothl1'])
        parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to sync bn')
        parser.add_argument('--use_apex', action='store_true', default=False, help='whether to use apex')
        parser.add_argument('--regular_ratio', type=float, default=0.5, help='regular ratio')
        parser.add_argument('--regular_coeff', type=float, default=5, help='regular coefficient')
        parser.add_argument('--fix_head', action='store_true', default=False, help='whether to fix head')
        parser.add_argument('--share_module', action='store_true', default=False, help='whether to share stem and aux head')

        ########### data augument ############
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        parser.add_argument('--use_aa', action='store_true', default=False, help='whether to use aa')
        parser.add_argument('--mixup_alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')

        ########### distributed ############
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument("--world_size", default=1, type=int)
        parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
        parser.add_argument('--distributed', action='store_true', help='run model distributed mode')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


class RetrainConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Retrain config")
        parser.add_argument('--dataset', default="cifar10", choices=['cifar10', 'cifar100', 'imagenet'])
        parser.add_argument('--data_dir', type=str, default='data/cifar', help='cifar dataset')
        parser.add_argument('--output_path', type=str, default='./outputs', help='')
        parser.add_argument("--arc_checkpoint", default="epoch_02.json")
        parser.add_argument('--log_frequency', type=int, default=10, help='print frequency')

        ########### model settings ############
        parser.add_argument('--n_classes', type=int, default=10)
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--stem_multiplier', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--eval_batch_size', type=int, default=500, help='batch size for validation')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--warmup_epochs', type=int, default=5, help='# warmup')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
        parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path prob')

        ########### data augmentation ############
        parser.add_argument('--use_aa', action='store_true', default=False, help='whether to use aa')
        parser.add_argument('--mixup_alpha', default=1., type=float, help='mixup interpolation coefficient')

        ########### distributed ############
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument("--world_size", default=1, type=int)
        parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
        parser.add_argument('--distributed', action='store_true', help='run model distributed mode')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
