import argparse
import importlib
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nni.nas.pytorch.random import RandomMutator


from sdk.graph import Graph, _debug_dump_graph
from sdk.jit_optimzer.simple_strategy import (SimpleOptimizationStrategy,
                                              dump_multi_graph)
from sdk.trainer.dataloaders import fake_imagenet_dataloader
from sdk.trainer.imagenet_dataloader import (imagenet_loader_gpu,
                                             torch_dataloader)
from sdk.trainer.optimizers import pytorch_builtin_optimizers
from sdk.trainer.steps import (classifier_train_step,
                               classifier_validation_test_step)
from sdk.trainer.train import train
from apex.parallel import DistributedDataParallel, convert_syncbn_model

from utils import CrossEntropyLabelSmooth, accuracy, prepare_experiment, AverageMeterGroup

_logger = logging.getLogger(__name__)

class SPOSSupernetTrainingMutator(RandomMutator):
    def __init__(self, model, flops_lb=290E6, flops_ub=360E6,
                 flops_bin_num=7, flops_sample_timeout=500):

        super().__init__(model)

        with open("./data/op_flops_dict.pkl", "rb") as fp:
            self._op_flops_dict = pickle.load(fp)
        self._parsed_flops = {
            'LayerChoice1': [13396992, 15805440, 19418112, 13146112],
            'LayerChoice2': [7325696, 8931328, 11339776, 12343296],
            'LayerChoice3': [7325696, 8931328, 11339776, 12343296],
            'LayerChoice4': [7325696, 8931328, 11339776, 12343296],
            'LayerChoice5': [26304768, 28111104, 30820608, 20296192],
            'LayerChoice6': [10599680, 11603200, 13108480, 16746240],
            'LayerChoice7': [10599680, 11603200, 13108480, 16746240],
            'LayerChoice8': [10599680, 11603200, 13108480, 16746240],
            'LayerChoice9': [30670080, 31673600, 33178880, 21199360],
            'LayerChoice10': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice11': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice12': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice13': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice14': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice15': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice16': [10317440, 10819200, 11571840, 15899520],
            'LayerChoice17': [30387840, 30889600, 31642240, 20634880],
            'LayerChoice18': [10176320, 10427200, 10803520, 15476160],
            'LayerChoice19': [10176320, 10427200, 10803520, 15476160],
            'LayerChoice20': [10176320, 10427200, 10803520, 15476160]
        }
        self._flops_bin_num = flops_bin_num
        self._flops_bins = [
            flops_lb + (flops_ub - flops_lb) / flops_bin_num * i for i in range(flops_bin_num + 1)]
        self._flops_sample_timeout = flops_sample_timeout

    def get_candidate_flops(self, candidate):
        conv1_flops = self._op_flops_dict["conv1"][(3, 16, 224, 224, 2)]
        # Should use `last_conv_channels` here, but megvii insists that it's `n_classes`. Keeping it.
        # https://github.com/megvii-model/SinglePathOneShot/blob/36eed6cf083497ffa9cfe7b8da25bb0b6ba5a452/src/Supernet/flops.py#L313
        rest_flops = self._op_flops_dict["rest_operation"][(
            640, 1000, 7, 7, 1)]
        total_flops = conv1_flops + rest_flops
        for k, m in candidate.items():
            parsed_flops_dict = self._parsed_flops[k]
            if isinstance(m, dict):  # to be compatible with classical nas format
                total_flops += parsed_flops_dict[m["_idx"]]
            else:
                total_flops += parsed_flops_dict[torch.max(m, 0)[1]]
        return total_flops

    def sample_search(self):
        for times in range(self._flops_sample_timeout):
            idx = np.random.randint(self._flops_bin_num)
            cand = super().sample_search()
            if self._flops_bins[idx] <= self.get_candidate_flops(cand) <= self._flops_bins[idx + 1]:
                _logger.debug(
                    "Sampled candidate flops %f in %d times.", cand, times)
                return cand
        _logger.warning(
            "Failed to sample a flops-valid candidate within %d tries.", self._flops_sample_timeout)
        return super().sample_search()

    def sample_final(self):
        return self.sample_search()


def test_simple_strategy():
    with open('./examples/graphs/mnist_3.json') as fp:
        graph = Graph.load(json.load(fp))
        # graph_1 = prepare_for_jit(graph_1)
    graph.generate_code(framework='pytorch', output_file=f'out_spos.py')

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _verbose_arch(arch):
    return {k: torch.argmax(v.long()).item() for k, v in arch.items()}


def train_one_epoch(args, epoch, model, mutator, criterion, optimizer, train_loader):
    model.train()
    meters = AverageMeterGroup()
    mp_rank = args.rank % args.model_parallel
    _logger.info("Current learning rate: %.6f", optimizer.param_groups[0]["lr"])
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        # find the correct model to train
        for _ in range(mp_rank + 1):
            mutator.reset()
        # train the model
        x, y, logits = model(x, y)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        # iterate over the rest in a loop
        for _ in range(mp_rank + 1, args.model_parallel):
            mutator.reset()

        metrics = accuracy(logits, y)
        metrics["loss"] = loss.item()
        meters.update(metrics)
        if args.log_frequency is not None and (step % args.log_frequency == 0 or step + 1 == len(train_loader)):
            _logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                        args.epochs, step + 1, len(train_loader), meters)


def validate_one_epoch(args, epoch, model, mutator, criterion, valid_loader):
    model.eval()
    meters = AverageMeterGroup()
    with torch.no_grad():
        for step, (x, y) in enumerate(valid_loader):
            mutator.reset()
            x, y, logits = model(x, y)
            loss = criterion(logits, y)
            metrics = accuracy(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if args.log_frequency is not None and (step % args.log_frequency == 0 or step + 1 == len(valid_loader)):
                _logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                            args.epochs, step + 1, len(valid_loader), meters)

if __name__ == '__main__':
    # test_simple_strategy()
    parser = argparse.ArgumentParser("Imagenet Training")
    parser.add_argument("--imagenet-dir", type=str,
                        default="data/imagenet")
    parser.add_argument("--output-dir", type=str, default="./spos_ckpt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.125)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--enable-gpu-dataloader",
                        default=False, action="store_true")
    parser.add_argument("--num-threads", default=12, type=int)
    parser.add_argument("--model-parallel", default=1, type=int)
    parser.add_argument("--sync-bn", default=False, action="store_true")
    parser.add_argument("--job_id", default=0, type=int)
    parser.add_argument('--mix_para', default=False, action="store_true")
    parser.add_argument('--module_path', action="store",
                        dest="module_path", type=str)
    parser.add_argument('--profile_out', default="", type=str)
    parser.add_argument('--use_dali', default=False, action="store_true")
    parser.add_argument('--file_list', default="./partial_list.txt", type=str)
    parser.add_argument('--disable_shared', default=False, action="store_true")
    parser.add_argument('--use_fake_input', default=False, action="store_true")
    #parser.add_argument('--ckpt_dir', default=".", type=str)
    #parser.add_argument('--resume', default=False, action="store_true")
    #parser.add_argument('--eval', default=False, action="store_true")

    args = parser.parse_args()
    # reset_seed(args.seed)

    is_distributed = False
    if "is_distributed" in os.environ:
        is_distributed = True

        args.world_size = int(os.environ["world_size"])
        args.rank = int(os.environ["rank"])
        args.local_rank = int(os.environ["rank"])
        args.model_parallel = args.world_size

    prepare_experiment(args)
    #import generated.debug as out_spos
    mod = importlib.import_module(args.module_path)
    model_cls = mod.Graph

    def __callable_loader(args, mode, distributed, dali=False):
        is_train = True if mode == 'train' else False

        def _dali_dataloader(trainer):
            return imagenet_loader_gpu(args, mode, distributed=distributed)

        def _torch_dataloader(trainer):
            return torch_dataloader(args, train=is_train)
        if dali:
            return _dali_dataloader
        else:
            return _torch_dataloader

        # (args, "train", distributed=(args.model_parallel == 1 or (args.model_parallel > 1 and not args.shared_input)))
    if (not args.use_fake_input) or args.disable_shared:
        train_dataloader = __callable_loader(
            args, "train", is_distributed, dali=args.use_dali)
        val_dataloader = __callable_loader(
            args, "val", False, dali=args.use_dali)
        use_fake_input = False
    else:
        train_dataloader = fake_imagenet_dataloader(args.imagenet_dir, [args.batch_size, 3, 224, 224],
                                                    batch_size=args.batch_size, train=True)
        val_dataloader = fake_imagenet_dataloader(args.imagenet_dir, [args.batch_size, 3, 224, 224],
                                                   batch_size=args.batch_size, train=False)
        use_fake_input = True

    criterion = CrossEntropyLabelSmooth(1000, args.label_smoothing).cuda()
    #criterion = CrossEntropyLoss().cuda()

    def imagenet_optimizer(trainer):
        optimizer = optim.SGD(trainer.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda step: (
                                                    1.0 - step / args.epochs)
                                                if step <= args.epochs else 0,
                                                last_epoch=-1)

        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'epoch'
        }]

    def imagenet_train_step(trainer, batch, batch_idx, use_mix_para=False,
                            mutator=None, n_model_parallel=None, **kwargs):
        if use_mix_para:
            # skip networks of this batch
            for _ in range(n_model_parallel):
                mutator.reset()
        x, y = batch
        x, y, y_hat = trainer(x, y)
        _, predicted = torch.max(y_hat.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
        loss = criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'train_acc': accuracy}
    if not args.mix_para:
        train(model_cls, train_dataloader, val_dataloader, imagenet_optimizer,
            imagenet_train_step, classifier_validation_test_step, classifier_validation_test_step,
            use_fake_input=use_fake_input, use_mix_para=args.mix_para, mutator_cls=SPOSSupernetTrainingMutator,
            profile_out=args.profile_out)
    else:
        # FIXME: Pytorch Lightning introduces unproducability issue to SPOS
        # Fallback to use traditional training loop
        use_distributed = False
        mix_para_idx = 0
        n_model_parallel = 1
        self_rank = 0
        if 'is_distributed' in os.environ:
            use_distributed = True
            # torch.cuda.set_device(os.environ["rank"])
            world_size = int(os.environ["world_size"])
            backend = os.environ["distributed_backend"]
            self_rank = int(os.environ["rank"])
            torch.distributed.init_process_group(
                backend=backend, init_method=f'tcp://127.0.0.1:{12400}', rank=self_rank, world_size=world_size)
            torch.cuda.set_device(int(os.environ["DEVICE_ID"]))
            mix_para_idx = self_rank
            n_model_parallel = world_size

        model = model_cls()
        from op_libs.spos import ShuffleNetV2OneShot
        ShuffleNetV2OneShot._initialize_weights(model)
        model.cuda()
        if n_model_parallel > 1:
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            model = DistributedDataParallel(model, delay_allreduce=True)
            torch.distributed.barrier()

        mutator = SPOSSupernetTrainingMutator(model)

        optimizer, _ = imagenet_optimizer(model)
        optimizer = optimizer[0]
        scheduler = _[0]['scheduler']
        train_loader = train_dataloader(model)
        valid_loader = val_dataloader(model)
        for i in range(args.epochs):
            train_one_epoch(args, i, model, mutator, criterion, optimizer, train_loader)
            scheduler.step()
            #if (i + 1) % 5 == 0:
            validate_one_epoch(args, i, model, mutator, criterion, valid_loader)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'GPU{self_rank}_epoch_{i}_{int(time.time())}.ckpt'))
            # exit()
