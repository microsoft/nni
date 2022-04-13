import time
import json
import random
import logging
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import nni
from nn_meter import load_latency_predictor
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator.functional import FunctionalEvaluator
from nni.retiarii.utils import original_state_dict_hooks
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment

from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import CrossEntropyLabelSmooth, accuracy, ToBGRTensor, get_archchoice_by_model

logger = logging.getLogger("nni.spos.search")


def retrain_bn(model, criterion, max_iters, log_freq, loader):
    with torch.no_grad():
        logger.info("Clear BN statistics...")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        logger.info("Train BN with training set (BN sanitize)...")
        model.train()
        meters = AverageMeterGroup()
        for step in range(max_iters):
            inputs, targets = next(iter(loader))
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == max_iters:
                logger.info("Train Step [%d/%d] %s", step + 1, max_iters, meters)


def test_acc(model, criterion, log_freq, loader):
    logger.info("Start testing...")
    model.eval()
    meters = AverageMeterGroup()
    start_time = time.time()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == len(loader):
                logger.info("Valid Step [%d/%d] time %.3fs acc1 %.4f acc5 %.4f loss %.4f",
                            step + 1, len(loader), time.time() - start_time,
                            meters.acc1.avg, meters.acc5.avg, meters.loss.avg)
    return meters.acc1.avg


def evaluate_acc(class_cls, criterion, args):
    model = class_cls()
    with original_state_dict_hooks(model):
        model.load_state_dict(load_and_parse_state_dict(args.checkpoint), strict=False)
    model.cuda()

    if args.spos_preprocessing:
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor()
        ])
    else:
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ])
    val_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        ToBGRTensor()
        ])
    train_dataset = datasets.ImageNet(args.imagenet_dir, split='train', transform=train_trans)
    val_dataset = datasets.ImageNet(args.imagenet_dir, split='val', transform=val_trans)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.workers, shuffle=True)

    acc_before = test_acc(model, criterion, args.log_frequency, test_loader)
    nni.report_intermediate_result(acc_before)

    retrain_bn(model, criterion, args.train_iters, args.log_frequency, train_loader)
    acc = test_acc(model, criterion, args.log_frequency, test_loader)
    assert isinstance(acc, float)
    nni.report_intermediate_result(acc)
    nni.report_final_result(acc)


class LatencyFilter:
    def __init__(self, threshold, predictor, predictor_version=None, reverse=False):
        """
        Filter the models according to predicted latency. If the predicted latency of the ir model is larger than
        the given threshold, the ir model will be filtered and will not be considered as a searched architecture.

        Parameters
        ----------
        threshold: `float`
            the threshold of latency
        config, hardware:
            determine the targeted device
        reverse: `bool`
            if reverse is `False`, then the model returns `True` when `latency < threshold`,
            else otherwise
        """
        self.predictors = load_latency_predictor(predictor, predictor_version)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni-ir')
        return latency < self.threshold


def _main():
    parser = argparse.ArgumentParser("SPOS Evolutional Search")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--checkpoint", type=str, default="./data/checkpoint-150000.pth.tar")
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR "
                             "(as in original repo).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--train-iters", type=int, default=200)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--evolution-sample-size", type=int, default=10)
    parser.add_argument("--evolution-population-size", type=int, default=50)
    parser.add_argument("--evolution-cycles", type=int, default=10)
    parser.add_argument("--latency-filter", type=str, default=None,
                        help="Apply latency filter by calling the name of the applied hardware.")
    parser.add_argument("--latency-threshold", type=float, default=100)

    args = parser.parse_args()

    # use a fixed set of image will improve the performance
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    assert torch.cuda.is_available()

    base_model = ShuffleNetV2OneShot()
    criterion = CrossEntropyLabelSmooth(1000, args.label_smoothing)

    if args.latency_filter:
        latency_filter = LatencyFilter(threshold=args.latency_threshold, predictor=args.latency_filter)
    else:
        latency_filter = None

    evaluator = FunctionalEvaluator(evaluate_acc, criterion=criterion, args=args)
    evolution_strategy = strategy.RegularizedEvolution(
        model_filter=latency_filter,
        sample_size=args.evolution_sample_size, population_size=args.evolution_population_size, cycles=args.evolution_cycles)
    exp = RetiariiExperiment(base_model, evaluator, strategy=evolution_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 2
    exp_config.trial_gpu_number = 1
    exp_config.max_trial_number = args.evolution_cycles
    exp_config.training_service.use_active_gpu = False
    exp_config.execution_engine = 'base'
    exp_config.dummy_input = [1, 3, 224, 224]

    exp.run(exp_config, args.port)

    print('Exported models:')
    for i, model in enumerate(exp.export_top_models(formatter='dict')):
        print(model)
        with open(f'architecture_final_{i}.json', 'w') as f: 
            json.dump(get_archchoice_by_model(model), f, indent=4)

if __name__ == "__main__":
    _main()
