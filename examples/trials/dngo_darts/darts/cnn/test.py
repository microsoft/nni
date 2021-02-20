import os
import sys
sys.path.insert(0, os.getcwd())
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import darts.cnn.genotypes as genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from darts.cnn.model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='BANANAS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():

  np.random.seed(args.seed)

  if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.gpu))
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
  else:
    device = torch.device('cpu')
    logging.info('No gpu device available')
    torch.manual_seed(args.seed)

  logging.info("args = %s", args)
  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.to(device)
  utils.load(model, args.model_path, args.gpu)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, test_transform = utils._data_transforms_cifar10(args.cutout, args.cutout_length)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj = infer(test_queue, model, criterion, args.gpu)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion, gpu_id=0):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() \
    else 'cpu')

  for step, (input, target) in enumerate(test_queue):
    input = input.to(device)
    target = target.to(device)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)  

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

