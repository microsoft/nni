from sdk.trainer.train import train
from sdk.trainer.steps import classifier_train_step, classifier_validation_test_step
from sdk.trainer.optimizers import pytorch_builtin_optimizers
from sdk.trainer.dataloaders import mnist_dataloader
from sdk.trainer.dataloaders import fake_mnist_dataloader
import importlib

import argparse
import os

parser = argparse.ArgumentParser(description='mnist_launcher')
parser.add_argument('--module_path', action="store", dest="module_path", type=str)
parser.add_argument('--job_id', action="store", dest="job_id", type=int)
parser.add_argument('--profile_out', default="", type=str)
args = parser.parse_args()

# bash cmd:
# CUDA_VISIBLE_DEVICES=0,1 is_distributed=1 rank=0 distributed_backend=nccl world_size=2 python test_train.py -x=0
# CUDA_VISIBLE_DEVICES=0,1 is_distributed=1 rank=1 distributed_backend=nccl world_size=2 python test_train.py -x=1

if __name__ == '__main__':    
    env = os.environ.copy()
    #model_map = {0:out0, 1:out1}
    mod = importlib.import_module(args.module_path)
    model_cls = mod.Graph
    if args.job_id == 0:
        train_dataloader = mnist_dataloader()
        use_fake_input = False
    else:    
        train_dataloader = fake_mnist_dataloader([32, 1, 28, 28])
        use_fake_input = True
    train(model_cls, train_dataloader, mnist_dataloader(), pytorch_builtin_optimizers("SGD", lr=0.001),classifier_train_step, classifier_validation_test_step, classifier_validation_test_step, use_fake_input, profile_out=args.profile_out, profile_steps=1000)