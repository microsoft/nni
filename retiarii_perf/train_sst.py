from sdk.trainer.train import train
from sdk.trainer.steps import classifier_train_step, classifier_validation_test_step
from sdk.trainer.optimizers import pytorch_builtin_optimizers
from sdk.trainer.sst_dataset import read_data_sst
from sdk.trainer.fake_dataloader import FakeDataLoader
from sdk.trainer.dataloaders import bert_dataloader
import argparse

import importlib
import torch

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--module_path', action="store", dest="module_path", type=str)
parser.add_argument('--job_id', action="store", dest="job_id", type=int)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument('--profile_out', default="", type=str)
parser.add_argument("--not_training", default=False, action="store_true")
parser.add_argument('--use_fake_input', default=False, action="store_true")
args = parser.parse_args()

# bash cmd:
# CUDA_VISIBLE_DEVICES=0,1,2 is_distributed=1 rank=0 distributed_backend=nccl world_size=3 python test_train_textnas.py -x=0
# CUDA_VISIBLE_DEVICES=0,1,2 is_distributed=1 rank=1 distributed_backend=nccl world_size=3 python test_train_textnas.py -x=1
# CUDA_VISIBLE_DEVICES=0,1,2 is_distributed=1 rank=2 distributed_backend=nccl world_size=3 python test_train_textnas.py -x=2

if __name__ == '__main__':    
    
    mod = importlib.import_module(args.module_path)
    model_cls = mod.Graph
    
    train_dataset, _, test_dataset = read_data_sst("data", train_with_valid=True)
    # def _callable_loader(dataset, batch_size, is_test=False):
    #     def dataloader(trainer):
    #         if not is_test:
    #             return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=12, shuffle=True, drop_last=True)
    #         else:
    #             return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=12)
    #     return dataloader
    
    #torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    if not args.use_fake_input:
        train_dataloader = bert_dataloader(train_dataset, args.batch_size, 0, shuffle=True, drop_last=True)
        test_dataloader = bert_dataloader(test_dataset, args.batch_size, 0)
        use_fake_input = False
    else:
        def _callable_loader_fake(dataset, input_size, batch_size):
            def dataloader(trainer):
                return FakeDataLoader(train_dataset, input_size, args.batch_size, has_mask=True, mask_size=[batch_size, 64])
            return dataloader
        train_dataloader = _callable_loader_fake(train_dataset, [args.batch_size, 64, 768], args.batch_size)
        test_dataloader = _callable_loader_fake(train_dataset, [args.batch_size, 64, 768], args.batch_size)
        use_fake_input = True
    
        
    train(model_cls, train_dataloader, test_dataloader, pytorch_builtin_optimizers("Adam", lr=5E-4, eps=1E-3, weight_decay=3E-6),classifier_train_step, classifier_validation_test_step, classifier_validation_test_step, use_fake_input,
          not_training=args.not_training, has_mask=True, profile_out=args.profile_out)