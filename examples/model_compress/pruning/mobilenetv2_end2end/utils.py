# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.utils import get_module_by_name
from nni.compression.pytorch.speedup import ModelSpeedup
import sys
sys.path.append('../../models')
from mobilenet import MobileNet
from mobilenet_v2 import MobileNetV2


def create_model(model_type=None, n_classes=120, input_size=224, checkpoint=None, pretrained=False, width_mult=1.):
    if model_type == 'mobilenet_v1':
        model = MobileNet(n_class=n_classes, profile='normal')
        
        # assert checkpoint is not None
        # model = MobileNet(n_class=1000, profile='normal')
        # state_dict = torch.load(checkpoint)
        # if 'state_dict' in state_dict:
        #    state_dict = state_dict['state_dict']
        # model.load_state_dict(state_dict)
        # feature_size = model.classifier[0].weight.size(1)
        # new_linear = torch.nn.Linear(feature_size, n_classes)
        # model.classifier[0] = new_linear
        # return model
    elif model_type == 'mobilenet_v2':
        model = MobileNetV2(n_class=n_classes, input_size=input_size, width_mult=width_mult)
    elif model_type == 'mobilenet_v2_torchhub':
        model = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=pretrained)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
        feature_size = model.classifier[1].weight.data.size()[1]
        replace_classifier = torch.nn.Linear(feature_size, n_classes)
        model.classifier[1] = replace_classifier
    elif model_type is None:
        model = None
    else:
        raise RuntimeError('Unknown model_type.')

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model


def get_dataloader(dataset_type, data_path, batch_size=32, shuffle=True):
    assert dataset_type in ['train', 'eval']
    if dataset_type == 'train':
        ds = TrainDataset(data_path)
    else:
        ds = EvalDataset(data_path)
    return DataLoader(ds, batch_size, shuffle=shuffle)


class TrainDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]
        
        transform_set = [transforms.Lambda(lambda x: x),
                         transforms.RandomRotation(30),
                         transforms.ColorJitter(),
                         transforms.RandomHorizontalFlip(p=1)]
        self.transform = transforms.RandomChoice(transform_set)
        
    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index], allow_pickle=True).item()
        x = instance['input'].transpose(2, 0, 1)     # (C, H, W)
        x = torch.from_numpy(x).type(torch.float)    # convert to Tensor to use torchvision.transforms
        x = self.transform(x)
        return x, instance['label']


class EvalDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index], allow_pickle=True).item()
        x = instance['input'].transpose(2, 0, 1)
        x = torch.from_numpy(x).type(torch.float)
        return x, instance['label']


def count_flops(model, log=None, device=None):
    dummy_input = torch.rand([1, 3, 256, 256])
    if device is not None:
        dummy_input = dummy_input.to(device)
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")
    if log is not None:
        log.write(f"FLOPs: {flops}, params: {params}\n")
    return flops, params

def align_speedup(model, dummy_input, mask_file):
    ms = ModelSpeedup(model, dummy_input, mask_file)
    ms.speedup_model()
    
    align = 16
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            in_channel = module.in_channels
            out_channel = module.out_channels
            depthwise = False
            if in_channel == out_channel and in_channel == module.groups:
                depthwise =  True
            if name != "conv1.0":
                new_in_c = in_channel + align - in_channel % align
            else:
                new_in_c = in_channel
            new_out_c = out_channel + align - out_channel % align
            print(f"{name} In Channel:{in_channel} -> {new_in_c}  Out Channel:{out_channel}->{new_out_c}")
            new_conv = torch.nn.Conv2d(in_channels = new_in_c, out_channels=new_out_c, groups=module.groups if not depthwise else new_in_c, kernel_size=module.kernel_size, stride=module.stride, padding = module.padding, bias = module.bias is not None).to(module.weight.device)
            new_conv.weight.data[:] = 0
            new_conv.weight.data[:out_channel, :in_channel] = module.weight.data
            if new_conv.bias is not None:
                new_conv.bias.data[:] = 0
                new_conv.bias.data[:out_channel] = module.bias.data
            father, leaf = get_module_by_name(model, name)
            setattr(father, name.split('.')[-1], new_conv)
        elif isinstance(module, torch.nn.BatchNorm2d):
            num_features = module.num_features
            new_features = num_features + align - num_features % align
            new_norm = torch.nn.BatchNorm2d(num_features=new_features,
                            eps=module.eps,
                            momentum=module.momentum,
                            affine=module.affine,
                            track_running_stats=module.track_running_stats).to(module.weight.device)
            new_norm.bias.data[:] = 0
            new_norm.bias.data[: num_features] = module.bias.data 
            new_norm.weight.data[:] = 0
            new_norm.weight.data[: num_features] = module.weight.data 

            new_norm.running_mean.data[:] = 0
            new_norm.running_mean.data[: num_features] = module.running_mean.data 

            new_norm.running_var.data[:] = 0
            new_norm.running_var.data[: num_features] = module.running_var.data 
            father, leaf = get_module_by_name(model, name)
            setattr(father, name.split('.')[-1], new_norm)
        elif isinstance(module, torch.nn.Linear):
            in_feat= module.in_features
            out_feat = module.out_features
            new_in_feat = in_feat + align - in_feat % align
            new_linear = torch.nn.Linear(new_in_feat, out_feat, bias=module.bias is not None).to(module.weight.device)
            new_linear.weight.data[:] = 0
            new_linear.weight.data[:,:in_feat] = module.weight.data
            if new_linear.bias is not None:
                new_linear.bias.data = module.bias.data
            father, leaf = get_module_by_name(model, name)
            setattr(father, name.split('.')[-1], new_linear)
    return model