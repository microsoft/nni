# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch

class TextLogger(object):
    """Write log immediately to the disk"""
    def __init__(self, filepath):
        self.f = open(filepath, 'w')
        self.fid = self.f.fileno()
        self.filepath = filepath

    def close(self):
        self.f.close()

    def write(self, content):
        self.f.write(content)
        self.f.flush()
        os.fsync(self.fid)

    def write_buf(self, content):
        self.f.write(content)

    def print_and_write(self, content):
        print(content)
        self.write(content+'\n')

def to_numpy(var):
    use_cuda = torch.cuda.is_available()
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()


def to_tensor(ndarray, requires_grad=False):  # return a float tensor by default
    tensor = torch.from_numpy(ndarray).float()  # by default does not require grad
    if requires_grad:
        tensor.requires_grad_()
    return tensor.cuda() if torch.cuda.is_available() else tensor


def measure_layer_for_pruning(wrapper, x):
    def get_layer_type(layer):
        layer_str = str(layer)
        return layer_str[:layer_str.find('(')].strip()

    def get_layer_param(model):
        import operator
        import functools

        return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])

    multi_add = 1
    layer = wrapper.module
    type_name = get_layer_type(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        wrapper.flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        wrapper.params = get_layer_param(layer)
    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        wrapper.flops = weight_ops + bias_ops
        wrapper.params = get_layer_param(layer)
    return


def least_square_sklearn(X, Y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    return reg.coef_


def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir
