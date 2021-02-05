import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.cifar10.vgg import VGG
from models.mnist.lenet import LeNet
from nni.compression.pytorch import apply_compression_results, ModelSpeedup

torch.manual_seed(0)
use_mask = True
use_speedup = True
compare_results = True

config = {
    'apoz': {
        'model_name': 'lenet',
        'input_shape': [64, 1, 28, 28],
        'masks_file': './experiment_data/mask_vgg16_cifar10_apoz.pth'
    },
    'l1filter': {
        'model_name': 'vgg16',
        'input_shape': [64, 3, 32, 32],
        'masks_file': './experiment_data/mask_vgg16_cifar10_l1filter.pth'
    },
    'fpgm': {
        'model_name': 'vgg16',
        'input_shape': [64, 3, 32, 32],
        'masks_file': './experiment_data/mask_vgg16_cifar10_fpgm.pth'
    },
    'slim': {
        'model_name': 'vgg19',
        'input_shape': [64, 3, 32, 32],
        'masks_file': './experiment_data/mask_vgg19_cifar10_slim.pth'
    }
}

def model_inference(config):
    masks_file = config['masks_file']
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    # device = torch.device(config['device'])
    if config['model_name'] == 'vgg16':
        model = VGG(depth=16)
    elif config['model_name'] == 'vgg19':
        model = VGG(depth=19)
    elif config['model_name'] == 'lenet':
        model = LeNet()

    model.to(device)
    model.eval()

    dummy_input = torch.randn(config['input_shape']).to(device)
    use_mask_out = use_speedup_out = None
    # must run use_mask before use_speedup because use_speedup modify the model
    if use_mask:
        apply_compression_results(model, masks_file, device)
        start = time.time()
        for _ in range(32):
            use_mask_out = model(dummy_input)
        print('elapsed time when use mask: ', time.time() - start)
    if use_speedup:
        m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
        m_speedup.speedup_model()
        start = time.time()
        for _ in range(32):
            use_speedup_out = model(dummy_input)
        print('elapsed time when use speedup: ', time.time() - start)
    if compare_results:
        if torch.allclose(use_mask_out, use_speedup_out, atol=1e-07):
            print('the outputs from use_mask and use_speedup are the same')
        else:
            raise RuntimeError('the outputs from use_mask and use_speedup are different')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("speedup")
    parser.add_argument("--example_name", type=str, default="slim", help="the name of pruning example")
    parser.add_argument("--masks_file", type=str, default=None, help="the path of the masks file")
    args = parser.parse_args()

    if args.example_name != 'all':
        if args.masks_file is not None:
            config[args.example_name]['masks_file'] = args.masks_file
        if not os.path.exists(config[args.example_name]['masks_file']):
            msg = '{} does not exist! You should specify masks_file correctly, ' \
                  'or use default one which is generated by model_prune_torch.py'
            raise RuntimeError(msg.format(config[args.example_name]['masks_file']))
        model_inference(config[args.example_name])
    else:
        model_inference(config['fpgm'])
        model_inference(config['slim'])
        model_inference(config['l1filter'])
        model_inference(config['apoz'])
