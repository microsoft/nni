import os
import nni
from nni.utils import merge_parameter
import logging
import time
import numpy as np
import torch
from collections import namedtuple
from data import AlignedDataset, CustomDatasetDataLoader
from pix2pix_model import Pix2PixModel


_logger = logging.getLogger('example_pix2pix')


def download_dataset(dataset_name):
    # code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    assert(dataset_name in ['facades', 'night2day', 'edges2handbags', 'edges2shoes', 'maps'])
    if os.path.exists('./data/' + dataset_name):
        _logger.info("Already downloaded dataset " + dataset_name)
    else:
        _logger.info("Downloading dataset " + dataset_name)
        if not os.path.exists('./data/'):
            os.system('mkdir ./data')
        os.system('mkdir ./data/' + dataset_name)
        URL = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{}.tar.gz'.format(dataset_name)
        TAR_FILE = './data/{}.tar.gz'.format(dataset_name)
        TARGET_DIR = './data/{}/'.format(dataset_name)
        os.system('wget -N {} -O {}'.format(URL, TAR_FILE))
        os.system('mkdir -p {}'.format(TARGET_DIR))
        os.system('tar -zxvf {} -C ./data/'.format(TAR_FILE))
        os.system('rm ' + TAR_FILE)        
    

def setup_trial_checkpoint_dir():
    checkpoint_dir = os.environ['NNI_OUTPUT_DIR'] + '/checkpoints/'
    os.system('mkdir ' + checkpoint_dir) 
    return checkpoint_dir

        
def get_config(dataset_name, checkpoint_dir):
    params = {}

    # change name and gpuid later
    basic_params = {'dataset': dataset_name,
                    'dataroot': './data/' + dataset_name,
                    'name': '',
                    'gpu_ids': [0],  
                    'checkpoints_dir': checkpoint_dir,
                    'verbose': False,
                    'print_freq': 100
                    }
    params.update(basic_params)
    
    dataset_params = {'dataset_mode': 'aligned',
                      'direction': 'BtoA',
                      'num_threads': 4,
                      'max_dataset_size': float('inf'),
                      'preprocess': 'resize_and_crop',
                      'display_winsize': 256,
                      'input_nc': 3,
                      'output_nc': 3}
    params.update(dataset_params)  

    model_params = {'model': 'pix2pix',
                    'ngf': 64,
                    'ndf': 64,
                    'netD': 'basic',
                    'netG': 'unet_256',
                    'n_layers_D': 3, 
                    'norm': 'batch',
                    'gan_mode': 'lsgan',
                    'init_type': 'normal',
                    'init_gain': 0.02,
                    'no_dropout': False}
    params.update(model_params)
    
    train_params = {'phase': 'train',
                    'isTrain': True,
                    'serial_batches': False,
                    'load_size': 286,
                    'crop_size': 256,
                    'no_flip': False,
                    'batch_size': 1,
                    'beta1': 0.5,
                    'pool_size': 0,
                    'lr_policy': 'linear',
                    'lr_decay_iters': 50,
                    'lr': 0.0002,
                    'lambda_L1': 100,
                    'epoch_count': 1,
                    'n_epochs': 10,           # 100
                    'n_epochs_decay': 0,      # 100
                    'continue_train': False}
    train_params.update(params)
    
    test_params = {'phase': 'test',
                   'isTrain': False,
                   'load_size': 256,
                   'crop_size': 256,
                   'batch_size': 1,
                   'serial_batches': True,
                   'no_flip': True,
                   'eval': True}
    test_params.update(params)
    
    return train_params, test_params
    

def train(config, model, dataset):
    total_iters = 0                # the total number of training iterations
    for epoch in range(config.epoch_count, config.n_epochs + config.n_epochs_decay + 1):
        _logger.info('Training epoch {}'.format(epoch))
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0
        model.update_learning_rate()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += config.batch_size
            epoch_iter += config.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            iter_data_time = time.time()


def evaluate_L1(config, model, dataset):
    if config.eval:
        model.eval()
    scores = []
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()
        score = torch.mean(torch.abs(visuals['fake_B']-visuals['real_B'])).detach().cpu().numpy()
        scores.append(score)
    return np.mean(np.array(scores))
            

def main(dataset_name, train_params, test_params):
    download_dataset(dataset_name)
    
    train_config = namedtuple('Struct', train_params.keys())(*train_params.values())
    test_config = namedtuple('Struct', test_params.keys())(*test_params.values())
    
    train_dataset, test_dataset = AlignedDataset(train_config), AlignedDataset(test_config)
    train_dataset = CustomDatasetDataLoader(train_config, train_dataset)
    test_dataset = CustomDatasetDataLoader(test_config, test_dataset)
    _logger.info('Number of training images = {}'.format(len(train_dataset)))
    _logger.info('Number of testing images = {}'.format(len(test_dataset)))    

    model = Pix2PixModel(train_config)
    model.setup(train_config)

    train(train_config, model, train_dataset)

    model.save_networks('latest')
    
    l1_score = evaluate_L1(test_config, model, test_dataset)
    nni.report_final_result(l1_score)

    
if __name__ == '__main__':
    dataset_name = 'facades'
   
    params_for_tuning = nni.get_next_parameter()
    checkpoint_dir = setup_trial_checkpoint_dir()
        
    train_config, test_config = get_config(dataset_name, checkpoint_dir)
    train_config = merge_parameter(train_config, params_for_tuning)
        
    main(dataset_name, train_config, test_config)
    
