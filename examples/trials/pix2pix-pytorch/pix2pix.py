import os
import logging
import time
import argparse
from collections import namedtuple
import numpy as np
import torch
import nni
from nni.utils import merge_parameter
from data import AlignedDataset, CustomDatasetDataLoader
from pix2pix_model import Pix2PixModel
from base_params import get_base_params


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
        

def parse_args():
    # Settings that may be overrided by parameters from nni
    parser = argparse.ArgumentParser(description='PyTorch Pix2pix Example')
    parser.add_argument('--ngf', type=int, default=64,
                        help='# of generator filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64,
                        help='# of discriminator filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic',
                        help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--lambda_L1', type=float, default=100,
                        help='weight of L1 loss in the generator objective')
    
    # Additional training settings 
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs to linearly decay learning rate to zero')
    
    args, _ = parser.parse_known_args()
    return args
  

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

    # training 
    total_iters = 0                # the total number of training iterations
    for epoch in range(train_config.epoch_count, train_config.n_epochs + train_config.n_epochs_decay + 1):
        _logger.info('Training epoch {}'.format(epoch))
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0
        model.update_learning_rate()
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_config.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += train_config.batch_size
            epoch_iter += train_config.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            iter_data_time = time.time()
        _logger.info('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, train_config.n_epochs + train_config.n_epochs_decay, time.time() - epoch_start_time))
    
    model.save_networks('latest')
    _logger.info("Training done. Saving the final model.")
    
    l1_score = evaluate_L1(test_config, model, test_dataset)
    _logger.info("The final L1 loss the test set is {}".format(l1_score))  
    nni.report_final_result(l1_score)

    
if __name__ == '__main__':
    dataset_name = 'facades'

    checkpoint_dir = setup_trial_checkpoint_dir()

    params_from_cl = vars(parse_args())
    params_for_tuning = nni.get_next_parameter()       
    train_params, test_params = get_base_params(dataset_name, checkpoint_dir)
    train_params.update(params_from_cl)
    test_params.update(params_from_cl)
    train_params = merge_parameter(train_params, params_for_tuning)
        
    main(dataset_name, train_params, test_params)
    
