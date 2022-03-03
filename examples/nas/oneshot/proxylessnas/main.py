import json
import logging
import os
import sys
from argparse import ArgumentParser

import torch
from torchvision import transforms
from nni.retiarii.fixed import fixed_arch

import datasets
from model import SearchMobileNet
from putils import LabelSmoothingLoss, accuracy, get_parameters
from retrain import Retrain

logger = logging.getLogger('nni_proxylessnas')

if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    # configurations of the model
    parser.add_argument("--n_cell_stages", default='4,4,4,4,4,1', type=str)
    parser.add_argument("--stride_stages", default='2,2,2,1,2,1', type=str)
    parser.add_argument("--width_stages", default='24,40,80,96,192,320', type=str)
    parser.add_argument("--bn_momentum", default=0.1, type=float)
    parser.add_argument("--bn_eps", default=1e-3, type=float)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str, choices=['add#linear', 'mul#log'])
    parser.add_argument('--grad_reg_loss_lambda', default=1e-1, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_alpha', default=0.2, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_beta',  default=0.3, type=float)  # grad_reg_loss_params
    parser.add_argument("--applied_hardware", default=None, type=str, help='the hardware to predict model latency')
    parser.add_argument("--reference_latency", default=None, type=float, help='the reference latency in specified hardware')
    # configurations of imagenet dataset
    parser.add_argument("--data_path", default='/data/imagenet/', type=str)
    parser.add_argument("--train_batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=500, type=int)
    parser.add_argument("--n_worker", default=32, type=int)
    parser.add_argument("--resize_scale", default=0.08, type=float)
    parser.add_argument("--distort_color", default='normal', type=str, choices=['normal', 'strong', 'None'])
    # configurations for training mode
    parser.add_argument("--train_mode", default='search', type=str, choices=['search', 'retrain'])
    # configurations for search
    parser.add_argument("--checkpoint_path", default='./search_mobile_net.pt', type=str)
    parser.add_argument("--arch_path", default='./arch_path.pt', type=str)
    parser.add_argument("--no-warmup", dest='warmup', action='store_false')
    # configurations for retrain
    parser.add_argument("--exported_arch_path", default=None, type=str)

    args = parser.parse_args()
    if args.train_mode == 'retrain' and args.exported_arch_path is None:
        logger.error('When --train_mode is retrain, --exported_arch_path must be specified.')
        sys.exit(-1)

    if args.train_mode == 'retrain':
        assert os.path.isfile(args.exported_arch_path), \
            "exported_arch_path {} should be a file.".format(args.exported_arch_path)
        with fixed_arch(args.exported_arch_path):
            model = SearchMobileNet(width_stages=[int(i) for i in args.width_stages.split(',')],
                                    n_cell_stages=[int(i) for i in args.n_cell_stages.split(',')],
                                    stride_stages=[int(i) for i in args.stride_stages.split(',')],
                                    n_classes=1000,
                                    dropout_rate=args.dropout_rate,
                                    bn_param=(args.bn_momentum, args.bn_eps))
    else:
        model = SearchMobileNet(width_stages=[int(i) for i in args.width_stages.split(',')],
                                n_cell_stages=[int(i) for i in args.n_cell_stages.split(',')],
                                stride_stages=[int(i) for i in args.stride_stages.split(',')],
                                n_classes=1000,
                                dropout_rate=args.dropout_rate,
                                bn_param=(args.bn_momentum, args.bn_eps))
    logger.info('SearchMobileNet model create done')
    model.init_model()
    logger.info('SearchMobileNet model init done')

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info('Creating data provider...')
    data_provider = datasets.ImagenetDataProvider(save_path=args.data_path,
                                                  train_batch_size=args.train_batch_size,
                                                  test_batch_size=args.test_batch_size,
                                                  valid_size=None,
                                                  n_worker=args.n_worker,
                                                  resize_scale=args.resize_scale,
                                                  distort_color=args.distort_color)
    logger.info('Creating data provider done')

    if args.no_decay_keys:
        keys = args.no_decay_keys
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.05, momentum=momentum, nesterov=nesterov)
    else:
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD(get_parameters(model), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    if args.grad_reg_loss_type == 'add#linear':
        grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == 'mul#log':
        grad_reg_loss_params = {
            'alpha': args.grad_reg_loss_alpha,
            'beta': args.grad_reg_loss_beta,
        }
    else:
        args.grad_reg_loss_params = None

    if args.train_mode == 'search':
        from nni.retiarii.oneshot.pytorch import ProxylessTrainer
        from torchvision.datasets import ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset = ImageNet(args.data_path, transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        trainer = ProxylessTrainer(model,
                                   loss=LabelSmoothingLoss(),
                                   dataset=dataset,
                                   optimizer=optimizer,
                                   metrics=lambda output, target: accuracy(output, target, topk=(1, 5,)),
                                   num_epochs=120,
                                   log_frequency=10,
                                   grad_reg_loss_type=args.grad_reg_loss_type, 
                                   grad_reg_loss_params=grad_reg_loss_params, 
                                   applied_hardware=args.applied_hardware, dummy_input=(1, 3, 224, 224),
                                   ref_latency=args.reference_latency)
        trainer.fit()
        print('Final architecture:', trainer.export())
        json.dump(trainer.export(), open('checkpoint.json', 'w'))
    elif args.train_mode == 'retrain':
        # this is retrain
        trainer = Retrain(model, optimizer, device, data_provider, n_epochs=300)
        trainer.run()
