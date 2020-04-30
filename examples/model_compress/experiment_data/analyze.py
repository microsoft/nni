import argparse
import json
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt

from nni.compression.torch import LevelPruner

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.mnist.lenet import LeNet
# from models.Pytorch_Retinaface.models.retinaface import RetinaFace
# from models.Pytorch_Retinaface.data import cfg_mnet, cfg_re50


def get_config_lists(files):
    pruning_histories = []
    performances = []
    config_lists = []
    overall_sparsities = []

    for f in files:
        pruning_histories.append(pd.read_csv(f))

    for history in pruning_histories:
        overall_sparsities.append(history.loc[0, 'sparsity'])
        performances.append(history['performance'].max())
        idx = history['performance'].idxmax()
        config_list = history.loc[idx, 'config_list']
        config_lists.append(json.loads(config_list))

    return config_lists, overall_sparsities, performances


def get_original_op(model):
    op_names = []
    op_weights = []

    pruner = LevelPruner(model, [{
        'sparsity': 0.1,
        'op_types': ['default']
    }])

    for wrapper in pruner.get_modules_wrapper():
        op_names.append(wrapper.name)
        op_weights.append(wrapper.module.weight.data.numel())

    return op_names, op_weights


def draw(model):
    # if model == 'lenet':
    #     files = list(os.walk('lenet'))[0][-1]
    #     model = LeNet()
    # elif model == 'mobilenet_v2':
    #     files = list(os.walk('mobilenet'))[0][-1]
    #     model = models.mobilenet_v2()
    # elif model == 'retinaface':
    #     files = list(os.walk('retinaface'))[0][-1]
    #     # model = models.mobilenet_v2()
    if model == 'lenet':
        files = ['lenet/pruning_history.csv']
        model = LeNet()
    elif model == 'mobilenet_v2':
        files = ['mobilenet/pruning_history_01.csv', 'mobilenet/pruning_history_03.csv', 'mobilenet/pruning_history_04.csv', 'mobilenet/pruning_history_05.csv']
        model = models.mobilenet_v2()
    # elif model == 'retinaface':
    #     files = ['retinaface/pruning_history_01.csv', 'retinaface/pruning_history_03.csv']
    #     cfg = cfg_mnet
    #     model = RetinaFace(cfg=cfg, phase='test')

    config_lists, overall_sparsities, performances = get_config_lists(files)

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle("Sparsities distribution ")

    # Fig 0 : layer weights
    op_names_original, op_weights_original = get_original_op(model)
    axs[0].plot(op_names_original, op_weights_original, label='op_weights')
    axs[0].set_title("op weights")
    axs[0].legend()

    # Fig 1 : original sparsities
    sparsities = [0]*len(op_names_original)
    for idx, config_list in enumerate(config_lists):
        for config in config_list:
            op_name = config['op_names'][0]
            i = op_names_original.index(op_name)
            sparsities[i] = config['sparsity']
        axs[1].plot(op_names_original, sparsities,
                    label='sparsity: {}, performance: {}'.format(overall_sparsities[idx], performances[idx]))
    axs[1].set_title('original order')
    axs[1].legend()

    # Fig 2 : storted sparsities
    for idx, config_list in enumerate(config_lists):
        op_names_sorted = []
        sparsities = []
        # sorted by layer weights
        for config in config_list:
            sparsities.append(config['sparsity'])
            op_names_sorted.append(config['op_names'][0])

        axs[2].plot(op_names_sorted, sparsities,
                    label='sparsity: {}'.format(overall_sparsities[idx]))
    axs[2].set_title('Sorted by op weights')
    axs[2].legend()

    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=90)

    # plt.tight_layout()
    plt.savefig('./sparsities_distribution.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='lenet', help='lenet, mobilenet_v2 or retinaface')
    args = parser.parse_args()

    draw(args.model)
