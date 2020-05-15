import argparse
import json
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt

from nni.compression.torch import LevelPruner

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models.mnist.lenet import LeNet


def get_config_lists_from_pruning_history(files):
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


def get_config_lists_from_search_result(files):
    performances = []
    config_lists = []

    for f in files:
        with open(f, 'r') as jsonfile:
            j = json.load(jsonfile)
            performances.append(j['performance'])
            config_lists.append(json.loads(j['config_list']))

    return config_lists, performances


def get_performances_fine_tuned(files):
    performances = []

    for f in files:
        with open(f, 'r') as jsonfile:
            j = json.load(jsonfile)
            performances.append(j['finetuned'])

    return performances



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


def draw(args):
    # if model == 'lenet':
    #     files = list(os.walk('lenet'))[0][-1]
    #     model = LeNet()
    # elif model == 'mobilenet_v2':
    #     files = list(os.walk('mobilenet'))[0][-1]
    #     model = models.mobilenet_v2()
    if args.model == 'lenet':
        files = ['lenet/pruning_history.csv']
        model = LeNet()
        notes = 'LeNet, MNIST, SAPruner, fine-grained'
        config_lists, overall_sparsities, performances = get_config_lists_from_pruning_history(
            files)
    elif args.model == 'mobilenet_v2' and args.pruning_mode == 'fine-grained':
        files = ['mobilenet_sapruner_fine_grained/pruning_history_01.csv', 'mobilenet_sapruner_fine_grained/pruning_history_03.csv',
                 'mobilenet_sapruner_fine_grained/pruning_history_04.csv', 'mobilenet_sapruner_fine_grained/pruning_history_05.csv']
        model = models.mobilenet_v2()
        notes = 'MobileNet V2, ImageNet, SAPruner, fine-grained'
        config_lists, overall_sparsities, performances = get_config_lists_from_pruning_history(
            files)
    elif args.model == 'mobilenet_v2' and args.pruning_mode == 'channel':
        files = ['imagenet_sapruner_channel/search_result_01.json', 'imagenet_sapruner_channel/search_result_02.json',
                 'imagenet_sapruner_channel/search_result_03.json', 'imagenet_sapruner_channel/search_result_04.json', 'imagenet_sapruner_channel/search_result_05.json']
        model = models.mobilenet_v2()
        notes = 'MobileNet V2, ImageNet, SAPruner, channel pruning'
        config_lists, performances = get_config_lists_from_search_result(
            files)
        overall_sparsities = [0.1, 0.2, 0.3, 0.4, 0.5]
        fine_tune_epochs = 3
        performances_fine_tuned = [
            0.43526, 0.43616, 0.41408, 0.38422, 0.39246]
    elif args.model == 'vgg16' and args.pruning_mode == 'channel':
        files = ['cifar10_sapruner_channel/01/search_result.json', 'cifar10_sapruner_channel/02/search_result.json',
                 'cifar10_sapruner_channel/03/search_result.json', 'cifar10_sapruner_channel/04/search_result.json', 'cifar10_sapruner_channel/05/search_result.json']
        model = models.vgg16()
        notes = 'VGG16, CIFAR10, SAPruner, channel pruning'
        config_lists, performances = get_config_lists_from_search_result(
            files)
        overall_sparsities = [0.1, 0.2, 0.3, 0.4, 0.5]
        fine_tune_epochs = 50
        files = ['cifar10_sapruner_channel/01/performance.json', 'cifar10_sapruner_channel/02/performance.json',
                 'cifar10_sapruner_channel/03/performance.json', 'cifar10_sapruner_channel/04/performance.json', 'cifar10_sapruner_channel/05/performance.json']
        performances_fine_tuned = get_performances_fine_tuned(files)

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle("Pruning Sparsities Distribution ({})".format(notes))
    fig.subplots_adjust(hspace=1)

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
                    label='sparsity: {}, performance: {:.4f}, fine-tuned performance ({} epochs): {:.4f}'.format(overall_sparsities[idx], performances[idx], fine_tune_epochs, performances_fine_tuned[idx]))
        # label='sparsity: {}, performance: {:.4f}'.format(overall_sparsities[idx], performances[idx]))
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
                    label='sparsity: {}, performance: {:.4f}, fine-tuned performance ({} epochs): {:.4f}'.format(overall_sparsities[idx], performances[idx], fine_tune_epochs, performances_fine_tuned[idx]))
        # label='sparsity: {}, performance: {:.4f}'.format(overall_sparsities[idx], performances[idx]))
    axs[2].set_title('Sorted by op weights')
    axs[2].legend()

    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=90)

    # plt.tight_layout()
    plt.savefig(
        './sparsities_distribution_{}_{}.png'.format(args.model, args.pruning_mode))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='lenet',
                        help='lenet, mobilenet_v2 or retinaface')
    parser.add_argument('--pruning-mode', type=str, default='channel',
                        help='channel, or fine-grained')
    args = parser.parse_args()

    draw(args)
