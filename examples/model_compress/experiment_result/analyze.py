import argparse
import json
import os
import matplotlib.pyplot as plt


def plot_performance_comparison(args, normalize=False):
    references = {
        'AutoCompressPruner':{
            'cifar10':{
                'vgg16':{
                    'performance': 0.9321,
                    'params':52.2,
                    'flops':8.8
                },
                'resnet18':{
                    'performance': 0.9381,
                    'params':54.2,
                    'flops':12.2
                }
            }
        }
    }
    target_sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
    pruners = ['L1FilterPruner', 'L2FilterPruner', 'FPGMPruner',
                'NetAdaptPruner', 'SimulatedAnnealingPruner', 'AutoCompressPruner']

    performances = {}
    flops = {}
    params = {}
    sparsities = {}
    for pruner in pruners:
        performances[pruner] = []
        flops[pruner] = []
        params[pruner] = []
        sparsities[pruner] = []
        for sparsity in target_sparsities:
            f = os.path.join('cifar10/', args.model, pruner, str(sparsity).replace('.', ''), 'result.json')
            if os.path.exists(f):
                with open(f, 'r') as jsonfile:
                    result = json.load(jsonfile)
                    sparsities[pruner].append(sparsity)
                    performances[pruner].append(result['performance']['finetuned'])
                    if normalize:
                        flops[pruner].append(result['flops']['original']/result['flops']['speedup'])
                        params[pruner].append(result['params']['original']/result['params']['speedup'])
                    else:
                        flops[pruner].append(result['flops']['speedup'])
                        params[pruner].append(result['params']['speedup'])

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle('Channel Pruning Comparison on {}/CIFAR10'.format(args.model))
    fig.subplots_adjust(hspace=0.5)

    for pruner in pruners:
        axs[0].scatter(params[pruner], performances[pruner], label=pruner)
        axs[1].scatter(flops[pruner], performances[pruner], label=pruner)

    if normalize:
        axs[0].annotate("original", (1, result['performance']['original']))
        axs[0].set_xscale('log')
    else:
        axs[0].plot(result['params']['original'], result['performance']['original'], 'rx', label='original model')
        if args.model in ['vgg16', 'resnet18']:
            axs[0].plot(result['params']['original']/references['AutoCompressPruner']['cifar10'][args.model]['params'], references['AutoCompressPruner']['cifar10'][args.model]['performance'], 'bx', label='AutoCompress Paper')
    axs[0].set_title("Performance v.s. Number of Parameters")
    axs[0].set_xlabel("Number of Parameters")
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    if normalize:
        axs[1].annotate("original", (1, result['performance']['original']))
        axs[1].set_xscale('log')
    else:
        axs[1].plot(result['flops']['original'], result['performance']['original'], 'rx', label='original model')
        if args.model in ['vgg16', 'resnet18']:
            axs[1].plot(result['flops']['original']/references['AutoCompressPruner']['cifar10'][args.model]['flops'], references['AutoCompressPruner']['cifar10'][args.model]['performance'], 'bx', label='AutoCompress Paper')
    axs[1].set_title("Performance v.s. FLOPs")
    axs[1].set_xlabel("FLOPs")
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.savefig('img/performance_comparison_{}.png'.format(args.model))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='vgg16',
                        help='vgg16, resnet18 or resnet50')
    args = parser.parse_args()

    plot_performance_comparison(args)
