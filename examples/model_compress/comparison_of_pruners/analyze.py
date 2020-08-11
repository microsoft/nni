import argparse
import json
import matplotlib.pyplot as plt


def plot_performance_comparison(args):
    # reference data, performance of the original model and the performance declared in the AutoCompress Paper
    references = {
        'original':{
            'cifar10':{
                'vgg16':{
                    'performance': 0.9298,
                    'params':14987722.0,
                    'flops':314018314.0
                },
                'resnet18':{
                    'performance': 0.9433,
                    'params':11173962.0,
                    'flops':556651530.0
                },
                'resnet50':{
                    'performance': 0.9488,
                    'params':23520842.0,
                    'flops':1304694794.0
                }
            }
        },
        'AutoCompressPruner':{
            'cifar10':{
                'vgg16':{
                    'performance': 0.9321,
                    'params':52.2, # times
                    'flops':8.8
                },
                'resnet18':{
                    'performance': 0.9381,
                    'params':54.2,  # times
                    'flops':12.2
                }
            }
        }
    }

    markers = ['v', '^', '<', '1', '2', '3', '4', '8', '*', '+', 'o']

    with open('cifar10/comparison_result_{}.json'.format(args.model), 'r') as jsonfile:
        result = json.load(jsonfile)

    pruners = result.keys()

    performances = {}
    flops = {}
    params = {}
    sparsities = {}
    for pruner in pruners:
        performances[pruner] = [val['performance'] for val in result[pruner]]
        flops[pruner] = [val['flops'] for val in result[pruner]]
        params[pruner] = [val['params'] for val in result[pruner]]
        sparsities[pruner] = [val['sparsity'] for val in result[pruner]]

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle('Channel Pruning Comparison on {}/CIFAR10'.format(args.model))
    fig.subplots_adjust(hspace=0.5)

    for idx, pruner in enumerate(pruners):
        axs[0].scatter(params[pruner], performances[pruner], marker=markers[idx], label=pruner)
        axs[1].scatter(flops[pruner], performances[pruner], marker=markers[idx], label=pruner)

    # references
    params_original = references['original']['cifar10'][args.model]['params']
    performance_original = references['original']['cifar10'][args.model]['performance']
    axs[0].plot(params_original, performance_original, 'rx', label='original model')
    if args.model in ['vgg16', 'resnet18']:
        axs[0].plot(params_original/references['AutoCompressPruner']['cifar10'][args.model]['params'],
                    references['AutoCompressPruner']['cifar10'][args.model]['performance'],
                    'bx', label='AutoCompress Paper')

    axs[0].set_title("Performance v.s. Number of Parameters")
    axs[0].set_xlabel("Number of Parameters")
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # references
    flops_original = references['original']['cifar10'][args.model]['flops']
    performance_original = references['original']['cifar10'][args.model]['performance']
    axs[1].plot(flops_original, performance_original, 'rx', label='original model')
    if args.model in ['vgg16', 'resnet18']:
        axs[1].plot(flops_original/references['AutoCompressPruner']['cifar10'][args.model]['flops'],
                    references['AutoCompressPruner']['cifar10'][args.model]['performance'],
                    'bx', label='AutoCompress Paper')

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
