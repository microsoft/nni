import argparse
import re

import tqdm
import torch

from .constants import ZERO, SKIPCONNECT, CONV1X1, CONV3X3, AP3X3
from .model import db, Nb201RunConfig, Nb201ComputedStats, Nb201IntermediateStats


def parse_arch_str(arch_str):
    mp = {
        'none': ZERO,
        'skip_connect': SKIPCONNECT,
        'nor_conv_1x1': CONV1X1,
        'nor_conv_3x3': CONV3X3,
        'avg_pool_3x3': AP3X3
    }
    m = re.match(r'\|(.*)~0\|\+\|(.*)~0\|(.*)~1\|\+\|(.*)~0\|(.*)~1\|(.*)~2\|', arch_str)
    return {
        '0_1': mp[m.group(1)],
        '0_2': mp[m.group(2)],
        '1_2': mp[m.group(3)],
        '0_3': mp[m.group(4)],
        '1_3': mp[m.group(5)],
        '2_3': mp[m.group(6)]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='Path to the file to be converted, e.g., NAS-Bench-201-v1_1-096897.pth.')
    args = parser.parse_args()
    dataset_split = {
        'cifar10-valid': ['train', 'x-valid', 'ori-test', 'ori-test'],
        'cifar10': ['train', 'ori-test', 'ori-test', 'ori-test'],
        'cifar100': ['train', 'x-valid', 'x-test', 'ori-test'],
        'imagenet16-120': ['train', 'x-valid', 'x-test', 'ori-test'],
    }

    with db:
        db.create_tables([Nb201RunConfig, Nb201ComputedStats, Nb201IntermediateStats])
        print('Loading NAS-Bench-201 pickle...')
        nb201_data = torch.load(args.input_file)
        print('Dumping architectures...')
        for arch_str in nb201_data['meta_archs']:
            arch_json = parse_arch_str(arch_str)
            for epochs in [12, 200]:
                for dataset in Nb201RunConfig.dataset.choices:
                    Nb201RunConfig.create(arch=arch_json, num_epochs=epochs, dataset=dataset,
                                          num_channels=16, num_cells=5)
        for arch_info in tqdm.tqdm(nb201_data['arch2infos'].values(),
                                   desc='Processing architecture statistics'):
            for epochs_verb, d in arch_info.items():
                if epochs_verb == 'less':
                    epochs = 12
                else:
                    epochs = 200
                arch_json = parse_arch_str(d['arch_str'])
                for (dataset, seed), r in d['all_results'].items():
                    sp = dataset_split[dataset.lower()]
                    data_parsed = {
                        'train_acc': r['train_acc1es'][epochs - 1],
                        'valid_acc': r['eval_acc1es']['{}@{}'.format(sp[1], epochs - 1)],
                        'test_acc': r['eval_acc1es']['{}@{}'.format(sp[2], epochs - 1)],
                        'ori_test_acc': r['eval_acc1es']['{}@{}'.format(sp[3], epochs - 1)],
                        'train_loss': r['train_losses'][epochs - 1],
                        'valid_loss': r['eval_losses']['{}@{}'.format(sp[1], epochs - 1)],
                        'test_loss': r['eval_losses']['{}@{}'.format(sp[2], epochs - 1)],
                        'ori_test_loss': r['eval_losses']['{}@{}'.format(sp[3], epochs - 1)],
                        'parameters': r['params'],
                        'flops': r['flop'],
                        'latency': r['latency'][0],
                        'training_time': r['train_times'][epochs - 1] * epochs,
                        'valid_evaluation_time': r['eval_times']['{}@{}'.format(sp[1], epochs - 1)],
                        'test_evaluation_time': r['eval_times']['{}@{}'.format(sp[2], epochs - 1)],
                        'ori_test_evaluation_time': r['eval_times']['{}@{}'.format(sp[3], epochs - 1)],
                    }
                    config = Nb201RunConfig.get((Nb201RunConfig.num_epochs == epochs) & \
                                                (Nb201RunConfig.arch == arch_json) & \
                                                (Nb201RunConfig.dataset == dataset.lower()))
                    computed_stats = Nb201ComputedStats.create(config=config, seed=seed, **data_parsed)
                    intermediate_stats = []
                    for epoch in range(epochs):
                        data_parsed = {
                            'train_acc': r['train_acc1es'].get(epoch),
                            'valid_acc': r['eval_acc1es'].get('{}@{}'.format(sp[1], epoch)),
                            'test_acc': r['eval_acc1es'].get('{}@{}'.format(sp[2], epoch)),
                            'ori_test_acc': r['eval_acc1es'].get('{}@{}'.format(sp[3], epoch)),
                            'train_loss': r['train_losses'].get(epoch),
                            'valid_loss': r['eval_losses'].get('{}@{}'.format(sp[1], epoch)),
                            'test_loss': r['eval_losses'].get('{}@{}'.format(sp[2], epoch)),
                            'ori_test_loss': r['eval_losses'].get('{}@{}'.format(sp[3], epoch)),
                        }
                        if all([v is None for v in data_parsed.values()]):
                            continue
                        data_parsed.update(current_epoch=epoch + 1, run=computed_stats)
                        intermediate_stats.append(data_parsed)
                    Nb201IntermediateStats.insert_many(intermediate_stats).execute(db)


if __name__ == '__main__':
    main()
