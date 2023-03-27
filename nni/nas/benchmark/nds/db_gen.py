# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
import os

import numpy as np
import tqdm

from nni.nas.benchmark.utils import load_benchmark
from .schema import NdsTrialConfig, NdsTrialStats, NdsIntermediateStats


def inject_item(db, item, proposer, dataset, generator):
    if 'genotype' in item['net']:
        model_family = 'nas_cell'
        num_nodes_normal = len(item['net']['genotype']['normal']) // 2
        num_nodes_reduce = len(item['net']['genotype']['reduce']) // 2
        model_spec = {
            'num_nodes_normal': num_nodes_normal,
            'num_nodes_reduce': num_nodes_reduce,
            'depth': item['net']['depth'],
            'width': item['net']['width'],
            'aux': item['net']['aux'],
            'drop_prob': item['net']['drop_prob'],
        }
        cell_spec = {}
        for cell_type in ['normal', 'reduce']:
            for i in range(num_nodes_normal):
                for j, label in enumerate(['x', 'y']):
                    cell_spec['{}_{}_op_{}'.format(cell_type, i, label)] = \
                        item['net']['genotype'][cell_type][i * 2 + j][0]
                    cell_spec['{}_{}_input_{}'.format(cell_type, i, label)] = \
                        item['net']['genotype'][cell_type][i * 2 + j][1]
            cell_spec['{}_concat'.format(cell_type)] = item['net']['genotype']['{}_concat'.format(cell_type)]
    else:
        if item['net']['block_type'].startswith('res_bottleneck'):
            model_family = 'residual_bottleneck'
        elif item['net']['block_type'].startswith('res_basic'):
            model_family = 'residual_basic'
        elif item['net']['block_type'].startswith('double_plain'):
            model_family = 'vanilla'
        else:
            raise ValueError('Unrecognized block type')
        model_spec = {k: v for k, v in item['net'].items() if v and k != 'block_type'}
        cell_spec = {}
    trial_config, _ = NdsTrialConfig.get_or_create(
        model_family=model_family,
        model_spec=model_spec,
        cell_spec=cell_spec,
        proposer=proposer,
        base_lr=item['optim']['base_lr'],
        weight_decay=item['optim']['wd'],
        num_epochs=item['optim']['max_ep'],
        dataset=dataset,
        generator=generator
    )
    assert len(item['train_ep_top1']) == len(item['test_ep_top1']) == trial_config.num_epochs
    trial = NdsTrialStats.create(
        config=trial_config,
        seed=item['rng_seed'],
        final_train_acc=100 - item['train_ep_top1'][-1],
        final_train_loss=item['train_ep_loss'][-1],
        final_test_acc=100 - item['test_ep_top1'][-1],
        best_train_acc=100 - min(item['train_ep_top1']),
        best_train_loss=np.nanmin(item['train_ep_loss']).item(),
        best_test_acc=100 - min(item['test_ep_top1']),
        parameters=item['params'] / 1e6,
        flops=item['flops'] / 1e6,
        iter_time=item['iter_time']
    )
    intermediate_stats = []
    for i in range(trial_config.num_epochs):
        intermediate_stats.append({
            'trial': trial,
            'current_epoch': i + 1,
            'train_loss': item['train_ep_loss'][i],
            'train_acc': 100 - item['train_ep_top1'][i],
            'test_acc': 100 - item['test_ep_top1'][i]
        })
    NdsIntermediateStats.insert_many(intermediate_stats).execute(db)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Path to extracted NDS data dir.')
    args = parser.parse_args()

    sweep_list = [
        'Amoeba.json',
        'Amoeba_in.json',
        'DARTS.json',
        'DARTS_fix-w-d.json',
        'DARTS_in.json',
        'DARTS_lr-wd.json',
        'DARTS_lr-wd_in.json',
        'ENAS.json',
        'ENAS_fix-w-d.json',
        'ENAS_in.json',
        'NASNet.json',
        'NASNet_in.json',
        'PNAS.json',
        'PNAS_fix-w-d.json',
        'PNAS_in.json',
        'ResNeXt-A.json',
        'ResNeXt-A_in.json',
        'ResNeXt-B.json',
        'ResNeXt-B_in.json',
        'ResNet-B.json',
        'ResNet.json',
        'ResNet_lr-wd.json',
        'ResNet_lr-wd_in.json',
        'ResNet_reruns.json',
        'ResNet_rng1.json',
        'ResNet_rng2.json',
        'ResNet_rng3.json',
        'Vanilla.json',
        'Vanilla_lr-wd.json',
        'Vanilla_lr-wd_in.json',
        'Vanilla_reruns.json',
        'Vanilla_rng1.json',
        'Vanilla_rng2.json',
        'Vanilla_rng3.json'
    ]

    db = load_benchmark('nds')

    with db:
        db.create_tables([NdsTrialConfig, NdsTrialStats, NdsIntermediateStats])
        for json_idx, json_file in enumerate(sweep_list, start=1):
            if 'fix-w-d' in json_file:
                generator = 'fix_w_d'
            elif 'lr-wd' in json_file:
                generator = 'tune_lr_wd'
            else:
                generator = 'random'
            if '_in' in json_file:
                dataset = 'imagenet'
            else:
                dataset = 'cifar10'
            proposer = json_file.split(".")[0].split("_")[0].lower()
            with open(os.path.join(args.input_dir, json_file), 'r') as f:
                data = json.load(f)
            if 'top' in data and 'mid' in data:
                for t in tqdm.tqdm(data['top'],
                                   desc='[{}/{}] Processing {} (top)'.format(json_idx, len(sweep_list), json_file)):
                    inject_item(db, t, proposer, dataset, generator)
                for t in tqdm.tqdm(data['mid'],
                                   desc='[{}/{}] Processing {} (mid)'.format(json_idx, len(sweep_list), json_file)):
                    inject_item(db, t, proposer, dataset, generator)
            else:
                for job in tqdm.tqdm(data,
                                     desc='[{}/{}] Processing {}'.format(json_idx, len(sweep_list), json_file)):
                    inject_item(db, job, proposer, dataset, generator)


if __name__ == '__main__':
    main()
