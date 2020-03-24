import pandas as pd
import os
import pdb

def write_row(path, data, reset=False):
    with open(path, 'w' if reset else 'a') as o:
        row = ''
        for i, c in enumerate(data):
            row += ('' if i == 0 else ',') + str(c)
        row += '\n'
        o.write(row)

models = ['vgg16', 'resnet56', 'densenet40']
methods = ['FPGMPruner', 'L1FilterPruner', 'ActivationAPoZRankFilterPruner', 'GradientWeightRankFilterPruner', 'SlimPruner', 'TaylorFOWeightFilterGlobalPruner']
seeds = [2333, 2222, 1000]
file_path = f'./20200319_results.csv'

write_row(file_path, ['model', 'method'] + [f'seed_{i}' for i in seeds], True)

for model in models:
    for method in methods:
        results = []
        for seed in seeds:
            file_name = f'/home/yanchenqian/workspace/nni-dev/examples/model_compress/experiments/{model}_cifar10_{method}/Task_01_seed_{seed}/results.csv'
            if os.path.isfile(file_name):
                df = pd.read_csv(file_name)
                if 'top1' in df:
                    acc1 = df['top1'].max()
                    # results.append(acc1)
                elif 'acc1' in df:
                    acc1 = df['acc1'].max()

                acc = acc1 * 100
                results.append(acc)
                # else:
                    # results.append(0)
            else:
                results.append(0)

        write_row(file_path, [model, method] + results)

        
            



