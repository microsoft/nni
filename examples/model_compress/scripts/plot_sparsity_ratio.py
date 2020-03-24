import matplotlib

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch

def get_config_list(model):
    config_list = []
    op_type = 'conv1'
    if model == 'resnet56':
        for layer in range(1, 4):
            for block in range(9):
                config_list.append(f'layer{layer}.{block}.{op_type}')
    elif model == 'densenet40':
        for layer in range(1, 4):
            config_list.append(f'trans{layer}.conv1')
            for block in range(12):
                config_list.append(f'dense{layer}.{block}.{op_type}')
    elif model == 'vgg16':
        config_list = ['feature.0', 'feature.3', 'feature.7', 'feature.10', 'feature.14', 'feature.17', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']

    return config_list


def plot_mix_distribution_bar(method, model, seed):
    path = f'./experiments/{model}_cifar10_{method}/Task_01_seed_{seed}/checkpoints/pruned.pth'
    loaded_ckpt = torch.load(path, map_location='cpu')
    if 'state_dict' in loaded_ckpt:
        loaded_ckpt = loaded_ckpt['state_dict']

    remain_list, filters_list = [], []
    for k, v in model.state_dict().items():
        if '.'.join(k.split('.')[:2]) in names:
            filters = v.size(0)
            pruend = (v.view(filters, -1).sum(1) == 0).sum().item()
            remained = filters - pruend
            filters_list.append(filters)
            remain_list.append(remained)

    names = get_config_list(model)
    kernels = [v for k, v in loaded_ckpt.items() if 'ks_list' in k]
    distribution = {0: [], 3: [], 5: [], 7: []}
    for i, ks_list in enumerate(kernels):
        cur_ks_list = ks_list.cpu().numpy()
        for ks in [0, 3, 5, 7]:
            distribution[ks].append(sum(cur_ks_list == ks))

    ind = np.arange(1, len(distribution[0])+1)    # the x locations for the groups
    width = 0.6       # the width of the bars: can also be len(x) sequence
    for k, v in distribution.items():
        distribution[k] = np.array(v)

    f, ax = plt.subplots(1, 1, figsize=(5.5, 4), sharex=False)

    p1 = plt.bar(ind, distribution[0], width, color='darkgrey')
    p3 = plt.bar(ind, distribution[3], width, bottom=distribution[0], color='lightsteelblue')
    p5 = plt.bar(ind, distribution[5], width, bottom=(distribution[0]+distribution[3]), color='wheat')
    p7 = plt.bar(ind, distribution[7], width, bottom=(distribution[0]+distribution[3]+distribution[5]), color='tomato')
    
    plt.legend((p1[0], p3[0], p5[0], p7[0]), ('0', '3', '5', '7'), fontsize=18)
    
    font = {'family' : 'Times New Roman', 'size': 16,}

    ax.set_ylabel(r'Filters', font)
    ax.set_xlabel(f"Layers", font)
    # ax.set_xticklabels(fontsize=18)
    func = lambda x, pos: "" if np.isclose(x,0) else int(x)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # print(len(kernels))

    plt.tight_layout()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    # plt.show()

    plt.savefig(f'/Users/colorjam/Documents/Go/eccv2020kit_MARA/pics/{label}.png')