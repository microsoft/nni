import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def trt_parse_log(fpath):
    f = open(fpath)
    lines = f.readlines()
    key_str = "average"
    latency = None
    for line in lines:
        if line.find(key_str) != -1:
            latency = float(line.split()[-2])
            break
    assert(latency is not None, "invalid tensorrt log")
    return latency

def rammer_parse_log(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        result = list(filter(lambda x: 'Summary: [min, max, mean] =' in x, lines))
        assert len(result) == 1
        tmp = re.split(' ', result[0])
        return float(tmp[5][1:-1])

def jit_parse_log(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        result = list(filter(lambda x: 'RunningTime =' in x, lines))
        assert len(result) == 1
        tmp = re.split(' ', result[0])
        return float(tmp[2])

def tvm_parse_log(fpath):
    f = open(fpath)
    lines = f.readlines()
    key_str = "mean"
    latency = None
    for idx, line in enumerate(lines):
        if line.find(key_str) != -1:
            latency_line = lines[idx+1]
            latency = float(latency_line.split()[0])
            break
    assert(latency is not None, "invalid tvm log")
    return latency 

def tvm_sparse_parse_log(fpath):
    f = open(fpath)
    lines = f.readlines()
    key_str = "batch"
    latency = None
    for line in lines:
        if line.find(key_str) != -1:
            latency = float(line.split()[-2])
            break
    assert(latency is not None, "invalid tvm-s log")
    return latency

func_map = {
    'rammer': rammer_parse_log,
    'rammer-s': rammer_parse_log,
    'sparta': rammer_parse_log,
    'tvm': tvm_parse_log,
    'jit': jit_parse_log,
    'trt': trt_parse_log,
    'tvm-s': tvm_sparse_parse_log
}

def draw_figure10(data):
    device = ['GPU', 'ROCM', 'CPU']
    solutions = ['PyTorch', 'TensorRT', 'OpenVINO', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']


    pattern = {
        'PyTorch': {'color':"white", 'hatch':'///'},
        'TensorRT': {'color': "white", 'hatch': '\\\\\\'},
        'OpenVINO': {'color':'white', 'hatch':'-//'},
        'TVM': {'color': "white", 'hatch': '--'},
        'TVM-S': {'color': "whitesmoke", 'hatch': 'xxx'},
        'Rammer': {'color': "lightgray", 'hatch': 'oo'},
        'Rammer-S': {'color': "gray", 'hatch': '|||'},
        'SparTA': {'color': "black", 'hatch': None},

    }



    x = np.arange(len(solutions))
    width = 0.75

    fig, axes = plt.subplots(1, 3, figsize=[8, 3])

    (ax1, ax2, ax3) = axes
    barwidth = 0.3
    Bars = []
    #====================
    # first row

    gpu_backend = ['PyTorch', 'TensorRT', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']
    gpu_data = [data['jit'], data['trt'], data['tvm'], data['tvm-s'], data['rammer'], data['rammer-s'], data['sparta']]
    x = np.arange(1, len(gpu_backend)+1)

    # divider = make_axes_locatable(ax1)
    # ax1_2 = divider.new_vertical(size="50%", pad=0.1)
    # fig.add_axes(ax1_2)


    # ax1.bar([10], [0], label='OpenVINO', **pattern['OpenVINO'])
    for i, solution in enumerate(gpu_backend):
        _b = ax1.bar(x[i], gpu_data[i], width, label=gpu_backend[i], **pattern[gpu_backend[i]], edgecolor='black')
        Bars.append(_b)
    ax1.set_xlim(0, max(x)+1)
    ax1.set_ylabel('   Latency (ms)', labelpad=14, loc='bottom', fontsize=16)
    ax1.set_xlabel('2080Ti', fontsize=16)
    ax1.set_yticks([0,50,100,130])
    ax1.set_yticklabels([0,50,100,130], fontsize=14)
    # ax1.set_yticks([0, 100, 200, 300])
    ax1.set_ylim(0, 130)
    ax1.tick_params(axis='y', direction='in')
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # for i, solution in enumerate(gpu_backend):
    #     ax1_2.bar(x[i], gpu_data[i], width, label=gpu_backend[i], **pattern[gpu_backend[i]], edgecolor='black')

    # ax1_2.set_ylim(1730, 1780)
    # ax1_2.set_yticks([1750, 1780])
    # ax1_2.set_yticklabels([1750, 1780], fontsize=14)
    # ax1_2.tick_params(bottom=False, labelbottom=False)
    # ax1_2.spines['bottom'].set_visible(False)
    # ax1_2.spines['top'].set_visible(False)
    # ax1_2.spines['right'].set_visible(False)

    # ax1_2.tick_params(axis='y', direction='in')

    # # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax1_2.transAxes, color='k', clip_on=False)
    # ax1_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
    # ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    #====================


    rocm_backend = ['PyTorch', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']
    rocm_data = [0, 0, 0, 0, 0, 0]
    x = np.arange(1, len(rocm_backend)+1)
    divider = make_axes_locatable(ax2)
    ax2_2 = divider.new_vertical(size="50%", pad=0.1)
    fig.add_axes(ax2_2)

    for i, solution in enumerate(rocm_backend):
        ax2.bar(x[i], rocm_data[i], width, label=rocm_backend[i], **pattern[rocm_backend[i]], edgecolor='black')

    # ax2.set_ylabel('    Latency (ms)', labelpad=10, loc='bottom', fontsize=16)
    ax2.set_xlabel('ROCM GPU', fontsize=16)
    ax2.set_yticks([0,50,100])
    ax2.set_yticklabels([0,50,100], fontsize=14)
    # ax2.set_yticks([0, 100, 200, 300])
    ax2.set_ylim(0, 130)
    ax2.tick_params(axis='y', direction='in')
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for i, solution in enumerate(rocm_backend):
        ax2_2.bar(x[i], rocm_data[i], width, label=rocm_backend[i], **pattern[rocm_backend[i]], edgecolor='black')

    ax2_2.set_ylim(1530, 1680)
    ax2_2.set_yticks([1550, 1600])
    ax2_2.set_yticklabels([1550, 1600], fontsize=14)
    ax2_2.tick_params(bottom=False, labelbottom=False)
    ax2_2.spines['bottom'].set_visible(False)
    ax2_2.spines['top'].set_visible(False)
    ax2_2.spines['right'].set_visible(False)

    ax2_2.tick_params(axis='y', direction='in')

    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax2_2.transAxes, color='k', clip_on=False)
    ax2_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    #====================

    cpu_backend = ['PyTorch', 'OpenVINO', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']
    cpu_data = [0, 0, 0, 0, 0, 0, 0]
    x = np.arange(1, len(cpu_backend)+1)


    for i, solution in enumerate(cpu_backend):
        _b = ax3.bar(x[i], cpu_data[i], width, label=cpu_backend[i], **pattern[cpu_backend[i]], edgecolor='black')
        if cpu_backend[i] == 'OpenVINO':
            Bars.append(_b)

    ax3.set_xlabel('CPU', fontsize=16)
    ax3.set_yticks([500,1000,1500,2000])
    ax3.set_yticklabels([500,1000,1500, 2000], fontsize=14)
    # ax2.set_yticks([0, 100, 200, 300])
    ax3.set_ylim(500, 2300)
    ax3.tick_params(axis='y', direction='in')
    ax3.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    #====================
    # second row
    # ['PyTorch', 'TensorRT', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']

    handles = Bars
    labels = [b.get_label() for b in handles]

    fig.legend(handles, labels, bbox_to_anchor = (0.5, 0.8), loc='lower center', fontsize=14,
            borderaxespad=0, ncol=4, columnspacing=1.0, handletextpad=0.4, frameon=False)
    fig.subplots_adjust( hspace=0.6)

    # #fig.suptitle('Structured Sparsity', x=0.5, y=0.42, fontsize='medium')
    # fig.text(0.5, 0.515, 'Structured Sparsity', fontsize='medium',
    #          horizontalalignment='center', verticalalignment='center')
    # fig.text(0.5, 0.285, 'Unstructured Sparsity', fontsize='medium',
    #          horizontalalignment='center', verticalalignment='center')
    # fig.text(0.5, 0.055, 'Structured+8bit Sparsity', fontsize='medium',
    #          horizontalalignment='center', verticalalignment='center')

    fig.tight_layout(rect=(0,0,1,0.85))
    # plt.tight_layout()

    fig.savefig("mix_sparsity_end2end.pdf", bbox_inches='tight',dpi=1000)
    # plt.show()


if __name__ == '__main__':
    data = {}
    for framework in ['jit', 'rammer', 'rammer-s', 'tvm', 'trt', 'tvm-s', 'sparta']:
        time = 0
        fpath = os.path.join('log', '{}.log'.format(framework))
        try:
            time = func_map[framework](fpath)
        except Exception as err:
            print(f'failed at the framework {framework} with ', err)
        data[framework] = time
    draw_figure10(data)