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

def draw_figure8(data):
    models = ['BERT', 'MobileNet', 'HuBERT']
    solutions = ['PyTorch', 'TensorRT', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']
    structured = {'BERT': [data['bert_coarse_jit'], data['bert_coarse_trt'], data['bert_coarse_tvm'], data['bert_coarse_tvm-s'], data['bert_coarse_rammer'], data['bert_coarse_rammer-s'], data['bert_coarse_sparta']],
                'MobileNet': [data['mobilenet_coarse_jit'], data['mobilenet_coarse_trt'], data['mobilenet_coarse_tvm'], data['mobilenet_coarse_tvm-s'], data['mobilenet_coarse_rammer'], data['mobilenet_coarse_rammer-s'], data['mobilenet_coarse_sparta']],
                'HuBERT': [data['hubert_coarse_jit'], data['hubert_coarse_trt'], data['hubert_coarse_tvm'], data['hubert_coarse_tvm-s'], data['hubert_coarse_rammer'], data['hubert_coarse_rammer-s'], data['hubert_coarse_sparta']]}

    hatches = ['///', '\\\\\\', '---', 'xxx', 'oo', '|||', None]
    colors = ['white', 'white', 'white', 'whitesmoke', 'lightgray', 'gray', 'black']


    x = np.arange(len(solutions))
    width = 0.75

    fig, axes = plt.subplots(3, 3, figsize=[6.4, 1.8*3])

    (ax1, ax2, ax3) = axes[0]

    #====================
    # first row

    # divider = make_axes_locatable(ax1)
    # ax1_2 = divider.new_vertical(size="100%", pad=0.1)
    # fig.add_axes(ax1_2)

    for i, solution in enumerate(solutions):
        ax1.bar(x[i], structured['BERT'][i], width, label=solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    ax1.set_ylabel('Latency (ms)', labelpad=8, loc='bottom')
    ax1.set_xlabel('BERT')
    ax1.set_yticks([0, 100, 200, 300])
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

    # for i, solution in enumerate(solutions):
    #     ax1_2.bar(x[i], structured['BERT'][i], width, label = solution,
    #             hatch=hatches[i], color=colors[i], edgecolor='black')

    # ax1_2.set_ylim(1730, 1780)
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

    for i, solution in enumerate(solutions):
        ax2.bar(x[i], structured['MobileNet'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    #ax1.set_ylabel('Latency (ms)')
    ax2.set_xlabel('MobileNet')
    ax2.set_yticks([0, 4, 8, 12])

    ax2.tick_params(axis='y', direction='in')
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    #====================

    # divider = make_axes_locatable(ax3)
    # ax3_2 = divider.new_vertical(size="100%", pad=0.1)
    # fig.add_axes(ax3_2)

    for i, solution in enumerate(solutions):
        ax3.bar(x[i], structured['HuBERT'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    ax3.set_xlabel('HuBERT')
    ax3.set_ylim(0, 170)
    ax3.set_yticks([0, 50,100, 150])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax3.tick_params(axis='y', direction='in')
    ax3.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # for i, solution in enumerate(solutions):
    #     ax3_2.bar(x[i], structured['HuBERT'][i], width, label = solution,
    #             hatch=hatches[i], color=colors[i], edgecolor='black')

    # ax3_2.set_ylim(195, 220)
    # ax3_2.tick_params(bottom=False, labelbottom=False)
    # ax3_2.spines['bottom'].set_visible(False)
    # ax3_2.spines['top'].set_visible(False)
    # ax3_2.spines['right'].set_visible(False)

    # ax3_2.tick_params(axis='y', direction='in')

    # # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax3_2.transAxes, color='k', clip_on=False)
    # ax3_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
    # ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    #====================
    # second row
    # ['PyTorch', 'TensorRT', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']

    unstructured = {'BERT': [data['bert_finegrained_jit'], data['bert_finegrained_trt'], data['bert_finegrained_tvm'], data['bert_finegrained_tvm-s'], data['bert_finegrained_rammer'], data['bert_finegrained_rammer-s'], data['bert_finegrained_sparta']],
                'MobileNet': [data['mobilenet_finegrained_jit'], data['mobilenet_finegrained_trt'], data['mobilenet_finegrained_tvm'], data['mobilenet_finegrained_tvm-s'], data['mobilenet_finegrained_rammer'], data['mobilenet_finegrained_rammer-s'], data['mobilenet_finegrained_sparta']],
                'HuBERT': [data['hubert_finegrained_jit'], data['hubert_finegrained_trt'], data['hubert_finegrained_tvm'], data['hubert_finegrained_tvm-s'], data['hubert_finegrained_rammer'], data['hubert_finegrained_rammer-s'], data['hubert_finegrained_sparta']]}
    (ax21, ax22, ax23) = axes[1]


    divider = make_axes_locatable(ax21)
    ax21_2 = divider.new_vertical(size="50%", pad=0.1)
    fig.add_axes(ax21_2)

    for i, solution in enumerate(solutions):
        ax21.bar(x[i], unstructured['BERT'][i], width, label=solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')


    ax21.set_yticks([0, 100, 200, 300])
    ax21.set_ylim(0,100)
    ax21.tick_params(axis='y', direction='in')
    ax21.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax21.spines['top'].set_visible(False)
    ax21.spines['right'].set_visible(False)


    for i, solution in enumerate(solutions):
        ax21_2.bar(x[i], unstructured['BERT'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    ax21_2.set_ylim(205, 230)
    ax21_2.tick_params(bottom=False, labelbottom=False)
    ax21_2.spines['bottom'].set_visible(False)
    ax21_2.spines['top'].set_visible(False)
    ax21_2.spines['right'].set_visible(False)
    ax21_2.set_yticks([215])
    ax21_2.tick_params(axis='y', direction='in')


    ax21.set_ylabel('Latency (ms)', labelpad=8, loc='bottom')
    ax21.set_xlabel('BERT')
    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax21_2.transAxes, color='k', clip_on=False)
    ax21_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax21.transAxes)  # switch to the bottom axes
    ax21.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    #====================

    for i, solution in enumerate(solutions):
        ax22.bar(x[i], unstructured['MobileNet'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    #ax1.set_ylabel('Latency (ms)')
    ax22.set_xlabel('MobileNet')
    ax22.set_yticks([0, 4, 8, 12])

    ax22.tick_params(axis='y', direction='in')
    ax22.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax22.spines['top'].set_visible(False)
    ax22.spines['right'].set_visible(False)

    #====================


    # divider = make_axes_locatable(ax23)
    # ax23_2 = divider.new_vertical(size="100%", pad=0.1)
    # fig.add_axes(ax23_2)

    for i, solution in enumerate(solutions):
        ax23.bar(x[i], unstructured['HuBERT'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    ax23.set_xlabel('HuBERT')
    ax23.set_yticks([0, 50, 100])

    ax23.tick_params(axis='y', direction='in')
    ax23.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax23.spines['top'].set_visible(False)
    ax23.spines['right'].set_visible(False)
    ax23.set_yticks([0,50,100,150])
    # ax23.set_ylim(0,100)

    # for i, solution in enumerate(solutions):
    #     ax23_2.bar(x[i], unstructured['HuBERT'][i], width, label = solution,
    #             hatch=hatches[i], color=colors[i], edgecolor='black')

    # ax23_2.set_ylim(4031, 4050)
    # ax23_2.tick_params(bottom=False, labelbottom=False)
    # ax23_2.spines['bottom'].set_visible(False)
    # ax23_2.spines['top'].set_visible(False)
    # ax23_2.spines['right'].set_visible(False)

    # ax23_2.tick_params(axis='y', direction='in')

    # # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax23_2.transAxes, color='k', clip_on=False)
    # ax23_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax23.transAxes)  # switch to the bottom axes
    # ax23.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    #====================
    # third row
    structured8bit = {'BERT': [data['bert_coarse_jit'], data['bert_coarse_int8_trt'], data['bert_coarse_int8_tvm'], data['bert_coarse_int8_tvm-s'], data['bert_coarse_int8_rammer'], data['bert_coarse_int8_rammer-s'], data['bert_coarse_int8_sparta']],
                'MobileNet': [data['mobilenet_coarse_int8_jit'], data['mobilenet_coarse_int8_trt'], data['mobilenet_coarse_int8_tvm'], data['mobilenet_coarse_int8_tvm-s'], data['mobilenet_coarse_int8_rammer'], data['mobilenet_coarse_int8_rammer-s'], data['mobilenet_coarse_int8_sparta']],
                'HuBERT': [data['hubert_coarse_int8_jit'], data['hubert_coarse_int8_trt'], data['hubert_coarse_int8_tvm'], data['hubert_coarse_int8_tvm-s'], data['hubert_coarse_int8_rammer'], data['hubert_coarse_int8_rammer-s'], data['hubert_coarse_int8_sparta']]}
    (ax31, ax32, ax33) = axes[2]

    # divider = make_axes_locatable(ax31)
    # ax31_2 = divider.new_vertical(size="100%", pad=0.1)
    # fig.add_axes(ax31_2)

    for i, solution in enumerate(solutions):
        ax31.bar(x[i], structured8bit['BERT'][i], width, label=solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    ax31.set_ylabel('Latency (ms)', labelpad=8, loc='bottom')
    ax31.set_xlabel('BERT')
    # ax31.set_yticks([0, 100, 200, 300])

    ax31.tick_params(axis='y', direction='in')
    ax31.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax31.spines['top'].set_visible(False)
    ax31.spines['right'].set_visible(False)

    # ax31.set_ylim(0, 130)
    # for i, solution in enumerate(solutions):
    #     ax31_2.bar(x[i], structured8bit['BERT'][i], width, label = solution,
    #             hatch=hatches[i], color=colors[i], edgecolor='black')

    # ax31_2.set_ylim(1749, 1770)
    # ax31_2.tick_params(bottom=False, labelbottom=False)
    # ax31_2.spines['bottom'].set_visible(False)
    # ax31_2.spines['top'].set_visible(False)
    # ax31_2.spines['right'].set_visible(False)

    # ax31_2.tick_params(axis='y', direction='in')

    # # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax31_2.transAxes, color='k', clip_on=False)
    # ax31_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax31.transAxes)  # switch to the bottom axes
    # ax31.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    #====================

    for i, solution in enumerate(solutions):
        ax32.bar(x[i], structured8bit['MobileNet'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    #ax1.set_ylabel('Latency (ms)')
    ax32.set_xlabel('MobileNet')
    ax32.set_yticks([0, 4, 8, 12])

    ax32.tick_params(axis='y', direction='in')
    ax32.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax32.spines['top'].set_visible(False)
    ax32.spines['right'].set_visible(False)

    #====================

    # divider = make_axes_locatable(ax33)
    # ax33_2 = divider.new_vertical(size="100%", pad=0.1)
    # fig.add_axes(ax33_2)

    for i, solution in enumerate(solutions):
        ax33.bar(x[i], structured8bit['HuBERT'][i], width, label = solution,
                hatch=hatches[i], color=colors[i], edgecolor='black')

    ax33.set_xlabel('HuBERT')
    ax33.set_ylim(0, 170)
    ax33.set_yticks([0, 50, 100, 150])
    ax33.spines['top'].set_visible(False)
    ax33.spines['right'].set_visible(False)

    ax33.tick_params(axis='y', direction='in')
    ax33.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # for i, solution in enumerate(solutions):
    #     ax33_2.bar(x[i], structured8bit['HuBERT'][i], width, label = solution,
    #             hatch=hatches[i], color=colors[i], edgecolor='black')

    # ax33_2.set_ylim(3910, 4010)
    # ax33_2.tick_params(bottom=False, labelbottom=False)
    # ax33_2.spines['bottom'].set_visible(False)
    # ax33_2.spines['top'].set_visible(False)
    # ax33_2.spines['right'].set_visible(False)

    # ax33_2.tick_params(axis='y', direction='in')

    # # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax33_2.transAxes, color='k', clip_on=False)
    # ax33_2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # #ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax33.transAxes)  # switch to the bottom axes
    # ax33.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # #ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    #====================

    handles, labels = ax33.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor = (0.5, 0.72), loc='lower center',
            borderaxespad=0, ncol=4, columnspacing=1.0, handletextpad=0.4, frameon=False)
    fig.subplots_adjust(top=0.7, hspace=0.6)

    #fig.suptitle('Structured Sparsity', x=0.5, y=0.42, fontsize='medium')
    fig.text(0.5, 0.515, 'Structured Sparsity', fontsize='medium',
            horizontalalignment='center', verticalalignment='center')
    fig.text(0.5, 0.285, 'Unstructured Sparsity', fontsize='medium',
            horizontalalignment='center', verticalalignment='center')
    fig.text(0.5, 0.055, 'Structured+8bit Sparsity', fontsize='medium',
            horizontalalignment='center', verticalalignment='center')

    #fig.tight_layout()


    fig.savefig("nvidia_three_sparsity.pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    models = ['bert', 'mobilenet', 'hubert']
    patterns = ['coarse', 'coarse_int8', 'finegrained']
    # frameworks = ['jit']
    frameworks = ['jit', 'tvm', 'tvm-s', 'rammer', 'rammer-s', 'sparta', 'trt']
    data = {}
    for model in models:
        for pattern in patterns:
            for framework in frameworks:
                file_name = f'{model}_{pattern}_{framework}.log'
                time = 0.0
                try:
                    tmp = func_map[framework](os.path.join('log', file_name))
                    if tmp is not None:
                        time = tmp
                except Exception as err:
                    print(file_name, err)
                data[f'{model}_{pattern}_{framework}'] = time
    draw_figure8(data)