import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = {}

for model in ['bert', 'mobilenet', 'hubert']:
    with open(f'{model}_data.json', 'r') as f:
        _tmp = json.load(f)
        data.update(_tmp)

models = ['BERT', 'MobileNet', 'HuBERT']
solutions = ['PyTorch', 'TensorRT', 'TVM', 'TVM-S', 'Rammer', 'Rammer-S', 'SparTA']
structured = {'BERT': [data['bert_coarse_jit'], data['bert_coarse_trt'], data['bert_coarse_tvm'], data['bert_coarse_tvm-s'], data['bert_coarse_rammer'], data['bert_coarse_rammer-s'], data['bert_coarse_sparta']],
            'MobileNet': [data['mobilenet_coarse_jit'], data['mobilenet_coarse_trt'], data['mobilenet_coarse_tvm'], data['mobilenet_coarse_tvm-s'], data['mobilenet_coarse_rammer'], data['mobilenet_coarse_rammer-s'], data['mobilenet_coarse_sparta']],
            'HuBERT': [data['hubert_coarse_jit'], data['hubert_coarse_trt'], data['hubert_coarse_tvm'], data['hubert_coarse_tvm-s'], data['hubert_coarse_rammer'], data['hubert_coarse_rammer-s'], data['hubert_coarse_sparta']]}


hatches = ['///', '\\\\\\', '--', 'xxx', 'oo', '|||', None]
colors = ['white', 'white', 'white', 'whitesmoke', 'lightgray', 'gray', 'black']


x = np.arange(len(solutions))
width = 0.75

fig, axes = plt.subplots(3, 3, figsize=[6.4, 1.8*3])

(ax1, ax2, ax3) = axes[0]

#====================
# first row

for i, solution in enumerate(solutions):
    ax1.bar(x[i], structured['BERT'][i], width, label=solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

ax1.set_ylabel('Memory (MB)')
ax1.set_xlabel('BERT')
ax1.set_yticks([0, 3000, 5000])

ax1.tick_params(axis='y', direction='in')
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

#====================

for i, solution in enumerate(solutions):
    ax2.bar(x[i], structured['MobileNet'][i], width, label = solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

#ax1.set_ylabel('Latency (ms)')
ax2.set_xlabel('MobileNet')
#ax2.set_yticks([0, 4, 8, 12])

ax2.tick_params(axis='y', direction='in')
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax2.set_yticks([0, 2000, 3000])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

#====================

for i, solution in enumerate(solutions):
    ax3.bar(x[i], structured['HuBERT'][i], width, label = solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

ax3.set_xlabel('HuBERT')
#ax3.set_yticks([0, 50, 100])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_yticks([0, 1000, 2000])

ax3.tick_params(axis='y', direction='in')
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

#====================
# second row
unstructured = {'BERT': [data['bert_finegrained_jit'], data['bert_finegrained_trt'], data['bert_finegrained_tvm'], data['bert_finegrained_tvm-s'], data['bert_finegrained_rammer'], data['bert_finegrained_rammer-s'], data['bert_finegrained_sparta']],
            'MobileNet': [data['mobilenet_finegrained_jit'], data['mobilenet_finegrained_trt'], data['mobilenet_finegrained_tvm'], data['mobilenet_finegrained_tvm-s'], data['mobilenet_finegrained_rammer'], data['mobilenet_finegrained_rammer-s'], data['mobilenet_finegrained_sparta']],
            'HuBERT': [data['hubert_finegrained_jit'], data['hubert_finegrained_trt'], data['hubert_finegrained_tvm'], data['hubert_finegrained_tvm-s'], data['hubert_finegrained_rammer'], data['hubert_finegrained_rammer-s'], data['hubert_finegrained_sparta']]}

(ax21, ax22, ax23) = axes[1]

for i, solution in enumerate(solutions):
    ax21.bar(x[i], unstructured['BERT'][i], width, label=solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

ax21.set_ylabel('Memory (MB)')
ax21.set_xlabel('BERT')
#ax21.set_yticks([0, 100, 200, 300])
ax21.set_yticks([0, 3000, 5000])

ax21.tick_params(axis='y', direction='in')
ax21.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax21.spines['top'].set_visible(False)
ax21.spines['right'].set_visible(False)

#====================

for i, solution in enumerate(solutions):
    ax22.bar(x[i], unstructured['MobileNet'][i], width, label = solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

#ax1.set_ylabel('Latency (ms)')
ax22.set_xlabel('MobileNet')
#ax22.set_yticks([0, 4, 8, 12])
ax22.set_yticks([0, 2000, 3000])

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

for i, solution in enumerate(solutions):
    ax23.bar(x[i], unstructured['HuBERT'][i], width, label = solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

ax23.set_xlabel('HuBERT')
#ax23.set_yticks([0, 50, 100])

ax23.tick_params(axis='y', direction='in')
ax23.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax23.set_yticks([0, 1000, 2000])

ax23.spines['top'].set_visible(False)
ax23.spines['right'].set_visible(False)

#====================
# third row
structured8bit = {'BERT': [data['bert_coarse_jit'], data['bert_coarse_int8_trt'], data['bert_coarse_int8_tvm'], data['bert_coarse_int8_tvm-s'], data['bert_coarse_int8_rammer'], data['bert_coarse_int8_rammer-s'], data['bert_coarse_int8_sparta']],
                'MobileNet': [data['mobilenet_coarse_int8_jit'], data['mobilenet_coarse_int8_trt'], data['mobilenet_coarse_int8_tvm'], data['mobilenet_coarse_int8_tvm-s'], data['mobilenet_coarse_int8_rammer'], data['mobilenet_coarse_int8_rammer-s'], data['mobilenet_coarse_int8_sparta']],
                'HuBERT': [data['hubert_coarse_int8_jit'], data['hubert_coarse_int8_trt'], data['hubert_coarse_int8_tvm'], data['hubert_coarse_int8_tvm-s'], data['hubert_coarse_int8_rammer'], data['hubert_coarse_int8_rammer-s'], data['hubert_coarse_int8_sparta']]}

(ax31, ax32, ax33) = axes[2]

for i, solution in enumerate(solutions):
    ax31.bar(x[i], structured8bit['BERT'][i], width, label=solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

ax31.set_ylabel('Memory (MB)')
ax31.set_xlabel('BERT')
#ax31.set_yticks([0, 100, 200, 300])
ax31.set_yticks([0, 3000, 5000])

ax31.tick_params(axis='y', direction='in')
ax31.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax31.spines['top'].set_visible(False)
ax31.spines['right'].set_visible(False)

#====================

for i, solution in enumerate(solutions):
    ax32.bar(x[i], structured8bit['MobileNet'][i], width, label = solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

#ax1.set_ylabel('Latency (ms)')
ax32.set_xlabel('MobileNet')
#ax32.set_yticks([0, 4, 8, 12])
ax32.set_yticks([0, 2000, 3000])

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

for i, solution in enumerate(solutions):
    ax33.bar(x[i], structured8bit['HuBERT'][i], width, label = solution,
            hatch=hatches[i], color=colors[i], edgecolor='black')

ax33.set_xlabel('HuBERT')
#ax33.set_yticks([0, 50, 100])
ax33.spines['top'].set_visible(False)
ax33.spines['right'].set_visible(False)
ax33.set_yticks([0, 1000, 2000])

ax33.tick_params(axis='y', direction='in')
ax33.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

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


fig.savefig("nvidia_three_sparsity_mem.pdf", bbox_inches='tight')

plt.show()
