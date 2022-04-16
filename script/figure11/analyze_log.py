import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylab import rcParams
import matplotlib as mpl
#mpl.style.use('ggplot')
# mpl.style.use('seaborn-whitegrid')
#mpl.style.use('grayscale')

# plt.rc('xtick', labelsize=13) 
# plt.rc('ytick', labelsize=13)

# font = {'weight' : 'normal',
#         'size'   : 20}
# plt.rc('font', **font)

def rammer_parse_log(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        result = list(filter(lambda x: 'Summary: [min, max, mean] =' in x, lines))
        assert len(result) == 1
        tmp = re.split(' ', result[0])
        return float(tmp[5][1:-1])

scenarios = ["Structured Sparsity", 'Unstructured Sparsity', 'Structured+8bit', 'Mixed Sparsity']

patterns = ["bert_coarse", "bert_finegrained", "bert_coarse_int8", "bert_mixed"]

optimizations = ["sparse_kernel", "propagation", "transformation", "specialization"]

parse_data = {}
for optimization in optimizations:
    tmp_result = []
    for pattern in patterns:
        log_name = os.path.join("log", pattern+"_"+optimization+'.log')
        tmp_time = 0
        try:
            tmp_time = rammer_parse_log(log_name)
        except Exception as err:
            print(err)
        tmp_result.append(tmp_time)
    parse_data[optimization] = tmp_result
dense_baseline = 80
try:
    dense_baseline = rammer_parse_log('log/bert_baseline.log')
except Exception as err:
    print(err)
data = {
        "Rammer":[dense_baseline, dense_baseline, dense_baseline, dense_baseline],
        "+Sparse Kernel":parse_data['sparse_kernel'],
        "+Propagation":parse_data['propagation'],
        "+Transformation":parse_data['transformation'],
        "+Specialization":parse_data['specialization']
}
keys = list(data.keys())
mpl.rcParams['figure.figsize'] = (8, 3)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax2 = ax.twinx()
divider = make_axes_locatable(ax)
ax2 = divider.new_vertical(size="35%", pad=0.1)

fig.add_axes(ax2)
x = np.arange(1,5,1)
bar_width=0.125
x1 = x - bar_width*2
x2 = x1 + bar_width
x3 = x2 + bar_width
x4 = x3 + bar_width
x5 = x4 + bar_width
hatches = ['///', '\\\\\\', '--', 'xxx', 'oo', '|||', None]
colors = ['white', 'white', 'white', 'whitesmoke', 'lightgray', 'gray', 'black']
ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.5)
ax.grid(False, axis='x')
# ax2.grid(False, axis='y')
ax.bar(x1, data[keys[0]], width=bar_width, label="Rammer", color=colors[0], edgecolor='black', hatch='///')
ax.bar(x2, data[keys[1]], width=bar_width, label=keys[1], edgecolor='black', color=colors[1], hatch='\\\\\\')
ax.bar(x3, data[keys[2]], width=bar_width, label=keys[2], edgecolor='black', color=colors[2], hatch='--',)
ax.bar(x4, data[keys[3]], width=bar_width, label=keys[3], edgecolor='black', color=colors[3], hatch='xxx')
ax.bar(x5, data[keys[4]], width=bar_width, label=keys[4], edgecolor='black', color=colors[4], hatch='oo')
ax.set_ylim(10, 50)
ax.set_xticks(x3)
ax.set_xticklabels(scenarios, fontsize=13, rotation=5)
ax.tick_params(axis='y', direction='in')
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
ax.set_yticks([10,45])
ax.set_yticklabels([10,45], fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2.set_ylim(60, 80)
ax2.set_yticks([70,80])
ax2.set_yticklabels([70,80], fontsize=14)
ax2.tick_params(bottom=False, labelbottom=False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.bar(x1, data[keys[0]], width=bar_width, label="Rammer", color=colors[0], edgecolor='black', hatch='///')
ax2.bar(x2, data[keys[1]], width=bar_width, label=keys[1], edgecolor='black', color=colors[1], hatch='\\\\\\')
ax2.bar(x3, data[keys[2]], width=bar_width, label=keys[2], edgecolor='black', color=colors[2], hatch='---',)
ax2.bar(x4, data[keys[3]], width=bar_width, label=keys[3], edgecolor='black', color=colors[3], hatch='xxx')
ax2.bar(x5, data[keys[4]], width=bar_width, label=keys[4], edgecolor='black', color=colors[4], hatch='oo')
ax2.tick_params(axis='y', direction='in')

# From https://matplotlib.org/examples/pylab_examples/broken_axis.html
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#ax3_2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# ax2.bar([1.25,2.25,3.25],[makespan['FIFO-SiloD'], makespan['FIFO-Uniform'], makespan['FIFO-LRU'],], label = 'Makespan\n  (right)',  hatch="/", width = 0.25, zorder=3, color = '#A25951')

# ax.set_yticks([0, 0.25e3, 0.5e3, 0.75e3, 1.0e3, 1.25e3, 1.5e3])
# ax2.set_yticks([0, 1.5e3, 3e3, 4.5e3, 6e3, 7.5e3, 9e3])
plt.tick_params(axis="y",direction="in")
plt.tick_params(axis="x",direction="in")
# plt.xlim(0.8,2.4)
# ax2.set_ylim(0,10500)
ax2.set_ylabel('Latency (ms)', fontsize=14, loc='top')
# ax2.set_ylabel('Makespan (mins)')
ax2.legend(loc="lower center", ncol=3, fontsize =12, bbox_to_anchor=(0.5, 0.98), borderaxespad=0, columnspacing=1.0, handletextpad=0.4, frameon=False,labelspacing=.3)
# ax2.legend(loc="upper left",fontsize =13,  bbox_to_anchor= (0.45, 1.055))
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.tick_params(axis='x', which='major', pad=15)
plt.tight_layout()
plt.savefig('breakdown_bert_2080ti.pdf', dpi=1000)

# plt.show()
