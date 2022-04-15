import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import matplotlib as mpl
#mpl.style.use('ggplot')
# mpl.style.use('seaborn-whitegrid')
#mpl.style.use('grayscale')
def get_time_ms(fpath):
    lines = []
    kernel_name = "Time="
    with open(fpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.find(kernel_name) == -1:
            continue
        else:
            run_time = float(line.split()[1])
            break
    return run_time*1000 # convert to us
def get_time_us(fpath):
    lines = []
    kernel_name = "Time="
    with open(fpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.find(kernel_name) == -1:
            continue
        else:
            run_time = float(line.split()[1])
            break
    return run_time # convert to us


log_data = {}
for framework in ['cusparse', 'sputnik', 'cublas']:
    log_data[framework] = {}
    for sparsity  in ['0.5', '0.7', '0.8', '0.9', '0.95', '0.99']:
        fpath = f'log/{framework}_{sparsity}.log'
        run_time = get_time_ms(fpath)
        log_data[framework][sparsity] = run_time
for framework in ['sparta']:
    log_data[framework] = {}
    for sparsity  in ['0.5', '0.7', '0.8', '0.9', '0.95', '0.99']:
        fpath = f'log/{framework}_{sparsity}.log'
        run_time = get_time_us(fpath)
        log_data[framework][sparsity] = run_time
with open('log/taco_latency.txt', 'r') as f:
    lines = f.readlines()
    log_data['taco'] = {}
    for i, sparsity in enumerate(['0.5', '0.7', '0.8', '0.9', '0.95', '0.99']):
        tmp = re.split(' ', lines[i])
        log_data['taco'][sparsity] = float(tmp[1])
        pass


plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)

font = {'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)

hatches = ['///', '\\\\\\', '--', 'xxx', 'oo', '|||', None]
colors = [ 'white', 'whitesmoke', 'lightgray', 'gray', 'black']
# MobileNet
jct = {"cu-50": log_data['cusparse']['0.5'], "cu-70":log_data['cusparse']['0.7'], "cu-80": log_data['cusparse']['0.8'], "cu-90": log_data['cusparse']['0.9'], "cu-95": log_data['cusparse']['0.95'], "cu-99": log_data['cusparse']['0.99'],
       "spgen-50": log_data['sparta']['0.5'], "spgen-70":log_data['sparta']['0.7'], "spgen-80": log_data['sparta']['0.8'], "spgen-90": log_data['sparta']['0.9'], "spgen-95": log_data['sparta']['0.95'], "spgen-99": log_data['sparta']['0.99'],
        "tc-50": log_data['taco']['0.5'], "tc-70":log_data['taco']['0.7'], "tc-80":log_data['taco']['0.8'], "tc-90":log_data['taco']['0.9'], "tc-95":log_data['taco']['0.95'], "tc-99":log_data['taco']['0.99'], "rt-50":344.99, "rt-70":214.5,
        "rt-80":159.94, "rt-90":105.38, "rt-95":82.144, "rt-99":30.271, "[27]-50":log_data['sputnik']['0.5'] , "[27]-70":log_data['sputnik']['0.7'], "[27]-80":log_data['sputnik']['0.8'], "[27]-90":log_data['sputnik']['0.9'],
        "[27]-95": log_data['sputnik']['0.95'], "[27]-99": log_data['sputnik']['0.99']


}
cublas = log_data['cublas']['0.5']
x_ticks = ['50%', '70%', '80%', '90%', '95%', '99%' ]
mpl.rcParams['figure.figsize'] = (8, 3.5)
fig = plt.figure()
ylimit = 2000
ax = fig.add_subplot(111)
# ax2 = ax.twinx()
divider = make_axes_locatable(ax)
ax2 = divider.new_vertical(size="50%", pad=0.1)
fig.add_axes(ax2)

# ax.grid(True, axis='y', which='major', linestyle='--', alpha=0.5)
# ax.grid(False, axis='x')
# ax2.grid(False, axis='y')
bar_width = 0.15
x_pos = np.arange(1, len(x_ticks)+1, 1)
# x3 = x_pos
# x2 = x3 - bar_width
# x1 = x2 - bar_width
# x4 = x3 + bar_width
# x5 = x4 + bar_width
# # print(x1)
# ax.bar(x1, [jct['cu-50'], jct['cu-70'], jct['cu-80'], jct['cu-90'],jct['cu-95'], jct['cu-99']], label='cuSparse', width=bar_width, hatch=hatches[0],  color = colors[0], edgecolor='black')
# ax.bar(x2, [jct['tc-50'], jct['tc-70'], jct['tc-80'], jct['tc-90'],jct['tc-95'], jct['tc-99']], label='TACO', hatch=hatches[1],  width=bar_width, color = colors[1], edgecolor='black')
# ax.bar(x3, [jct['[27]-50'], jct['[27]-70'], jct['[27]-80'], jct['[27]-90'],jct['[27]-95'], jct['[27]-99']], label='Sputnik', hatch=hatches[2],  width=bar_width, color = colors[2], edgecolor='black')
# ax.bar(x4, [jct['rt-50'], jct['rt-70'], jct['rt-80'], jct['rt-90'],jct['rt-95'], jct['rt-99']], label='SparseRT', hatch=hatches[3],  width=bar_width, color = colors[3], edgecolor='black')
# tc_keys = ['tc-50', 'tc-70','tc-80','tc-90','tc-95','tc-99']
# for i in range(len(x2)):
#     ax.text(x2[i]+bar_width/2, ylimit-150, str('%.1f'%jct[tc_keys[i]]), fontsize=12)
# ax.bar(x5, [jct['spgen-50'], jct['spgen-70'], jct['spgen-80'], jct['spgen-90'],jct['spgen-95'], jct['spgen-99']], label='SparGen', hatch=hatches[4],  width=bar_width,color = colors[4], edgecolor='black')



x2 = x_pos - 0.5 * bar_width
x1 = x2 - bar_width
x3 = x2 + bar_width
x4 = x3 + bar_width
# print(x1)
ax.bar(x1, [jct['cu-50'], jct['cu-70'], jct['cu-80'], jct['cu-90'],jct['cu-95'], jct['cu-99']], label='cuSparse', width=bar_width, hatch=hatches[0],  color = colors[0], edgecolor='black')
ax.bar(x2, [jct['tc-50'], jct['tc-70'], jct['tc-80'], jct['tc-90'],jct['tc-95'], jct['tc-99']], label='taco', hatch=hatches[1],  width=bar_width, color = colors[1], edgecolor='black')
ax.bar(x3, [jct['[27]-50'], jct['[27]-70'], jct['[27]-80'], jct['[27]-90'],jct['[27]-95'], jct['[27]-99']], label='Sputnik', hatch=hatches[2],  width=bar_width, color = colors[2], edgecolor='black')
# ax.bar(x4, [jct['rt-50'], jct['rt-70'], jct['rt-80'], jct['rt-90'],jct['rt-95'], jct['rt-99']], label='SparseRT', hatch=hatches[3],  width=bar_width, color = colors[3], edgecolor='black')
tc_keys = ['tc-50', 'tc-70','tc-80','tc-90','tc-95','tc-99']
# for i in range(len(x2)):
#     ax.text(x2[i]+bar_width/2, ylimit-150, str('%.1f'%jct[tc_keys[i]]), fontsize=12)
ax.bar(x4, [jct['spgen-50'], jct['spgen-70'], jct['spgen-80'], jct['spgen-90'],jct['spgen-95'], jct['spgen-99']], label='SparGen', hatch=hatches[4],  width=bar_width,color = colors[4], edgecolor='black')
ax.set_ylim(0, 820)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax2.bar(x1, [jct['cu-50'], jct['cu-70'], jct['cu-80'], jct['cu-90'],jct['cu-95'], jct['cu-99']], label='cuSparse', width=bar_width, hatch=hatches[0],  color = colors[0], edgecolor='black')
ax2.bar(x2, [jct['tc-50'], jct['tc-70'], jct['tc-80'], jct['tc-90'],jct['tc-95'], jct['tc-99']], label='taco', hatch=hatches[1],  width=bar_width, color = colors[1], edgecolor='black')
ax2.bar(x3, [jct['[27]-50'], jct['[27]-70'], jct['[27]-80'], jct['[27]-90'],jct['[27]-95'], jct['[27]-99']], label='Sputnik', hatch=hatches[2],  width=bar_width, color = colors[2], edgecolor='black')
# ax.bar(x4, [jct['rt-50'], jct['rt-70'], jct['rt-80'], jct['rt-90'],jct['rt-95'], jct['rt-99']], label='SparseRT', hatch=hatches[3],  width=bar_width, color = colors[3], edgecolor='black')
tc_keys = ['tc-50', 'tc-70','tc-80','tc-90','tc-95','tc-99']
# for i in range(len(x2)):
#     ax.text(x2[i]+bar_width/2, ylimit-150, str('%.1f'%jct[tc_keys[i]]), fontsize=12)
ax2.bar(x4, [jct['spgen-50'], jct['spgen-70'], jct['spgen-80'], jct['spgen-90'],jct['spgen-95'], jct['spgen-99']], label='SparTA', hatch=hatches[4],  width=bar_width,color = colors[4], edgecolor='black')

ax2.set_ylim(820, 6600)
ax2.tick_params(bottom=False, labelbottom=False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

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




x_from = min(x1)- bar_width
x_to = max(x4)+bar_width
ax.plot((x_from, x_to), (cublas, cublas), linestyle='--', label='cuBLAS', color='gray')
ax2.plot((x_from, x_to), (cublas, cublas), linestyle='--', label='cuBLAS', color='gray')

# ax.bar([1.0, 1.7],[jct['cu-50'], jct['spgen-50'] ], label = 'sparsity-50%', width = 0.1, zorder=3, hatch='x',  color = 'lightblue')
# ax.bar([1.1, 1.8],[jct['cu-70'], jct['spgen-70'] ], label = 'sparsity-70%', width = 0.1, zorder=3, hatch='\\',  color = 'slategray')
# ax.bar([1.2, 1.9],[jct['cu-80'], jct['spgen-80'] ], label = 'sparsity-80%', width = 0.1, zorder=3, hatch='/', edgecolor = 'white', color = 'tan')
# ax.bar([1.3, 2.0],[jct['cu-90'], jct['spgen-90'] ], label = 'sparsity-90%', width = 0.1, zorder=3, hatch='+', edgecolor = 'white', color = 'tan')
# ax.bar([1.4, 2.1],[jct['cu-95'], jct['spgen-95'] ], label = 'sparsity-95%', width = 0.1, zorder=3, hatch='o', edgecolor = 'white', color = 'tan')
# ax.bar([1.5, 2.2],[jct['cu-99'], jct['spgen-99'] ], label = 'sparsity-99%', width = 0.1, zorder=3, hatch='-', edgecolor = 'white', color = 'tan')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_ticks)
# for bar_idx in range(12):
#     bar = ax.patches[bar_idx]
#     bar.set_edgecolor([1.0,1.0,1.0])


#ax.text(bar_lru[0].get_x()-0.03, 1.35*bar_lru[0].get_height(), "3.56x", fontsize=13, rotation=80)
#ax.text(bar_lru[1].get_x()-0.03, 1.7*bar_lru[1].get_height(), "1.69x", fontsize=13, rotation=80)
#ax.text(bar_lru[2].get_x()-0.03, 1.7*bar_lru[1].get_height(), "1.82x", fontsize=13, rotation=80)

#ax.text(bar_uniform[0].get_x()+0.0, 1.11*bar_uniform[0].get_height(), "7.4x", fontsize=13, rotation=80)
#ax.text(bar_uniform[1].get_x()+0.03, 1.5*bar_uniform[1].get_height(), "2.74x", fontsize=13, rotation=80)
#ax.text(bar_uniform[2].get_x()+0.03, 1.5*bar_uniform[1].get_height(), "3.11x", fontsize=13, rotation=80)

#plt.xticks([1.1, 1.6], ['Avg. JCT', 'Makespan'], rotation='0')

# ax2.bar([1.25,2.25,3.25],[makespan['FIFO-SiloD'], makespan['FIFO-Uniform'], makespan['FIFO-LRU'],], label = 'Makespan\n  (right)',  hatch="/", width = 0.25, zorder=3, color = '#A25951')
# plt.xticks([1.2, 2.0], ['cuSPARSE', 'SparGen'], rotation='0')
# ax.set_yticks([0, 0.25e3, 0.5e3, 0.75e3, 1.0e3, 1.25e3, 1.5e3])
# ax2.set_yticks([0, 1.5e3, 3e3, 4.5e3, 6e3, 7.5e3, 9e3])
# plt.tick_params(axis="y",direction="in")
# plt.tick_params(axis="x",direction="in")

ax.set_xlim(min(x1)- bar_width, max(x4)+bar_width)
ax2.set_xlim(min(x1)- bar_width, max(x4)+bar_width)

# ax2.set_ylim(0,10500)
ax.set_ylabel(' Latency (us)', fontsize=17, loc="bottom")
ax.set_xlabel('Sparsity Ratio', fontsize=17)
# ax2.set_ylabel('Makespan (mins)')
ax2.legend(loc="lower center", ncol=5, bbox_to_anchor=[0.5, 1.02], fontsize =14, columnspacing=1.0, handletextpad=0.4, frameon=False)
# ax2.legend(loc="upper left",fontsize =13,  bbox_to_anchor= (0.45, 1.055))
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.tick_params(axis='x', which='major', pad=15)
plt.tight_layout()
# plt.show()
plt.savefig('micro_sparse.pdf', dpi=1000)
#plt.savefig('micro_sparse.jpg')
plt.show()
