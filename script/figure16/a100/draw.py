import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
#mpl.style.use('ggplot')
# mpl.style.use('seaborn-whitegrid')
#mpl.style.use('grayscale')
def get_kernel_run_time(file_name):
    lines = []
    kernel_name = "Time="
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.find(kernel_name) == -1:
            continue
        else:
            run_time = float(line.split()[-2])
            break
    return run_time

log_data = {}
frameworks = ['sparta', 'sputnik', 'cublas', 'openai']
for framework in frameworks:
    log_data[framework] = []
    for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
        fpath = 'log/{}_{}.log'.format(framework, s)
        latency = get_kernel_run_time(fpath) * 1000
        log_data[framework].append(latency)

            

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)

font = {'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)

data = {
    'block': log_data['openai'],
    'sputnik': log_data['sputnik'],
    'sparta': log_data['sparta']
}
hatches = ['///',  '---', None]
colors = [ 'white', 'lightgray', 'black']
# MobileNet

cublas = 183.76
x_ticks = ['50%', '60%' ,'70%', '80%', '90%']
mpl.rcParams['figure.figsize'] = (8, 3)
fig = plt.figure()
ylimit = 2000
ax = fig.add_subplot(111)
bar_width = 0.15
x_pos = np.arange(1, len(x_ticks)+1, 1)


x2 = x_pos 
x1 = x2 - bar_width
x3 = x2 + bar_width
# print(x1)
ax.bar(x1, data['block'], label='BlockSparse Kernel', width=bar_width, hatch=hatches[0],  color = colors[0], edgecolor='black')
ax.bar(x2, data['sputnik'], label='Sputnik', hatch=hatches[1],  width=bar_width, color = colors[1], edgecolor='black')
ax.bar(x3, data['sparta'], label='SparTA', hatch=hatches[2],  width=bar_width, color = colors[2], edgecolor='black')
# ax.bar(x4, [jct['rt-50'], jct['rt-70'], jct['rt-80'], jct['rt-90'],jct['rt-95'], jct['rt-99']], label='SparseRT', hatch=hatches[3],  width=bar_width, color = colors[3], edgecolor='black')

x_from = min(x1)- bar_width
x_to = max(x3)+bar_width
ax.plot((x_from, x_to), (cublas, cublas), linestyle='--', label='cuBLAS', color='gray')

ax.set_xticks(x_pos)
ax.set_xticklabels(x_ticks)


ax.set_xlim(min(x1)- bar_width, max(x3)+bar_width)

# ax2.set_ylim(0,10500)
ax.set_ylabel(' Latency (us)', fontsize=17, loc="bottom")
ax.set_xlabel('Sparsity Ratio', fontsize=17)
# ax2.set_ylabel('Makespan (mins)')
ax.legend(loc="lower center", ncol=4, bbox_to_anchor=[0.5, 1.02], fontsize =16,  columnspacing=1.0, handletextpad=0.4, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.show()
plt.savefig('a100.pdf', dpi=1000)
plt.show()