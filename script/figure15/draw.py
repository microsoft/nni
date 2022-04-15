import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl

with open('raw_data.json', 'r') as f:
    data = json.load(f)

data1 = {
        'openai':[data['openai'][0], data['openai'][1], data['openai'][2]],
        'sputnik': [data['sputnik'][0], data['sputnik'][1], data['sputnik'][2]],
        'spargen': [data['sparta'][0], data['sparta'][1], data['sparta'][2]]
}
data2 = {
        'openai':[data['openai'][0], data['openai'][1], data['openai'][2]],
        'sputnik': [data['sputnik'][0], data['sputnik'][1], data['sputnik'][2]],
        'spargen': [data['sparta_int8'][0], data['sparta_int8'][1], data['sparta_int8'][2]]

}

fig, axes = plt.subplots(1, 2, figsize=[8, 2.5])
hatches = ['///', '---', None]
colors = ['white', 'lightgray', 'black']



ax = axes[0]


X = np.arange(1, 4)
bar_width = 0.2
x1=X-bar_width
x2=X
x3=x2+bar_width
x_ticklabels = [ '70%-block', '80%-block', '90%-block']
x_ticklabels2 = [ '70%-block(8bit)', '80%-block(8bit)', '90%-block(8bit)']

ax.bar(x1, data1['openai'],width=bar_width, label='BlockSparse Kernel', color=colors[0], hatch=hatches[0], edgecolor='black')
ax.bar(x2, data1['sputnik'],width=bar_width, label='Sputnik', color=colors[1], hatch=hatches[1],edgecolor='black')
ax.bar(x3, data1['spargen'],width=bar_width, label='SparTA', color=colors[2], hatch=hatches[2],edgecolor='black')

ax.set_xticks(X)
ax.set_xticklabels(x_ticklabels, rotation=10, fontsize=14)
ax.set_yticks([0.05, 0.15, 0.25])
ax.set_yticklabels([0.05, 0.15, 0.25], fontsize=15)
ax.set_xlim(x1[0]-bar_width,x3[2]+bar_width)
ax.set_ylabel('Latency (ms)', fontsize=17)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = axes[1]
ax2.bar(x1, data2['openai'],width=bar_width, label='BlockSparse Kernel', color=colors[0], hatch=hatches[0], edgecolor='black')
ax2.bar(x2, data2['sputnik'],width=bar_width, label='Sputnik', color=colors[1], hatch=hatches[1],edgecolor='black')
ax2.bar(x3, data2['spargen'],width=bar_width, label='SparTA', color=colors[2], hatch=hatches[2],edgecolor='black')
ax2.set_xticks(X)
ax2.set_xticklabels(x_ticklabels2, rotation=10, fontsize=14)
ax2.set_yticks([0.05, 0.15, 0.25])
ax2.set_yticklabels([0.05, 0.15, 0.25], fontsize=15)
ax2.set_xlim(x1[0]-bar_width,x3[2]+bar_width)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.set_ylabel('Latency (ms)', fontsize=17)

# fig.legend(  loc='lower center', bbox_to_anchor = (0.6, 0.98), fontsize=14, ncol=3)
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor = (0.5, 0.91), loc='lower center',fontsize=15, ncol=3,borderaxespad=0,columnspacing=1.0, handletextpad=0.4,frameon=False)
fig.subplots_adjust( hspace=0.6)

fig.savefig("mix_pattern.pdf", bbox_inches='tight')
# plt.tight_layout()
plt.show()
