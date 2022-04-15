#matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pylab import rcParams
import matplotlib as mpl
import csv


# plt.rc('xtick', labelsize=15) 
# plt.rc('ytick', labelsize=15)

# font = {'weight' : 'normal',
#         'size'   : 20}
# plt.rc('font', **font)

# #data = pd.read_csv('share.csv')
# data = {
#     "layer": [1, 2, 3, 4],

#     "b-50": [0.5,	0.5312,	0.7344,	0.5],
#     "b-60": [0.6,	0.6211,	0.7578,	0.6],
#     "b-70": [0.7,	0.8164,	0.8906,	0.7],
#     "b-80": [0.8,	0.9336,	0.9453,	0.8],
#     "b-90": [0.9453,	0.9961,	0.9922,	0.9],

#     "f-50": [0.5,	0.5,	0.5029,	0.5],
#     "f-60": [0.6,	0.6,	0.6039,	0.6],
#     "f-70": [0.7,	0.7,	0.7081,	0.7],
#     "f-80": [0.8,	0.8,	0.8214,	0.8],
#     "f-90": [0.9,	0.9,	0.9344,	0.9],
#     "f-95": [0.95,	0.95,	0.9797,	0.95],
#     "f-98": [0.98,	0.983,	0.9963,	0.98],
#     "f-99": [0.99,	0.9962,	0.999,	0.99],

#     "c-50": [0.5,	0.75,	0.75,	0.75],
#     "c-60": [0.6,	0.8397,	0.8397,	0.8397],
#     "c-70": [0.7,	0.9095,	0.9095,	0.9098],
#     "c-80": [0.8,	0.9599,	0.9599,	0.9599],
#     "c-90": [0.9,	0.9899,	0.9899,	0.9899],
# }

# mpl.rcParams['figure.figsize'] = (4, 3.5)
ratios = [0.5, 0.7, 0.9]
labels = ['50%', '70%', '90%']
pattern=['--.', 'v-', 'x-', 'o-', '^-', 'x-']
colors = ['black','black','black','black','black','black','black']
sizes= [7,7,8]
data = {}
with open('finegrained_propagate.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(iter(reader))
    for row in reader:
        print(row[0])
        sparsity = float(row[0])
        value = [float(x) for x in row[1:]]
        data[sparsity] = value

fig = plt.figure(figsize= (12,3))
ax = fig.add_subplot(111)
x = np.arange(1, len(data[0.5])+1, 1)
for i,_ in enumerate(ratios):
    key = ratios[i]
    plt.plot(x, data[key], pattern[i], label=labels[i], color=colors[i],markersize=sizes[i])
ax.set_xlim(min(x)-1, max(x)+1)
ax.set_xticks([0,10,20])
ax.set_xticklabels([0,10,20], fontsize=17)
# plt.plot(data['layer'], data['c-50'], 'o-', label='50%',  color = '#E0BB81')
# plt.plot(data['layer'], data['c-60'],'v-', label='60%', color = '#A25951')
# plt.plot(data['layer'], data['c-70'],'s-' , label='70%',  color = '#5C412B')
# plt.plot(data['layer'], data['c-80'],'x-' , label='80%',  color = 'lightblue')
# plt.plot(data['layer'], data['c-90'],'^-' , label='90%',  color = 'tan')

# plt.xticks(data['layer'], [1,2,3,4], fontsize=17)
#plt.yticks([1.6e3,2.0e3,2.5e3,3e3],fontsize=20)
plt.ylabel('Sparsity Ratio',fontsize=17)
plt.xlabel('Layer Index', fontsize=17)
#plt.ylim([1.6e3,3.0e3])
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.5, 0.6, 0.7, 0.8, 0.9,1.0], fontsize=17)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.yaxis.get_offset_text().set_fontsize(24)

# from matplotlib.font_manager import FontProperties
# fontP = FontProperties()
# fontP.set_size('xx-small')

plt.legend(bbox_to_anchor=(0.5, 1.01), ncol=6, loc='lower center', fontsize=17)
plt.tight_layout()
plt.savefig('propagate_sparsity_fine.pdf',dpi=1000)
#plt.savefig('propagate_sparsity_coarse.jpg')
# plt.show()