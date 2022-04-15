#matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from pylab import rcParams
import matplotlib as mpl
#mpl.style.use('ggplot')
# mpl.style.use('seaborn-whitegrid')
# mpl.style.use('grayscale')

with open('bit_propagation.csv', 'r') as f:
    csv_reader = csv.reader(f)

    before = next(iter(csv_reader))[1:29]
    after = next(iter(csv_reader))[1:29]

before = [int(x) for x in before]
after = [int(x) for x in after]
print(len(before))
print(len(after))
fig = plt.figure(figsize=(8, 3))

x = np.arange(0, len(before))

plt.plot(x, before, 'o--', label='Before Propagation', color='black', zorder=10, dashes=(5, 5), linewidth=2)
plt.plot(x, after, 'v-', label='After Propagation', color='gray', zorder=0)

plt.xticks([0,10,20], [0,10,20], fontsize=18)
plt.yticks(fontsize=17)
#plt.yticks([1.6e3,2.0e3,2.5e3,3e3],fontsize=20)
plt.ylabel('Bit Width',fontsize=17)
plt.xlabel('Layer Index', fontsize=17)
#plt.ylim([1.6e3,3.0e3])
# plt.ticklabel_format(axis="y", style="plain", scilimits=(0,0)
#from matplotlib.font_manager import FontProperties
#fontP = FontProperties()
#fontP.set_size('xx-small')

plt.legend(bbox_to_anchor=(0.5, 1.01), ncol=2, loc='lower center', fontsize=16, frameon=False)#, prop=fontP)
plt.tight_layout()
# plt.show()
plt.savefig('propagate_precision.pdf', dpi=1000)
#plt.savefig('propagate_precision.jpg')
# plt.show()
