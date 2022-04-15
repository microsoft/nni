
import matplotlib.pyplot as plt
import os
import re
import matplotlib
import numpy as np
# matplotlib.use('Agg')


def load_data(path):
    training_time = []
    total_time = None
    with open(path, 'r') as logf:
        lines = logf.readlines()
        for line in lines:
            if line.startswith('Training Time Cost:'):
                line = re.split(':', line)
                _time = float(line[1])
                training_time.append(_time)
            elif line.startswith('Total Time:'):
                line = re.split(':', line)
                total_time = float(line[1])
    return training_time, total_time


marlin, _ = load_data('./Iterative_SA.log')
baseline, _ = load_data('./Iterative_SA.log.baseline')

x = list(range(len(marlin)))
x = np.asarray(x)
plt.figure(figsize=(8, 3))

plt.plot(x, marlin, '-', label='With SparTA', color='black')
# plt.scatter(x, marlin, marker='v')
plt.plot(x, baseline, '--', label='Without SparTA',color='black', dashes=(5,6))
plt.legend(loc='lower center', fontsize=13, ncol=2, bbox_to_anchor=(0.5, 1.01), columnspacing=0.7, handletextpad=0.4, frameon=False)
zeros = np.asarray( [0] * len(marlin))
plt.yticks(fontsize=14)
# print(marlin > zeros)
plt.fill_between(x, marlin, zeros, where=marlin > zeros, color='dimgray', alpha=0.3)
plt.fill_between(x, baseline, zeros, where=baseline >
                 zeros, color='lightgray', alpha=0.3)
# plt.fill_between(x, marlin, baseline, where=marlin>0, color='g', alpha=0.3)
plt.xlabel('Training Epoch', fontsize=14)
plt.ylabel('Training Time\nPer Epoch(s)', fontsize=14)
plt.ylim(min(marlin)-10, max(baseline)+10)
plt.xlim(0, max(x))
plt.xticks(list(range(0, max(x), 200)), list(range(0, max(x), 200)), fontsize=14)
plt.plot((300, 300), (0, 75), linestyle='--', color='black')
plt.plot((600, 600), (0, 75), linestyle='--', color='black')
plt.tight_layout()
plt.savefig('./training_time.pdf', dpi=1000)
plt.show()
