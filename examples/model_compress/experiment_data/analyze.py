import json
import torch
# import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt

files = ['pruning_history_01.csv', 'pruning_history_03.csv', 'pruning_history_04.csv', 'pruning_history_05.csv']
# files = ['pruning_history.csv']

pruning_histories = []
performances = []
config_lists = []

for f in files:
    pruning_histories.append(pd.read_csv(f))


for history in pruning_histories:
    performances.append(history['performance'].max())
    idx = history['performance'].idxmax()
    print("idx: ", idx)
    config_list = history.loc[idx, 'config_list']
    # print(config_list)
    config_lists.append(json.loads(config_list))

plt.xticks(rotation=90)

for idx, config_list in enumerate(config_lists):
    layers = []
    sparsities = []
    for config in config_list:
        sparsities.append(config['sparsity'])
        layers.append(config['op_names'][0])


    plt.plot(layers, sparsities, scalex=True, scaley=True, label='sparsity: {}'.format(0.1*(2*idx+1)))

plt.title("Sparsities distribution")
plt.legend()
plt.savefig('./sparsities.png')

