import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import scipy.stats as stats
basepath = './results/'

names = []
names.append(['AlexNet', '{0}alexnet_dense_feat_data_layer_0.npy'.format(basepath), '{0}alexnet_sparse_feat_data_layer_0.npy'.format(basepath)])
#names.append(['VGG', 'VGG_dense_feat_data_layer_0.npy', 'VGG_sparse_feat_data_layer_0.npy'])
#names.append(['WRN-16-10', 'WRN-28-2_dense_feat_data_layer_0.npy', 'WRN-28-2_sparse_feat_data_layer_0.npy'])

# taken from: https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

layers = [5, 13, 24]


for (name, dense_path, sparse_path), max_layers in zip(names, layers):
    print(name, dense_path, sparse_path)
    densities = np.load(sparse_path.replace('feat_data_layer_0', 'density_data'))

    anova_all = []
    anova_all.append(['', 'y', 'layer_id', 'is_sparse'])
    data_id = 1
    sparse = []
    dense = []
    for layer_id in range(max_layers):

        dense_data = np.load(dense_path.replace('0', str(layer_id)))
        sparse_data = np.load(sparse_path.replace('0', str(layer_id)))
        density = densities[layer_id]

        for value in sparse_data:
            anova_all.append([data_id, value , layer_id, 1])
            data_id += 1
            sparse.append(value)

        for value in dense_data:
            anova_all.append([data_id, value , layer_id, 0])
            data_id += 1
            dense.append(value)





        hist, bins = np.histogram(dense_data, bins=np.linspace(0.09, 0.51, 50))
        hist2, bins2 = np.histogram(sparse_data, bins=np.linspace(0.09, 0.51, 50))

        xlim = np.max([np.max(hist), np.max(hist2)])

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        width2 = 0.7 * (bins2[1] - bins2[0])
        center2 = (bins2[:-1] + bins2[1:]) / 2


        fig, axes = plt.subplots(ncols=2, sharey=True)

        axes[0].barh(center, hist, align='center', height=width)
        axes[0].set(title='Dense')
        axes[1].barh(center2, hist2, align='center', height=width2)
        axes[1].set(title='Sparse')
        axes[0].set_xlim(0, xlim+5)
        axes[1].set_xlim(0, xlim+5)

        #axes[0].set_xlabel('Channel Count')
        axes[1].set_xlabel('Channel Count', x=0.0)
        axes[0].set_ylabel('Class-Specialization')

        axes[0].invert_xaxis()
        #axes[0].set(yticks=y, yticklabels=states)
        #axes[0].yaxis.tick_right()

        for ax in axes.flat:
            ax.margins(0.00)
            ax.grid(True)


        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        fig.subplots_adjust(wspace=0.0)
        title = '{0} Conv2D Layer {1}'.format(name, layer_id+1)

        plt.suptitle(title, x=0.55, y=1.0)
        if not os.path.exists('./feat_plots'):
            os.mkdir('feat_plots')
        plt.savefig('./feat_plots/{0}.png'.format(title))
        #fig.savefig("foo.pdf", bbox_inches='tight')
        plt.clf()

    anova_all = np.array(anova_all)
    df = pd.DataFrame(data=anova_all[1:,1:],index=anova_all[1:,0].tolist(),columns=anova_all[0,1:].tolist())
    df.colums = ['id', 'y', 'layer_id', 'is_sparse']
    df = df.astype({'y' : 'float32', 'layer_id' : 'int32', 'is_sparse' : 'int32'})
    formula = 'y ~ C(layer_id) + C(is_sparse) + C(layer_id)*C(is_sparse)'
    model = ols(formula, df).fit()
    aov_table = anova_lm(model, typ=1)

    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)
