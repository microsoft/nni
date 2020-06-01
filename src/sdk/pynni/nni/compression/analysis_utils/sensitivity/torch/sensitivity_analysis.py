# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import csv
import logging
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import torch.nn as nn

from nni.compression.torch import LevelPruner
from nni.compression.torch import L1FilterPruner
from nni.compression.torch import L2FilterPruner

# use Agg backend
matplotlib.use('Agg')
SUPPORTED_OP_NAME = ['Conv2d', 'Conv1d']
SUPPORTED_OP_TYPE = [getattr(nn, name) for name in SUPPORTED_OP_NAME]

logger = logging.getLogger('Sensitivity_Analysis')
logger.setLevel(logging.INFO)


class SensitivityAnalysis:
    def __init__(self, model, val_func, sparsities=None, prune_type='l1', early_stop=1.0):
        """
        Perform sensitivity analysis for this model.
        Parameters
        ----------
        model : torch.nn.Module
            the model to perform sensitivity analysis
        val_func : function
            validation function for the model. Due to
            different models may need different dataset/criterion
            , therefore the user need to cover this part by themselves.
            In the val_func, the model should be tested on the validation dateset,
            and the validation accuracy should be returned as the output of val_func.
            There are no restrictions on the input parameters of the val_function.
            User can use the val_args, val_kwargs parameters in analysis
            to pass all the parameters that val_func needed.

        sparsities : list
            The sparsity list provided by users. This parameter is set when the user
            only wants to test some specific sparsities. In the sparsity list, each element
            is a sparsity value which means how much weight the pruner should prune. Take
            [0.25, 0.5, 0.75] for an example, the SensitivityAnalysis will prune 25% 50% 75%
            weights gradually for each layer.

        prune_type : str
            The pruner type used to prune the conv layers, default is 'l1',
            and 'l2', 'fine-grained' is also supported.
        early_stop : float
            If this flag is set, the sensitivity analysis
            for a conv layer will early stop when the accuracy
            drop already reach the value of early_stop (0.05 for example).
            The default value is 1.0, which means the analysis won't stop
            until all given sparsities are tested.

        """
        self.model = model
        self.val_func = val_func
        self.target_layer = OrderedDict()
        self.ori_state_dict = copy.deepcopy(self.model.state_dict())
        self.target_layer = {}
        self.sensitivities = {}
        if sparsities is not None:
            self.sparsities = sorted(sparsities)
        else:
            self.sparsities = np.arange(0.1, 1.0, 0.1)
        self.sparsities = [np.round(x, 2) for x in self.sparsities]
        self.Pruner = L1FilterPruner
        if prune_type == 'l2':
            self.Pruner = L2FilterPruner
        elif prune_type == 'fine-grained':
            self.Pruner = LevelPruner
        self.early_stop = early_stop
        self.ori_acc = None  # original accuracy for the model
        # already_pruned is for the iterative sensitivity analysis
        # For example, sensitivity_pruner iteratively prune the target
        # model according to the sensitivity. After each round of
        # pruning, the sensitivity_pruner will test the new sensitivity
        # for each layer
        self.already_pruned = {}
        self.model_parse()

    @property
    def layers_count(self):
        return len(self.target_layer)

    def model_parse(self):
        for name, submodel in self.model.named_modules():
            for op_type in SUPPORTED_OP_TYPE:
                if isinstance(submodel, op_type):
                    self.target_layer[name] = submodel
                    self.already_pruned[name] = 0

    def analysis(self, val_args=None, val_kwargs=None, start=0, end=None):
        """
        This function analyze the sensitivity to pruning for
        each conv layer in the target model.
        If start and end are not set, we analyze all the conv
        layers by default. Users can specify several layers to
        analyze or parallelize the analysis process easily through
        the start and end parameter.

        Parameters
        ----------
        val_args : list
            args for the val_function
        val_kwargs : dict
            kwargs for the val_funtion
            The val_funtion will be called as:
                val_function(*val_args, **val_kwargs).
        start : int
            Layer index of the sensitivity analysis start.
        end : int
            Layer index of the sensitivity analysis end.

        Returns
        -------
        sensitivities : dict
            dict object that stores the trajectory of the
            accuracy when the prune ratio changes
        """
        if not end:
            end = self.layers_count
        assert start >= 0 and end <= self.layers_count
        assert start <= end
        if val_args is None:
            val_args = []
        if val_kwargs is None:
            val_kwargs = {}
        # Get the validation accuracy before pruning
        if self.ori_acc is None:
            self.ori_acc = self.val_func(*val_args, **val_kwargs)
        namelist = list(self.target_layer.keys())
        for layerid in range(start, end):
            name = namelist[layerid]
            self.sensitivities[name] = {}
            for sparsity in self.sparsities:
                # Calculate the actual prune ratio based on the already pruned ratio
                sparsity = (
                    1.0 - self.already_pruned[name]) * sparsity + self.already_pruned[name]
                # TODO In current L1/L2 Filter Pruner, the 'op_types' is still necessary
                # I think the L1/L2 Pruner should specify the op_types automaticlly
                # according to the op_names
                cfg = [{'sparsity': sparsity, 'op_names': [
                    name], 'op_types': ['Conv2d']}]
                pruner = self.Pruner(self.model, cfg)
                pruner.compress()
                val_acc = self.val_func(*val_args, **val_kwargs)
                logger.info('Layer: %s Sparsity: %.2f Accuracy: %.4f',
                            name, sparsity, val_acc)

                self.sensitivities[name][sparsity] = val_acc
                pruner._unwrap_model()
                del pruner
                # if the accuracy drop already reach the 'early_stop'
                if val_acc + self.early_stop < self.ori_acc:
                    break

            # reset the weights pruned by the pruner, because
            # out sparsities is sorted, so we donnot need to reset
            # weight of the layer when the sparsity changes, instead,
            # we only need reset the weight when the pruning layer changes.
            self.model.load_state_dict(self.ori_state_dict)

        return self.sensitivities

    def visualization(self, outdir, merge=False):
        """
        Visualize the sensitivity curves of the model

        Parameters
        ----------
        outdir : str
            output directory of the image
        merge : bool
            if merge all the sensitivity curves into a
            single image. If not, we will draw a picture
            for each target layer of the model.
        """
        os.makedirs(outdir, exist_ok=True)
        LineStyles = [':', '-.', '--', '-']
        Markers = list(Line2D.markers.keys())
        if not merge:
            # Draw the sensitivity curves for each layer first
            for name in self.sensitivities:
                X = list(self.sensitivities[name].keys())
                X = sorted(X)
                Y = [self.sensitivities[name][x] for x in X]
                if 0.00 not in X:
                    # add the original accuracy into the figure
                    X = [0.00] + X
                    Y = [self.ori_acc] + Y
                plt.figure(figsize=(8, 4))
                plt.plot(X, Y, marker='*')
                plt.xlabel('Prune Ratio')
                plt.ylabel('Validation Accuracy')
                plt.title(name)
                plt.tight_layout()
                filepath = os.path.join(outdir, '%s.jpg' % name)
                plt.savefig(filepath)
                plt.close()
        else:
            plt.figure()
            styleid = 0
            for name in self.sensitivities:
                X = list(self.sensitivities[name].keys())
                X = sorted(X)
                Y = [self.sensitivities[name][x] for x in X]
                if 0.00 not in X:
                    # add the original accuracy into the figure
                    X = [0.00] + X
                    Y = [self.ori_acc] + Y
                linestyle = LineStyles[styleid % len(LineStyles)]
                marker = Markers[styleid % len(Markers)]
                plt.plot(X, Y, label=name, linestyle=linestyle, marker=marker)
                plt.xlabel('Prune Ratio')
                plt.ylabel('Validation Accuracy')
                plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
                plt.tight_layout()
                filepath = os.path.join(outdir, 'all.jpg')
                plt.savefig(filepath, dpi=1000, bbox_inches='tight')
                styleid += 1
            plt.close()

    def export(self, filepath):
        """
        Export the results of the sensitivity analysis
        to a csv file. The firstline of the csv file describe the content
        structure. The first line is constructed by 'layername' and sparsity
        list. Each line below records the accuracy of a layer under different
        sparsities. Note that, due to the early_stop option, some layers may
        not have all accuracies under different sparsities, because his accuracy
        drop has alreay exceeded the threshold set by the user. The following is an
        example output for export.

        layername, 0.25, 0.5, 0.75
        conv1, 0.6, 0.55
        conv2, 0.61, 0.57, 0.56
        
        Parameters
        ----------
        filepath : str
            Path of the output file
        """
        str_sparsities = [str(x) for x in self.sparsities]
        header = ['layername'] + str_sparsities
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf)
            csv_w.writerow(header)
            for layername in self.sensitivities:
                row = []
                row.append(layername)
                for sparsity in sorted(self.sensitivities[layername].keys()):
                    row.append(self.sensitivities[layername][sparsity])
                csv_w.writerow(row)

    def update_already_pruned(self, layername, ratio):
        """
        Set the already pruned ratio for the target layer.
        """
        self.already_pruned[layername] = ratio

    def load_state_dict(self, state_dict):
        """
        Update the weight of the model
        """
        self.ori_state_dict = copy.deepcopy(state_dict)
        self.model.load_state_dict(self.ori_state_dict)
