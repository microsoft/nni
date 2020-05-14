# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import copy
import json
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from nni.compression.torch import LevelPruner
from nni.compression.torch import L1FilterPruner
from nni.compression.torch import L2FilterPruner


SUPPORTED_OP_NAME = ['Conv2d', 'Conv1d']
SUPPORTED_OP_TYPE = [getattr(nn, name) for name in SUPPORTED_OP_NAME]


class SensitivityAnalysis:
    def __init__(self, model, val_func, ratio_step=0.1):
        # TODO Speedup by ratio_threshold or list
        # TODO l1 or l2 seted here
        """
        Perform sensitivity analysis for this model.
        Parameters
        ----------
            model:
                the model to perform sensitivity analysis
            val_func:
                validation function for the model. Due to 
                different models may need different dataset/criterion
                , therefore the user need to cover this part by themselves.
                val_func take the model as the first input parameter, and 
                return the accuracy as output.
            ratio_step: 
                the step to change the prune ratio during the analysis
        """
        self.model = model
        self.val_func = val_func
        self.ratio_step = ratio_step
        self.target_layer = OrderedDict()
        self.ori_state_dict = copy.deepcopy(self.model.state_dict())
        self.target_layer = {}
        self.sensitivities = {}
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

    def analysis(self, start=0, end=None, type='l1'):
        """
        This function analyze the sensitivity to pruning for 
        each conv layer in the target model.
        If %start and %end are not set, we analyze all the conv
        layers by default. Users can specify several layers to 
        analyze or parallelize the analysis process easily through
        the %start and %end parameter.

        Parameters
        ----------
            start: 
                Layer index of the sensitivity analysis start
            end:  
                Layer index of the sensitivity analysis end
            type: 
                Prune type of the Conv layers (l1/l2)

        Returns
        -------
            sensitivities:
                dict object that stores the trajectory of the 
                accuracy when the prune ratio changes
        """
        if not end:
            end = self.layers_count
        assert start >= 0 and end <= self.layers_count
        assert start <= end
        namelist = list(self.target_layer.keys())
        for layerid in range(start, end):
            name = namelist[layerid]
            self.sensitivities[name] = {}
            for prune_ratio in np.arange(self.ratio_step, 1.0, self.ratio_step):
                print('PruneRatio: ', prune_ratio)
                prune_ratio = np.round(prune_ratio, 2)
                # Calculate the actual prune ratio based on the already pruned ratio
                prune_ratio = (
                    1.0 - self.already_pruned[name]) * prune_ratio + self.already_pruned[name]
                cfg = [{'sparsity': prune_ratio, 'op_names': [
                    name], 'op_types': ['Conv2d']}]
                pruner = L1FilterPruner(self.model, cfg)
                pruner.compress()
                val_acc = self.val_func(self.model)
                self.sensitivities[name][prune_ratio] = val_acc
                pruner._unwrap_model()
                # TODO outside the ratio loop
                # reset the weights pruned by the pruner
                self.model.load_state_dict(self.ori_state_dict)
                # print('Reset')
                # print(self.val_func(self.model))
                del pruner
        return self.sensitivities

    def visualization(self, outdir, merge=False):
        """
        # 
        Visualize the sensitivity curves of the model

        Parameters
        ----------
            outdir: 
                output directory of the image
            merge:
                if merge all the sensitivity curves into a 
                single image. If not, we will draw a picture 
                for each target layer of the model.
        """
        os.makedirs(outdir, exist_ok=True)
        import matplotlib
        # use Agg backend
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        LineStyles = [':', '-.', '--', '-']
        Markers = list(Line2D.markers.keys())
        if not merge:
            # Draw the sensitivity curves for each layer first
            for name in self.sensitivities:
                X = list(self.sensitivities[name].keys())
                X = sorted(X)
                Y = [self.sensitivities[name][x] for x in X]
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
        #TODO CSV
        Export the results of the sensitivity analysis
        to a json file.

        Parameters
        ----------
            filepath:
                Path of the output file
        """
        # TODO csv
        with open(filepath, 'w') as jf:
            json.dump(self.sensitivities, jf, indent=4)

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
