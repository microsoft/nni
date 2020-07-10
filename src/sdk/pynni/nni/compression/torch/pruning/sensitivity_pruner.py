# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import csv
import copy
import json
import logging
import torch
import torch.nn as nn

from ...analysis_utils.sensitivity.torch.sensitivity_analysis import SensitivityAnalysis
from ...analysis_utils.sensitivity.torch.sensitivity_analysis import SUPPORTED_OP_TYPE
from ...analysis_utils.sensitivity.torch.sensitivity_analysis import SUPPORTED_OP_NAME
from nni.compression.torch import L1FilterPruner
from nni.compression.torch import L2FilterPruner

MAX_PRUNE_RATIO_PER_ITER = 0.95
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('Sensitivity_Pruner')
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class SensitivityPruner:
    def __init__(self, model, val_func, finetune_func=None, sparsity_proportion_calc=None):
        """
        Parameters
        ----------
            model:
                model to be compressed
            val_function:
                validation function for the model. This function should return the accuracy
                of the validation dataset.
            finetune_func:
                finetune function for the model. This parameter is not essential, if is not None, 
                the sensitivity pruner will finetune the model after pruning in each iteration.
            sparsity_proportion_calc:
                This function generate the sparsity proportion between the conv layers according to the
                sensitivity analysis results. The input of this function is a dict, for example :
                {'conv1' : {0.1: 0.9, 0.2 : 0.8}, 'conv2' : {0.1: 0.9, 0.2 : 0.8}}, in which, 'conv1' and
                is the name of the conv layer, and 0.1:0.9 means when the sparsity of conv1 is 0.1 (10%), 
                the model's val accuracy equals to 0.9.

        """
        self.model = model
        self.val_func = val_func
        self.finetune_func = finetune_func
        self.analyzer = SensitivityAnalysis(self.model, self.val_func)
        # Get the original accuracy of the pretrained model
        self.ori_acc = None
        # Copy the original weights before pruning
        self.ori_state_dict = copy.deepcopy(self.model.state_dict())
        self.sensitivities = {}
        # Save the weight count for each layer
        self.weight_count = {}
        self.weight_sum = 0
        # Map the layer name to the layer module
        self.named_module = {}
        for name, submodule in self.model.named_modules():
            self.named_module[name] = submodule
            if name in self.analyzer.target_layer:
                # Currently, only count the weights in the conv layers
                # else the fully connected layer (which contains
                # the most weights) may make the pruner prune the
                # model too hard
                # if hasattr(submodule, 'weight'): # Count all the weights of the model
                self.weight_count[name] = submodule.weight.data.numel()
                self.weight_sum += self.weight_count[name]
        # function to generate the sparsity proportion betweent the conv layers
        if sparsity_proportion_calc is None:
            self.sparsity_proportion_calc = self._max_prune_ratio
        else:
            self.sparsity_proportion_calc = sparsity_proportion_calc
        # The ratio of remained weights is 1.0 at the begining
        self.remained_ratio = 1.0

    def load_sensitivitis(self, filepath):
        # TODO load from a csv file
        """
        Load the sensitivity analysis result from file
        """
        assert os.path.exists(filepath)
        with open(filepath, 'r') as jf:
            sensitivities = json.load(jf)
            # convert string type to float
            for name in sensitivities:
                sensitivities[name] = {float(k): float(v)
                                       for k, v in sensitivities[name].items()}
            return sensitivities

    def load_sensitivitis_csv(self, filepath):
        """
        load the sensitivity results exported by the sensitivity analyzer
        """
        assert os.path.exists(filepath)
        with open(filepath, 'r') as csvf:
            csv_r = csv.reader(csvf)
            header = next(csv_r)
            sparsities = [float(x) for x in header[1:]]
            sensitivities = {}
            for row in csv_r:
                layername = row[0]
                accuracies = [float(x) for x in row[1:]]
                sensitivities[layername] = {}
                for i, accuracy in enumerate(accuracies):
                    sensitivities[layername][sparsities[i]] = accuracy
            return sensitivities

    def _max_prune_ratio(self, ori_acc, threshold, sensitivities):
        """
        Find the maximum prune ratio for a single layer whose accuracy 
        drop is lower than the threshold.

        Parameters
        ----------
            ori_acc:
                Original accuracy 
            threshold:
                Accuracy drop threshold 
            sensitivities:
                The dict object that stores the sensitivity results for each layer.
                For example: {'conv1' : {0.1: 0.9, 0.2 : 0.8}}
        Returns
        -------
            max_ratios:
                return the maximum prune ratio for each layer. For example:
                {'conv1':0.1, 'conv2':0.2}
        """
        max_ratio = {}
        for layer in sensitivities:
            prune_ratios = sorted(sensitivities[layer].keys())
            last_ratio = 0
            for ratio in prune_ratios:
                cur_acc = sensitivities[layer][ratio]
                if cur_acc + threshold < ori_acc:
                    break
                last_ratio = ratio
            max_ratio[layer] = last_ratio
        return max_ratio

    def normalize(self, ratios, target_pruned):
        """
        Normalize the prune ratio of each layer according to the
        total already pruned ratio and the finnal target total prune 
        ratio

        Parameters
        ----------
            ratios:
                Dict object that save the prune ratio for each layer
            target_pruned:
                The amount of the weights expected to be pruned in this
                iteration

        Returns
        -------
            new_ratios:
                return the normalized prune ratios for each layer.

        """
        w_sum = 0
        _Max = 0
        for layername, ratio in ratios.items():
            wcount = self.weight_count[layername]
            w_sum += ratio * wcount * \
                (1-self.analyzer.already_pruned[layername])
        target_count = self.weight_sum * target_pruned
        for layername in ratios:
            ratios[layername] = ratios[layername] * target_count / w_sum
            _Max = max(_Max, ratios[layername])
        # Cannot Prune too much in a single iteration
        # If a layer's prune ratio is larger than the
        # MAX_PRUNE_RATIO_PER_ITER we rescal all prune
        # ratios under this threshold
        if _Max > MAX_PRUNE_RATIO_PER_ITER:
            for layername in ratios:
                ratios[layername] = ratios[layername] * \
                    MAX_PRUNE_RATIO_PER_ITER / _Max
        return ratios

    def create_cfg(self, ratios):
        """
        Generate the cfg_list for the pruner according to the prune ratios.

        Parameters
        ---------
            ratios:
                For example: {'conv1' : 0.2}

        Returns
        -------
            cfg_list:
                For example: [{'sparsity':0.2, 'op_names':['conv1'], 'op_types':['Conv2d']}]
        """
        cfg_list = []
        for layername in ratios:
            prune_ratio = ratios[layername]
            remain = 1 - self.analyzer.already_pruned[layername]
            sparsity = remain * prune_ratio + \
                self.analyzer.already_pruned[layername]
            if sparsity > 0:
                # Pruner does not allow the prune ratio to be zero
                cfg = {'sparsity': sparsity, 'op_names': [
                    layername], 'op_types': ['Conv2d']}
                cfg_list.append(cfg)
        return cfg_list

    def current_sparsity(self):
        """
        The sparisity of the weight.
        """
        pruned_weight = 0
        for layer_name in self.analyzer.already_pruned:
            w_count = self.weight_count[layer_name]
            prune_ratio = self.analyzer.already_pruned[layer_name]
            pruned_weight += w_count * prune_ratio
        return pruned_weight / self.weight_sum

    def compress(self, target_ratio, val_args=None, val_kwargs=None,
                 finetune_args=None, finetune_kwargs=None, resume_sensitivity=None, 
                 ratio_step=0.1, threshold=0.05, MAX_ITERATION=None, checkpoint_dir=None):
        """
        This function iteratively prune the model according to the results of 
        the sensitivity analysis.

        Parameters
        ----------
            target_ratio:
                Target sparsity for the model
            val_args & val_kwargs:
                Parameters for the val_funtion, the val_function will be called like
                val_func(*val_args, **val_kwargs)
            finetune_args & finetune_kwargs:
                Parameters for the finetune function.
            resume_sensitivity:
                resume the sensitivity results from this file. In the one shot pruning mode(in which
                the maximum iteration is set to 1), user can avoid the repeated sensitivity analysis 
                for the same model.
            ratio_step:
                The ratio of the weights that sensitivity pruner try to prune in each 
                iteration.
            threshold:
                The hyperparameter used to determine the sparsity ratio for each layer.
            MAX_ITERATION:
                The maximum number of the iterations.
            checkpoint_dir:
                If not None, save the checkpoint of each iteration into the checkpoint_dir.
        """

        if not val_args:
            val_args = []
        if not val_kwargs:
            val_kwargs = {}
        if not finetune_args:
            finetune_args = []
        if not finetune_kwargs:
            finetune_kwargs = {}
        if self.ori_acc is None:
            self.ori_acc = self.val_func(*val_args, **val_kwargs)
        if not resume_sensitivity:
            self.sensitivities = self.analyzer.analysis(
                    val_args=val_args, val_kwargs=val_kwargs, early_stop=threshold)
        else:
            self.sensitivities = self.load_sensitivitis(resume_sensitivity)

        cur_ratio = self.remained_ratio
        ori_acc = self.ori_acc
        iteration_count = 0
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        while cur_ratio > target_ratio:
            iteration_count += 1
            if MAX_ITERATION is not None and iteration_count > MAX_ITERATION:
                break
            # Each round have three steps:
            # 1) Get the current sensitivity for each layer
            # 2) Prune each layer according the sensitivies
            # 3) finetune the model
            logger.info('Current base accuracy %f' % ori_acc)
            logger.info('Remained %f weights' % cur_ratio)
            # determine the sparsity proportion between different
            # layers according to the sensitivity result
            proportion = self.sparsity_proportion_calc(
                ori_acc, threshold, self.sensitivities)
            new_pruneratio = self.normalize(proportion, ratio_step)
            cfg_list = self.create_cfg(new_pruneratio)
            logger.debug('Pruner Config' + str(cfg_list))
            pruner = L1FilterPruner(self.model, cfg_list)
            pruner.compress()
            pruned_acc = self.val_func(*val_args, **val_kwargs)
            logger.info('Accuracy after pruning: %f' % pruned_acc)
            finetune_acc = pruned_acc
            if self.finetune_func is not None:
                # if the finetune function is None, then skip the finetune
                self.finetune_func(*finetune_args, **finetune_kwargs)
                finetune_acc = self.val_func(*val_args, **val_kwargs)
            logger.info('Accuracy after finetune:' % finetune_acc)
            ori_acc = finetune_acc
            # unwrap the pruner
            pruner._unwrap_model()
            # update the already prune ratio of each layer befor the new
            # sensitivity analysis
            for layer_cfg in cfg_list:
                name = layer_cfg['op_names'][0]
                sparsity = layer_cfg['sparsity']
                self.analyzer.already_pruned[name] = sparsity
            # update the cur_ratio
            cur_ratio = 1 - self.current_sparsity()
            del pruner
            logger.info('Currently remained weights: %f' % cur_ratio)

            if checkpoint_dir is not None:
                checkpoint_name = 'Iter_%d_finetune_acc_%.3f_sparsity_%.2f' % (
                    iteration_count, finetune_acc, cur_ratio)
                checkpoint_path = os.path.join(
                    checkpoint_dir, '%s.pth' % checkpoint_name)
                cfg_path = os.path.join(
                    checkpoint_dir, '%s_pruner.json' % checkpoint_name)
                torch.save(self.model.state_dict(), checkpoint_path)
                with open(cfg_path, 'w') as jf:
                    json.dump(cfg_list, jf)
            if MAX_ITERATION is not None and iteration_count < MAX_ITERATION:
                # If this is the last prune iteration, skip the time-consuming
                # sensitivity analysis
                self.analyzer.load_state_dict(self.model.state_dict())
                self.sensitivities = self.analyzer.analysis(
                    val_args=val_args, val_kwargs=val_kwargs, early_stop=threshold)

        logger.info('After Pruning: %.2f weights remains' % cur_ratio)
        return self.model

    def export(self, model_path, pruner_path=None):
        """
        Export the pruned results of the target model.

        Parameters
        ----------
            model_path:
                Path of the checkpoint of the pruned model.
            pruner_path:
                If not none, save the config of the pruner to this file.
        """
        torch.save(self.model.state_dict(), model_path)
        if pruner_path is not None:
            sparsity_ratios = {}
            for layername in self.analyzer.already_pruned:
                sparsity_ratios[layername] = self.analyzer.already_pruned[layername]
                cfg_list = self.create_cfg(sparsity_ratios)
            with open(pruner_path, 'w') as pf:
                json.dump(cfg_list, pf)

    def resume(self, checkpoint, pruner_cfg):
        """
        Resume from the checkpoint and continue to prune the model.

        Parameters
        ----------
            checkpoint:
                checkpoint of the model
            pruner_cfg:
                configuration of the previous pruner.
        """
        assert os.path.exists(checkpoint)
        assert os.path.exists(pruner_cfg)
        self.ori_state_dict = torch.load(checkpoint)
        # the ori_state_dict of the sensitivity analyzer also needs update
        # self.analyzer.load_state_dict(self.ori_state_dict)
        self.analyzer.ori_state_dict = copy.deepcopy(self.ori_state_dict)
        self.model.load_state_dict(self.ori_state_dict)
        with open(pruner_cfg, 'r') as jf:
            cfgs = json.load(jf)
            # reset the already pruned for the sensitivity analyzer
            for cfg in cfgs:
                for layername in cfg['op_names']:
                    self.analyzer.already_pruned[layername] = float(cfg['sparsity'])
            self.remained_ratio = 1.0 - self.current_sparsity()
