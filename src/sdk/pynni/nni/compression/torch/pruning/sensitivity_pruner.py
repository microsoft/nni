# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import csv
import copy
import json
import logging
import torch

from schema import And, Optional
from ..compressor import Pruner
from ..utils.config_validation import CompressorSchema
from .constants_pruner import PRUNER_DICT
from ..utils.sensitivity_analysis import SensitivityAnalysis


MAX_PRUNE_RATIO_PER_ITER = 0.95


root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# root.addHandler(handler)

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_logger = logging.getLogger('Sensitivity_Pruner')
_logger.addHandler(handler)
# _logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class SensitivityPruner(Pruner):
    """
    This function prune the model based on the sensitivity
    for each layer.
    """

    def __init__(self, model, config_list, evaluator,
                 finetuner=None, base_algo='l1', sparsity_proportion_calc=None,
                 sparsity_per_iter=0.1, acc_drop_threshold=0.05, checkpoint_dir=None):
        """
        Parameters
        ----------
        model: pytorch model
            model to be compressed
        evaluator: function
            validation function for the model. This function should return the accuracy
            of the validation dataset. The input parameters of evaluator can be specified
            in the parameter `eval_args` and 'eval_kwargs' of the funciton compress if needed. 
        finetuner: function
            finetune function for the model. This parameter is not essential, if is not None,
            the sensitivity pruner will finetune the model after pruning in each iteration.
            The input parameters of finetuner can be specified in the parameter `finetune_args`
            and `finetune_kwargs` if needed.
        base_algo: str
            base pruning algorithm. `level`, `l1` or `l2`, by default `l1`.
        sparsity_proportion_calc: function
            This function generate the sparsity proportion between the conv layers according to the
            sensitivity analysis results. We provide a default function to quantify the sparsity
            proportion according to the sensitivity analysis results. Users can also customize
            this function according to their needs. The input of this function is a dict,
            for example : {'conv1' : {0.1: 0.9, 0.2 : 0.8}, 'conv2' : {0.1: 0.9, 0.2 : 0.8}},
            in which, 'conv1' and is the name of the conv layer, and 0.1:0.9 means when the 
            sparsity of conv1 is 0.1 (10%), the model's val accuracy equals to 0.9.
        sparsity_per_iter: float
            The sparsity of the model that the pruner try to prune in each iteration.
        acc_drop_threshold : float
            The hyperparameter used to quantifiy the sensitivity for each layer.
        checkpoint_dir: str
            The dir path to save the checkpoints during the pruning.

        """
        self.base_algo = base_algo
        self.model = model
        super(SensitivityPruner, self).__init__(model, config_list)
        # unwrap the model
        self._unwrap_model()
        _logger.debug(str(self.model))
        self.evaluator = evaluator
        self.finetuner = finetuner
        self.analyzer = SensitivityAnalysis(
            self.model, self.evaluator, prune_type=base_algo, \
            early_stop_mode='dropped', early_stop_value=acc_drop_threshold)
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

        self.Pruner = PRUNER_DICT[self.base_algo]
        # Count the total weight count of the model
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
        self.sparsity_per_iter = sparsity_per_iter
        self.acc_drop_threshold = acc_drop_threshold
        self.checkpoint_dir = checkpoint_dir

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        if self.base_algo == 'level':
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                Optional('op_types'): [str],
                Optional('op_names'): [str],
            }], model, _logger)
        elif self.base_algo in ['l1', 'l2']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

        schema.validate(config_list)

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
        ori_acc: float
            Original accuracy
        threshold: float
            Accuracy drop threshold
        sensitivities: dict
            The dict object that stores the sensitivity results for each layer.
            For example: {'conv1' : {0.1: 0.9, 0.2 : 0.8}}
        Returns
        -------
        max_ratios: dict
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
        total already pruned ratio and the finnal target total pruning
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

    def compress(self, eval_args=None, eval_kwargs=None,
                 finetune_args=None, finetune_kwargs=None, resume_sensitivity=None):
        """
        This function iteratively prune the model according to the results of 
        the sensitivity analysis.

        Parameters
        ----------
        eval_args: list 
        eval_kwargs: list& dict
            Parameters for the val_funtion, the val_function will be called like
            evaluator(*eval_args, **eval_kwargs)
        finetune_args: list
        finetune_kwargs: dict
            Parameters for the finetuner function if needed.
        resume_sensitivity:
            resume the sensitivity results from this file. In the one shot pruning mode(in which
            the maximum iteration is set to 1), user can avoid the repeated sensitivity analysis 
            for the same model.

        """
        # pylint suggest not use the empty list and dict
        # as the default input parameter
        if not eval_args:
            eval_args = []
        if not eval_kwargs:
            eval_kwargs = {}
        if not finetune_args:
            finetune_args = []
        if not finetune_kwargs:
            finetune_kwargs = {}
        if self.ori_acc is None:
            self.ori_acc = self.evaluator(*eval_args, **eval_kwargs)
        if not resume_sensitivity:
            self.sensitivities = self.analyzer.analysis(
                val_args=eval_args, val_kwargs=eval_kwargs)
        else:
            self.sensitivities = self.load_sensitivitis(resume_sensitivity)
            self.analyzer.sensitivities = self.sensitivities
        # the final target sparsity of the model
        target_ratio = 1 - self.config_list[0]['sparsity']
        cur_ratio = self.remained_ratio
        ori_acc = self.ori_acc
        iteration_count = 0
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        while cur_ratio > target_ratio:
            iteration_count += 1
            # Each round have three steps:
            # 1) Get the current sensitivity for each layer(the sensitivity
            # of each layer may change during the pruning)
            # 2) Prune each layer according the sensitivies
            # 3) finetune the model
            _logger.info('Current base accuracy %f', ori_acc)
            _logger.info('Remained %f weights', cur_ratio)
            # determine the sparsity proportion between different
            # layers according to the sensitivity result
            proportion = self.sparsity_proportion_calc(
                ori_acc, self.acc_drop_threshold, self.sensitivities)
            new_pruneratio = self.normalize(proportion, self.sparsity_per_iter)
            cfg_list = self.create_cfg(new_pruneratio)
            _logger.debug('Pruner Config: %s', str(cfg_list))
            pruner = self.Pruner(self.model, cfg_list)
            pruner.compress()
            pruned_acc = self.evaluator(*eval_args, **eval_kwargs)
            _logger.info('Accuracy after pruning: %f', pruned_acc)
            finetune_acc = pruned_acc
            if self.finetuner is not None:
                # if the finetune function is None, then skip the finetune
                self.finetuner(*finetune_args, **finetune_kwargs)
                finetune_acc = self.evaluator(*eval_args, **eval_kwargs)
            _logger.info('Accuracy after finetune: %f', finetune_acc)
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
            _logger.info('Currently remained weights: %f', cur_ratio)

            if self.checkpoint_dir is not None:
                checkpoint_name = 'Iter_%d_finetune_acc_%.5f_sparsity_%.4f' % (
                    iteration_count, finetune_acc, cur_ratio)
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, '%s.pth' % checkpoint_name)
                cfg_path = os.path.join(
                    self.checkpoint_dir, '%s_pruner.json' % checkpoint_name)
                sensitivity_path = os.path.join(
                    self.checkpoint_dir, '%s_sensitivity.csv' % checkpoint_name)
                torch.save(self.model.state_dict(), checkpoint_path)
                with open(cfg_path, 'w') as jf:
                    json.dump(cfg_list, jf)
                self.analyzer.export(sensitivity_path)
            if cur_ratio > target_ratio:
                # If this is the last prune iteration, skip the time-consuming
                # sensitivity analysis
                self.analyzer.load_state_dict(self.model.state_dict())
                self.sensitivities = self.analyzer.analysis(
                    val_args=eval_args, val_kwargs=eval_kwargs)

        _logger.info('After Pruning: %.2f weights remains' % cur_ratio)
        return self.model




    # def resume(self, checkpoint, pruner_cfg):
    #     """
    #     Resume from the checkpoint and continue to prune the model.

    #     Parameters
    #     ----------
    #         checkpoint:
    #             checkpoint of the model
    #         pruner_cfg:
    #             configuration of the previous pruner.
    #     """
    #     assert os.path.exists(checkpoint)
    #     assert os.path.exists(pruner_cfg)
    #     self.ori_state_dict = torch.load(checkpoint)
    #     # the ori_state_dict of the sensitivity analyzer also needs update
    #     # self.analyzer.load_state_dict(self.ori_state_dict)
    #     self.analyzer.ori_state_dict = copy.deepcopy(self.ori_state_dict)
    #     self.model.load_state_dict(self.ori_state_dict)
    #     with open(pruner_cfg, 'r') as jf:
    #         cfgs = json.load(jf)
    #         # reset the already pruned for the sensitivity analyzer
    #         for cfg in cfgs:
    #             for layername in cfg['op_names']:
    #                 self.analyzer.already_pruned[layername] = float(
    #                     cfg['sparsity'])
    #         self.remained_ratio = 1.0 - self.current_sparsity()