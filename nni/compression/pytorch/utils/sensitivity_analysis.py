# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import csv
import logging
from collections import OrderedDict

import numpy as np
import torch.nn as nn

# FIXME: I don't know where "utils" should be
SUPPORTED_OP_NAME = ['Conv2d', 'Conv1d']
SUPPORTED_OP_TYPE = [getattr(nn, name) for name in SUPPORTED_OP_NAME]

logger = logging.getLogger('Sensitivity_Analysis')
logger.setLevel(logging.INFO)


class SensitivityAnalysis:
    def __init__(self, model, val_func, sparsities=None, prune_type='l1', early_stop_mode=None, early_stop_value=None):
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
            and the validation accuracy/loss should be returned as the output of val_func.
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
        early_stop_mode : str
            If this flag is set, the sensitivity analysis
            for a conv layer will early stop when the validation metric(
            for example, accurracy/loss) has alreay meet the threshold. We
            support four different early stop modes: minimize, maximize, dropped,
            raised. The default value is None, which means the analysis won't stop
            until all given sparsities are tested. This option should be used with
            early_stop_value together.

            minimize: The analysis stops when the validation metric return by the val_func
            lower than early_stop_value.
            maximize: The analysis stops when the validation metric return by the val_func
            larger than early_stop_value.
            dropped: The analysis stops when the validation metric has dropped by early_stop_value.
            raised: The analysis stops when the validation metric has raised by early_stop_value.
        early_stop_value : float
            This value is used as the threshold for different earlystop modes.
            This value is effective only when the early_stop_mode is set.

        """
        from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT

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
        self.Pruner = PRUNER_DICT[prune_type]
        self.early_stop_mode = early_stop_mode
        self.early_stop_value = early_stop_value
        self.ori_metric = None  # original validation metric for the model
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

    def _need_to_stop(self, ori_metric, cur_metric):
        """
        Judge if meet the stop conditon(early_stop, min_threshold,
        max_threshold).
        Parameters
        ----------
        ori_metric : float
            original validation metric
        cur_metric : float
            current validation metric

        Returns
        -------
        stop : bool
            if stop the sensitivity analysis
        """
        if self.early_stop_mode is None:
            # early stop mode is not enable
            return False
        assert self.early_stop_value is not None
        if self.early_stop_mode == 'minimize':
            if cur_metric < self.early_stop_value:
                return True
        elif self.early_stop_mode == 'maximize':
            if cur_metric > self.early_stop_value:
                return True
        elif self.early_stop_mode == 'dropped':
            if cur_metric < ori_metric - self.early_stop_value:
                return True
        elif self.early_stop_mode == 'raised':
            if cur_metric > ori_metric + self.early_stop_value:
                return True
        return False

    def analysis(self, val_args=None, val_kwargs=None, specified_layers=None):
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
        specified_layers : list
            list of layer names to analyze sensitivity.
            If this variable is set, then only analyze
            the conv layers that specified in the list.
            User can also use this option to parallelize
            the sensitivity analysis easily.
        Returns
        -------
        sensitivities : dict
            dict object that stores the trajectory of the
            accuracy/loss when the prune ratio changes
        """
        if val_args is None:
            val_args = []
        if val_kwargs is None:
            val_kwargs = {}
        # Get the original validation metric(accuracy/loss) before pruning
        # Get the accuracy baseline before starting the analysis.
        self.ori_metric = self.val_func(*val_args, **val_kwargs)
        namelist = list(self.target_layer.keys())
        if specified_layers is not None:
            # only analyze several specified conv layers
            namelist = list(filter(lambda x: x in specified_layers, namelist))
        for name in namelist:
            self.sensitivities[name] = {}
            for sparsity in self.sparsities:
                # here the sparsity is the relative sparsity of the
                # the remained weights
                # Calculate the actual prune ratio based on the already pruned ratio
                real_sparsity = (
                    1.0 - self.already_pruned[name]) * sparsity + self.already_pruned[name]
                # TODO In current L1/L2 Filter Pruner, the 'op_types' is still necessary
                # I think the L1/L2 Pruner should specify the op_types automaticlly
                # according to the op_names
                cfg = [{'sparsity': real_sparsity, 'op_names': [
                    name], 'op_types': ['Conv2d']}]
                pruner = self.Pruner(self.model, cfg)
                pruner.compress()
                val_metric = self.val_func(*val_args, **val_kwargs)
                logger.info('Layer: %s Sparsity: %.2f Validation Metric: %.4f',
                            name, real_sparsity, val_metric)

                self.sensitivities[name][sparsity] = val_metric
                pruner._unwrap_model()
                del pruner
                # check if the current metric meet the stop condition
                if self._need_to_stop(self.ori_metric, val_metric):
                    break

            # reset the weights pruned by the pruner, because the
            # input sparsities is sorted, so we donnot need to reset
            # weight of the layer when the sparsity changes, instead,
            # we only need reset the weight when the pruning layer changes.
            self.model.load_state_dict(self.ori_state_dict)

        return self.sensitivities

    def export(self, filepath):
        """
        Export the results of the sensitivity analysis
        to a csv file. The firstline of the csv file describe the content
        structure. The first line is constructed by 'layername' and sparsity
        list. Each line below records the validation metric returned by val_func
        when this layer is under different sparsities. Note that, due to the early_stop
        option, some layers may not have the metrics under all sparsities.

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
