# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import copy
import logging
import os
import random
import statistics
import sys
from enum import Enum, unique
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import nni.metis_tuner.lib_constraint_summation as lib_constraint_summation
import nni.metis_tuner.lib_data as lib_data
import nni.metis_tuner.Regression_GMM.CreateModel as gmm_create_model
import nni.metis_tuner.Regression_GMM.Selection as gmm_selection
import nni.metis_tuner.Regression_GP.CreateModel as gp_create_model
import nni.metis_tuner.Regression_GP.OutlierDetection as gp_outlier_detection
import nni.metis_tuner.Regression_GP.Prediction as gp_prediction
import nni.metis_tuner.Regression_GP.Selection as gp_selection
from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

logger = logging.getLogger("Metis_Tuner_AutoML")

@unique
class OptimizeMode(Enum):
    """
    Optimize Mode class
    """
    Minimize = 'minimize'
    Maximize = 'maximize'


NONE_TYPE = ''
CONSTRAINT_LOWERBOUND = None
CONSTRAINT_UPPERBOUND = None
CONSTRAINT_PARAMS_IDX = []


class MetisTuner(Tuner):
    """
    Metis Tuner

    More algorithm information you could reference here:
    https://www.microsoft.com/en-us/research/publication/metis-robustly-tuning-tail-latencies-cloud-systems/
    """

    def __init__(self, optimize_mode="maximize", no_resampling=True, no_candidates=False,
                 selection_num_starting_points=600, cold_start_num=10, exploration_probability=0.9):
        """
        Parameters
        ----------
        optimize_mode : str
            optimize_mode is a string that including two mode "maximize" and "minimize"

        no_resampling : bool
            True or False. Should Metis consider re-sampling as part of the search strategy?
        If you are confident that the training dataset is noise-free, then you do not need re-sampling.

        no_candidates: bool
            True or False. Should Metis suggest parameters for the next benchmark?
        If you do not plan to do more benchmarks, Metis can skip this step.

        selection_num_starting_points: int
            how many times Metis should try to find the global optimal in the search space?
        The higher the number, the longer it takes to output the solution.

        cold_start_num: int
            Metis need some trial result to get cold start. when the number of trial result is less than
        cold_start_num, Metis will randomly sample hyper-parameter for trial.

        exploration_probability: float
            The probability of Metis to select parameter from exploration instead of exploitation.
        """

        self.samples_x = []
        self.samples_y = []
        self.samples_y_aggregation = []
        self.total_data = []
        self.space = None
        self.no_resampling = no_resampling
        self.no_candidates = no_candidates
        self.optimize_mode = optimize_mode
        self.key_order = []
        self.cold_start_num = cold_start_num
        self.selection_num_starting_points = selection_num_starting_points
        self.exploration_probability = exploration_probability
        self.minimize_constraints_fun = None
        self.minimize_starting_points = None
        self.supplement_data_num = 0


    def update_search_space(self, search_space):
        """Update the self.x_bounds and self.x_types by the search_space.json

        Parameters
        ----------
        search_space : dict
        """
        self.x_bounds = [[] for i in range(len(search_space))]
        self.x_types = [NONE_TYPE for i in range(len(search_space))]

        for key in search_space:
            self.key_order.append(key)

        key_type = {}
        if isinstance(search_space, dict):
            for key in search_space:
                key_type = search_space[key]['_type']
                key_range = search_space[key]['_value']
                idx = self.key_order.index(key)
                if key_type == 'quniform':
                    if key_range[2] == 1:
                        self.x_bounds[idx] = [key_range[0], key_range[1]]
                        self.x_types[idx] = 'range_int'
                    else:
                        bounds = []
                        for value in np.arange(key_range[0], key_range[1], key_range[2]):
                            bounds.append(value)
                        self.x_bounds[idx] = bounds
                        self.x_types[idx] = 'discrete_int'
                elif key_type == 'randint':
                    self.x_bounds[idx] = [0, key_range[0]]
                    self.x_types[idx] = 'range_int'
                elif key_type == 'uniform':
                    self.x_bounds[idx] = [key_range[0], key_range[1]]
                    self.x_types[idx] = 'range_continuous'
                elif key_type == 'choice':
                    self.x_bounds[idx] = key_range

                    for key_value in key_range:
                        if not isinstance(key_value, (int, float)):
                            raise RuntimeError("Metis Tuner only support numerical choice.")

                    self.x_types[idx] = 'discrete_int'
                else:
                    logger.info("Metis Tuner doesn't support this kind of variable: " + str(key_type))
                    raise RuntimeError("Metis Tuner doesn't support this kind of variable: " + str(key_type))
        else:
            logger.info("The format of search space is not a dict.")
            raise RuntimeError("The format of search space is not a dict.")

        self.minimize_starting_points = _rand_init(self.x_bounds, self.x_types, \
                                                        self.selection_num_starting_points)


    def _pack_output(self, init_parameter):
        """Pack the output

        Parameters
        ----------
        init_parameter : dict

        Returns
        -------
        output : dict
        """
        output = {}
        for i, param in enumerate(init_parameter):
            output[self.key_order[i]] = param
        return output


    def generate_parameters(self, parameter_id):
        """Generate next parameter for trial
        If the number of trial result is lower than cold start number,
        metis will first random generate some parameters.
        Otherwise, metis will choose the parameters by the Gussian Process Model and the Gussian Mixture Model.

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        result : dict
        """
        if len(self.samples_x) < self.cold_start_num:
            init_parameter = _rand_init(self.x_bounds, self.x_types, 1)[0]
            results = self._pack_output(init_parameter)
        else:
            self.minimize_starting_points = _rand_init(self.x_bounds, self.x_types, \
                                                       self.selection_num_starting_points)
            results = self._selection(self.samples_x, self.samples_y_aggregation, self.samples_y,
                                      self.x_bounds, self.x_types,
                                      threshold_samplessize_resampling=(None if self.no_resampling is True else 50),
                                      no_candidates=self.no_candidates,
                                      minimize_starting_points=self.minimize_starting_points,
                                      minimize_constraints_fun=self.minimize_constraints_fun)

        logger.info("Generate paramageters:\n" + str(results))
        return results


    def receive_trial_result(self, parameter_id, parameters, value):
        """Tuner receive result from trial.

        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
        """
        value = extract_scalar_reward(value)
        if self.optimize_mode == OptimizeMode.Maximize:
            value = -value

        logger.info("Received trial result.")
        logger.info("value is :" + str(value))
        logger.info("parameter is : " + str(parameters))

        # parse parameter to sample_x
        sample_x = [0 for i in range(len(self.key_order))]
        for key in parameters:
            idx = self.key_order.index(key)
            sample_x[idx] = parameters[key]

        # parse value to sample_y
        temp_y = []
        if sample_x in self.samples_x:
            idx = self.samples_x.index(sample_x)
            temp_y = self.samples_y[idx]
            temp_y.append(value)
            self.samples_y[idx] = temp_y

            # calculate y aggregation
            median = get_median(temp_y)
            self.samples_y_aggregation[idx] = [median]
        else:
            self.samples_x.append(sample_x)
            self.samples_y.append([value])

            # calculate y aggregation
            self.samples_y_aggregation.append([value])


    def _selection(self, samples_x, samples_y_aggregation, samples_y,
                   x_bounds, x_types, max_resampling_per_x=3,
                   threshold_samplessize_exploitation=12,
                   threshold_samplessize_resampling=50, no_candidates=False,
                   minimize_starting_points=None, minimize_constraints_fun=None):

        next_candidate = None
        candidates = []
        samples_size_all = sum([len(i) for i in samples_y])
        samples_size_unique = len(samples_y)

        # ===== STEP 1: Compute the current optimum =====
        gp_model = gp_create_model.create_model(samples_x, samples_y_aggregation)
        lm_current = gp_selection.selection(
            "lm",
            samples_y_aggregation,
            x_bounds,
            x_types,
            gp_model['model'],
            minimize_starting_points,
            minimize_constraints_fun=minimize_constraints_fun)
        if not lm_current:
            return None

        if no_candidates is False:
            candidates.append({'hyperparameter': lm_current['hyperparameter'],
                               'expected_mu': lm_current['expected_mu'],
                               'expected_sigma': lm_current['expected_sigma'],
                               'reason': "exploitation_gp"})

            # ===== STEP 2: Get recommended configurations for exploration =====
            results_exploration = gp_selection.selection(
                "lc",
                samples_y_aggregation,
                x_bounds,
                x_types,
                gp_model['model'],
                minimize_starting_points,
                minimize_constraints_fun=minimize_constraints_fun)

            if results_exploration is not None:
                if _num_past_samples(results_exploration['hyperparameter'], samples_x, samples_y) == 0:
                    candidates.append({'hyperparameter': results_exploration['hyperparameter'],
                                       'expected_mu': results_exploration['expected_mu'],
                                       'expected_sigma': results_exploration['expected_sigma'],
                                       'reason': "exploration"})
                    logger.info("DEBUG: 1 exploration candidate selected\n")
            else:
                logger.info("DEBUG: No suitable exploration candidates were")

            # ===== STEP 3: Get recommended configurations for exploitation =====
            if samples_size_all >= threshold_samplessize_exploitation:
                print("Getting candidates for exploitation...\n")
                try:
                    gmm = gmm_create_model.create_model(samples_x, samples_y_aggregation)
                    results_exploitation = gmm_selection.selection(
                        x_bounds,
                        x_types,
                        gmm['clusteringmodel_good'],
                        gmm['clusteringmodel_bad'],
                        minimize_starting_points,
                        minimize_constraints_fun=minimize_constraints_fun)

                    if results_exploitation is not None:
                        if _num_past_samples(results_exploitation['hyperparameter'], samples_x, samples_y) == 0:
                            candidates.append({'hyperparameter': results_exploitation['hyperparameter'],\
                                               'expected_mu': results_exploitation['expected_mu'],\
                                               'expected_sigma': results_exploitation['expected_sigma'],\
                                               'reason': "exploitation_gmm"})
                            logger.info("DEBUG: 1 exploitation_gmm candidate selected\n")
                    else:
                        logger.info("DEBUG: No suitable exploitation_gmm candidates were found\n")

                except ValueError as exception:
                    # The exception: ValueError: Fitting the mixture model failed
                    # because some components have ill-defined empirical covariance
                    # (for instance caused by singleton or collapsed samples).
                    # Try to decrease the number of components, or increase reg_covar.
                    logger.info("DEBUG: No suitable exploitation_gmm candidates were found due to exception.")
                    logger.info(exception)

            # ===== STEP 4: Get a list of outliers =====
            if (threshold_samplessize_resampling is not None) and \
                        (samples_size_unique >= threshold_samplessize_resampling):
                logger.info("Getting candidates for re-sampling...\n")
                results_outliers = gp_outlier_detection.outlierDetection_threaded(samples_x, samples_y_aggregation)

                if results_outliers is not None:
                    for results_outlier in results_outliers:
                        if _num_past_samples(samples_x[results_outlier['samples_idx']], samples_x, samples_y) < max_resampling_per_x:
                            candidates.append({'hyperparameter': samples_x[results_outlier['samples_idx']],\
                                               'expected_mu': results_outlier['expected_mu'],\
                                               'expected_sigma': results_outlier['expected_sigma'],\
                                               'reason': "resampling"})
                    logger.info("DEBUG: %d re-sampling candidates selected\n")
                else:
                    logger.info("DEBUG: No suitable resampling candidates were found\n")

            if candidates:
                # ===== STEP 5: Compute the information gain of each candidate towards the optimum =====
                logger.info("Evaluating information gain of %d candidates...\n")
                next_improvement = 0

                threads_inputs = [[
                    candidate, samples_x, samples_y, x_bounds, x_types,
                    minimize_constraints_fun, minimize_starting_points
                ] for candidate in candidates]
                threads_pool = ThreadPool(4)
                # Evaluate what would happen if we actually sample each candidate
                threads_results = threads_pool.map(_calculate_lowest_mu_threaded, threads_inputs)
                threads_pool.close()
                threads_pool.join()

                for threads_result in threads_results:
                    if threads_result['expected_lowest_mu'] < lm_current['expected_mu']:
                        # Information gain
                        temp_improvement = threads_result['expected_lowest_mu'] - lm_current['expected_mu']

                        if next_improvement > temp_improvement:
                            next_improvement = temp_improvement
                            next_candidate = threads_result['candidate']
            else:
                # ===== STEP 6: If we have no candidates, randomly pick one =====
                logger.info(
                    "DEBUG: No candidates from exploration, exploitation,\
                                 and resampling. We will random a candidate for next_candidate\n"
                )

                next_candidate = _rand_with_constraints(x_bounds, x_types) \
                                    if minimize_starting_points is None else minimize_starting_points[0]
                next_candidate = lib_data.match_val_type(next_candidate, x_bounds, x_types)
                expected_mu, expected_sigma = gp_prediction.predict(next_candidate, gp_model['model'])
                next_candidate = {'hyperparameter': next_candidate, 'reason': "random",
                                  'expected_mu': expected_mu, 'expected_sigma': expected_sigma}

        # ===== STEP 7: If current optimal hyperparameter occurs in the history or exploration probability is less than the threshold, take next config as exploration step  =====
        outputs = self._pack_output(lm_current['hyperparameter'])
        ap = random.uniform(0, 1)
        if outputs in self.total_data or ap<=self.exploration_probability:
            if next_candidate is not None:
                outputs = self._pack_output(next_candidate['hyperparameter'])
            else:
                random_parameter = _rand_init(x_bounds, x_types, 1)[0]
                outputs = self._pack_output(random_parameter)
        self.total_data.append(outputs)
        return outputs

    def import_data(self, data):
        """Import additional data for tuning
        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s" %(_completed_num, len(data)))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data." %_value)
                continue
            self.supplement_data_num += 1
            _parameter_id = '_'.join(["ImportData", str(self.supplement_data_num)])
            self.total_data.append(_params)
            self.receive_trial_result(parameter_id=_parameter_id, parameters=_params, value=_value)
        logger.info("Successfully import data to metis tuner.")

def _rand_with_constraints(x_bounds, x_types):
    outputs = None
    x_bounds_withconstraints = [x_bounds[i] for i in CONSTRAINT_PARAMS_IDX]
    x_types_withconstraints = [x_types[i] for i in CONSTRAINT_PARAMS_IDX]

    x_val_withconstraints = lib_constraint_summation.rand(x_bounds_withconstraints,\
                                x_types_withconstraints, CONSTRAINT_LOWERBOUND, CONSTRAINT_UPPERBOUND)
    if not x_val_withconstraints:
        outputs = [None] * len(x_bounds)

        for i, _ in enumerate(CONSTRAINT_PARAMS_IDX):
            outputs[CONSTRAINT_PARAMS_IDX[i]] = x_val_withconstraints[i]

        for i, output in enumerate(outputs):
            if not output:
                outputs[i] = random.randint(x_bounds[i][0], x_bounds[i][1])
    return outputs


def _calculate_lowest_mu_threaded(inputs):
    [candidate, samples_x, samples_y, x_bounds, x_types, minimize_constraints_fun, minimize_starting_points] = inputs

    outputs = {"candidate": candidate, "expected_lowest_mu": None}

    for expected_mu in [candidate['expected_mu'] + 1.96 * candidate['expected_sigma'],
                        candidate['expected_mu'] - 1.96 * candidate['expected_sigma']]:
        temp_samples_x = copy.deepcopy(samples_x)
        temp_samples_y = copy.deepcopy(samples_y)

        try:
            idx = temp_samples_x.index(candidate['hyperparameter'])
            # This handles the case of re-sampling a potential outlier
            temp_samples_y[idx].append(expected_mu)
        except ValueError:
            temp_samples_x.append(candidate['hyperparameter'])
            temp_samples_y.append([expected_mu])

        # Aggregates multiple observation of the sample sampling points
        temp_y_aggregation = [statistics.median(temp_sample_y) for temp_sample_y in temp_samples_y]
        temp_gp = gp_create_model.create_model(temp_samples_x, temp_y_aggregation)
        temp_results = gp_selection.selection(
            "lm",
            temp_y_aggregation,
            x_bounds,
            x_types,
            temp_gp['model'],
            minimize_starting_points,
            minimize_constraints_fun=minimize_constraints_fun)

        if outputs["expected_lowest_mu"] is None or outputs["expected_lowest_mu"] > temp_results['expected_mu']:
            outputs["expected_lowest_mu"] = temp_results['expected_mu']

    return outputs


def _num_past_samples(x, samples_x, samples_y):
    try:
        idx = samples_x.index(x)
        return len(samples_y[idx])
    except ValueError:
        logger.info("x not in sample_x")
        return 0


def _rand_init(x_bounds, x_types, selection_num_starting_points):
    '''
    Random sample some init seed within bounds.
    '''
    return [lib_data.rand(x_bounds, x_types) for i \
                    in range(0, selection_num_starting_points)]


def get_median(temp_list):
    """Return median
    """
    num = len(temp_list)
    temp_list.sort()
    print(temp_list)
    if num % 2 == 0:
        median = (temp_list[int(num/2)] + temp_list[int(num/2) - 1]) / 2
    else:
        median = temp_list[int(num/2)]
    return median
