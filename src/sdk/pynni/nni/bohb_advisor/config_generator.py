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
import logging
from copy import deepcopy
import traceback

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import statsmodels.api as sm

logger = logging.getLogger('BOHB_Advisor')

class CG_BOHB(object):
    def __init__(self, configspace, min_points_in_model=None,
                 top_n_percent=15, num_samples=64, random_fraction=1/3,
                 bandwidth_factor=3, min_bandwidth=1e-3):
        """Fits for each given budget a kernel density estimator on the best N percent of the
        evaluated configurations on this budget.


        Parameters:
        -----------
        configspace: ConfigSpace
            Configuration space object
        top_n_percent: int
            Determines the percentile of configurations that will be used as training data
            for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
            for training.
        min_points_in_model: int
            minimum number of datapoints needed to fit a model
        num_samples: int
            number of samples drawn to optimize EI via sampling
        random_fraction: float
            fraction of random configurations returned
        bandwidth_factor: float
            widens the bandwidth for contiuous parameters for proposed points to optimize EI
        min_bandwidth: float
            to keep diversity, even when all (good) samples have the same value for one of the parameters,
            a minimum bandwidth (Default: 1e-3) is used instead of zero. 

        """
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters())+1
        
        if self.min_points_in_model < len(self.configspace.get_hyperparameters())+1:
            logger.warning('Invalid min_points_in_model value. Setting it to %i'%(len(self.configspace.get_hyperparameters())+1))
            self.min_points_in_model =len(self.configspace.get_hyperparameters())+1
        
        self.num_samples = num_samples
        self.random_fraction = random_fraction

        hps = self.configspace.get_hyperparameters()

        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]
        
        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        self.configs = dict()
        self.losses = dict()
        self.good_config_rankings = dict()
        self.kde_models = dict()

    def largest_budget_with_model(self):
        if len(self.kde_models) == 0:
            return(-float('inf'))
        return(max(self.kde_models.keys()))

    def get_config(self, budget):
        """Function to sample a new configuration
        This function is called inside BOHB to query a new configuration

        Parameters:
        -----------
        budget: float
            the budget for which this configuration is scheduled

        Returns
        -------
        config
            should return a valid configuration
        """
        logger.debug('start sampling a new configuration.') 
        sample = None
        info_dict = {}
   
        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            sample =  self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False

        best = np.inf
        best_vector = None

        if sample is None:
            try:             
                # sample from largest budget
                budget = max(self.kde_models.keys())

                l = self.kde_models[budget]['good'].pdf
                g = self.kde_models[budget]['bad' ].pdf

                minimize_me = lambda x: max(1e-32, g(x))/max(l(x),1e-32)

                kde_good = self.kde_models[budget]['good']
                kde_bad = self.kde_models[budget]['bad']

                for i in range(self.num_samples):
                    idx = np.random.randint(0, len(kde_good.data))
                    datum = kde_good.data[idx]
                    vector = []

                    for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                        bw = max(bw, self.min_bandwidth)
                        if t == 0:
                            bw = self.bw_factor*bw
                            try:
                                vector.append(sps.truncnorm.rvs(-m/bw,(1-m)/bw, loc=m, scale=bw))
                            except:
                                logger.warning("Truncated Normal failed")
                                logger.warning("data in the KDE:\n%s"%kde_good.data)
                        else:
                            if np.random.rand() < (1-bw):
                                vector.append(int(m))
                            else:
                                vector.append(np.random.randint(t))
                    val = minimize_me(vector)

                    if not np.isfinite(val):
                        logger.warning('sampled vector: %s has EI value %s'%(vector, val))
                        logger.warning("data in the KDEs:\n%s\n%s"%(kde_good.data, kde_bad.data))
                        logger.warning("bandwidth of the KDEs:\n%s\n%s"%(kde_good.bw, kde_bad.bw))
                        logger.warning("l(x) = %s"%(l(vector)))
                        logger.warning("g(x) = %s"%(g(vector)))

                        # right now, this happens because a KDE does not contain all values for a categorical parameter
                        # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                        # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                        if np.isfinite(l(vector)):
                            best_vector = vector
                            break

                    if val < best:
                        best = val
                        best_vector = vector

                if best_vector is None:
                    logger.debug("Sampling based optimization with %i samples failed -> using random configuration"%self.num_samples)
                    sample = self.configspace.sample_configuration().get_dictionary()
                    info_dict['model_based_pick'] = False
                else:
                    logger.debug('best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                    for i, hp_value in enumerate(best_vector):
                        if isinstance(
                            self.configspace.get_hyperparameter(
                                self.configspace.get_hyperparameter_by_idx(i)
                            ),
                            ConfigSpace.hyperparameters.CategoricalHyperparameter
                        ):
                            best_vector[i] = int(np.rint(best_vector[i]))
                    sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()
                    
                    try:
                        sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                                    configuration_space=self.configspace,
                                    configuration=sample
                                    )
                        info_dict['model_based_pick'] = True

                    except Exception as e:
                        logger.warning(("="*50 + "\n")*3 +\
                                "Error converting configuration:\n%s"%sample+\
                                "\n here is a traceback:" +\
                                traceback.format_exc())
                        raise(e)
            except:
                logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
                sample = self.configspace.sample_configuration()
                info_dict['model_based_pick'] = False

        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary()
            ).get_dictionary()
        except Exception as e:
            logger.warning("Error (%s) converting configuration: %s -> "
                                "using random configuration!", e, sample)
            sample = self.configspace.sample_configuration().get_dictionary()
        logger.debug('done sampling a new configuration.')
        sample['STEPS'] = budget
        return sample

    def impute_conditional_data(self, array):
        return_array = np.empty_like(array)
        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()
            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()
                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]
                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i,:] = datum
        return(return_array)

    def new_result(self, loss, budget, parameters, update_model=True):
        """function to register finished runs

        Every time a run has finished, this function should be called
        to register it with the loss.
        """
        if loss is None:
            # One could skip crashed results, but we decided
            # assign a +inf loss and count them as bad configurations
            loss = np.inf

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        # skip model building if we already have a bigger model
        if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
            return

        # We want to get a numerical representation of the configuration in the original space
        conf = ConfigSpace.Configuration(self.configspace, parameters)
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)

        # skip model building:
        # a) if not enough points are available
        if len(self.configs[budget]) <= self.min_points_in_model-1:
            logger.debug("Only %i run(s) for budget %f available, need more than %s -> can't build model!"%(len(self.configs[budget]), budget, self.min_points_in_model+1))
            return
        # b) during warnm starting when we feed previous results in and only update once
        if not update_model:
            return

        train_configs = np.array(self.configs[budget])
        train_losses = np.array(self.losses[budget])

        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100)
        #n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
        n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return
        
        #more expensive crossvalidation method
        #bw_estimation = 'cv_ls'
        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes, bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models[budget] = {
                'good': good_kde,
                'bad' : bad_kde
        }

        # update probs for the categorical parameters for later sampling
        logger.debug('done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n'%(budget, n_good, n_bad, np.min(train_losses)))
