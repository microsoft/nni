# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted

import torch

from nni.feature_engineering.feature_selector import FeatureSelector
from . import constants
from .fginitialize import PrepareData
from .fgtrain import _train


class FeatureGradientSelector(FeatureSelector, BaseEstimator, SelectorMixin):
    def __init__(self,
                 order=4,
                 penalty=1,
                 n_features=None,
                 max_features=None,
                 learning_rate=1e-1,
                 init='zero',
                 n_epochs=1,
                 shuffle=True,
                 batch_size=1000,
                 target_batch_size=1000,
                 max_time=np.inf,
                 classification=True,
                 ordinal=False,
                 balanced=True,
                 preprocess='zscore',
                 soft_grouping=False,
                 verbose=0,
                 device='cpu'):
        """
            FeatureGradientSelector is a class that selects features for a machine
            learning model using a gradient based search.

            Parameters
            ----------
            order : int
                What order of interactions to include. Higher orders
                may be more accurate but increase the run time. 12 is the maximum allowed order.
            penatly : int
                Constant that multiplies the regularization term.
            n_features: int
                If None, will automatically choose number of features based on search.
                Otherwise, number of top features to select.
            max_features : int
                If not None, will use the 'elbow method' to determine the number of features
                with max_features as the upper limit.
            learning_rate : float
            init : str
                How to initialize the vector of scores. 'zero' is the default.
                Options: {'zero', 'on', 'off', 'onhigh', 'offhigh', 'sklearn'}
            n_epochs : int
                number of epochs to run
            shuffle : bool
                Shuffle "rows" prior to an epoch.
            batch_size : int
                Nnumber of "rows" to process at a time
            target_batch_size : int
                Number of "rows" to accumulate gradients over.
                Useful when many rows will not fit into memory but are needed for accurate estimation.
            classification : bool
                If True, problem is classification, else regression.
            ordinal : bool
                If True, problem is ordinal classification. Requires classification to be True.
            balanced : bool
                If true, each class is weighted equally in optimization, otherwise
                weighted is done via support of each class. Requires classification to be True.
            prerocess : str
                'zscore' which refers to centering and normalizing data to unit variance or
                'center' which only centers the data to 0 mean
            soft_grouping : bool
                if True, groups represent features that come from the same source.
                Used to encourage sparsity of groups and features within groups.
            verbose : int
                Controls the verbosity when fitting. Set to 0 for no printing
                1 or higher for printing every verbose number of gradient steps.
            device : str
                'cpu' to run on CPU and 'cuda' to run on GPU. Runs much faster on GPU
        """
        assert order <= 12 and order >= 1, 'order must be an integer between 1 and 12, inclusive'
        assert n_features is None or max_features is None, \
            'only specify one of n_features and max_features at a time'

        self.order = order
        self.penalty = penalty
        self.n_features = n_features
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.init = init
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.max_time = max_time
        self.dftol_stop = -1
        self.freltol_stop = -1
        self.classification = classification
        self.ordinal = ordinal
        self.balanced = balanced
        self.preprocess = preprocess
        self.soft_grouping = soft_grouping
        self.verbose = verbose
        self.device = device

        self.model_ = None
        self.scores_ = None
        self._prev_checkpoint = None
        self._data_train = None

    def partial_fit(self, X, y,
                    n_classes=None,
                    groups=None):
        """
        Select Features via a gradient based search on (X, y) on the given samples.
        Can be called repeatedly with different X and y to handle streaming datasets.

        Parameters
        ----------
        X : array-like
            Shape = [n_samples, n_features]
            The training input samples.
        y :  array-like
            Shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        n_classes : int
            Number of classes
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all).shape[0]`, where y_all is the
            target vector of the entire dataset.
            This argument is expected for the first call to partial_fit,
            otherwise will assume all classes are present in the batch of y given.
            It will be ignored in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        groups : array-like
            Optional, shape = [n_features]
            Groups of columns that must be selected as a unit
            e.g. [0, 0, 1, 2] specifies the first two columns are part of a group.
            This argument is expected for the first call to partial_fit,
            otherwise will assume all classes are present in the batch of y given.
            It will be ignored in the subsequent calls.
        """
        try:
            self._partial_fit(X, y, n_classes=n_classes, groups=groups)
        except constants.NanError:
            if hasattr(self, '_prev_checkpoint'):
                # if it's already done some batches successfully just ignore it
                print('failed fitting this batch, loss was nan')
            else:
                # if this is the first batch, reset and try with doubles
                if self.verbose:
                    print('Loss was nan, trying with Doubles')
                self._reset()
                torch.set_default_tensor_type(torch.DoubleTensor)
                self._partial_fit(X, y, n_classes=n_classes, groups=groups)

        return self

    def _partial_fit(self, X, y, n_classes=None, groups=None):
        """
        Private function for partial_fit to enable trying floats before doubles.
        """
        # pass in X and y in chunks
        if hasattr(self, '_data_train'):
            # just overwrite the X and y from the new chunk but make them tensors
            # keep dataset stats from previous
            self._data_train.X = X.values if isinstance(X, pd.DataFrame) else X
            self._data_train.N, self._data_train.D = self._data_train.X.shape
            self._data_train.dense_size_gb = self._data_train.get_dense_size()
            self._data_train.set_dense_X()

            self._data_train.y = y.values if isinstance(y, pd.Series) else y
            self._data_train.y = torch.as_tensor(
                y, dtype=torch.get_default_dtype())
        else:
            data_train = self._prepare_data(X, y, n_classes=n_classes)
            self._data_train = data_train

        batch_size, _, accum_steps, max_iter = self._set_batch_size(
            self._data_train)

        rng = None  # not used
        debug = 0  # {0,1} print messages and do other stuff?
        dn_logs = None  # tensorboard logs; only specify if debug=1
        path_save = None  # intermediate models saves; only specify if debug=1
        m, solver = _train(self._data_train,
                           batch_size,
                           self.order,
                           self.penalty,
                           rng,
                           self.learning_rate,
                           debug,
                           max_iter,
                           self.max_time,
                           self.init,
                           self.dftol_stop,
                           self.freltol_stop,
                           dn_logs,
                           accum_steps,
                           path_save,
                           self.shuffle,
                           device=self.device,
                           verbose=self.verbose,
                           prev_checkpoint=self._prev_checkpoint if hasattr(
                               self, '_prev_checkpoint') else None,
                           groups=groups if not self.soft_grouping else None,
                           soft_groups=groups if self.soft_grouping else None)

        self._prev_checkpoint = m
        self._process_results(m, solver, X, groups=groups)
        return self

    def fit(self, X, y,
            groups=None):
        """
        Select Features via a gradient based search on (X, y).

        Parameters
        ----------
        X : array-like
            Shape = [n_samples, n_features]
            The training input samples.
        y : array-like
            Shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        groups : array-like
            Optional, shape = [n_features]
            Groups of columns that must be selected as a unit
            e.g. [0, 0, 1, 2] specifies the first two columns are part of a group.
        """
        try:
            self._fit(X, y, groups=groups)
        except constants.NanError:
            if self.verbose:
                print('Loss was nan, trying with Doubles')
            torch.set_default_tensor_type(torch.DoubleTensor)
            self._fit(X, y, groups=groups)
        return self

    def get_selected_features(self):
        return self.selected_features_

    def _prepare_data(self, X, y, n_classes=None):
        """
        Returns a PrepareData object.
        """
        return PrepareData(X=X.values if isinstance(X, pd.DataFrame) else X,
                           y=y.values if isinstance(y, pd.Series) else y,
                           data_format=constants.DataFormat.NUMPY,
                           classification=int(self.classification),
                           ordinal=self.ordinal,
                           balanced=self.balanced,
                           preprocess=self.preprocess,
                           verbose=self.verbose,
                           device=self.device,
                           n_classes=n_classes)

    def _fit(self, X, y, groups=None):
        """
        Private function for fit to enable trying floats before doubles.
        """
        data_train = self._prepare_data(X, y)

        batch_size, _, accum_steps, max_iter = self._set_batch_size(
            data_train)

        rng = None  # not used
        debug = 0  # {0,1} print messages and log to tensorboard
        dn_logs = None  # tensorboard logs; only specify if debug=1
        path_save = None  # intermediate models saves; only specify if debug=1
        m, solver = _train(data_train,
                           batch_size,
                           self.order,
                           self.penalty,
                           rng,
                           self.learning_rate,
                           debug,
                           max_iter,
                           self.max_time,
                           self.init,
                           self.dftol_stop,
                           self.freltol_stop,
                           dn_logs,
                           accum_steps,
                           path_save,
                           self.shuffle,
                           device=self.device,
                           verbose=self.verbose,
                           groups=groups if not self.soft_grouping else None,
                           soft_groups=groups if self.soft_grouping else None)

        self._process_results(m, solver, X, groups=groups)
        return self

    def _process_torch_scores(self, scores):
        """
        Convert scores into flat numpy arrays.
        """
        if constants.Device.CUDA in scores.device.type:
            scores = scores.cpu()
        return scores.numpy().ravel()

    def _set_batch_size(self, data_train):
        """
        Ensures that batch_size is less than the number of rows.
        """
        batch_size = min(self.batch_size, data_train.N)
        target_batch_size = min(max(
            self.batch_size, self.target_batch_size), data_train.N)
        accum_steps = max(int(np.ceil(target_batch_size / self.batch_size)), 1)
        max_iter = self.n_epochs * (data_train.N // batch_size)
        return batch_size, target_batch_size, accum_steps, max_iter

    def _process_results(self, m, solver, X, groups=None):
        """
        Process the results of a run into something suitable for transform().
        """
        self.scores_ = self._process_torch_scores(
            torch.sigmoid(m[constants.Checkpoint.MODEL]['x'] * 2))
        if self.max_features:
            self.max_features = min([self.max_features, self.scores_.shape[0]])
            n_features = self._recommend_number_features(solver)
            self.set_n_features(n_features, groups=groups)
        elif self.n_features:
            self.set_n_features(self.n_features, groups=groups)
        else:
            self.selected_features_ = m['feats']

        # subtract elapsed time from max_time
        self.max_time -= m['t']

        self.model_ = m

        return self

    def transform(self, X):
        """
        Returns selected features from X.

        Paramters
        ---------
        X: array-like
            Shape = [n_samples, n_features]
            The training input samples.
        """

        self._get_support_mask()
        if self.selected_features_.shape[0] == 0:
            raise ValueError(
                'No Features selected, consider lowering the penalty or specifying n_features')
        return (X.iloc[:, self.selected_features_]
                if isinstance(X, pd.DataFrame)
                else X[:, self.selected_features_])

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool
            Default False
            If True, the return value will be an array of integers, rather than a boolean mask.

        Returns
        -------
        list :
            returns support: An index that selects the retained features from a feature vector.
            If indices is False, this is a boolean array of shape [# input features],
            in which an element is True iff its corresponding feature is selected for retention.
            If indices is True, this is an integer array of shape [# output features] whose values
            are indices into the input feature vector.
        """
        self._get_support_mask()
        if indices:
            return self.selected_features_

        mask = np.zeros_like(self.scores_, dtype=bool)
        # pylint: disable=E1137
        mask[self.selected_features_] = True
        return mask

    def inverse_transform(self, X):
        """
        Returns transformed X to the original number of column.
        This operation is lossy and all columns not in the transformed data
        will be returned as columns of 0s.
        """
        self._get_support_mask()
        X_new = np.zeros((X.shape[0], self.scores_.shape[0]))
        X_new[self.selected_features_] = X
        return X_new

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        params = self.__dict__
        params = {key: val for (key, val) in params.items()
                  if not key.endswith('_')}
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for param in params:
            if hasattr(self, param):
                setattr(self, param, params[param])
        return self

    def fit_transform(self, X, y):
        """
        Select features and then return X with the selected features.

        Parameters
        ----------
        X : array-like
            Shape = [n_samples, n_features]
            The training input samples.
        y : array-like
            Shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        """
        self.fit(X, y)
        return self.transform(X)

    def _get_support_mask(self):
        """
        Check if it is fitted.
        """
        check_is_fitted(self, 'scores_')

    def _generate_scores(self, solver, xsub, ysub, step_size, feature_order):
        """
        Generate forward passes to determine the number of features when max_features is set.
        """
        scores = []
        for i in np.arange(1, self.max_features + 1, step_size):
            # optimization possible since xsub is growing?
            i = int(np.ceil(i))
            # pylint: disable=E1102
            score = solver.f_train(torch.tensor(np.ones(i),
                                                dtype=torch.get_default_dtype()
                                                ).unsqueeze(1).to(self.device),
                                   xsub[:, feature_order[:i]],
                                   ysub)
            if constants.Device.CUDA in score.device.type:
                score = score.cpu()
            # score.numpy()[0][0]
            scores.append(score)
        return scores

    def set_n_features(self, n, groups=None):
        """
        Set the number of features to return after fitting.
        """
        self._get_support_mask()
        self.n_features = n
        return self._set_top_features(groups=groups)

    def _set_top_features(self, groups=None):
        """
        Set the selected features after a run.

        With groups, ensures that if any member of a group is selected, all members are selected
        """
        self._get_support_mask()
        assert self.n_features <= self.scores_.shape[0], \
            'n_features must be less than or equal to the number of columns in X'
        # pylint: disable=E1130
        self.selected_features_ = np.argpartition(
            self.scores_, -self.n_features)[-self.n_features:]
        if groups is not None and not self.soft_grouping:
            selected_feature_set = set(self.selected_features_.tolist())
            for _ in np.unique(groups):
                group_members = np.where(groups == groups)[0].tolist()
                if selected_feature_set.intersection(group_members):
                    selected_feature_set.update(group_members)
            self.selected_features_ = np.array(list(selected_feature_set))
        self.selected_features_ = np.sort(self.selected_features_)
        return self

    def set_top_percentile(self, percentile, groups=None):
        """
        Set the percentile of features to return after fitting.
        """
        self._get_support_mask()
        assert percentile <= 1 and percentile >= 0, \
            'percentile must between 0 and 1 inclusive'
        self.n_features = int(self.scores_.shape[0] * percentile)
        return self._set_top_features(groups=groups)

    def _recommend_number_features(self, solver, max_time=None):
        """
        Get the recommended number of features by doing forward passes when max_features is set.
        """
        max_time = max_time if max_time else self.max_time
        if max_time < 0:
            max_time = 60  # allow 1 minute extra if we already spent max_time
        MAX_FORWARD_PASS = 200
        MAX_FULL_BATCHES = 3  # the forward passes can take longer than the fitting
        # if we allow a full epoch of data to be included. By only doing 3 full batches at most
        # we get enough accuracy without increasing the time too much. This
        # constant may not be optimal
        accum_steps = solver.accum_steps
        step_size = max(self.max_features / MAX_FORWARD_PASS, 1)
        # pylint: disable=E1130
        feature_order = np.argsort(-self.scores_)  # note the negative
        t = time.time()

        dataloader_iterator = iter(solver.ds_train)
        full_scores = []
        # keep_going = True
        with torch.no_grad():
            # might want to only consider a batch valid if there are at least
            # two classes
            for _ in range(accum_steps * MAX_FULL_BATCHES):
                scores = []
                try:
                    xsub, ysub = next(dataloader_iterator)
                except StopIteration:
                    # done with epoch, don't do more than one epoch
                    break
                except Exception as e:
                    print(e)
                    break
                if max_time and time.time() - t > max_time:
                    if self.verbose:
                        print(
                            "Stoppinn forward passes because they reached max_time: ",
                            max_time)
                    if not full_scores:
                        # no forward passes worked, return half of max_features
                        return self.max_features // 2
                    break
                if solver.multiclass:
                    for target_class in range(solver.n_classes):
                        ysub_binary = solver.transform_y_into_binary(
                            ysub, target_class)
                        scaling_value = solver._get_scaling_value(
                            ysub, target_class)
                        if not solver._skip_y_forward(ysub_binary):
                            scores = self._generate_scores(
                                solver, xsub, ysub_binary, step_size, feature_order)
                            # one row will represent one class that is present in the data
                            # all classes are weighted equally
                            full_scores.append(
                                [score * scaling_value for score in scores])
                else:
                    if not solver._skip_y_forward(ysub):
                        scores = self._generate_scores(
                            solver, xsub, ysub, step_size, feature_order)
                        full_scores.append(scores)
        best_index = FeatureGradientSelector._find_best_index_elbow(
            full_scores)
        if self.verbose:
            print("Forward passes took: ", time.time() - t)
        # account for step size and off by one (n_features is 1 indexed, not 0
        # )
        return int(
            np.ceil(
                np.arange(
                    1,
                    self.max_features +
                    1,
                    step_size))[best_index])

    @staticmethod
    def _find_best_index_elbow(full_scores):
        """
        Finds the point on the curve that maximizes distance from the line determined by the endpoints.
        """
        scores = pd.DataFrame(full_scores).mean(0).values.tolist()
        first_point = np.array([0, scores[0]])
        last_point = np.array([len(scores) - 1, scores[-1]])
        elbow_metric = []
        for i in range(len(scores)):
            elbow_metric.append(
                FeatureGradientSelector._distance_to_line(
                    first_point, last_point, np.array([i, scores[i]])))
        return np.argmax(elbow_metric)

    @staticmethod
    def _distance_to_line(start_point, end_point, new_point):
        """
        Calculates the shortest distance from new_point to the line determined by start_point and end_point.
        """
        # for calculating elbow method
        return np.cross(new_point - start_point,
                        end_point - start_point) / np.linalg.norm(
                            end_point - start_point)

    def _reset(self):
        """
        Reset the estimator by deleting all private and fit parameters.
        """
        params = self.__dict__
        for key, _ in params.items():
            if key.endswith('_') or key.startswith('_'):
                delattr(self, key)
        return self
