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

"""
gbdt_selector.py including:
    class GBDTSelector
"""

import random
from sklearn.model_selection import train_test_split

# pylint: disable=E0401
import lightgbm as lgb

from nni.feature_engineering.feature_selector import FeatureSelector


class GBDTSelector(FeatureSelector):

    def __init__(self, **kwargs):
        self.selected_features_ = None
        self.X = None
        self.y = None
        self.feature_importance = None
        self.lgb_params = None
        self.eval_ratio = None
        self.early_stopping_rounds = None
        self.importance_type = None
        self.num_boost_round = None
        self.model = None


    def fit(self, X, y, **kwargs):
        """
        Fit the training data to FeatureSelector

        Paramters
        ---------
        X : array-like numpy matrix
            The training input samples, which shape is [n_samples, n_features].
        y : array-like numpy matrix
            The target values (class labels in classification, real numbers in
            regression). Which shape is [n_samples].
        lgb_params : dict
            Parameters of lightgbm
        eval_ratio : float
            The ratio of data size. It's used for split the eval data and train data from self.X.
        early_stopping_rounds : int
            The early stopping setting in lightgbm.
        importance_type : str
            Supporting type is 'gain' or 'split'.
        num_boost_round : int
            num_boost_round in lightgbm.
        """
        assert kwargs['lgb_params']
        assert kwargs['eval_ratio']
        assert kwargs['early_stopping_rounds']
        assert kwargs['importance_type']
        assert kwargs['num_boost_round']

        self.X = X
        self.y = y
        self.lgb_params = kwargs['lgb_params']
        self.eval_ratio = kwargs['eval_ratio']
        self.early_stopping_rounds = kwargs['early_stopping_rounds']
        self.importance_type = kwargs['importance_type']
        self.num_boost_round = kwargs['num_boost_round']

        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            test_size=self.eval_ratio,
                                                            random_state=random.seed(41))
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        self.model = lgb.train(self.lgb_params,
                               lgb_train,
                               num_boost_round=self.num_boost_round,
                               valid_sets=lgb_eval,
                               early_stopping_rounds=self.early_stopping_rounds)

        self.feature_importance = self.model.feature_importance(self.importance_type)


    def get_selected_features(self, topk):
        """
        Fit the training data to FeatureSelector

        Returns
        -------
        list :
                Return the index of imprtant feature.
        """
        assert topk > 0

        self.selected_features_ = self.feature_importance.argsort()[-topk:][::-1]

        return self.selected_features_
