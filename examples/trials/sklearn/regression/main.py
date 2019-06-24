# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import logging
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

LOG = logging.getLogger('sklearn_regression')

def load_data():
    '''Load dataset, use boston dataset'''
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=99, test_size=0.25)
    #normalize data
    ss_X = StandardScaler()
    ss_y = StandardScaler()

    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    y_train = ss_y.fit_transform(y_train[:, None])[:,0]
    y_test = ss_y.transform(y_test[:, None])[:,0]

    return X_train, X_test, y_train, y_test

def get_default_parameters():
    '''get default parameters'''
    params = {
        'model_name': 'LinearRegression'
    }
    return params

def get_model(PARAMS):
    '''Get model according to parameters'''
    model_dict = {
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor()
    }
    if not model_dict.get(PARAMS['model_name']):
        LOG.exception('Not supported model!')
        exit(1)

    model = model_dict[PARAMS['model_name']]

    try:
        if PARAMS['model_name'] == 'SVR':
            model.kernel = PARAMS['svr_kernel']
        elif PARAMS['model_name'] == 'KNeighborsRegressor':
            model.weights = PARAMS['knr_weights']
    except Exception as exception:
        LOG.exception(exception)
        raise
    return model


def run(X_train, X_test, y_train, y_test, PARAMS):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    predict_y = model.predict(X_test)
    score = r2_score(y_test, predict_y)
    LOG.debug('r2 score: %s' % score)
    nni.report_final_result(score)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
