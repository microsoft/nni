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

# import nni
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import logging
from sklearn.model_selection import KFold

LOG = logging.getLogger('mnist_keras')

model_dict = {
    "LinearRegression": linear_model.LinearRegression(),
    "Lasso": linear_model.Lasso(),
    "Ridge": linear_model.Ridge(),
    "BayesianRidge": linear_model.BayesianRidge(),
    "RidgeCV": linear_model.RidgeCV()
}

def load_data():
    '''Load dataset, use boston dataset'''
    boston = load_boston()
    return boston.data, boston.target

def train(params, n_splits):
    '''Train model and predict result'''
    X, y = load_data()
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=0)
    if not model_dict.get(params.get('model_name')):
        raise Exception('model_name error!')
    clf = model_dict.get(params.get('model_name'))
    total = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        LOG.debug('Intermediate result is: %d', score)
        nni.report_intermediate_result(score)
        total += score
    total /= n_splits
    nni.report_final_result(total)
    LOG.debug('Final result is: %d', total)

if __name__ == '__main__':
    RECEIVED_PARAMS = nni.get_parameters()
    train(RECEIVED_PARAMS, 5)
    train({'model_name':'Lasso'}, 5)
