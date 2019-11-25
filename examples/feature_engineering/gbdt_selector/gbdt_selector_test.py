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

import bz2
import urllib.request
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from nni.feature_engineering.gbdt_selector import GBDTSelector

url_zip_train = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
urllib.request.urlretrieve(url_zip_train, filename='train.bz2')

f_svm = open('train.svm', 'wt')
with bz2.open('train.bz2', 'rb') as f_zip:
    data = f_zip.read()
    f_svm.write(data.decode('utf-8'))
f_svm.close()

X, y = load_svmlight_file('train.svm')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 20,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0}

eval_ratio = 0.1
early_stopping_rounds = 10
importance_type = 'gain'
num_boost_round = 1000
topk = 10

selector = GBDTSelector()
selector.fit(X_train, y_train,
             lgb_params = lgb_params,
             eval_ratio = eval_ratio,
             early_stopping_rounds = early_stopping_rounds,
             importance_type = importance_type,
             num_boost_round = num_boost_round)

print("selected features\t", selector.get_selected_features(topk=topk))

