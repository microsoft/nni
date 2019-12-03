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

import os

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from nni.feature_engineering.gradient_selector import FeatureGradientSelector


class Benchmark():

    def __init__(self, files, test_size = 0.2):
        self.files =  files
        self.test_size = test_size


    def run_all_test(self, pipeline):
        for file_name in self.files:
            file_path = self.files[file_name]

            self.run_test(pipeline, file_name, file_path)


    def run_test(self, pipeline, name, path):
        print("download " + name)
        update_name = self.download(name, path)
        X, y = load_svmlight_file(update_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
        pipeline.fit(X_train, y_train)
        print("[Benchmark "+ name + " Score]: ", pipeline.score(X_test, y_test))


    def download(self, name, path):
        old_name = name + '_train.bz2'
        update_name = name + '_train.svm'

        if os.path.exists(old_name) and os.path.exists(update_name):
            return update_name

        urllib.request.urlretrieve(path, filename=old_name)

        f_svm = open(update_name, 'wt')
        with bz2.open(old_name, 'rb') as f_zip:
            data = f_zip.read()
            f_svm.write(data.decode('utf-8'))
        f_svm.close()

        return update_name


if __name__ == "__main__":
    LIBSVM_DATA = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        # "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        # "kdd2010" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.bz2",
        # "kdd2012" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdd12.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2"
    }

    test_benchmark = Benchmark(LIBSVM_DATA)

    pipeline1 = make_pipeline(LogisticRegression())
    print("Test all data in LogisticRegression.")
    print()
    test_benchmark.run_all_test(pipeline1)

    pipeline2 = make_pipeline(FeatureGradientSelector(), LogisticRegression())
    print("Test data selected by FeatureGradientSelector in LogisticRegression.")
    print()
    test_benchmark.run_all_test(pipeline2)

    pipeline3 = make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
    print("Test data selected by TreeClssifier in LogisticRegression.")
    print()
    test_benchmark.run_all_test(pipeline3)

    pipeline4 = make_pipeline(FeatureGradientSelector(n_features=20), LogisticRegression())
    print("Test data selected by FeatureGradientSelector top 20 in LogisticRegression.")
    print()
    test_benchmark.run_all_test(pipeline4)
    
    print("Done.")