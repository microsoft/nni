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

import os
import sys
from operator import itemgetter

import sklearn.mixture as mm

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def create_model(samples_x, samples_y_aggregation, percentage_goodbatch=0.34):
    '''
    Create the Gaussian Mixture Model
    '''
    samples = [samples_x[i] + [samples_y_aggregation[i]] for i in range(0, len(samples_x))]

    # Sorts so that we can get the top samples
    samples = sorted(samples, key=itemgetter(-1))
    samples_goodbatch_size = int(len(samples) * percentage_goodbatch)
    samples_goodbatch = samples[0:samples_goodbatch_size]
    samples_badbatch = samples[samples_goodbatch_size:]

    samples_x_goodbatch = [sample_goodbatch[0:-1] for sample_goodbatch in samples_goodbatch]
    #samples_y_goodbatch = [sample_goodbatch[-1] for sample_goodbatch in samples_goodbatch]
    samples_x_badbatch = [sample_badbatch[0:-1] for sample_badbatch in samples_badbatch]

    # === Trains GMM clustering models === #
    #sys.stderr.write("[%s] Train GMM's GMM model\n" % (os.path.basename(__file__)))
    bgmm_goodbatch = mm.BayesianGaussianMixture(n_components=max(1, samples_goodbatch_size - 1))
    bad_n_components = max(1, len(samples_x) - samples_goodbatch_size - 1)
    bgmm_badbatch = mm.BayesianGaussianMixture(n_components=bad_n_components)
    bgmm_goodbatch.fit(samples_x_goodbatch)
    bgmm_badbatch.fit(samples_x_badbatch)

    model = {}
    model['clusteringmodel_good'] = bgmm_goodbatch
    model['clusteringmodel_bad'] = bgmm_badbatch
    return model
    