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

from .constants import API_ROOT_URL, BASE_URL, STDOUT_API, NNI_TRIAL_JOB_ID, NNI_EXP_ID, VERSION_API

def gen_send_stdout_url(ip, port):
    '''Generate send stdout url'''
    return '{0}:{1}{2}{3}/{4}/{5}'.format(BASE_URL.format(ip), port, API_ROOT_URL, STDOUT_API, NNI_EXP_ID, NNI_TRIAL_JOB_ID)

def gen_send_version_url(ip, port):
    '''Generate send error url'''
    return '{0}:{1}{2}{3}/{4}/{5}'.format(BASE_URL.format(ip), port, API_ROOT_URL, VERSION_API, NNI_EXP_ID, NNI_TRIAL_JOB_ID)