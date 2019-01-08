# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
Util Module
'''

import time

import tensorflow as tf


def shape(tensor):
    '''
    Get shape of variable.
    Return type is tuple.
    '''
    temp_s = tensor.get_shape()
    return tuple([temp_s[i].value for i in range(0, len(temp_s))])


def get_variable(name, temp_s):
    '''
    Get variable by name.
    '''
    return tf.Variable(tf.zeros(temp_s), name=name)


def dropout(tensor, drop_prob, is_training):
    '''
    Dropout except test.
    '''
    if not is_training:
        return tensor
    return tf.nn.dropout(tensor, 1.0 - drop_prob)


class Timer:
    '''
    Class Timer is for calculate time.
    '''
    def __init__(self):
        self.__start = time.time()

    def start(self):
        '''
        Start to calculate time.
        '''
        self.__start = time.time()

    def get_elapsed(self, restart=True):
        '''
        Calculate time span.
        '''
        end = time.time()
        span = end - self.__start
        if restart:
            self.__start = end
        return span
