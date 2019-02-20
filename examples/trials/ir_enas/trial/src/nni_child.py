from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import fcntl
import numpy as np
import tensorflow as tf
import logging
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.cifar10.data_utils import read_data
from src.cifar10.general_child import GeneralChild
from src.cifar10.micro_child import MicroChild


class ENASBaseTrial(object):

    def __init__(self):
        return


    def get_csvaa(self, controller_total_steps, child_arc):
        valid_acc_arr = []
        for idx in range(0, controller_total_steps):
            cur_valid_acc = self.sess.run\
                (self.child_model.cur_valid_acc,
                 feed_dict={self.child_model.sample_arc: child_arc[idx]})
            valid_acc_arr.append(cur_valid_acc)
        return valid_acc_arr


    def parset_child_arch(self, child_arc):
        result_arc = []
        for i in range(0,len(child_arc)):
            arc = child_arc[i]['__ndarray__']
            result_arc.append(arc)
        return result_arc

