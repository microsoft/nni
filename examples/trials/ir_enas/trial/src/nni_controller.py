from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import logging
import tensorflow as tf
import fcntl
from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.cifar10.general_controller import GeneralController
import nni
from nni.multi_phase.multi_phase_tuner import MultiPhaseTuner

class ENASBaseTuner(MultiPhaseTuner):

    def __init__(self):
        return


    def get_controller_arc_macro(self, child_totalsteps):
        child_arc = []
        for _ in range(0, child_totalsteps):
            arc = self.sess.run(self.controller_model.sample_arc)
            child_arc.append(arc)
        return child_arc
