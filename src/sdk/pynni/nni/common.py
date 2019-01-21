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


from collections import namedtuple
from datetime import datetime
from io import TextIOBase
import logging
import os
import sys


def _load_env_args():
    args = {
        'platform': os.environ.get('NNI_PLATFORM'),
        'trial_job_id': os.environ.get('NNI_TRIAL_JOB_ID'),
        'log_dir': os.environ.get('NNI_LOG_DIRECTORY'),
        'role': os.environ.get('NNI_ROLE'),
    }
    return namedtuple('EnvArgs', args.keys())(**args)

env_args = _load_env_args()
'''Arguments passed from environment'''


_time_format = '%Y-%m-%d %H:%M:%S'
class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.file = logger_file

    def write(self, s):
        if s != '\n':
            time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(time) + s + '\n')
            self.file.flush()
        return len(s)


def init_logger(logger_file_path):
    """Initialize root logger.
    This will redirect anything from logging.getLogger() as well as stdout to specified file.
    logger_file_path: path of logger file (path-like object).
    """
    if env_args.platform == 'unittest':
        logger_file_path = 'unittest.log'
    elif env_args.log_dir is not None:
        logger_file_path = os.path.join(env_args.log_dir, logger_file_path)
    logger_file = open(logger_file_path, 'w')

    fmt = '[%(asctime)s] %(levelname)s (%(name)s) %(message)s'
    formatter = logging.Formatter(fmt, _time_format)

    handler = logging.StreamHandler(logger_file)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    # these modules are too verbose
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    sys.stdout = _LoggerFileWrapper(logger_file)

_multi_thread = False

def enable_multi_thread():
    global _multi_thread
    _multi_thread = True

def multi_thread_enabled():
    return _multi_thread
