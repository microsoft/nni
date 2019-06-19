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
import json
import logging
import logging.handlers
import time
import threading
import re

from datetime import datetime
from enum import Enum, unique
from logging import StreamHandler

from queue import Queue

from .rest_utils import rest_get, rest_post, rest_put, rest_delete
from .constants import NNI_EXP_ID, NNI_TRIAL_JOB_ID, STDOUT_API
from .url_utils import gen_send_stdout_url

@unique
class LogType(Enum):
    Trace = 'TRACE'
    Debug = 'DEBUG'
    Info = 'INFO'
    Warning = 'WARNING'
    Error = 'ERROR'
    Fatal = 'FATAL'

@unique
class StdOutputType(Enum):
    Stdout = 'stdout',
    Stderr = 'stderr'

def nni_log(log_type, log_message):
    '''Log message into stdout'''
    dt = datetime.now()
    print('[{0}] {1} {2}'.format(dt, log_type.value, log_message))

class NNIRestLogHanlder(StreamHandler):
    def __init__(self, host, port, tag, std_output_type=StdOutputType.Stdout):
        StreamHandler.__init__(self)
        self.host = host
        self.port = port
        self.tag = tag
        self.std_output_type = std_output_type
        self.orig_stdout = sys.__stdout__
        self.orig_stderr = sys.__stderr__

    def emit(self, record):
        log_entry = {}
        log_entry['tag'] = self.tag
        log_entry['stdOutputType'] = self.std_output_type.name
        log_entry['msg'] = self.format(record)

        try:
            response = rest_post(gen_send_stdout_url(self.host, self.port), json.dumps(log_entry), 10, True)
        except Exception as e:
            self.orig_stderr.write(str(e) + '\n')
            self.orig_stderr.flush()

class RemoteLogger(object):
    """
    NNI remote logger
    """
    def __init__(self, syslog_host, syslog_port, tag, std_output_type, log_collection, log_level=logging.INFO):
        '''
        constructor
        '''
        self.logger = logging.getLogger('nni_syslog_{}'.format(tag))
        self.log_level = log_level
        self.logger.setLevel(self.log_level)
        handler = NNIRestLogHanlder(syslog_host, syslog_port, tag)
        self.logger.addHandler(handler)
        if std_output_type == StdOutputType.Stdout:
            self.orig_stdout = sys.__stdout__
        else:
            self.orig_stdout = sys.__stderr__
        self.log_collection = log_collection

    def get_pipelog_reader(self):
        '''
        Get pipe for remote logger
        '''
        return PipeLogReader(self.logger, self.log_collection, logging.INFO)

    def write(self, buf):
        '''
        Write buffer data into logger/stdout
        '''
        for line in buf.rstrip().splitlines():
            self.orig_stdout.write(line.rstrip() + '\n')
            self.orig_stdout.flush()
            try:
                self.logger.log(self.log_level, line.rstrip())
            except Exception as e:
                pass

class PipeLogReader(threading.Thread):
    """
    The reader thread reads log data from pipe
    """
    def __init__(self, logger, log_collection, log_level=logging.INFO):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        threading.Thread.__init__(self)
        self.queue = Queue()
        self.logger = logger
        self.daemon = False
        self.log_level = log_level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.orig_stdout = sys.__stdout__
        self._is_read_completed = False
        self.process_exit = False
        self.log_collection = log_collection
        self.log_pattern = re.compile(r'^NNISDK_MEb\'.*\'$')

        def _populateQueue(stream, queue):
            '''
            Collect lines from 'stream' and put them in 'quque'.
            '''
            time.sleep(5)
            while True:
                cur_process_exit = self.process_exit
                try:
                    line = self.queue.get(True, 5)
                    try:
                        self.logger.log(self.log_level, line.rstrip())
                    except Exception as e:
                        pass
                except Exception as e:
                    if cur_process_exit == True:
                        self._is_read_completed = True
                        break

        self.pip_log_reader_thread = threading.Thread(target = _populateQueue,
                args = (self.pipeReader, self.queue))
        self.pip_log_reader_thread.daemon = True
        self.start()
        self.pip_log_reader_thread.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
           If the log_collection is 'none', the log content will not be enqueued
        """
        for line in iter(self.pipeReader.readline, ''):
            self.orig_stdout.write(line.rstrip() + '\n')
            self.orig_stdout.flush()
            if self.log_collection == 'none':
                # If not match metrics, do not put the line into queue
                if not self.log_pattern.match(line):
                    continue
            self.queue.put(line)

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)

    @property
    def is_read_completed(self):
        """Return if read is completed
        """
        return self._is_read_completed

    def set_process_exit(self):
        self.process_exit = True
        return self.process_exit