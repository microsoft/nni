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

#import json_tricks
import os
import threading
import logging
from multiprocessing.dummy import Pool as ThreadPool
from queue import Queue, Empty
import json_tricks

from .common import multi_thread_enabled
from .env_vars import dispatcher_env_vars
from .utils import init_dispatcher_logger
from .recoverable import Recoverable
from .protocol import CommandType, receive

init_dispatcher_logger()

_logger = logging.getLogger(__name__)

QUEUE_LEN_WARNING_MARK = 20
_worker_fast_exit_on_terminate = True

class MsgDispatcherBase(Recoverable):
    def __init__(self):
        if multi_thread_enabled():
            self.pool = ThreadPool()
            self.thread_results = []
        else:
            self.stopping = False
            self.default_command_queue = Queue()
            self.assessor_command_queue = Queue()
            self.default_worker = threading.Thread(target=self.command_queue_worker, args=(self.default_command_queue,))
            self.assessor_worker = threading.Thread(target=self.command_queue_worker, args=(self.assessor_command_queue,))
            self.default_worker.start()
            self.assessor_worker.start()
            self.worker_exceptions = []

    def run(self):
        """Run the tuner.
        This function will never return unless raise.
        """
        _logger.info('Start dispatcher')
        if dispatcher_env_vars.NNI_MODE == 'resume':
            self.load_checkpoint()

        while True:
            command, data = receive()
            if data:
                data = json_tricks.loads(data)

            if command is None or command is CommandType.Terminate:
                break
            if multi_thread_enabled():
                result = self.pool.map_async(self.process_command_thread, [(command, data)])
                self.thread_results.append(result)
                if any([thread_result.ready() and not thread_result.successful() for thread_result in self.thread_results]):
                    _logger.debug('Caught thread exception')
                    break
            else:
                self.enqueue_command(command, data)
                if self.worker_exceptions:
                    break

        _logger.info('Dispatcher exiting...')
        self.stopping = True
        if multi_thread_enabled():
            self.pool.close()
            self.pool.join()
        else:
            self.default_worker.join()
            self.assessor_worker.join()

        _logger.info('Terminated by NNI manager')

    def command_queue_worker(self, command_queue):
        """Process commands in command queues.
        """
        while True:
            try:
                # set timeout to ensure self.stopping is checked periodically
                command, data = command_queue.get(timeout=3)
                try:
                    self.process_command(command, data)
                except Exception as e:
                    _logger.exception(e)
                    self.worker_exceptions.append(e)
                    break
            except Empty:
                pass
            if self.stopping and (_worker_fast_exit_on_terminate or command_queue.empty()):
                break

    def enqueue_command(self, command, data):
        """Enqueue command into command queues
        """
        if command == CommandType.TrialEnd or (command == CommandType.ReportMetricData and data['type'] == 'PERIODICAL'):
            self.assessor_command_queue.put((command, data))
        else:
            self.default_command_queue.put((command, data))

        qsize = self.default_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('default queue length: %d', qsize)

        qsize = self.assessor_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('assessor queue length: %d', qsize)

    def process_command_thread(self, request):
        """Worker thread to process a command.
        """
        command, data = request
        if multi_thread_enabled():
            try:
                self.process_command(command, data)
            except Exception as e:
                _logger.exception(str(e))
                raise
        else:
            pass

    def process_command(self, command, data):
        _logger.debug('process_command: command: [{}], data: [{}]'.format(command, data))

        command_handlers = {
            # Tunner commands:
            CommandType.Initialize: self.handle_initialize,
            CommandType.RequestTrialJobs: self.handle_request_trial_jobs,
            CommandType.UpdateSearchSpace: self.handle_update_search_space,
            CommandType.ImportData: self.handle_import_data,
            CommandType.AddCustomizedTrialJob: self.handle_add_customized_trial,

            # Tunner/Assessor commands:
            CommandType.ReportMetricData: self.handle_report_metric_data,

            CommandType.TrialEnd: self.handle_trial_end,
            CommandType.Ping: self.handle_ping,
        }
        if command not in command_handlers:
            raise AssertionError('Unsupported command: {}'.format(command))
        command_handlers[command](data)

    def handle_ping(self, data):
        pass

    def handle_initialize(self, data):
        raise NotImplementedError('handle_initialize not implemented')

    def handle_request_trial_jobs(self, data):
        raise NotImplementedError('handle_request_trial_jobs not implemented')

    def handle_update_search_space(self, data):
       raise NotImplementedError('handle_update_search_space not implemented')

    def handle_import_data(self, data):
        raise NotImplementedError('handle_import_data not implemented')

    def handle_add_customized_trial(self, data):
        raise NotImplementedError('handle_add_customized_trial not implemented')

    def handle_report_metric_data(self, data):
        raise NotImplementedError('handle_report_metric_data not implemented')

    def handle_trial_end(self, data):
        raise NotImplementedError('handle_trial_end not implemented')
