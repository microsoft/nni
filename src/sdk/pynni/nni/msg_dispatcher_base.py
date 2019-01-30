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
import logging
import json_tricks

from .common import init_logger
from .recoverable import Recoverable
from .protocol import CommandType, receive

init_logger('dispatcher.log')
_logger = logging.getLogger(__name__)

class MsgDispatcherBase(Recoverable):
    def run(self):
        """Run the tuner.
        This function will never return unless raise.
        """
        mode = os.getenv('NNI_MODE')
        if mode == 'resume':
            self.load_checkpoint()

        while self.handle_request():
            pass

        _logger.info('Terminated by NNI manager')

    def handle_request(self):
        _logger.debug('waiting receive_message')

        command, data = receive()
        if command is None:
            return False

        _logger.debug('handle request: command: [{}], data: [{}]'.format(command, data))

        if command is CommandType.Terminate:
            return False

        data = json_tricks.loads(data)

        command_handlers = {
            # Tunner commands:
            CommandType.RequestTrialJobs: self.handle_request_trial_jobs,
            CommandType.UpdateSearchSpace: self.handle_update_search_space,
            CommandType.AddCustomizedTrialJob: self.handle_add_customized_trial,

            # Tunner/Assessor commands:
            CommandType.ReportMetricData: self.handle_report_metric_data,

            CommandType.TrialEnd: self.handle_trial_end,
        }
        if command not in command_handlers:
            raise AssertionError('Unsupported command: {}'.format(command))

        return command_handlers[command](data)

    def handle_request_trial_jobs(self, data):
        raise NotImplementedError('handle_request_trial_jobs not implemented')

    def handle_update_search_space(self, data):
       raise NotImplementedError('handle_update_search_space not implemented')

    def handle_add_customized_trial(self, data):
        raise NotImplementedError('handle_add_customized_trial not implemented')

    def handle_report_metric_data(self, data):
        raise NotImplementedError('handle_report_metric_data not implemented')

    def handle_trial_end(self, data):
        raise NotImplementedError('handle_trial_end not implemented')
