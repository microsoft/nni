# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.tuner import Tuner

class DummyTuner(Tuner):
    def generate_parameters(self, parameter_id):
        return 'unit-test-parm'

    def generate_multiple_parameters(self, parameter_id_list):
        return ['unit-test-param1', 'unit-test-param2']

    def receive_trial_result(self, parameter_id, parameters, value):
        pass

    def receive_customized_trial_result(self, parameter_id, parameters, value):
        pass

    def update_search_space(self, search_space):
        pass
