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


import logging

import nni
from .recoverable import Recoverable

_logger = logging.getLogger(__name__)


class Tuner(Recoverable):
    # pylint: disable=no-self-use,unused-argument

    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial (hyper-)parameters, as a serializable object.
        User code must override either this function or 'generate_multiple_parameters()'.
        parameter_id: int
        """
        raise NotImplementedError('Tuner: generate_parameters not implemented')

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.
        Call 'generate_parameters()' by 'count' times by default.
        User code must override either this function or 'generate_parameters()'.
        If there's no more trial, user should raise nni.NoMoreTrialError exception in generate_parameters().
        If so, this function will only return sets of trial (hyper-)parameters that have already been collected.
        parameter_id_list: list of int
        """
        result = []
        for parameter_id in parameter_id_list:
            try:
                _logger.debug("generating param for {}".format(parameter_id))
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                return result
            result.append(res)
        return result

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """Invoked when a trial reports its final result. Must override.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        reward: object reported by trial
        """
        raise NotImplementedError('Tuner: receive_trial_result not implemented')

    def receive_customized_trial_result(self, parameter_id, parameters, value, **kwargs):
        """Invoked when a trial added by WebUI reports its final result. Do nothing by default.
        parameter_id: int
        parameters: object created by user
        value: object reported by trial
        """
        _logger.info('Customized trial job %s ignored by tuner', parameter_id)

    def trial_end(self, parameter_id, success, **kwargs):
        """Invoked when a trial is completed or terminated. Do nothing by default.
        parameter_id: int
        success: True if the trial successfully completed; False if failed or terminated
        """
        pass

    def update_search_space(self, search_space):
        """Update the search space of tuner. Must override.
        search_space: JSON object
        """
        raise NotImplementedError('Tuner: update_search_space not implemented')

    def load_checkpoint(self):
        """Load the checkpoint of tuner.
        path: checkpoint directory for tuner
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by tuner, checkpoint path: %s' % checkpoin_path)

    def save_checkpoint(self):
        """Save the checkpoint of tuner.
        path: checkpoint directory for tuner
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by tuner, checkpoint path: %s' % checkpoin_path)

    def import_data(self, data):
        """Import additional data for tuning
        data: a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        pass

    def _on_exit(self):
        pass

    def _on_error(self):
        pass

    @staticmethod
    def convert_nas_search_space(search_space):
        """
        :param search_space: raw search space
        :return: the new search space, mutable_layers will be converted into choice
        """
        ret = dict()
        for k, v in search_space.items():
            if "_type" not in v:
                # this should not happen
                _logger.warning("There is no _type in one of your search space values with key '%s'"
                                ". Please check your search space" % k)
                ret[k] = v
            elif v["_type"] != "mutable_layer":
                ret[k] = v
            else:
                _logger.info("Converting mutable_layer search space with key '%s'" % k)
                # v["_value"] looks like {'mutable_layer_1': {'layer_choice': ...} ...}
                values = v["_value"]
                for layer_name, layer_data in values.items():
                    # there should be at most layer_choice, optional_inputs, optional_input_size in layer_data
                    layer_key = k + "/" + layer_name

                    if layer_data.get("layer_choice"):  # filter out empty choice and no choice
                        layer_choice = layer_data["layer_choice"]
                    else:
                        raise ValueError("No layer choice found in %s" % layer_key)

                    if layer_data.get("optional_inputs") and layer_data.get("optional_input_size"):
                        input_choice = []
                        input_size = layer_data["optional_input_size"]
                        if isinstance(input_size, int):
                            input_size = [input_size, input_size]
                        input_pool = layer_data["optional_inputs"]
                        for chosen_size in range(max(input_size[0], 0), input_size[1] + 1):
                            if chosen_size == 0:
                                input_choice.append([])
                                continue
                            # enumerate all the possible situations
                            for i in range(0, len(input_pool) ** chosen_size):
                                tmp_state, tmp_chosen = i, []
                                for j in range(chosen_size):
                                    tmp_chosen.append(input_pool[tmp_state % len(input_pool)])
                                    tmp_state //= len(input_pool)
                    else:
                        _logger.info("Optional input choices are set to empty by default")
                        input_choice = [[]]

                    ret[layer_key + "/layer_choice"] = layer_choice
                    ret[layer_key + "/input_choice"] = input_choice

        return ret
