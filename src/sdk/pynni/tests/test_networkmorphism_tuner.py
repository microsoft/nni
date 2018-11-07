import logging
import random
from io import BytesIO

import nni
import nni.protocol
from nni.protocol import CommandType, send, receive

from unittest import TestCase, main

from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner


class NetworkMorphismTestCase(TestCase):
    def test_generate_parameters(self):
        pass

    def test_receive_trial_result(self):
        pass

    def test_update_search_space(self):
        pass

    def test__choose_tuner(self):
        pass

    def test_init_search(self):
        pass

    def test_add_model(self):
        pass

    def test_get_best_model_id(self):
        pass


if __name__ == '__main__':
    main()
