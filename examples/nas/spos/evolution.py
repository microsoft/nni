import json
import logging
import re
from collections import deque

import numpy as np
from nni.tuner import Tuner

from network import ShuffleNetV2OneShot

_logger = logging.getLogger("nni")


class SPOSEvolution(Tuner):

    def __init__(self, max_epochs=20, num_select=10, num_population=50, m_prob=0.1,
                 num_crossover=25, num_mutation=25, flops_limit=330E6):
        assert num_population >= num_select
        self.max_epochs = max_epochs
        self.num_select = num_select
        self.num_population = num_population
        self.m_prob = m_prob
        self.num_crossover = num_crossover
        self.num_mutation = num_mutation
        self.flops_limit = flops_limit
        self.model = ShuffleNetV2OneShot()
        self.epoch = 0
        self.candidates = []
        self.search_space = None
        self.random_state = np.random.RandomState(0)

        # async status
        self._to_evaluate_queue = deque()
        self._sending_parameter_queue = deque()
        self._pending_result_ids = set()
        self._reward_dict = dict()
        self._id2candidate = dict()
        self._st_callback = None

    def update_search_space(self, search_space):
        self._search_space = search_space
        self._next_round()

    def _next_round(self):
        if self.epoch >= self.max_epochs:
            return
        _logger.info("Epoch %d, generating...", self.epoch)
        if self.epoch == 0:
            self._get_random_population()
        else:
            best_candidates = self._select_top_candidates()
            self.candidates = self._get_mutation(best_candidates) + self._get_crossover(best_candidates)
            self._get_random_population()
        self.epoch += 1

    def _random_candidate(self):
        chosen_arch = dict()
        for key, val in self._search_space.items():
            if val["_type"] == "layer_choice":
                choices = val["_value"]
                index = self.random_state.randint(len(choices))
                chosen_arch[key] = {"_value": choices[index], "_idx": index}
            elif val["_type"] == "input_choice":
                raise NotImplementedError("Input choice is not implemented yet.")
        return chosen_arch

    def _add_to_evaluate_queue(self, cand):
        _logger.info("Generate candidate %s with flops %d, adding to eval queue.",
                     self._get_architecture_repr(cand), self.model.get_candidate_flops(cand))
        self._reward_dict[self._hashcode(cand)] = 0.
        self._to_evaluate_queue.append(cand)

    def _get_random_population(self):
        while len(self.candidates) < self.num_population:
            cand = self._random_candidate()
            if self._is_legal(cand):
                _logger.info("Random candidate generated.")
                self._add_to_evaluate_queue(cand)
                self.candidates.append(cand)

    def _get_crossover(self, best):
        result = []
        for _ in range(10 * self.num_crossover):
            cand_p1 = best[self.random_state.randint(len(best))]
            cand_p2 = best[self.random_state.randint(len(best))]
            assert cand_p1.keys() == cand_p2.keys()
            cand = {k: cand_p1[k] if self.random_state.randint(2) == 0 else cand_p2[k]
                    for k in cand_p1.keys()}
            if self._is_legal(cand):
                result.append(cand)
                self._add_to_evaluate_queue(cand)
            if len(result) >= self.num_crossover:
                break
        _logger.info("Found %d architectures with crossover.", len(result))
        return result

    def _get_mutation(self, best):
        result = []
        for _ in range(10 * self.num_mutation):
            cand = best[self.random_state.randint(len(best))].copy()
            mutation_sample = np.random.random_sample(len(cand))
            for s, k in zip(mutation_sample, cand):
                if s < self.m_prob:
                    choices = self._search_space[k]["_value"]
                    index = self.random_state.randint(len(choices))
                    cand[k] = {"_value": choices[index], "_idx": index}
            if self._is_legal(cand):
                result.append(cand)
                self._add_to_evaluate_queue(cand)
            if len(result) >= self.num_mutation:
                break
        _logger.info("Found %d architectures with mutation.", len(result))
        return result

    def _get_architecture_repr(self, cand):
        return re.sub(r"\".*?\": \{\"_idx\": (\d+), \"_value\": \".*?\"\}", r"\1",
                      self._hashcode(cand))

    def _is_legal(self, cand):
        if self._hashcode(cand) in self._reward_dict:
            return False
        if self.model.get_candidate_flops(cand) > self.flops_limit:
            return False
        return True

    def _select_top_candidates(self):
        reward_query = lambda cand: self._reward_dict[self._hashcode(cand)]
        _logger.info("All candidate rewards: %s", list(map(reward_query, self.candidates)))
        result = sorted(self.candidates, key=reward_query, reverse=True)[:self.num_select]
        _logger.info("Best candidate rewards: %s", list(map(reward_query, result)))
        return result

    @staticmethod
    def _hashcode(d):
        return json.dumps(d, sort_keys=True)

    def _bind_and_send_parameters(self, use_st_callback=False):
        result = []
        while self._sending_parameter_queue and self._to_evaluate_queue:
            parameter_id = self._sending_parameter_queue.popleft()
            parameters = self._to_evaluate_queue.popleft()
            self._id2candidate[parameter_id] = parameters
            _logger.info("Send parameter [%d] %s.", parameter_id, self._get_architecture_repr(parameters))
            result.append(parameters)
            self._pending_result_ids.add(parameter_id)
            if use_st_callback:
                self._st_callback(parameter_id, parameters)
                _logger.info("Extra parameter with callback.")
        return result

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        if "st_callback" in kwargs:
            self._st_callback = kwargs["st_callback"]
        for parameter_id in parameter_id_list:
            self._sending_parameter_queue.append(parameter_id)
        result = self._bind_and_send_parameters()
        _logger.info("Requested %d parameters, %d sent.", len(parameter_id_list), len(result))
        return result

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        _logger.info("Candidate %d, reported reward %f", parameter_id, value)
        self._reward_dict[self._hashcode(self._id2candidate[parameter_id])] = value

    def trial_end(self, parameter_id, success, **kwargs):
        self._pending_result_ids.remove(parameter_id)
        if not self._pending_result_ids and not self._to_evaluate_queue:
            # a new epoch now
            self._next_round()
            if self._st_callback is not None:
                self._bind_and_send_parameters(use_st_callback=True)
            else:
                _logger.warning("No send callback found.")
