import math
import logging
import copy
import random
import numpy as np
from itertools import permutations, combinations

import nni
from nni.tuner import Tuner


class Parameter(object):
    """Base class for all types of parameters
    """
    def mutate(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def pick_out(self):
        raise NotImplementedError

    def get_cardinality(self):
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


class Choice(Parameter):
    """choice type parameter
    """
    def __init__(self, choices, mutate_rate):
        self.choices = choices
        self.value = random.choice(self.choices)
        self.mutate_rate = mutate_rate

    def get_cardinality(self):
        return len(self.choices)

    def reset(self):
        self.value = random.choice(self.choices)

    def mutate(self):
        child = copy.deepcopy(self)
        while random.uniform(0, 1) < child.mutate_rate:
            choices = copy.deepcopy(child.choices)
            choices.remove(child.value)
            if choices:
                child.value = random.choice(choices)
            else:
                break

        return child

    def pick_out(self):
        return self.value


class Discrete(Parameter):
    """choice type parameter
    """
    def __init__(self, numbers, mutate_rate):
        numbers.sort()
        self.numbers = numbers
        self.value = random.choice(self.numbers)
        self.mutate_rate = mutate_rate

    def get_cardinality(self):
        return len(self.numbers)

    def reset(self):
        self.value = random.choice(self.numbers)

    def mutate(self):
        child = copy.deepcopy(self)
        while random.uniform(0, 1) < child.mutate_rate:
            idx = child.numbers.index(child.value)
            if idx == 0 and idx + 1 < len(child.numbers):
                child.value = child.numbers[idx + 1]
            elif idx + 1 == len(child.numbers) and idx - 1 >= 0:
                child.value = child.numbers[idx - 1]
            elif idx == 0 and idx + 1 == len(child.numbers):
                break
            else:
                shift = random.choice([-1, 1])
                child.value = child.numbers[idx + shift]

        return child

    def pick_out(self):
        return self.value


class Factor(Parameter):
    """factor type parameter
    """
    def __init__(self, value, mutate_rate):
        self.product, self.num = value
        self.mutate_rate = mutate_rate
        self.all_partitions = self._get_all_partitions(self.product, self.num)
        self.partition = random.choice(self.all_partitions)

    def reset(self):
        self.partition = random.choice(self.all_partitions)

    def get_cardinality(self):
        return len(self.all_partitions)

    def mutate(self):
        child = copy.deepcopy(self)
        while random.uniform(0, 1) < self.mutate_rate:
            action = random.choice(child._get_actions())
            child._step(action)

        return child

    def pick_out(self):
        return self.partition

    def _step(self, action):
        self.partition[action[0]] = int(self.partition[action[0]] / action[2])
        self.partition[action[1]] = int(self.partition[action[1]] * action[2])

    def _get_actions(self):
        actions = []
        prime_factors = self._get_prime_factors(self.product, False)
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    for k in range(len(prime_factors)):
                        action = [i]
                        action.append(j)
                        action.append(prime_factors[k])
                        if self.partition[action[0]] % action[2] == 0:
                            actions.append(action)
        return actions

    def __repr__(self):
        string = "["
        for factor in self.partition:
            string += factor.__repr__() + " "
        string = string[:-1] + "]"

        return string

    def _get_all_partitions(self, product, num):
        # get all prime factors with repetition
        prime_factors = self._get_prime_factors(product)

        # group all prime factors
        groups = {}
        for prime_factor in prime_factors:
            if prime_factor in groups.keys():
                groups[prime_factor] += 1
            else:
                groups[prime_factor] = 1

        # partition each group
        for key, value in groups.items():
            partitions = []
            for comb in combinations(range(value + num - 1), num - 1):
                # print(comb)
                partition = []
                start_idx = -1
                for idx in comb:
                    partition.append(key**(idx - start_idx - 1))
                    start_idx = idx
                partition.append(key**(value + num - 2 - start_idx))
                partitions.append(partition)
            groups[key] = partitions

        # generate partitions
        partitions = []

        def part(groups, mul=[]):
            if not groups:
                partition = [1] * num
                for i in range(num):
                    for m in mul:
                        partition[i] *= m[i]
                partitions.append(partition)

            for key, group in groups.items():
                for partition in group:
                    mul.append(partition)
                    tmp = copy.deepcopy(groups)
                    del tmp[key]
                    part(tmp, mul)
                    mul.pop()
                break

        part(groups)
        return partitions

    def _get_prime_factors(self, n, repeat=True):
        prime_factors = []

        while n % 2 == 0:
            if 2 not in prime_factors:
                prime_factors.append(2)
            elif repeat:
                prime_factors.append(2)
            n = n / 2

        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                if i not in prime_factors:
                    prime_factors.append(i)
                elif repeat:
                    prime_factors.append(i)
                n = n / i

        if n > 2:
            prime_factors.append(int(n))

        return prime_factors


class Individual(object):
    """Individual class
    """
    def __init__(self, search_space, mutate_rate):
        self.params = {}
        for key in search_space.keys():
            if search_space[key]['_type'] == 'choice':
                self.params[key] = \
                    Choice(search_space[key]['_value'], mutate_rate)
            elif search_space[key]['_type'] == 'discrete':
                self.params[key] = \
                    Discrete(search_space[key]['_value'], mutate_rate)
            elif search_space[key]['_type'] == 'factor':
                self.params[key] = \
                    Factor(search_space[key]['_value'], mutate_rate)
            else:
                raise RuntimeError(
                    "OpEvo Tuner doesn't support this kind of parameter: "
                    + str(search_space[key]['_type'])
                )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __repr__(self):
        string = ""
        for param in self.params:
            string += param.__repr__() + '\n'

        return string

    def mutate(self):
        child = copy.deepcopy(self)
        for key in child.params.keys():
            child.params[key] = child.params[key].mutate()

        return child

    def reset(self):
        for key in self.params.keys():
            self.params[key].reset()

        return self

    def pick_out(self):
        output = {}
        for key in self.params.keys():
            output[key] = self.params[key].pick_out()

        return output


class Population(object):
    """Population class
    """

    def __init__(self, search_space, mutate_rate, opt_mode='maximize'):
        self.search_space = search_space
        self.mutate_rate = mutate_rate
        self.opt_mode = opt_mode
        self.population = []
        self.fitness = []

        self.individual = Individual(self.search_space, self.mutate_rate)
        self.volume = 1
        for key, value in self.individual.params.items():
            self.volume *= self.individual.params[key].get_cardinality()

    def append(self, individual, fitness):
        if self.opt_mode == "minimize":
            fitness = -1 * fitness

        self.population.insert(0, individual)
        self.fitness.insert(0, fitness)

        i = 0
        while (i < len(self.fitness) - 1
                and self.fitness[i] < self.fitness[i + 1]):
            self.fitness[i], self.fitness[i + 1] = \
                self.fitness[i + 1], self.fitness[i]
            self.population[i], self.population[i + 1] = \
                self.population[i + 1], self.population[i]
            i += 1

    def get_offspring(self, parents_size, offspring_size):
        children = []
        if len(self.fitness) < parents_size:
            for _ in range(offspring_size):
                child = copy.deepcopy(self.individual.reset())
                while child in self.population or child in children:
                    child = child.mutate()
                children.append(child)
        elif self.fitness[0] < 1e-3:
            for _ in range(offspring_size):
                child = copy.deepcopy(self.individual.reset())
                while child in self.population or child in children:
                    child = child.mutate()
                children.append(child)
        else:
            prob = np.array(self.fitness[:parents_size]) / \
                np.sum(self.fitness[:parents_size])

            for _ in range(offspring_size):
                child = copy.deepcopy(self.population[0])
                for key in child.params.keys():
                    idx = np.random.choice(range(parents_size), p=prob)
                    child.params[key] = self.population[idx].params[key]
                child = child.mutate()
                while child in self.population or child in children:
                    child = child.mutate()
                children.append(child)

        return children


class OpEvo(Tuner):
    """OpEvo Tuner

    Parameters
    ----------
    optimize_mode: str, 'maximize' or 'minimize'
    parents_size: int
    offspring_size: int
        parents_size and offspring_size govern the diversity in evolutionary
        process. OpEvo with large parents_size and offspring_size tends to get
        rid of suboptimum but sacrifice data efficiency, while one with smaller
        parants_size and offspring_size is easier to converge but suffers suboptimum.
    mutate_rate: float, (0, 1)
        Mutation rate ranging from 0 to 1. It trade-offs the exploration and
        exploitation. OpEvo tends to exploration as q approaches 0, while tends
        to exploitation as q approaches 1.
    """

    def __init__(self,
                 optimize_mode="maximize",
                 parents_size=20,
                 offspring_size=20,
                 mutate_rate=0.5):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel('DEBUG')

        self.optimize_mode = optimize_mode
        self.parents_size = parents_size
        self.offspring_size = offspring_size
        self.mutate_rate = mutate_rate

        self.request_list = []
        self.serve_list = []
        self.wait_dict = {}

    def update_search_space(self, search_space):
        """Update the self.bounds and self.types by the search_space.json file.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if not isinstance(search_space, dict):
            self.logger.info("The format of search space is not a dict.")
            raise RuntimeError("The format of search space is not a dict.")

        self.population = Population(search_space,
                                     self.mutate_rate,
                                     self.optimize_mode)
        self.logger.debug('Total search space volume: ', str(self.population.volume))

        if not self.serve_list:
            self.serve_list = self.population.get_offspring(
                self.parents_size, self.offspring_size)

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """Returns multiple sets of trial (hyper-)parameters,
        as iterable of serializable objects.
        """
        result = []
        self.send_trial_callback = kwargs['st_callback']
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                self.logger.debug("generating param for %s", parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
        return result

    def generate_parameters(self, parameter_id, **kwargs):
        """Method which provides one set of hyper-parameters.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if self.serve_list:
            self.wait_dict[parameter_id] = self.serve_list.pop()
            return self.wait_dict[parameter_id].pick_out()
        else:
            self.request_list.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """Method invoked when a trial reports its final result.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if isinstance(value, dict):
            value = value['default']

        self.population.append(self.wait_dict[parameter_id], value)
        del self.wait_dict[parameter_id]

        if not self.serve_list:
            self.serve_list = self.population.get_offspring(
                self.parents_size, self.offspring_size)

        while self.request_list and self.serve_list:
            param_id = self.request_list[0]
            self.wait_dict[param_id] = self.serve_list.pop()
            self.send_trial_callback(
                param_id, self.wait_dict[param_id].pick_out())
            self.request_list.pop(0)

    def trial_end(self, parameter_id, success, **kwargs):
        """Method invoked when a trial is completed or terminated.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if not success:
            self.population.append(self.wait_dict[parameter_id], 0.0)
            del self.wait_dict[parameter_id]
