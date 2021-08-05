import copy
import logging
import random
from collections import deque

from schema import Schema, Optional
import nni
from nni.tuner import Tuner
from nni import ClassArgsValidator
from nni.utils import OptimizeMode, extract_scalar_reward

logger = logging.getLogger(__name__)


class FinishedIndividual:
    def __init__(self, parameter_id, parameters, result):
        """
        Parameters
        ----------
        parameter_id: int
            the index of the parameter
        parameters : dict
            chosen architecture and parameters
        result : float
            final metric of the chosen one
        """
        self.parameter_id = parameter_id
        self.parameters = parameters
        self.result = result


class EvolutionClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('population_size'): self.range('population_size', int, 0, 99999),
            Optional('sample_size'): self.range('sample_size', int, 0, 9999),
        }).validate(kwargs)


class RegularizedEvolutionTuner(Tuner):
    """
    RegularizedEvolutionTuner is tuner using Evolution NAS Tuner.
    See ``Regularized Evolution for Image Classifier Architecture Search`` for details.

    Parameters
    ---
    optimize_mode: str
        whether to maximize metric or not. default: 'maximize'
    population_size: int
        the maximum number of kept models
    sample_size: int
        the number of models chosen from population each time when evolution
    """
    def __init__(self, optimize_mode="maximize", population_size=100, sample_size=25):
        super(RegularizedEvolutionTuner, self).__init__()
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.population_size = population_size
        self.sample_size = sample_size
        self.initial_population = deque()
        self.population = deque()
        self.history = {}
        self.search_space = None
        self._from_initial = {}  # whether the parameter is from initial population

    def generate_parameters(self, parameter_id, **kwargs):
        """
        This function will returns a dict of trial (hyper-)parameters, as a serializable object.

        Parameters
        ---
        parameter_id: int
            the index of current set of parameters
        """
        if self.initial_population:
            arch = self.initial_population.popleft()
            self.history[parameter_id] = arch
            self._from_initial[parameter_id] = True
            return arch
        elif self.population:
            sample = []
            while len(sample) < self.sample_size:
                sample.append(random.choice(list(self.population)))

            candidate = max(sample, key=lambda x: x.result)
            arch = self._mutate_model(candidate)
            self.history[parameter_id] = arch
            self._from_initial[parameter_id] = False
            return arch
        else:
            raise nni.NoMoreTrialError

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Record the result from a trial

        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
            value is final metrics of the trial.
        """
        reward = extract_scalar_reward(value)
        if parameter_id not in self.history:
            raise RuntimeError('Received parameter_id not in total_data.')
        params = self.history[parameter_id]

        if self.optimize_mode == OptimizeMode.Minimize:
            reward = -reward

        self.population.append(FinishedIndividual(parameter_id, params, reward))
        if len(self.population) > self.population_size:
            self.population.popleft()

    def update_search_space(self, search_space):
        """
        Update search space.
        Search_space contains the information that user pre-defined.

        Parameters
        ----------
        search_space : dict
        """
        logger.info('update search space %s', search_space)
        assert self.search_space is None
        self.search_space = search_space

        for _, val in search_space.items():
            if val['_type'] != 'layer_choice' and val['_type'] != 'input_choice':
                raise ValueError('Unsupported search space type: %s' % (val['_type']))

        self._generate_initial_population()

    def trial_end(self, parameter_id, success, **kwargs):
        if not success:
            del self.history[parameter_id]
            if self._from_initial[parameter_id]:
                self.initial_population.append(self._random_model())
            del self._from_initial[parameter_id]

    def _mutate(self, key, individual):
        mutate_val = self.search_space[key]
        if mutate_val['_type'] == 'layer_choice':
            idx = random.randint(0, len(mutate_val['_value']) - 1)
            individual[key] = {'_value': mutate_val['_value'][idx], '_idx': idx}
        elif mutate_val['_type'] == 'input_choice':
            candidates = mutate_val['_value']['candidates']
            n_chosen = mutate_val['_value']['n_chosen']
            idxs = [random.randint(0, len(candidates) - 1) for _ in range(n_chosen)]
            vals = [candidates[k] for k in idxs]
            individual[key] = {'_value': vals, '_idx': idxs}
        else:
            raise KeyError

    def _random_model(self):
        individual = {}
        for key in self.search_space.keys():
            self._mutate(key, individual)
        return individual

    def _mutate_model(self, model):
        new_individual = copy.deepcopy(model.parameters)
        mutate_key = random.choice(list(new_individual.keys()))
        self._mutate(mutate_key, new_individual)
        return new_individual

    def _generate_initial_population(self):
        while len(self.initial_population) < self.population_size:
            self.initial_population.append(self._random_model())
        logger.info('init population done.')
