import logging
import time

from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner

from .. import Sampler, submit_models, wait_models, query_available_resources, is_stopped_exec
from .strategy import BaseStrategy

_logger = logging.getLogger(__name__)


class TPESampler(Sampler):
    def __init__(self, optimize_mode='minimize'):
        self.tpe_tuner = HyperoptTuner('tpe', optimize_mode)
        self.cur_sample = None
        self.index = None
        self.total_parameters = {}

    def update_sample_space(self, sample_space):
        search_space = {}
        for i, each in enumerate(sample_space):
            search_space[str(i)] = {'_type': 'choice', '_value': each}
        self.tpe_tuner.update_search_space(search_space)

    def generate_samples(self, model_id):
        self.cur_sample = self.tpe_tuner.generate_parameters(model_id)
        self.total_parameters[model_id] = self.cur_sample
        self.index = 0

    def receive_result(self, model_id, result):
        self.tpe_tuner.receive_trial_result(model_id, self.total_parameters[model_id], result)

    def choice(self, candidates, mutator, model, index):
        chosen = self.cur_sample[str(self.index)]
        self.index += 1
        return chosen


class TPEStrategy(BaseStrategy):
    def __init__(self):
        self.tpe_sampler = TPESampler()
        self.model_id = 0
        self.running_models = {}

    def run(self, base_model, applied_mutators):
        sample_space = []
        new_model = base_model
        for mutator in applied_mutators:
            recorded_candidates, new_model = mutator.dry_run(new_model)
            sample_space.extend(recorded_candidates)
        self.tpe_sampler.update_sample_space(sample_space)

        _logger.info('stargety start...')
        while True:
            avail_resource = query_available_resources()
            if avail_resource > 0:
                model = base_model
                _logger.info('apply mutators...')
                _logger.info('mutators: %s', str(applied_mutators))
                self.tpe_sampler.generate_samples(self.model_id)
                for mutator in applied_mutators:
                    _logger.info('mutate model...')
                    mutator.bind_sampler(self.tpe_sampler)
                    model = mutator.apply(model)
                # run models
                submit_models(model)
                self.running_models[self.model_id] = model
                self.model_id += 1
                _logger.info('Strategy says: %s', model.metric)
            else:
                time.sleep(2)

            _logger.warning('num of running models: %d', len(self.running_models))
            to_be_deleted = []
            for _id, _model in self.running_models.items():
                if is_stopped_exec(_model):
                    if _model.metric is not None:
                        self.tpe_sampler.receive_result(_id, _model.metric)
                        _logger.warning('tpe receive results: %d, %s', _id, _model.metric)
                    to_be_deleted.append(_id)
            for _id in to_be_deleted:
                del self.running_models[_id]
