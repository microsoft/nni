import random
import logging
import sdk
import numpy as np
from gym import spaces

from . import ppo

_logger = logging.getLogger(__name__)

def process_search_space(search_space: 'List[List]'):
    actions_spaces = search_space

    # calculate observation space
    dedup = {}
    for step in actions_spaces:
        for action in step:
            dedup[action] = 1
    full_act_space = [act for act, _ in dedup.items()]
    assert len(full_act_space) == len(dedup)
    observation_space = len(full_act_space)
    nsteps = len(actions_spaces)

    return actions_spaces, full_act_space, observation_space, nsteps

def generate_action_mask(actions_spaces, full_act_space):
        """
        Different step could have different action space. to deal with this case, we merge all the
        possible actions into one action space, and use mask to indicate available actions for each step
        """
        two_masks = []

        mask = []
        for acts in actions_spaces:
            one_mask = [0 for _ in range(len(full_act_space))]
            for act in acts:
                idx = full_act_space.index(act)
                one_mask[idx] = 1
            mask.append(one_mask)
        two_masks.append(mask)

        mask = []
        for acts in actions_spaces:
            one_mask = [-np.inf for _ in range(len(full_act_space))]
            for act in acts:
                idx = full_act_space.index(act)
                one_mask[idx] = 0
            mask.append(one_mask)
        two_masks.append(mask)

        return np.asarray(two_masks, dtype=np.float32)

def actions_to_config(actions, full_act_space):
    """
    Given actions, to generate the corresponding trial configuration
    """
    '''chosen_arch = copy.deepcopy(self.chosen_arch_template)
    for cnt, act in enumerate(actions):
        act_name = full_act_space[act]
        (_key, _type) = self.actions_to_config[cnt]
        if _type == 'input_choice':
            if act_name == 'None':
                chosen_arch[_key] = {'_value': [], '_idx': []}
            else:
                candidates = self.search_space[_key]['_value']['candidates']
                idx = candidates.index(act_name)
                chosen_arch[_key] = {'_value': [act_name], '_idx': [idx]}
        elif _type == 'layer_choice':
            idx = self.search_space[_key]['_value'].index(act_name)
            chosen_arch[_key] = {'_value': act_name, '_idx': idx}
        else:
            raise ValueError('unrecognized key: {0}'.format(_type))'''
    return [ full_act_space[act] for act in actions ]

def main():
    """
    this one can be seen as a scheduler in strategy, and there is another component which decides the choices.
    for simplicity, it only awares whether there is still available resource, but does not care the amount of available resource.
    the amount of available resource is handled by JIT engine (i.e., NasAdvisor)
    """
    try:
        _logger.info('start main')
        base_graph = sdk.experiment.base_model
        _logger.info('get graph object, start mutator class')
        mutators = sdk.experiment.mutators

        # dry run
        search_space = []
        new_graph = base_graph.duplicate()
        for mutator in mutators:
            new_graph, recorded_candidates = mutator.dry_run(new_graph)
            search_space.extend(recorded_candidates)

        actions_spaces, full_act_space, obs_space, nsteps = process_search_space(search_space)

        inf_batch_size = 20
        model_config = ppo.ModelConfig()
        model_config.num_envs = inf_batch_size
        model_config.noptepochs = 4
        model_config.nminibatches = 4
        model_config.observation_space = spaces.Discrete(obs_space)
        model_config.action_space = spaces.Discrete(obs_space)
        model_config.nsteps = nsteps

        # generate mask in numpy
        mask = generate_action_mask(actions_spaces, full_act_space)
        model = ppo.PPOModel(model_config, mask)

        while True:
            trials_result = [None for _ in range(inf_batch_size)]
            mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values = model.inference(num=inf_batch_size)
            trials_info = ppo.TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs,
                                            mb_dones, last_values, inf_batch_size)
            # trial_info_idx is None means no more trial in trials_info
            trial_info_idx, actions = trials_info.get_next()
            while trial_info_idx is not None:
                _logger.info('actions: {}'.format(actions))
                config = actions_to_config(actions, full_act_space)
                _logger.info('config: {}'.format(config))
                d_sampler = DeterministicSampler(config)
                new_graph = base_graph.duplicate()
                for mutator in mutators:
                    # Note: must use returned graph
                    new_graph = mutator.apply(new_graph, d_sampler)
                sdk.train_graph(new_graph)
                value = new_graph.metrics
                _logger.info('graph metrics: {}'.format(value))
                trials_result[trial_info_idx] = value

                trial_info_idx, actions = trials_info.get_next()

            model.compute_rewards(trials_info, trials_result)
            model.train(trials_info, inf_batch_size)
    except Exception as e:
        # has to catch error here, because main thread cannot show the error in this thread
        _logger.error(logging.exception('message'))

class DeterministicSampler(sdk.strategy.Sampler):
    def __init__(self, actions):
        self.actions = actions
        self.idx = 0

    def choice(self, candidates):
        act = self.actions[self.idx]
        self.idx += 1
        return act
