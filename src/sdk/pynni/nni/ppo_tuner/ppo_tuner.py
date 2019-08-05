# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
ppo_tuner.py including:
    class PPOTuner
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import logging
import numpy as np
import json_tricks
from gym import spaces

import nni
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward

from .model import Model
from .util import set_global_seeds
from .policy import build_lstm_policy


logger = logging.getLogger('ppo_tuner_AutoML')

def constfn(val):
    """wrap as function"""
    def f(_):
        return val
    return f


class ModelConfig:
    """
    Configurations of the PPO model
    """
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.num_envs = 0
        self.nsteps = 0

        self.ent_coef = 0.0
        self.lr = 3e-4
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.gamma = 0.99
        self.lam = 0.95
        self.cliprange = 0.2
        self.embedding_size = None  # the embedding is for each action

        self.noptepochs = 4         # number of training epochs per update
        self.total_timesteps = 5000 # number of timesteps (i.e. number of actions taken in the environment)
        self.nminibatches = 4       # number of training minibatches per update. For recurrent policies,
                                    # should be smaller or equal than number of environments run in parallel.

class TrialsInfo:
    """
    Informations of each trial from one model inference
    """
    def __init__(self, obs, actions, values, neglogpacs, dones, last_value, inf_batch_size):
        self.iter = 0
        self.obs = obs
        self.actions = actions
        self.values = values
        self.neglogpacs = neglogpacs
        self.dones = dones
        self.last_value = last_value

        self.rewards = None
        self.returns = None

        self.inf_batch_size = inf_batch_size
        #self.states = None

    def get_next(self):
        """
        get actions of the next trial
        """
        if self.iter >= self.inf_batch_size:
            return None, None
        actions = []
        for step in self.actions:
            actions.append(step[self.iter])
        self.iter += 1
        return self.iter - 1, actions

    def update_rewards(self, rewards, returns):
        """
        after the trial is finished, reward and return of this trial is updated
        """
        self.rewards = rewards
        self.returns = returns

    def convert_shape(self):
        """
        convert shape
        """
        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        self.obs = sf01(self.obs)
        self.returns = sf01(self.returns)
        self.dones = sf01(self.dones)
        self.actions = sf01(self.actions)
        self.values = sf01(self.values)
        self.neglogpacs = sf01(self.neglogpacs)


class PPOModel:
    """
    PPO Model
    """
    def __init__(self, model_config, mask):
        self.model_config = model_config
        self.states = None    # initial state of lstm in policy/value network
        self.nupdates = None  # the number of func train is invoked, used to tune lr and cliprange
        self.cur_update = 1   # record the current update
        self.np_mask = mask   # record the mask of each action within one trial

        set_global_seeds(None)
        assert isinstance(self.model_config.lr, float)
        self.lr = constfn(self.model_config.lr)
        assert isinstance(self.model_config.cliprange, float)
        self.cliprange = constfn(self.model_config.cliprange)

        # build lstm policy network, value share the same network
        policy = build_lstm_policy(model_config)

        # Get the nb of env
        nenvs = model_config.num_envs

        # Calculate the batch_size
        self.nbatch = nbatch = nenvs * model_config.nsteps
        nbatch_train = nbatch // model_config.nminibatches
        self.nupdates = self.model_config.total_timesteps//self.nbatch

        # Instantiate the model object (that creates act_model and train_model)
        self.model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train,
                           nsteps=model_config.nsteps, ent_coef=model_config.ent_coef, vf_coef=model_config.vf_coef,
                           max_grad_norm=model_config.max_grad_norm, np_mask=self.np_mask)

        self.states = self.model.initial_state

        logger.info('=== finished PPOModel initialization')

    def inference(self, num):
        """
        generate actions along with related info from policy network.
        observation is the action of the last step.

        Parameters:
        ----------
        num:             the number of trials to generate
        """
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], []
        # initial observation
        # use the (n+1)th embedding to represent the first step action
        first_step_ob = self.model_config.action_space.n
        obs = [first_step_ob for _ in range(num)]
        dones = [True for _ in range(num)]
        states = self.states
        # For n in range number of steps
        for cur_step in range(self.model_config.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, neglogpacs = self.model.step(cur_step, obs, S=states, M=dones)
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs[:] = actions
            if cur_step == self.model_config.nsteps - 1:
                dones = [True for _ in range(num)]
            else:
                dones = [False for _ in range(num)]

        #batch of steps to batch of rollouts
        np_obs = np.asarray(obs)
        mb_obs = np.asarray(mb_obs, dtype=np_obs.dtype)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(np_obs, S=states, M=dones)

        return mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values

    def compute_rewards(self, trials_info, trials_result):
        """
        compute the rewards of the trials in trials_info based on trials_result,
        and update the rewards in trials_info

        Parameters:
        ----------
        trials_info:             info of the generated trials
        trials_result:           final results (e.g., acc) of the generated trials
        """
        mb_rewards = np.asarray([trials_result for _ in trials_info.actions], dtype=np.float32)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        last_dones = np.asarray([True for _ in trials_result], dtype=np.bool) # ugly
        for t in reversed(range(self.model_config.nsteps)):
            if t == self.model_config.nsteps - 1:
                nextnonterminal = 1.0 - last_dones
                nextvalues = trials_info.last_value
            else:
                nextnonterminal = 1.0 - trials_info.dones[t+1]
                nextvalues = trials_info.values[t+1]
            delta = mb_rewards[t] + self.model_config.gamma * nextvalues * nextnonterminal - trials_info.values[t]
            mb_advs[t] = lastgaelam = delta + self.model_config.gamma * self.model_config.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + trials_info.values

        trials_info.update_rewards(mb_rewards, mb_returns)
        trials_info.convert_shape()

    def train(self, trials_info, nenvs):
        """
        train the policy/value network using trials_info

        Parameters:
        ----------
        trials_info:             complete info of the generated trials from the previous inference
        nenvs:                   the batch size of the (previous) inference
        """
        if self.cur_update <= self.nupdates:
            frac = 1.0 - (self.cur_update - 1.0) / self.nupdates
        else:
            logger.warning('current update (self.cur_update) %d has exceeded total updates (self.nupdates) %d',
                           self.cur_update, self.nupdates)
            frac = 1.0 - (self.nupdates - 1.0) / self.nupdates
        lrnow = self.lr(frac)
        cliprangenow = self.cliprange(frac)
        self.cur_update += 1

        states = self.states

        assert states is not None # recurrent version
        assert nenvs % self.model_config.nminibatches == 0
        envsperbatch = nenvs // self.model_config.nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * self.model_config.nsteps).reshape(nenvs, self.model_config.nsteps)
        for _ in range(self.model_config.noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (trials_info.obs, trials_info.returns, trials_info.dones,
                                                      trials_info.actions, trials_info.values, trials_info.neglogpacs))
                mbstates = states[mbenvinds]
                self.model.train(lrnow, cliprangenow, *slices, mbstates)


class PPOTuner(Tuner):
    """
    PPOTuner
    """

    def __init__(self, optimize_mode, trials_per_update=20, epochs_per_update=4, minibatch_size=4):
        """
        initialization, PPO model is not initialized here as search space is not received yet.

        Parameters:
        ----------
        optimize_mode:         maximize or minimize
        trials_per_update:     number of trials to have for each model update
        epochs_per_update:     number of epochs to run for each model update
        minibatch_size:        minibatch size (number of trials) for the update
        """
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.model_config = ModelConfig()
        self.model = None
        self.search_space = None
        self.running_trials = {}                  # key: parameter_id, value: actions/states/etc.
        self.inf_batch_size = trials_per_update   # number of trials to generate in one inference
        self.first_inf = True                     # indicate whether it is the first time to inference new trials
        self.trials_result = [None for _ in range(self.inf_batch_size)] # results of finished trials

        self.credit = 0 # record the unsatisfied trial requests
        self.param_ids = []
        self.finished_trials = 0
        self.chosen_arch_template = {}

        self.actions_spaces = None
        self.actions_to_config = None
        self.full_act_space = None
        self.trials_info = None

        self.all_trials = {} # used to dedup the same trial, key: config, value: final result

        self.model_config.num_envs = self.inf_batch_size
        self.model_config.noptepochs = epochs_per_update
        self.model_config.nminibatches = minibatch_size

        self.send_trial_callback = None
        logger.info('=== finished PPOTuner initialization')

    def _process_one_nas_space(self, block_name, block_space):
        """
        process nas space to determine observation space and action space

        Parameters:
        ----------
        block_name:              the name of the mutable block
        block_space:             search space of this mutable block

        Returns:
        ----------
        actions_spaces:          list of the space of each action
        actions_to_config:       the mapping from action to generated configuration
        """
        actions_spaces = []
        actions_to_config = []

        block_arch_temp = {}
        for l_name, layer in block_space.items():
            chosen_layer_temp = {}

            if len(layer['layer_choice']) > 1:
                actions_spaces.append(layer['layer_choice'])
                actions_to_config.append((block_name, l_name, 'chosen_layer'))
                chosen_layer_temp['chosen_layer'] = None
            else:
                assert len(layer['layer_choice']) == 1
                chosen_layer_temp['chosen_layer'] = layer['layer_choice'][0]

            if layer['optional_input_size'] not in [0, 1, [0, 1]]:
                raise ValueError('Optional_input_size can only be 0, 1, or [0, 1], but the pecified one is %s'
                                 % (layer['optional_input_size']))
            if isinstance(layer['optional_input_size'], list):
                actions_spaces.append(["None", *layer['optional_inputs']])
                actions_to_config.append((block_name, l_name, 'chosen_inputs'))
                chosen_layer_temp['chosen_inputs'] = None
            elif layer['optional_input_size'] == 1:
                actions_spaces.append(layer['optional_inputs'])
                actions_to_config.append((block_name, l_name, 'chosen_inputs'))
                chosen_layer_temp['chosen_inputs'] = None
            elif layer['optional_input_size'] == 0:
                chosen_layer_temp['chosen_inputs'] = []
            else:
                raise ValueError('invalid type and value of optional_input_size')

            block_arch_temp[l_name] = chosen_layer_temp

        self.chosen_arch_template[block_name] = block_arch_temp

        return actions_spaces, actions_to_config

    def _process_nas_space(self, search_space):
        """
        process nas search space to get action/observation space
        """
        actions_spaces = []
        actions_to_config = []
        for b_name, block in search_space.items():
            if block['_type'] != 'mutable_layer':
                raise ValueError('PPOTuner only accept mutable_layer type in search space, but the current one is %s'%(block['_type']))
            block = block['_value']
            act, act_map = self._process_one_nas_space(b_name, block)
            actions_spaces.extend(act)
            actions_to_config.extend(act_map)

        # calculate observation space
        dedup = {}
        for step in actions_spaces:
            for action in step:
                dedup[action] = 1
        full_act_space = [act for act, _ in dedup.items()]
        assert len(full_act_space) == len(dedup)
        observation_space = len(full_act_space)

        nsteps = len(actions_spaces)

        return actions_spaces, actions_to_config, full_act_space, observation_space, nsteps

    def _generate_action_mask(self):
        """
        different step could have different action space. to deal with this case, we merge all the
        possible actions into one action space, and use mask to indicate available actions for each step
        """
        two_masks = []

        mask = []
        for acts in self.actions_spaces:
            one_mask = [0 for _ in range(len(self.full_act_space))]
            for act in acts:
                idx = self.full_act_space.index(act)
                one_mask[idx] = 1
            mask.append(one_mask)
        two_masks.append(mask)

        mask = []
        for acts in self.actions_spaces:
            one_mask = [-np.inf for _ in range(len(self.full_act_space))]
            for act in acts:
                idx = self.full_act_space.index(act)
                one_mask[idx] = 0
            mask.append(one_mask)
        two_masks.append(mask)

        return np.asarray(two_masks, dtype=np.float32)

    def update_search_space(self, search_space):
        """
        get search space, currently the space only includes that for NAS

        Parameters:
        ----------
        search_space:                  search space for NAS

        Returns:
        -------
        no return
        """
        logger.info('=== update search space %s', search_space)
        assert self.search_space is None
        self.search_space = search_space

        assert self.model_config.observation_space is None
        assert self.model_config.action_space is None

        self.actions_spaces, self.actions_to_config, self.full_act_space, obs_space, nsteps = self._process_nas_space(search_space)

        self.model_config.observation_space = spaces.Discrete(obs_space)
        self.model_config.action_space = spaces.Discrete(obs_space)
        self.model_config.nsteps = nsteps

        # generate mask in numpy
        mask = self._generate_action_mask()

        assert self.model is None
        self.model = PPOModel(self.model_config, mask)

    def _actions_to_config(self, actions):
        """
        given actions, to generate the corresponding trial configuration
        """
        chosen_arch = copy.deepcopy(self.chosen_arch_template)
        for cnt, act in enumerate(actions):
            act_name = self.full_act_space[act]
            (block_name, layer_name, key) = self.actions_to_config[cnt]
            if key == 'chosen_inputs':
                if act_name == 'None':
                    chosen_arch[block_name][layer_name][key] = []
                else:
                    chosen_arch[block_name][layer_name][key] = [act_name]
            elif key == 'chosen_layer':
                chosen_arch[block_name][layer_name][key] = act_name
            else:
                raise ValueError('unrecognized key: {0}'.format(key))
        return chosen_arch

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """
        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.
        """
        result = []
        self.send_trial_callback = kwargs['st_callback']
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                logger.debug("generating param for %s", parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
        return result

    def generate_parameters(self, parameter_id, **kwargs):
        """
        generate parameters, if no trial configration for now, self.credit plus 1 to send the config later
        """
        if self.first_inf:
            self.trials_result = [None for _ in range(self.inf_batch_size)]
            mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values = self.model.inference(self.inf_batch_size)
            self.trials_info = TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs,
                                          mb_dones, last_values, self.inf_batch_size)
            self.first_inf = False

        trial_info_idx, actions = self.trials_info.get_next()
        if trial_info_idx is None:
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')

        self.running_trials[parameter_id] = trial_info_idx
        new_config = self._actions_to_config(actions)
        return new_config

    def _next_round_inference(self):
        """
        """
        self.finished_trials = 0
        self.model.compute_rewards(self.trials_info, self.trials_result)
        self.model.train(self.trials_info, self.inf_batch_size)
        self.running_trials = {}
        # generate new trials
        self.trials_result = [None for _ in range(self.inf_batch_size)]
        mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values = self.model.inference(self.inf_batch_size)
        self.trials_info = TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs,
                                        mb_dones, last_values, self.inf_batch_size)
        # check credit and submit new trials
        for _ in range(self.credit):
            trial_info_idx, actions = self.trials_info.get_next()
            if trial_info_idx is None:
                logger.warning('No enough trial config, trials_per_update is suggested to be larger than trialConcurrency')
                break
            assert self.param_ids
            param_id = self.param_ids.pop()
            self.running_trials[param_id] = trial_info_idx
            new_config = self._actions_to_config(actions)
            self.send_trial_callback(param_id, new_config)
            self.credit -= 1

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        receive trial's result. if the number of finished trials equals self.inf_batch_size, start the next update to
        train the model
        """
        trial_info_idx = self.running_trials.pop(parameter_id, None)
        assert trial_info_idx is not None

        value = extract_scalar_reward(value)
        if self.optimize_mode == OptimizeMode.Minimize:
            value = -value

        self.trials_result[trial_info_idx] = value
        self.finished_trials += 1

        if self.finished_trials == self.inf_batch_size:
            self._next_round_inference()

    def trial_end(self, parameter_id, success, **kwargs):
        """
        to deal with trial failure
        """
        if not success:
            if parameter_id not in self.running_trials:
                logger.warning('The trial is failed, but self.running_trial does not have this trial')
                return
            trial_info_idx = self.running_trials.pop(parameter_id, None)
            assert trial_info_idx is not None
            # use mean of finished trials as the result of this failed trial
            values = [val for val in self.trials_result if val is not None]
            self.trials_result[trial_info_idx] = sum(values) / len(values) if len(values) > 0 else 0
            self.finished_trials += 1
            if self.finished_trials == self.inf_batch_size:
                self._next_round_inference()

    def import_data(self, data):
        """
        Import additional data for tuning

        Parameters
        ----------
        data:               a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        logger.warning('PPOTuner cannot leverage imported data.')
