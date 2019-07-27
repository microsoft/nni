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
'''
ppo_tuner.py including:
    class PPOTuner
'''

import numpy as np
import logging
import random

import nni
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward
from nni.protocol import CommandType, send

from gym import spaces

from .model import Model
from .util import set_global_seeds
from .policy import build_lstm_policy


logger = logging.getLogger('ppo_tuner_AutoML')

def constfn(val):
    def f(_):
        return val
    return f


class ModelConfig(object):
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
        
        self.nminibatches = 4 # number of training minibatches per update. For recurrent policies,
                              # should be smaller or equal than number of environments run in parallel.
        self.noptepochs = 4 # number of training epochs per update
        self.total_timesteps = 5000 # number of timesteps (i.e. number of actions taken in the environment)

        self.embedding_size = None # the embedding is for each action

class TrialsInfo(object):
    def __init__(self, obs, actions, values, neglogpacs, dones, last_value):
        self.iter = 0
        self.obs = obs
        self.actions = actions
        self.values = values
        self.neglogpacs = neglogpacs
        self.dones = dones
        self.last_value = last_value

        self.rewards = None
        self.returns = None

        #self.states = None

    def get_next(self):
        if self.iter >= self.actions.size:
            return None, None
        actions = []
        for step in self.actions:
            actions.append(step[self.iter])
        self.iter += 1
        return self.iter - 1, actions

    def update_rewards(self, rewards, returns):
        self.rewards = rewards
        self.returns = returns

    def convert_shape(self):
        # obs, returns, masks, actions, values, neglogpacs, states = runner.run()
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


class PPOModel(object):
    """
    """
    def __init__(self, model_config, mask):
        self.model_config = model_config
        self.states = None
        self.nupdates = None
        self.cur_update = 1
        self.np_mask = mask

        set_global_seeds(None)
        assert isinstance(self.model_config.lr, float)
        self.lr = constfn(self.model_config.lr)
        assert isinstance(self.model_config.cliprange, float)
        self.cliprange = constfn(self.model_config.cliprange)

        policy = build_lstm_policy(model_config)

        # Get the nb of env
        nenvs = model_config.num_envs

        # Get state_space and action_space
        ob_space = model_config.observation_space
        ac_space = model_config.action_space

        print('zql: ob_space shape', ob_space, ac_space)

        # Calculate the batch_size
        self.nbatch = nbatch = nenvs * model_config.nsteps
        nbatch_train = nbatch // model_config.nminibatches
        self.nupdates = self.model_config.total_timesteps//self.nbatch

        # Instantiate the model object (that creates act_model and train_model)
        self.model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=model_config.nsteps, ent_coef=model_config.ent_coef, vf_coef=model_config.vf_coef,
                        max_grad_norm=model_config.max_grad_norm, np_mask=self.np_mask)

        self.states = self.model.initial_state

        logger.info('zql: finished PPOModel initialization')

    def inference(self, num):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[]
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
            #print('=============zql: obs, type', self.obs.shape, type(self.obs), type(self.obs[0]))
            actions, values, states, neglogpacs = self.model.step(cur_step, obs, S=states, M=dones)
            #print('=============zql: actions, type', actions.shape, type(actions), type(actions[0]), actions)
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            #obs[:], rewards, self.dones, infos = self.env.step(actions)
            obs[:] = actions
            #rewards
            if cur_step == self.model_config.nsteps - 1:
                dones = [True for _ in range(num)]
            else:
                dones = [False for _ in range(num)]
            #for info in infos:
            #    maybeepinfo = info.get('episode')
            #    if maybeepinfo: epinfos.append(maybeepinfo)
            #mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        np_obs = np.asarray(obs)
        mb_obs = np.asarray(mb_obs, dtype=np_obs.dtype)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(np_obs, S=states, M=dones)

        return mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values#, states

    def compute_rewards(self, trials_info, trials_result):
        """
        """
        mb_rewards = np.asarray([trials_result for _ in trials_info.actions], dtype=np.float32)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        last_dones = np.asarray([True for _ in trials_result], dtype=np.bool) # ugly
        epinfos = []
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
        """
        assert self.cur_update <= self.nupdates # TODO:
        frac = 1.0 - (self.cur_update - 1.0) / self.nupdates
        lrnow = self.lr(frac)
        cliprangenow = self.cliprange(frac)
        self.cur_update += 1

        states = self.states

        mblossvals = []
        assert states is not None # recurrent version
        print('zql: recurrent version')
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
                slices = (arr[mbflatinds] for arr in (trials_info.obs, trials_info.returns, trials_info.dones, trials_info.actions, trials_info.values, trials_info.neglogpacs))
                mbstates = states[mbenvinds]
                mblossvals.append(self.model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        print('zql: lossvals: ', lossvals)


class PPOTuner(Tuner):
    """
    PPOTuner
    """

    def __init__(self, optimize_mode):
        self.model_config = ModelConfig()
        self.model = None
        self.search_space = None
        self.running_trials = {} # key: parameter_id, value: actions/states/etc.
        #self.ntrials_to_train = 20 # the number of new trials are finished, then start the next training
        #self.finished_trials = [] # finished trials after the previous training
        self.inf_batch_size = 20 # number of trials to generate in one inference
        self.first_inf = True # indicate whether it is the first time to inference new trials
        self.trials_result = [None for _ in range(self.inf_batch_size)] # results of finished trials

        self.credit = 0 # record the unsatisfied trial requests
        self.param_ids = []
        self.finished_trials = 0
        self.chosen_arch_template = {}

        self.all_trials = {} # used to dedup the same trial, key: config, value: final result

        self.model_config.num_envs = self.inf_batch_size
        logger.info('zql: finished PPOTuner initialization')

    def _process_one_nas_space(self, block_name, block_space):
        """
        process nas space to determine observation space and action space

        Parameters:
        ----------
        block_name: the name of the mutable block
        block_space: search space of this mutable block

        Returns:
        ----------
        """
        actions_spaces = []
        actions_to_config = []

        # TODO: currently optional_input_size is not allowed than 1
        block_arch_temp = {}
        for l_name, layer in block_space.items():
            chosen_layer_temp = {}

            if len(layer['layer_choice']) > 1:
                actions_spaces.append(layer['layer_choice'])
                actions_to_config.append((block_name, l_name, 'layer_choice'))
                chosen_layer_temp['layer_choice'] = None
            else:
                assert len(layer['layer_choice']) == 1
                chosen_layer_temp['layer_choice'] = layer['layer_choice'][0]

            assert layer['optional_input_size'] in [0, 1, [0, 1]]
            if isinstance(layer['optional_input_size'], list):
                actions_spaces.append(["None", *layer['optional_inputs']])
                actions_to_config.append((block_name, l_name, 'optional_inputs'))
                chosen_layer_temp['chosen_inputs'] = None
            elif layer['optional_input_size'] == 1:
                actions_spaces.append(layer['optional_inputs'])
                actions_to_config.append((block_name, l_name, 'optional_inputs'))
                chosen_layer_temp['chosen_inputs'] = None
            elif layer['optional_input_size'] == 0:
                chosen_layer_temp['chosen_inputs'] = []
            else:
                raise ValueError('invalid type and value of optional_input_size')

            block_arch_temp[l_name] = chosen_layer_temp

        self.chosen_arch_template[block_name] = block_arch_temp

        return actions_spaces, actions_to_config

    def _process_nas_space(self, search_space):
        # TODO: 
        actions_spaces = []
        actions_to_config = []
        for b_name, block in search_space.items():
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
        mask = []
        for acts in self.actions_spaces:
            one_mask = [0 for _ in range(len(self.full_act_space))]
            for act in acts:
                idx = self.full_act_space.index(act)
                one_mask[idx] = 1
            mask.append(one_mask)
        return np.asarray(mask, dtype=np.float32)

    def update_search_space(self, search_space):
        """
        get search space, currently the space only includes that for NAS

        Parameters:
        ----------
        search_space: json object                 search space for NAS
        
        Returns:
        -------
        no return
        """
        logger.info('zql: update search space %s', search_space)
        assert self.search_space is None
        self.search_space = search_space
        # TODO: determine observation/action space based on search_space
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

    def _trial_dedup(self):
        # TODO:
        pass

    def _actions_to_config(self, actions):
        new_config = {}
        for b_name, block in self.search_space.items():
            new_config[b_name] = {}
            for l_name, layer in block.items():
                chosen_layer = {}
                new_config[b_name][l_name] = chosen_layer
                if l_name == 'mutable_layer_0':
                    chosen_layer['chosen_layer'] = layer['layer_choice'][actions[0]]
                    chosen_layer['chosen_inputs'] = layer['optional_inputs']
                elif l_name == 'mutable_layer_1':
                    chosen_layer['chosen_layer'] = layer['layer_choice'][0]
                    chosen_layer['chosen_inputs'] = layer['optional_inputs']
                elif l_name == 'mutable_layer_2':
                    chosen_layer['chosen_layer'] = layer['layer_choice'][actions[1]]
                    chosen_layer['chosen_inputs'] = layer['optional_inputs']
                elif l_name == 'mutable_layer_3':
                    chosen_layer['chosen_layer'] = layer['layer_choice'][actions[2]]
                    chosen_layer['chosen_inputs'] = layer['optional_inputs']
                elif l_name == 'mutable_layer_4':
                    chosen_layer['chosen_layer'] = layer['layer_choice'][0]
                    chosen_layer['chosen_inputs'] = layer['optional_inputs']
                elif l_name == 'mutable_layer_5':
                    chosen_layer['chosen_layer'] = layer['layer_choice'][actions[3]]
                    chosen_layer['chosen_inputs'] = layer['optional_inputs'][1:]
        return new_config

    def generate_parameters(self, parameter_id, **kwargs):
        """
        """
        if self.first_inf:
            self.trials_result = [None for _ in range(self.inf_batch_size)]
            mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values = self.model.inference(self.inf_batch_size)
            self.trials_info = TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values)
            self.first_inf = False

        trial_info_idx, actions = self.trials_info.get_next()
        if trial_info_idx is None:
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')
        else:
            self.running_trials[parameter_id] = trial_info_idx
            new_config = self._actions_to_config(actions)
            return new_config

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        """
        def _pack_parameter(parameter_id, params, customized=False, trial_job_id=None, parameter_index=None):
            _trial_params[parameter_id] = params
            ret = {
                'parameter_id': parameter_id,
                'parameter_source': 'customized' if customized else 'algorithm',
                'parameters': params
            }
            if trial_job_id is not None:
                ret['trial_job_id'] = trial_job_id
            if parameter_index is not None:
                ret['parameter_index'] = parameter_index
            else:
                ret['parameter_index'] = 0
            return json_tricks.dumps(ret)

        trial_info_idx = self.running_trials.pop(parameter_id, None)
        assert trial_info_idx is not None

        value = extract_scalar_reward(value)
        #if self.optimize_mode == OptimizeMode.Minimize:
        #    value = -value

        self.trials_result[trial_info_idx] = value
        #self.finished_trials.append(trial_info_idx)
        self.finished_trials += 1

        if self.finished_trials == self.inf_batch_size:
            self.finished_trials = 0
            self.model.compute_rewards(self.trials_info, self.trials_result)
            self.model.train(self.trials_info, self.inf_batch_size)
            #self.finished_trials = []
            self.running_trials = {}
            # generate new trials
            self.trials_result = [None for _ in range(self.inf_batch_size)]
            mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values = self.model.inference(self.inf_batch_size)
            self.trials_info = TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values)
            # check credit and submit new trials
            for i in range(self.credit):
                trial_info_idx, actions = self.trial_info.get_next()
                assert trial_info_idx is not None
                assert self.param_ids
                param_id = self.param_ids.pop()
                new_config = self._actions_to_config(actions)
                send(CommandType.NewTrialJob, _pack_parameter(param_id, new_config))

    def import_data(self, data):
        """
        Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        pass
