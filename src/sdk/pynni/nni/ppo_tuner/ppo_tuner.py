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

from model import Model


logger = logging.getLogger('ppo_tuner_AutoML')

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
        
        self.nminibatches = 4
        self.noptepochs = 4 # number of training epochs per update

    def update(self):
        pass

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

        self.states = None

    def get_next(self):
        if self.iter >= self.actions.size():
            return None, None
        self.iter += 1
        return self.iter - 1, self.actions[self.iter - 1]

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
        pass


class PPOModel(object):
    """
    """
    def __init__(self, model_config):
        self.model_config = model_config
        self.states = None

        set_global_seeds(seed)
        assert isinstance(lr, float)
        self.lr = constfn(lr)
        assert isinstance(cliprange, float)
        self.cliprange = constfn(cliprange)

        policy = build_lstm_policy(model_config)

        # Get the nb of env
        nenvs = model_config.num_envs

        # Get state_space and action_space
        ob_space = model_config.observation_space
        ac_space = model_config.action_space

        print('zql: ob_space shape', ob_space, ac_space)

        # Calculate the batch_size
        nbatch = nenvs * model_config.nsteps
        nbatch_train = nbatch // model_config.nminibatches

        # Instantiate the model object (that creates act_model and train_model)
        self.model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=model_config.nsteps, ent_coef=model_config.ent_coef, vf_coef=model_config.vf_coef,
                        max_grad_norm=model_config.max_grad_norm)

        self.states = self.model.initial_state

    def inference(self, num):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        # initial observation
        # TODO: ====================
        obs = [random.randint(1, 6) for _ in range(num)]
        dones = [True for _ in range(num)]
        states = 
        # For n in range number of steps
        for cur_step in range(self.model_config.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            #print('=============zql: obs, type', self.obs.shape, type(self.obs), type(self.obs[0]))
            actions, values, states, neglogpacs = self.model.step(obs, S=states, M=dones)
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
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=dones)

        return mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values#, states

    def compute_rewards(self, trials_info, trials_result):
        """
        """
        mb_rewards = np.asarray([trials_result for _ in trials_info.actions], dtype=np.float32)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        last_dones = [True for _ in trials_result] # ugly
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

        return 
        #(*map(sf01, (obs, mb_returns, dones, actions, values, neglogpacs)), mb_states, epinfos)

    def train(self, trials_info, nenvs):
        """
        """
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
                slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mbstates = states[mbenvinds]
                mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)


class PPOTuner(Tuner):
    """
    PPOTuner
    """

    def __init__(self):
        self.model_config = ModelConfig()
        self.model = None
        self.running_trials = {} # key: parameter_id, value: actions/states/etc.
        #self.ntrials_to_train = 20 # the number of new trials are finished, then start the next training
        #self.finished_trials = [] # finished trials after the previous training
        self.inf_batch_size = 20 # number of trials to generate in one inference
        self.first_inf = True # indicate whether it is the first time to inference new trials
        self.trials_result = [None for _ in range(self.inf_batch_size)] # results of finished trials
        self.credit = 0 # record the unsatisfied trial requests

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
        self.model_config.update()
        assert self.model is None
        self.model = PPOModel(self.model_config)

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
            raise nni.NoMoreTrialError('no more parameters now.')
        else:
            self.running_trials[parameter_id] = trial_info_idx
            XXX = 
            return XXX

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
        assert trial_info is not None

        value = extract_scalar_reward(value)
        #if self.optimize_mode == OptimizeMode.Minimize:
        #    value = -value

        self.trials_result[trial_info_idx] = value
        #self.finished_trials.append(trial_info_idx)

        if not self.running_trials:
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
                XXX = 
                send(CommandType.NewTrialJob, _pack_parameter("ids[i]", "params_list[i]"))

    def import_data(self, data):
        """
        Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        pass
