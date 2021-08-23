import math
import random
import logging
import copy

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np

import nni
from nni.tuner import Tuner


class Factor(object):
    """factor type parameter
    """
    def __init__(self, value):
        self.product, self.num = value
        self.partition = [1] * self.num
        self.partition[0] = self.product

    def pick_out(self):
        return self.partition

    def step(self, action):
        if self.partition[action[0]] % action[2] == 0:
            self.partition[action[0]] /= action[2]
            self.partition[action[1]] *= action[2]
            status = True
        else:
            status = False

        return status

    def get_actions(self):
        actions = []
        prime_factors = self._get_prime_factors(self.product, False)
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    for k in range(len(prime_factors)):
                        action = [i]
                        action.append(j)
                        action.append(prime_factors[k])
                        actions.append(action)

        return actions

    def __repr__(self):
        return self.partition.__repr__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

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


class Configuration(object):
    """Configuration class
    """
    def __init__(self, search_space):
        self.params = {}
        self.key_order = []
        for key in search_space.keys():
            if search_space[key]['_type'] == 'factor':
                self.key_order.append(key)
                self.params[key] = \
                    Factor(search_space[key]['_value'])
            else:
                raise RuntimeError(
                    "N_A2C Tuner doesn't support this kind of parameter: "
                    + str(search_space[key]['_type'])
                )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __repr__(self):
        string = ""
        for key, value in self.params.items():
            string += key + ': ' + value.__repr__() + ' '

        return string

    def pick_out(self):
        output = {}
        for key in self.params.keys():
            output[key] = self.params[key].pick_out()

        return output

    def step(self, action):
        config = copy.deepcopy(self)
        status = config.params[action[0]].step(action[1])

        return status, config

    def get_actions(self):
        actions = []
        for key, value in self.params.items():
            subactions = value.get_actions()
            for subaction in subactions:
                action = [key]
                action.append(subaction)
                actions.append(action)

        return actions

    def to_torch(self):
        states = []
        for key in self.key_order:
            state = torch.tensor(self.params[key].partition).float() / \
                self.params[key].product - 0.5
            states.append(state)

        return torch.cat(states).float()


class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.fc = nn.Linear(num_states, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.fc(state))
        value = self.critic_linear2(x)
        policy_dist = F.softmax(self.actor_linear2(x))

        return value, policy_dist


class Population(object):
    """Population class
    """
    def __init__(self, search_space, opt_mode, n_states, n_steps,
                 hidden_size, lr):
        self.search_space = search_space
        self.opt_mode = opt_mode
        self.n_states = n_states
        self.n_steps = n_steps
        self.hidden_size = hidden_size
        self.lr = lr

        self.config = Configuration(search_space)
        self.max_reward = 0.0

        self.action_space = self.config.get_actions()
        self.dim_actions = len(self.action_space)
        self.dim_states = len(self.config.to_torch())
        self.log_probs = []
        self.values = []
        self.rewards = []

        self.population = []

        self.actor_critic = ActorCritic(
            self.dim_states, self.dim_actions, self.hidden_size
        )
        self.ac_optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.lr
        )

    def append(self, individual, fitness):
        if self.opt_mode == "minimize":
            fitness = -1 * fitness

        self.population.append(individual)

        if self.max_reward < fitness:
            self.max_reward = fitness
            self.config = individual

        if self.collect:
            idx = self.collect.index(individual)
            self.waiting_rewards[idx] = fitness
            del self.collect[idx]
        else:
            raise RuntimeError("Received unexpected trials.")

        if not self.collect:
            self.rewards.extend(self.waiting_rewards)

            self.ac_optimizer.zero_grad()
            gradient_loss = 0
            value_loss = 0
            for i in range(len(self.values)):
                advantage = self.rewards[i] - self.values[i]
                gradient_loss += self.log_probs[i] * advantage
                value_loss += torch.pow(advantage, 2)
            loss = gradient_loss + value_loss
            loss.backward()
            self.ac_optimizer.step()

            self.rewards = []
            self.values = []
            self.log_probs = []
            self.collect = []

    def generate(self):
        self.collect = []
        while len(self.collect) < self.n_states:
            config = self.config
            for i in range(self.n_steps):
                value, policy_dist = self.actor_critic(config.to_torch())
                dist = policy_dist.detach().numpy()

                if random.uniform(0, 1) < 0.1:
                    action = random.choice(range(self.dim_actions))
                else:
                    action = np.random.choice(
                        self.dim_actions, p=np.squeeze(dist))

                log_prob = torch.log(policy_dist.squeeze(0)[action])
                # entropy = -np.sum(np.mean(dist) * np.log(dist))
                flag, new_config = config.step(self.action_space[action])

                if (flag and new_config not in self.population
                        and new_config not in self.collect):
                    self.collect.append(new_config)
                    self.log_probs.append(log_prob)
                    self.values.append(value)

                config = new_config
        # print([math.exp(float(i)) for i in self.log_probs])

        self.waiting_rewards = [0.0] * len(self.collect)
        return copy.deepcopy(self.collect)


class N_A2C(Tuner):
    """N-A2C Tuner
    Based on paper Compiler-Level Matrix Multiplication Optimization for Deep Learning

    Parameters
    ----------
    optimize_mode: str, 'maximize' or 'minimize'
    n_states: int,
        The maximum search steps Tau
    n_steps: int
        number of steps to train the policy and critic networks each iteration
    hidden_size: int,
        number of hidden size of the policy and critic networks
    lr: float,
        learning rate of the policy and critic networks
    """

    def __init__(self,
                 optimize_mode="maximize",
                 n_states=6,
                 n_steps=3,
                 hidden_size=128,
                 lr=1e-3):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel('DEBUG')

        self.opt_mode = optimize_mode
        self.n_states = n_states
        self.n_steps = n_steps
        self.hidden_size = 128
        self.lr = lr

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

        self.population = \
            Population(
                search_space,
                self.opt_mode,
                self.n_states,
                self.n_steps,
                self.hidden_size,
                self.lr
            )

        if not self.serve_list:
            self.serve_list = self.population.generate()

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

        if not self.serve_list and not self.wait_dict:
            self.serve_list = self.population.generate()
            if not self.serve_list:
                raise RuntimeError("Tuner stopped since no candidates")

        while self.request_list and self.serve_list:
            param_id = self.request_list[0]
            self.wait_dict[param_id] = self.serve_list.pop()
            self.send_trial_callback(
                param_id, self.wait_dict[param_id].pick_out())
            self.request_list.pop(0)

        # print('request_list: ' + str(len(self.request_list)))
        # print('serve_list: ' + str(len(self.serve_list)))
        # print('wait_dict: ' + str(len(self.wait_dict.keys())))

    def trial_end(self, parameter_id, success, **kwargs):
        """Method invoked when a trial is completed or terminated.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if not success:
            self.population.append(self.wait_dict[parameter_id], 0.0)
            del self.wait_dict[parameter_id]

            if not self.serve_list and not self.wait_dict:
                self.serve_list = self.population.generate()
                if not self.serve_list:
                    raise RuntimeError("Tuner stopped since no candidates")

            while self.request_list and self.serve_list:
                param_id = self.request_list[0]
                self.wait_dict[param_id] = self.serve_list.pop()
                self.send_trial_callback(
                    param_id, self.wait_dict[param_id].pick_out())
                self.request_list.pop(0)

            # print('trial_end request_list: ' + str(len(self.request_list)))
            # print('trial_end serve_list: ' + str(len(self.serve_list)))
            # print('trial_end wait_dict: ' + str(len(self.wait_dict.keys())))
