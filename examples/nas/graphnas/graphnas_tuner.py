import copy
import logging
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import nni
import pickle
import os
from nni.tuner import Tuner
from collections import OrderedDict

class GraphNASController(nn.Module):
    def __init__(self, search_space,
                       arc_save_path,
                       hidden=100,
                       softmax_temp=5.0,
                       tanh_c=2.5,
                       ema_decay=0.95,
                       ctrl_lr=3.5e-4,
                       ctrl_grad_clip=0,
                       max_step=100,
                       entropy_coeff=1e-4,
                       num_layers=2,
                       max_epoch=10):
        super(GraphNASController, self).__init__()
        self.tanh_c = tanh_c
        self.search_space = search_space
        self.arc_save_path = arc_save_path
        self.hidden = hidden
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.softmax_temp = softmax_temp
        self.ema_decay = ema_decay
        self.ctrl_lr = ctrl_lr
        self.ctrl_grad_clip = ctrl_grad_clip
        self.max_step = max_step
        self.entropy_coeff = entropy_coeff
        self.num_layers = num_layers
        self.max_epoch = max_epoch
        self.epoch = 0

        self.step = 0
        self._parse_search_space(search_space)
        self.action_list = []
        for i in range(num_layers):
            for key in self.search_space.keys():
                self.action_list.append(key)

        
        self.history = []
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.encoder = nn.Embedding(self.num_total_tokens, hidden)

        self.ctrl_optim = torch.optim.Adam(self.parameters(), lr=self.ctrl_lr)

    def _parse_search_space(self, search_space):
        assert isinstance(search_space, dict)
        assert len(set(search_space.keys())) == len(search_space.keys())

        self.decoders = nn.ModuleDict()
        self.num_total_tokens = 0
        self.key2tokens = OrderedDict()

        for key, val in search_space.items():
            assert isinstance(val, dict)
            assert "_value" in val

            choices = val['_value']
            assert isinstance(choices, list) and len(choices) > 0

            self.decoders[key] = nn.Linear(self.hidden, len(choices))
            self.num_total_tokens += len(choices)
            self.key2tokens[key] = len(choices)

    def _parse_arcs(self, chosen_actions):
        arcs = []

        for actions in chosen_actions:
            arc = []
            for idx, key in zip(actions, self.action_list):
                idx = idx.item()
                arc.append(self.search_space[key]["_value"][idx])
            arcs.append(arc)

        return arcs

    def forward(self, x, hidden, decoder):
        h, c = self.lstm(x, hidden)
        out = decoder(h)
        out = out / self.softmax_temp
        out = self.tanh_c * torch.tanh(out)

        return out, (h, c)

    def sample(self, num_samples):
        x = torch.zeros([num_samples, self.hidden]).to(self.device)
        hidden = (torch.zeros([num_samples, self.hidden]).to(self.device), torch.zeros([num_samples, self.hidden]).to(self.device))

        entropys = []
        chosen_log_probs = []
        chosen_actions = []
        for key in self.action_list:
            decoder = self.decoders[key]
            out, hidden = self.forward(x, hidden, decoder)

            prob = F.softmax(out, dim=-1)
            log_prob = F.log_softmax(out, dim=-1)

            entropy = -(log_prob * prob).sum(1, keepdim=False)
            chosen_action = prob.multinomial(num_samples=1).data
            chosen_log_prob = log_prob.gather(1, chosen_action)

            entropys.append(entropy)
            chosen_actions.append(chosen_action[:, 0])
            chosen_log_probs.append(chosen_log_prob[:, 0])

            x = torch.tensor(chosen_action[:, 0])
            for k, v in self.key2tokens.items():
                if k != key:
                    x += v
                else:
                    break
            x = self.encoder(x)
        arcs = self._parse_arcs(torch.stack(chosen_actions).transpose(0, 1))

        self.entropys = torch.cat(entropys)
        self.chosen_log_probs = torch.cat(chosen_log_probs)
        return arcs

    def train_one_step(self, reward):
        if self.step == 0:
            self.baseline = None

        self.train()
        if reward is None:
            return

        assert self.entropys is not None and self.chosen_log_probs is not None
        reward = reward + self.entropy_coeff * self.entropys.data.cpu().numpy()

        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = self.ema_decay * self.baseline + (1 - self.ema_decay) * reward

        adv = reward - self.baseline
        self.history.append(adv)
        adv = self._scale(adv, scale_value=0.5)
        adv = torch.tensor(adv).to(self.device)

        loss = -self.chosen_log_probs * adv
        loss = loss.sum()

        self.ctrl_optim.zero_grad()
        loss.backward()

        if self.ctrl_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.ctrl_grad_clip)
        
        self.step += 1
        
        self.entropys = None
        self.chosen_log_probs = None

        if self.step == self.max_step:
            self.epoch += 1
            self.step = 0
            arcs = self.sample(num_samples=100)
            f = open(self.arc_save_path, 'wb')
            pickle.dump(arcs, f)
            f.close()

    def _scale(self, value, last_k=10, scale_value=0.5):
        max_reward = np.max(self.history[-last_k:])
        if max_reward == 0:
            return value
        return scale_value / max_reward * value

class GraphNASTuner(Tuner):
    def __init__(self, arc_save_path, derive_num_samples=100):
        self.controller = None
        self.arc_save_path = arc_save_path
        self.derive_num_samples = derive_num_samples

    def update_search_space(self, search_space):
        if self.controller is None:
            self.controller = GraphNASController(search_space, self.arc_save_path)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        self.controller.train_one_step(value)

    def generate_parameters(self, parameter_id, **kwargs):
        arcs = self.controller.sample(num_samples=1)
        return arcs[0]

    def trial_end(self, parameter_id, success, **kwargs):
        pass
