import copy
import logging
import random
from collections import deque
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import nni
from nni.tuner import Tuner
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


class Predictor(nn.Module):
    def __init__(self, arch_size, hp_size):
        super(Predictor, self).__init__()
        self.arch_encoder = nn.Sequential(
            nn.Linear(arch_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.device)
        self.proxy_predictor = nn.Linear(128, 1).to(self.device)
        self.accuracy_predictor = nn.Linear(128 + hp_size, 1).to(self.device)

    def forward(self, arch, param=None, mode="acc"):
        arch = self.arch_encoder(arch)
        if mode == "acc":
            assert param is not None
            hybrid = torch.cat([arch, param], dim=0)
            return self.accuracy_predictor(hybrid)
        elif mode == 'flops':
            return self.proxy_predictor(arch)
        else:
            raise KeyError("mode should be either acc or flops")

    def predict_acc(self, arch, param):
        self.eval()
        arch = self.arch_encoder(arch)
        hybrid = torch.cat([arch, param], dim=0)
        self.train()
        return self.accuracy_predictor(hybrid)

    def predict_flops(self, arch):
        self.eval()
        arch = self.arch_encoder(arch)
        return self.proxy_predictor(arch)


class PredictorDataset(Dataset):
    def __init__(self, data, label):
        super(PredictorDataset, self).__init__()
        assert len(data) == len(label)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class FBv3Tuner(Tuner):
    def __init__(self, batch_size=16, pool_size=1000, max_flops=10000, p=50, q=50, epsilon=1e-6):
        # TODO: max_flops need adjustment
        super(FBv3Tuner, self).__init__()
        # TODO: early stop strategy

        self.predictor = None
        self.search_space = None
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.max_flops = max_flops
        self.epsilon = epsilon
        self.p = p
        self.q = q
        self.evaluate_queue = deque()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optim_label = []
        self.arch = []
        self.history = []
        self.pool_model = []
        self._random_pool()
        self._encoder_pretrain()
        self.iter_point = 0
        self.key_len = dict()
        self._stage2 = False
        self._s = 0
        self.s = 0
        self.D = None

    def _encoder_pretrain(self):
        flops = [self._get_flops(self.dict2tensor(model, mode='arch', device=self.device).to(self.device))
                 for model in self.pool_model]
        dataloader = DataLoader(PredictorDataset(self.arch, flops), batch_size=64, shuffle=True)
        criterion = nn.MSELoss()
        for train_x, label in dataloader:
            pred = self.predictor(train_x, mode="flops")
            loss = criterion(pred, label)
            loss.backward()

    def dict2tensor(self, config, device, mode=''):
        keys = sorted(config.keys())
        arch, params = [], []
        for key in keys:
            choice = config[key]['_idx']
            if self.search_space[key]['_type'] == 'choice':
                params.append(F.one_hot(choice, classes=self.key_len[key]).to(device))
            elif self.search_space[key]['_type'] == 'input_choice':
                arch.append(F.one_hot(choice, classes=self.key_len[key]).to(device))
            else:
                raise KeyError
        arch = reduce(lambda x, y: torch.cat([x, y]), arch)
        params = reduce(lambda x, y: torch.cat([x, y]), params)
        if mode == 'arch':
            return arch
        elif mode == 'params':
            return params
        return (arch, params)

    def generate_parameters(self, parameter_id, **kwargs):
        if not self.evaluate_queue:
            if self.iter_point != 0:
                self._train_predictor()
            if self.iter_point < len(self.pool_model) - 1:
                for model in self.pool_model[self.iter_point: min(self.iter_point + self.batch_size, self.pool_size)]:
                    self.evaluate_queue.append(model)
                    self.iter_point += self.batch_size
        return self.evaluate_queue.popleft()

    def _get_flops(self, model_config):
        # TODO: use real flops counter
        return np.random.randint(1, 100000)

    def _stage2_setup(self):
        self._stage2 = True
        self.s = max(self.optim_label)
        top_indices = np.argpartition(np.array(self.optim_label), -1 * self.p)[-1 * self.p:]
        while self.s - self._s > self.epsilon:
            self.D = [(self.arch[idx], self.optim_label[idx]) for idx in top_indices]
            for _ in range(self.q):
                model_config = self._random_model()
                while model_config in self.D or self.predictor.predict_flops(
                        self.dict2tensor(model_config, mode='arch', device=self.device)) > self.max_flops:
                    model_config = self._random_model()
                arch_tensor, params_tensor = self.dict2tensor(model_config, device=self.device)
                pred_acc = self.predictor.predict_acc(arch_tensor, params_tensor)
                self.D.append((model_config, pred_acc))
        # TODO: *warning* no validation on trial in stage 2

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        self.history.append(FinishedIndividual(parameter_id=parameter_id,
                                               parameters=parameters,
                                               result=value))
        self.arch.append(parameters)
        self.optim_label.append(value)
        if self.iter_point > len(self.pool_model) - 1 and not self._stage2:
            self._stage2_setup()

    def trial_end(self, parameter_id, success, **kwargs):
        pass

    def update_search_space(self, search_space):
        assert self.search_space is None
        self.search_space = search_space
        search_space_key = sorted(search_space)
        arch_size, hp_size = 0, 0
        for key in search_space_key:
            val = search_space[key]
            if val['_type'] == 'input_choice' or val['_type'] == 'layer_choice':
                arch_size += len(val['_value'])
            elif val['_type'] == 'choice':
                hp_size += len(val['_value'])
        self.predictor = Predictor(arch_size, hp_size)

    def _random_pool(self):
        for _ in range(self.pool_size):
            model_config = self._random_model()
            while model_config in self.pool_model or self.predictor.predict_flops(model_config) > self.max_flops:
                model_config = self._random_model()
            self.pool_model.append(model_config)

    def _train_predictor(self):
        arch = [self.dict2tensor(model, device=self.device) for model in self.arch]
        dataloader = DataLoader(PredictorDataset(arch, self.optim_label),
                                batch_size=64, shuffle=True)
        criterion = nn.MSELoss()
        for train_x, label in dataloader:
            pred = self.predictor(train_x[0], train_x[1])
            loss = criterion(pred, label)
            loss.backward()

    def _random_model(self):
        search_space_key = sorted(self.search_space)
        individual = {}
        for key in search_space_key:
            val = self.search_space[key]
            self.key_len[key] = len(val['_value'])
            if val['_type'] == 'input_choice':
                choice = np.random.choice(range(len(val['_value'])))
                individual[key] = {'_value': val['_value'][choice], '_idx': choice}
            elif val['_type'] == 'layer_choice':
                choice = np.random.choice(range(len(val['_value'])))
                individual[key] = {'_value': val['_value'][choice], '_idx': choice}
            elif val['_type'] == 'choice':
                choice = np.random.choice(range(len(val['_value'])))
                individual[key] = val['_value'][choice]
            else:
                raise KeyError

        return individual
