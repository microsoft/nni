import importlib
import json
import sys
import os

from .graph import Graph
from .graph_v2 import Graph as GraphV2
from . import utils
from .translate_code import gen_pytorch_graph


class ExperimentConfig:
    def __init__(self) -> None:
        self._base_graph = None
        self._mutators = None
        self._strategy = None
        self._base_code = None

    @property
    def base_code(self) -> str:
        if not self._base_code:
            self._base_code = utils.experiment_config()['base_code']
        return self._base_code

    @property
    def base_graph(self) -> Graph:
        if not self._base_graph:
            json_path = utils.experiment_config('base_graph')
            data = json.load(open(json_path))
            if utils.experiment_config('framework') == 'tf':
                self._base_graph = GraphV2.load(data)
            else:
                self._base_graph = Graph.load(data)
        return self._base_graph

    def _attach_training(self, graph_json) -> 'JSON':
        graph_json['training'] = utils.experiment_config().training
        return graph_json

    def _get_dummy_input(self):
        trainer_constructor = utils.experiment_config().training_cls
        args = utils.experiment_config().training['args']
        kwargs = utils.experiment_config().training['kwargs']
        trainer = trainer_constructor(*args, **kwargs)
        data_loader = trainer.train_dataloader()
        # TODO: make sure there must be two returned variables
        data, _ = next(iter(data_loader))
        return data

    def _get_dummy_input_textnas(self):
        trainer_constructor = utils.experiment_config().training_cls
        args = utils.experiment_config().training['args']
        kwargs = utils.experiment_config().training['kwargs']
        trainer = trainer_constructor(*args, **kwargs)
        data_loader = trainer.train_dataloader()
        text, mask, label = next(iter(data_loader)) # for textnas
        text = text.to('cpu')
        mask = mask.to('cpu')
        #print('text: ', text.size())
        #print('mask', mask.size())
        return (text, mask)

    @property
    def base_model(self) -> Graph:
        if not self._base_graph:
            base_model = utils.experiment_config().base_model
            collapsed_nodes = utils.experiment_config().collapsed_nodes
            if collapsed_nodes == 'textnas':
                dummy_input = self._get_dummy_input_textnas()
            else:
                dummy_input = self._get_dummy_input()
            graph_json = gen_pytorch_graph(base_model, dummy_input, collapsed_nodes)
            graph_json = self._attach_training(graph_json)
            self._base_graph = Graph.load(graph_json)
        return self._base_graph

    @property
    def mutators(self) -> 'List[Mutator]':
        if not self._mutators:
            self._mutators = utils.experiment_config().mutators
        return self._mutators

    @property
    def strategy(self) -> 'ExperimentConfig_Strategy':
        if self._strategy is None:
            scheduler = utils.import_(utils.experiment_config()['strategy']['scheduler'])
            sampler = utils.import_(utils.experiment_config()['strategy']['sampler'])
            self._strategy = SimpleNamespace(scheduler=scheduler, sampler=sampler)
        return self._strategy

    @property
    def hyper_parameters(self) -> 'Any':
        return utils.experiment_config()['hyper_parameters']


class ExperimentConfig_Strategy:
    def __init__(self, scheduler: 'Union[Callable, AbstractStrategy, None]', sampler: 'Optional[Sampler]') -> None:
        self.scheduler = scheduler
        self.sampler = sampler
