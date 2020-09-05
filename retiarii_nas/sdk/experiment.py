import json
import inspect
import os
import random
import subprocess
import yaml

from . import utils
from . import strategy
from .graph import Graph

class Experiment:
    def __init__(self, exp_name, base_model):
        self.exp_name = exp_name
        self.base_model = base_model
        self.training = None
        self.mutators = None
        self.strategy = None
        self.trainin_cls = None
        # create working space
        #if not os.path.isdir(exp_name):
        #    os.mkdir(exp_name)
        self.collapsed_nodes = None

    def specify_training(self, training_cls, *args, **kwargs) -> None:
        self.training_cls = training_cls
        self.training = {
            'file': inspect.getfile(training_cls),
            'func': training_cls.__name__,
            'args': args,
            'kwargs': kwargs
        }

    def specify_collapsed_nodes(self, collapsed_nodes):
        """
        This is hack for pytorch trace: nodes clustering
        """
        self.collapsed_nodes = collapsed_nodes

    def specify_mutators(self, mutators: 'List[Mutator]') -> None:
        """
        Parameters
        ----------
        mutators : list
        """
        self.mutators = mutators

    def specify_strategy(self, strategy: str, sampler: str) -> None:
        self.strategy = {'scheduler': strategy, 'sampler': sampler}

    def _pre_run(self) -> None:
        if os.environ.get('RETIARII_PRERUN'):
            # register this instance
            print(self.training, self.strategy)
            utils.register_experiment_config(self)
        else:
            print(self.training, self.strategy)
            print("Warning: please use retiarii to run this python script")
            print("Now you are in standalone mode...")
            
            import torch
            from .translate_code import gen_pytorch_graph
            def _get_dummy_input(data_type):
                trainer_constructor = self.training_cls
                args = self.training['args']
                kwargs = self.training['kwargs']
                trainer = trainer_constructor(*args, **kwargs)
                data_loader = trainer.train_dataloader()
                if data_type == 'textnas':
                    text, mask, label = next(iter(data_loader)) # for textnas
                    print('text: ', text.size())
                    print('mask', mask.size())
                    return (text, mask)
                else:
                    data, _ = next(iter(data_loader))
                    return data
            #dummy_input = _get_dummy_input('image')
            #dummy_input = torch.rand((1, 3, 224, 224))
            '''dummy_text = torch.rand((256, 64, 768))
            dummy_mask = torch.rand((256, 64))
            #print('dummy input size: ', dummy_input.size())
            # convert user model to graph ir
            #graph = gen_pytorch_graph(self.base_model, dummy_input, self.collapsed_nodes)
            graph = gen_pytorch_graph(self.base_model, (dummy_text, dummy_mask), self.collapsed_nodes)
            print('Generated base graph: ', graph)
            # visualize the generated graph ir
            print('convert graph...')
            from .visualization import convert_to_visualize
            vgraph = convert_to_visualize(graph)
            print('convert graph done')
            # mutate the generated graph ir
            class RandomSampler(strategy.Sampler):
                def choice(self, candidates):
                    return random.choice(candidates)
            random.seed(10)
            graph_obj = Graph.load(graph)
            sampler = RandomSampler()'''
            # using dry-run to generate search space
            '''search_space = []
            for mutator in self.mutators:
                #mutator.apply(graph_obj, sampler)
                graph_obj, recorded_candidates = mutator.dry_run(graph_obj)
                search_space.extend(recorded_candidates)
            print('search space: ', search_space)'''
            '''for mutator in self.mutators:
                graph_obj = mutator.apply(graph_obj, sampler)'''
            # generate pytorch code from graph ir
            '''graph_obj.generate_code('pytorch', output_file=f'generated/debug.py')
            
            #===train the graph===#
            print('start training the generated graph')
            graph_cls = utils.import_(f'generated.debug.Graph')
            trainer_constructor = self.training_cls
            args = self.training['args']
            kwargs = self.training['kwargs']
            training_instance = trainer_constructor(*args, **kwargs)
            model = graph_cls()
            training_instance.bind_model(model)
            optimizer = training_instance.configure_optimizer()
            training_instance.set_optimizer(optimizer)
            training_instance.training_logic()
            print('training done')'''

    def run(self, config) -> None:
        if os.environ.get('RETIARII_PREPARE'):
            with open('nni.yaml', 'w') as fp:
                yaml.dump(config, fp)
        else:
            self._pre_run()


def create_experiment(exp_name: str,
                      base_model: 'Union[torch.nn.Module, json]'
                      ) -> 'Experiment':
    return Experiment(exp_name, base_model)
