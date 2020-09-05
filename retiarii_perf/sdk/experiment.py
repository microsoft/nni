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

    def specify_training(self, training_cls, *args, **kwargs) -> None:
        self.training_cls = training_cls
        self.training = {
            'file': inspect.getfile(training_cls),
            'func': training_cls.__name__,
            'args': args,
            'kwargs': kwargs
        }

    def specify_mutators(self, mutators: 'List[Mutator]') -> None:
        """
        Parameters
        ----------
        mutators : list
        """
        self.mutators = mutators

    def specify_strategy(self, strategy: str, sampler: str) -> None:
        self.strategy = {'scheduler': strategy, 'sampler': sampler}

    def _pre_run(self, pre_run_config=None) -> None:
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
            def _get_dummy_input():
                trainer_constructor = self.training_cls
                args = self.training['args']
                kwargs = self.training['kwargs']
                trainer = trainer_constructor(*args, **kwargs)
                data_loader = trainer.train_dataloader()
                # TODO: make sure there must be two returned variables
                if pre_run_config['mask']:
                    text, mask, label = next(iter(data_loader)) # for textnas
                    #print('text: ', text.size())
                    #print('mask', mask.size())
                    return text, mask
                else:
                    data, _ = next(iter(data_loader))
                    return data
            
            # dummy_input = _get_dummy_input()
            if pre_run_config['mask']:
                dummy_input = torch.rand((128, 64, 768))
                dummy_mask = torch.rand((128, 64))
                #print(dummy_input)
                # from transformers import BertTokenizer
                # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                #dummy_input = torch.tensor([tokenizer.encode(t, max_length=64, pad_to_max_length=True, truncation=True) for t in dummy_input]).cuda()
                #self.base_model.cuda()
                graph = gen_pytorch_graph(self.base_model, 
                                (dummy_input, dummy_mask),
                                name = pre_run_config['name'])
            else:
                dummy_input = torch.rand(pre_run_config['x_shape'])#(32, 3, 224, 224))
                graph = gen_pytorch_graph(self.base_model, dummy_input,
                                name = pre_run_config['name'])
            #print('dummy input size: ', dummy_input.size())
            # convert user model to graph ir

            with open('debug.json', 'w') as fp:
                json.dump(graph, fp)
            print('Generated base graph: ', graph)
            # visualize the generated graph ir
            print('convert graph...')
            #from .visualization import convert_to_visualize
            #vgraph = convert_to_visualize(graph)
            print('convert graph done')
            # mutate the generated graph ir
            class RandomSampler(strategy.Sampler):
                def choice(self, candidates):
                    return random.choice(candidates)
            
            graph['name'] = pre_run_config['name']
            x_node_name = graph['graph']['inputs'][0]['name']
            graph['graph']['inputs'][0]['attributes'] = {"shape": pre_run_config['x_shape'], "dtype": pre_run_config['x_dtype'], "position": "x"}
            if pre_run_config['mask']:
                graph['graph']['inputs'][1]['attributes'] = {"shape": pre_run_config['x_shape'][:2], "dtype": pre_run_config['x_dtype'], "position": "mask"}
                graph['graph']['hidden_nodes'].append({
                        "name": "breakpoint", 
                        "operation": {"type": 'Identity'},
                        "attributes": {
                            "shape": [128, 64, 768],
                            "dtype": "torch.float32",
                            "non-trainable": True}
                        })
                
                for e in graph['graph']['edges']:
                    if e['head'] == x_node_name:
                        e['head'] = 'breakpoint'
                graph['graph']['edges'].append({"head": x_node_name, "tail": "breakpoint"})
                
            graph['graph']['inputs'].append({"name": "input_2", 'attributes' : {"shape" : pre_run_config['y_shape'], "dtype": pre_run_config['y_dtype'], "position": "y"}})
            
            graph['graph']['outputs'] = [{"name": "origin_x"}, {"name": "origin_y"}, *graph['graph']['outputs']]
            if pre_run_config['mask']:
                graph['graph']['edges'].append({"head": "breakpoint", "tail": "origin_x"})
            else:
                graph['graph']['edges'].append({"head": x_node_name, "tail": "origin_x"})
            graph['graph']['edges'].append({"head": "input_2", "tail": "origin_y"})
            
            graph_obj = Graph.load(graph)
            sampler = RandomSampler()
            search_space = []
            for mutator in self.mutators:
                #mutator.apply(graph_obj, sampler)
                graph_obj, recorded_candidates = mutator.dry_run(graph_obj)
                search_space.extend(recorded_candidates)
            print('search space: ', search_space)
            # generate pytorch code from graph ir
            if 'imports' in pre_run_config:
                graph_obj.configs['imports'] = pre_run_config['imports']
            graph_obj.generate_code('pytorch',
                output_file=f"generated/{pre_run_config['name']}.py")
            
            with open(f"generated/{pre_run_config['name']}.json", 'w') as fp:
                json.dump(graph, fp)

    def run(self, config, pre_run_config = None) -> None:
        if os.environ.get('RETIARII_PREPARE'):
            with open('nni.yaml', 'w') as fp:
                yaml.dump(config, fp)
        else:
            self._pre_run(pre_run_config=pre_run_config)


def create_experiment(exp_name: str,
                      base_model: 'Union[torch.nn.Module, json]'
                      ) -> 'Experiment':
    return Experiment(exp_name, base_model)
