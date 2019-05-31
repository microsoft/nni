import numpy as np

from nni.tuner import Tuner

def random_archi_generator(nas_ss, random_state):
    '''random
    '''
    chosen_archi = {}
    print("zql: nas search space: ", nas_ss)
    for block_name, block in nas_ss.items():
        tmp_block = {}
        for layer_name, layer in block.items():
            tmp_layer = {}
            for key, value in layer.items():
                if key == 'layer_choice':
                    index = random_state.randint(len(value))
                    tmp_layer['chosen_layer'] = value[index]
                elif key == 'optional_inputs':
                    tmp_layer['chosen_inputs'] = []
                    print("zql: optional_inputs", layer['optional_inputs'])
                    if layer['optional_inputs']:
                        if isinstance(layer['optional_input_size'], int):
                            choice_num = layer['optional_input_size']
                        else:
                            choice_range = layer['optional_input_size']
                            choice_num = random_state.randint(choice_range[0], choice_range[1]+1)
                        for _ in range(choice_num):
                            index = random_state.randint(len(layer['optional_inputs']))
                            tmp_layer['chosen_inputs'].append(layer['optional_inputs'][index])
                elif key == 'optional_input_size':
                    pass
                else:
                    raise ValueError('Unknown field %s in layer %s of block %s' % (key, layer_name, block_name))
            tmp_block[layer_name] = tmp_layer
        chosen_archi[block_name] = tmp_block
    return chosen_archi

class RandomNASTuner(Tuner):
    '''RandomNASTuner
    '''

    def __init__(self):
        self.searchspace_json = None
        self.random_state = None

    def update_search_space(self, search_space):
        '''update
        '''
        self.searchspace_json = search_space
        self.random_state = np.random.RandomState()

    def generate_parameters(self, parameter_id):
        '''generate
        '''
        return random_archi_generator(self.searchspace_json, self.random_state)

    def receive_trial_result(self, parameter_id, parameters, value):
        '''receive
        '''
        pass
