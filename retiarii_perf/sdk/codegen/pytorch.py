
def graph_to_pytorch_script(graph: 'Graph') -> str:
    imports = []
    if 'imports' in graph.configs:
        imports.extend([f"from {_} import *" for _ in graph.configs['imports']])

    imports = "\n".join(imports)
    return _PyTorchScriptTemplate.format(imports=imports, codes=graph_to_pytorch_model(graph)).strip()

def graph_to_pytorch_model(graph: 'Graph') -> str:
    
    nodes = graph.topo_sort()

    # handle module node and function node differently
    # only need to generate code for module here
    node_codes = []
    for node in nodes:
        # 'input_arguments' means the node is `func` type
        if node.operation != None:
            if 'input_arguments' not in node.operation.params:
                node_codes.append('self.{} = {}'.format(node.name, node.operation.to_pytorch_init()))
        # else:
        #     node_codes.append('self.{} = None'.format(node.name))

    edge_codes = []
    for node in graph.input_nodes:
        if node.get_attribute("cuda") != False:
            line_code = node.name + " = " + node.name + ".cuda(non_blocking=True)"
            edge_codes.append(line_code)
    
    succ_count = {}
    for node in nodes:
        succ_count[node.name] = 0
        predecessors = graph.get_predecessors(node)
        for pred in predecessors:
            if pred.name in succ_count:
                #print(node.name, "pred", pred.name)
                succ_count[pred.name] += 1
    
    for node in graph.output_nodes:
        predecessors = graph.get_predecessors(node)
        for pred in predecessors:
            if pred.name in succ_count:
                #print(node.name, "pred", pred.name)
                succ_count[pred.name] += 1

    for node in nodes:
        predecessors = graph.get_predecessors(node)
        if not predecessors:
            inputs = ''
        elif len(predecessors) == 1:
            if node.operation and node.operation.type == 'FixedInputChoice':
                inputs = [predecessors[0].name]
            else:
                inputs = predecessors[0].name
        else:
            if not node.operation.type in ['aten::add', 'aten::stack', 'Mask', 'GlobalAvgPool', 'Attention', 'RNN', 'WrapperOp', 'Cell', 'CellStem0', 'CellStem1', 'FixedInputChoice', 'aten::view', 'aten::relu']:
                print('zql op: ', node.operation.type)
            assert node.operation.type in ['aten::add', 'aten::stack', 'Mask', 'GlobalAvgPool', 'Attention', 'RNN', 'WrapperOp', 'Cell', 'CellStem0', 'CellStem1', 'FixedInputChoice', 'aten::view', 'aten::relu']
            #inputs = 'sum([{}])'.format(', '.join(node.name for node in predecessors))
            inputs = [ node.name for node in predecessors ]
        
        if node.operation == None:
            edge_codes.append('{} = {}'.format(node.name, inputs))
        elif 'input_arguments' not in node.operation.params:
            if not isinstance(inputs, list):
                edge_codes.append('{} = self.{}({})'.format(node.name, node.name, inputs))
            else:
                assert node.operation.type in ['Mask', 'GlobalAvgPool', 'Attention', 'RNN', 'WrapperOp', 'Cell', 'CellStem0', 'CellStem1', 'FixedInputChoice']
                if node.operation.type == 'Mask': #FIXME: duplicated inputs for MASK
                    inputs = [inputs[1], inputs[0]] # FIXME: WRONG ORDER of input
                if node.operation.type == 'GlobalAvgPool': #FIXME: WRONG ORDER of input
                    inputs = [inputs[1], inputs[0]] 
                if node.operation.type == 'WrapperOp': #FIXME: WRONG ORDER of input
                    inputs = [inputs[1], inputs[0]] 
                if node.operation.type == 'FixedInputChoice':
                    edge_codes.append('{} = self.{}([{}])'.format(node.name, node.name, ', '.join(inputs)))
                else:
                    edge_codes.append('{} = self.{}({})'.format(node.name, node.name, ', '.join(inputs)))
        else: # `func` type
            if node.operation.type == 'aten::view':
                assert len(node.operation.params['input_arguments']) == 2
                if len(inputs) == 2:
                    # hack for batch size variable
                    #_aten__Int_190
                    first_in, second_in = None, None
                    if 'aten__Int' in inputs[0]:
                        first_in = inputs[1]
                        second_in = inputs[0]
                    elif 'aten__Int' in inputs[1]:
                        first_in = inputs[0]
                        second_in = inputs[1]
                    else:
                        raise
                    edge_codes.append('{} = {}.{}({}, {})'.format(node.name, first_in,
                                      node.operation.type.split('::')[-1],
                                      second_in, node.operation.params['input_arguments'][1][-1]))
                else:
                    edge_codes.append('{} = {}.{}({})'.format(node.name, inputs,
                                    node.operation.type.split('::')[-1],
                                    str(node.operation.params['input_arguments'][1])[1:-1]))
            elif node.operation.type == 'aten::size':
                assert len(node.operation.params['input_arguments']) == 2
                edge_codes.append('{} = {}.{}({})'.format(node.name, inputs,
                                  node.operation.type.split('::')[-1],
                                  node.operation.params['input_arguments'][1]))
            elif node.operation.type == 'aten::Int':
                assert len(node.operation.params['input_arguments']) == 1
                edge_codes.append('{} = int({})'.format(node.name, inputs))
            elif node.operation.type == 'aten::contiguous':
                assert len(node.operation.params['input_arguments']) == 2
                edge_codes.append('{} = {}.{}({})'.format(node.name, inputs,
                                  node.operation.type.split('::')[-1],
                                  ""))#node.operation.params['input_arguments'][1]
            elif node.operation.type == 'aten::mean':
                if len(node.operation.params['input_arguments']) == 4:
                    node.operation.params['input_arguments'] = node.operation.params['input_arguments'][0:3]
                assert len(node.operation.params['input_arguments']) == 3
                edge_codes.append('{} = {}.{}({}, {})'.format(node.name, inputs,
                                  node.operation.type.split('::')[-1],
                                  node.operation.params['input_arguments'][1],
                                  node.operation.params['input_arguments'][2]))
            elif node.operation.type == 'aten::add':
                assert len(node.operation.params['input_arguments']) == 3
                if isinstance(inputs, str):
                    edge_codes.append('{} = {}'.format(node.name, inputs))
                elif len(inputs) == 2:
                    edge_codes.append('{} = {} + {}'.format(node.name, inputs[0], inputs[1]))
                else:
                    assert(False)
            elif node.operation.type == 'aten::transpose':
                assert len(node.operation.params['input_arguments']) == 3
                edge_codes.append('{} = {}.{}({}, {})'.format(node.name, inputs,
                                  node.operation.type.split('::')[-1],
                                  node.operation.params['input_arguments'][1],
                                  node.operation.params['input_arguments'][2]))
            elif node.operation.type == 'aten::stack':
                assert len(node.operation.params['input_arguments']) == 2
                assert isinstance(inputs, list)
                edge_codes.append('{} = torch.stack([{}], {})'.format(node.name, ', '.join(inputs),
                                  node.operation.params['input_arguments'][1]))
            else:
                args_str = ''
                if node.operation.params['input_arguments']:
                    args_str = ', {}'.format(str(node.operation.params['input_arguments'][1:])[1:-1])
                edge_codes.append('{} = F.{}({}{})'.format(node.name,
                                  node.operation.type.split('::')[-1], inputs, args_str))
        # del nodes with no more successors to release GPU mem
        for pred in predecessors:
            if pred.name in succ_count:
                succ_count[pred.name] -= 1
                if succ_count[pred.name] == 0:
                    edge_codes.append("del {}".format(pred.name))

    for node in graph.output_nodes:
        predecessors = graph.get_predecessors(node)
        if not predecessors:
            value = 'None'
        elif len(predecessors) == 1:
            value = predecessors[0].name
        else:
            value = 'sum([{}])'.format(', '.join(node.name for node in predecessors))
        edge_codes.append('{} = {}'.format(node.name, value))

    try:
        input_code = ', '.join(node.name for node in graph.input_nodes)
        output_code = ', '.join(node.name for node in graph.output_nodes)
    except AttributeError:
        # to be compatible with the new format
        input_code = ', '.join(node.name for node in graph.input_nodes[0])
        output_code = ', '.join(node.name for node in graph.output_nodes[0])

    linebreak = '\n        '
    return _PyTorchModelTemplate.format(
        graph_name='Graph',
        inputs=input_code,
        outputs=output_code,
        nodes=linebreak.join(node_codes),
        edges=linebreak.join(edge_codes)
    )


_PyTorchScriptTemplate = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sdk.custom_ops_torch as CUSTOM
from nni.nas.pytorch import mutables

{imports}

{codes}
'''

_PyTorchModelTemplate = '''

class {graph_name}(nn.Module):
    def __init__(self):
        super().__init__()
        {nodes}

    def forward(self, {inputs}):
        {edges}
        return {outputs}
'''
