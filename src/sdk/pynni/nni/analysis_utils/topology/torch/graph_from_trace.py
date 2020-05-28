# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import os
import csv
import logging
import torch
import torch.jit as jit
import graphviz


__all__ = ["VisualGraph"]

TUPLE_UNPACK = 'prim::TupleUnpack'
CLASSTYPE_KIND = 'ClassType'
logger = logging.getLogger('Graph_From_Trace')


class PyNode:
    def __init__(self, cnode, isValue=False):
        self.cnode = cnode
        self.isValue = isValue
        self.isTensor = False
        self.isOp = not self.isValue
        if self.isValue:
            if isinstance(self.cnode.type(), torch._C.TensorType):
                self.isTensor = True
                self.shape = self.cnode.type().sizes()
        if self.isOp:
            scopename = cnode.scopeName()
            if torch.__version__ >= '1.4.0':
                # note, the scopeName of node may be empty
                scopename = re.split('/', scopename)
                self.name = scopename[-1] if len(scopename) > 0 else ''
            else:
                self.name = '.'.join(re.findall(r'\[(.*?)\]', scopename))
            # remove the __module prefix
            if self.name.startswith('__module.'):
                self.name = self.name[len('__module.'):]

    def __str__(self):
        if self.isTensor:
            _str = 'Tensor: {}'.format(self.shape)
        elif self.isOp:
            op_type = self.cnode.kind()
            op_type = re.split('::', op_type)[1]
            _str = self.name + '\nType: ' + op_type
        else:
            _str = str(self.cnode.type())
        return _str

    def parents(self):
        if self.isOp:
            return list(self.cnode.inputs())
        else:
            return [self.cnode.node()]


class VisualGraph:
    def __init__(self, model=None, data=None, graph=None):
        """
        We build the network architecture graph according the graph
        in the scriptmodule. However, the original graph from jit.trace
        has lots of detailed information which make the graph complicated
        and hard to understand. So we also store a copy of the network
        architecture in the self.forward_edge. We will simplify the network
        architecure (such as unpack_tuple, etc) stored in self.forward_edge
        to make the graph more clear.
        Parameters
        ----------
            model:
                The model to build the network architecture.
            data:
                The sample input data for the model.
            graph:
                Traced graph from jit.trace, if this option is set,
                we donnot need to trace the model again.
        """
        self.model = model
        self.data = data
        if graph is not None:
            self.graph = graph
        elif (model is not None) and (data is not None):
            with torch.onnx.set_training(model, False):
                self.traced_model = jit.trace(model, data)
                self.graph = self.traced_model.graph
                torch._C._jit_pass_inline(self.graph)
        else:
            raise Exception('Input parameters invalid!')
        self.forward_edge = {}
        self.c2py = {}
        self.visited = set()
        self.build_graph()
        self.unpack_tuple()

    def unpack_tuple(self):
        """
        jit.trace also traces the tuple creation and unpack, which makes
        the grapgh complex and difficult to understand. Therefore, we
        unpack the tuple handly to make the graph clear.
        """
        parent_node = None
        for node in self.graph.nodes():
            if node.kind() == TUPLE_UNPACK:
                in_tuple = list(node.inputs())[0]
                parent_node = in_tuple.node()
                in_tensors = list(parent_node.inputs())
                out_tensors = list(node.outputs())
                assert len(in_tensors) == len(out_tensors)
                for i, _ in enumerate(in_tensors):
                    ori_edges = self.forward_edge[in_tensors[i]]
                    # remove the out edge to the Tuple_construct OP node
                    self.forward_edge[in_tensors[i]] = list(
                        filter(lambda x: x != parent_node, ori_edges))
                    # Directly connect to the output nodes of the out_tensors
                    self.forward_edge[in_tensors[i]].extend(
                        self.forward_edge[out_tensors[i]])

    def build_graph(self):
        """
        Copy the architecture information from the traced_model into
        forward_edge.
        """
        for node in self.graph.nodes():
            self.c2py[node] = PyNode(node)
            for _input in node.inputs():
                if _input not in self.c2py:
                    self.c2py[_input] = PyNode(_input, True)
                if _input in self.forward_edge:
                    self.forward_edge[_input].append(node)
                else:
                    self.forward_edge[_input] = [node]
            for output in node.outputs():
                if output not in self.c2py:
                    self.c2py[output] = PyNode(output, True)
                if node in self.forward_edge:
                    self.forward_edge[node].append(output)
                else:
                    self.forward_edge[node] = [output]

    def visual_traverse(self, curnode, graph, lastnode, cfg):
        """"
        Traverse the network and draw the nodes and edges
        at the same time.
        Parameters
        ----------
            curnode:
                Current visiting node(tensor/module).
            graph:
                The handle of the Dgraph.
            lastnode:
                The last visited node.
            cfg:
                Dict object to specify the rendering
                configuration for operation node.
                key is the name of the operation,
                value is a also a dict. For example,
                {'conv1': {'shape':'box', 'color':'red'}}
        """
        if curnode in self.visited:
            if lastnode is not None:
                graph.edge(str(id(lastnode)), str(id(curnode)))
            return
        self.visited.add(curnode)
        tmp_str = str(self.c2py[curnode])
        if self.c2py[curnode].isOp:
            name = self.c2py[curnode].name
            # default render configuration
            render_cfg = {'shape': 'ellipse', 'style': 'solid'}
            if name in cfg:
                render_cfg = cfg[name]
            graph.node(str(id(curnode)), tmp_str, **render_cfg)
        else:
            graph.node(str(id(curnode)), tmp_str, shape='box',
                       color='lightblue', style='dashed')
        if lastnode is not None:
            graph.edge(str(id(lastnode)), str(id(curnode)))
        if curnode in self.forward_edge:
            for _next in self.forward_edge[curnode]:
                self.visual_traverse(_next, graph, curnode, cfg)

    def base_visualization(self, filename, save_format='jpg', cfg=None):
        """
        visualize the network architecture automaticlly.
        Parameters
        ----------
            filename:
                The filename of the saved image file.
            save_format:
                The output save_format.
        """
        # TODO and detailed mode for the visualization function
        # in which the graph will also contain all the weights/bias
        # information.
        if not cfg:
            cfg = {}
        graph = graphviz.Digraph(format=save_format)
        self.visited.clear()
        for _input in self.graph.inputs():
            if _input.type().kind() == CLASSTYPE_KIND:
                continue
            self.visual_traverse(_input, graph, None, cfg)
        graph.render(filename)

    def visualize_with_flops(self, filepath, save_format, flops_file):
        assert os.path.exists(flops_file)
        f_handle = open(flops_file, 'r')
        csv_r = csv.reader(f_handle)
        flops = {}
        # skip the header of the csv file
        _ = next(csv_r)
        for row in csv_r:
            if len(row) == 2:
                layername = row[0]
                _flops = float(row[1])
                flops[layername] = _flops

        f_handle.close()
        # Divide the flops of the layers into 11 levels
        # We use the 'rdylgn11 color scheme' to present
        # the number of the flops, in which we have 11 colors
        # range from green to red.
        _min_flops = min(flops.values())
        _max_flops = max(flops.values())
        color_scheme_count = 9
        flops_step = (_max_flops - _min_flops) / (color_scheme_count-1)

        cfgs = {}
        for layername in flops:
            flops_level = (flops[layername] - _min_flops) / flops_step
            # flops_level = color_scheme_count - int(round(flops_level))
            flops_level = int(round(flops_level)) + 1
            render_cfg = render_cfg = {'shape': 'ellipse',
                                       'fillcolor': "/reds9/"+str(flops_level), 'style': 'filled'}
            cfgs[layername] = render_cfg
        self.base_visualization(filepath, save_format=save_format, cfg=cfgs)

    def visualize_with_dependency(self, filepath, save_format, dependency_file):
        assert os.path.exists(dependency_file)
        f_handle = open(dependency_file, 'r')
        csv_r = csv.reader(f_handle)
        # skip the header of the csv file
        _ = next(csv_r)
        dependency_sets = []
        for row in csv_r:
            tmp_set = set()
            for i in range(1, len(row)):
                tmp_set.add(row[i])
            dependency_sets.append(tmp_set)
        f_handle.close()
        # Create the render configs, assign the same color for the
        # same dependency set
        cfgs = {}
        colorid = 0
        for tmp_set in dependency_sets:
            if len(tmp_set) == 1:
                # This layer has no dependency
                continue
            colorid = (colorid + 1) % 12
            str_color = "/paired12/%d" % (colorid + 1)
            for layername in tmp_set:
                render_cfg = {'shape': 'ellipse',
                              'fillcolor': str_color, 'style': 'filled'}
                cfgs[layername] = render_cfg
        self.base_visualization(filepath, save_format=save_format, cfg=cfgs)

    def visualize_with_sensitivity(self, filepath, save_format, sensitivity_file):
        assert os.path.exists(sensitivity_file)
        f_handle = open(sensitivity_file, 'r')
        csv_r = csv.reader(f_handle)
        header = next(csv_r)
        # sparsities is ordered in sensitivity analysis
        sparsities = [float(x) for x in header[1:]]
        sensitivity = {}
        for row in csv_r:
            layername = row[0]
            accs = [float(_acc) for _acc in row[1:]]
            sensitivity[layername] = accs
        f_handle.close()
        # Note: Due to the early stop in SensitivityAnalysis, the number of
        # accuracies of different sparsities may be different. The earlier
        # the layers stops, the higher the sensitivity is.
        cfgs = {}
        color_scheme_count = 9
        for layername in sensitivity:
            _max = sparsities[len(sensitivity[layername]) - 1]
            _max_all = max(sparsities)
            level = 1.0 - (_max / _max_all)  # [0, 1]
            level = int(color_scheme_count * level)  # [0, 9]
            # color number start from 1
            if level == 0:
                level = 1
            str_color = "/reds9/%d" % level
            render_cfg = {'shape': 'ellipse',
                          'fillcolor': str_color, 'style': 'filled'}
            cfgs[layername] = render_cfg
        self.base_visualization(filepath, save_format=save_format, cfg=cfgs)

    def visualization(self, filename, save_format='jpg',
                      flops_file=None,
                      sensitivity_file=None,
                      dependency_file=None):

        # First, visualize the network architecture only
        self.base_visualization(filename, save_format=save_format)
        # if the flops file is specified, we also render
        # a image with the flops information.
        if flops_file is not None:
            flops_img = filename + '_flops'
            self.visualize_with_flops(flops_img, save_format, flops_file)

        if dependency_file is not None:
            dependency_img = filename + '_dependency'
            self.visualize_with_dependency(
                dependency_img, save_format, dependency_file)

        if sensitivity_file is not None:
            sensitivity_img = filename + '_sensitivity'
            self.visualize_with_sensitivity(
                sensitivity_img, save_format, sensitivity_file)
