from .base_optimization_strategy import BaseOptimizationStrategy
from .dedup_rule import DeduplicationRule
from .batch_rule import BatchRule
#from .batch_rule import BatchingRule
#from .weight_sharing_rule import WeightSharingRule
from .profiler import Profiler
from .logical_plan import logical_plan_generator
from ..graph import Graph, NodeType, Node, Operation, Edge
from ..operations import Broadcast
from .dedup_rule import DedupNode

import json

import copy

profiler = Profiler()

def dump_multi_graph(graph, original_graphs):
    data = graph.dump()
    inputs = [[] for _ in original_graphs]
    outputs = [[] for _ in original_graphs]
    graph_idx = {}
    for idx, g in enumerate(original_graphs):
        graph_idx[g.name] = idx
    for node in graph.input_nodes:
        inputs[graph_idx[node.graph.name]].append(node.dump())
    for node in graph.output_nodes:
        outputs[graph_idx[node.graph.name]].append(node.dump())

    data['graph']['inputs'] = inputs
    data['graph']['outputs'] = outputs
    data['training_config'] = [_.configs for _ in original_graphs]
    data['utils'] = {}
    for g in original_graphs:
        data['utils'].update(g.dump_utils())
    return data
    
class SimpleOptimizationStrategy(BaseOptimizationStrategy):
    def __init__(self, config, 
                        disable_dedup = False, 
                        disable_batch = False, 
                        disbale_standalone = False,
                        batch_size = 1):
        self._rules = [#DeduplicationRule, 
                       #BatchingRule, 
                    ]
        if not disable_dedup:
            self._rules.append(DeduplicationRule)
        if not disable_batch:
            self._rules.append(BatchRule)
        self.disbale_standalone = disbale_standalone
        self.config = config
        self.rule_cfg = {'batch_size': batch_size}

    def optimize(self, trials : 'List[Graph]') -> 'List[List[Graph]]':
        logical_plan = logical_plan_generator(trials)
        rule_instances = [ _(self.rule_cfg) for _ in self._rules]
        for rule in rule_instances:
            rule.logical_transform(logical_plan)
        
        with open('debug.json', 'w') as fp:
            json.dump(logical_plan.graph.dump(), fp)
        
        optimized_trials = self._physical_optimize(logical_plan)
        for graph_group in optimized_trials:
            for graph, _ in graph_group:
                _name_replace_logical_to_physical(graph)
        return optimized_trials
    
    def _physical_optimize(self, 
                        logical_plan:'LogicalPlan') \
                            -> 'List[List[(Graph, List[Graph])]]':
        trial_groups = _simple_planner(logical_plan, 
                                        self.config, 
                                        disbale_standalone=self.disbale_standalone)
        #trial_groups = [logical_plan.all_trials] 
        # TODO: generate multiple groups using config

        optimized_trials = []
        for trials in trial_groups:
            phy_graph = copy.deepcopy(logical_plan.graph)
            
            # Logical node replaced with physical nodes
            has_optimized_hidden_nodes = _logical_node_repalce(phy_graph, trials)
            
            # graph partition; each graph runs in one process
            if self.config['use_multi_proc']:
                # graph partition
                final_phy_graphs, graph_rank, original_graph = \
                    _partition_breakpoint(trials, phy_graph)
                use_distributed = \
                    _generate_final_graphs(phy_graph, final_phy_graphs, graph_rank)
                    
                if use_distributed:
                    _add_distributed_config(final_phy_graphs, graph_rank)
                    _add_device_placement(final_phy_graphs, 
                                            self.config, 
                                            disbale_standalone=self.disbale_standalone)
                
                optimized_trials.append([ (final_phy_graphs[_], \
                    [original_graph[_]]) for _ in final_phy_graphs ])
            else: 
                # all graphs built into one graph; or just a single graph
                optimized_trials.append( [ (phy_graph, [phy_graph]) ] )
        return optimized_trials

def _get_max_batch_time(trials, perf):
    return max([perf[_.name]['avg_batch_time'] for _ in trials])

def _simple_planner(logical_plan, config, disbale_standalone=False):
    all_trials = logical_plan.all_trials
    if 'pack_all' in config and config['pack_all']:
        return [[all_trials]]
    trial_perf = {}
    for trial in all_trials:
        trial_perf[trial.name] = profiler.profile(trial)
    trials_in_one_gpu = _greedy_pack(all_trials, 
                                    trial_perf, 
                                    logical_plan=logical_plan, 
                                    max_mem_util = config['max_mem_util'], 
                                    max_trial_per_gpu = config['max_trial_per_gpu'], 
                                    disbale_standalone=disbale_standalone)
    trials_in_one_gpu.sort(key=lambda x:_get_max_batch_time(x, trial_perf))
    
    base_idx, search_idx = 0, 1
    current_group = [ *trials_in_one_gpu[base_idx] ]
    trials_groups = [ current_group ]
    while search_idx < len(trials_in_one_gpu):
        if config['disable_merge']==False and \
            search_idx - base_idx +1 <= config['max_num_gpu']:
            # Merge
            # trials_in_one_gpu[base_idx].merge(trials_in_one_gpu[search_idx])
            current_group.extend(trials_in_one_gpu[search_idx])
        else:
            base_idx = search_idx
            current_group = [ *trials_in_one_gpu[base_idx]]
            trials_groups.append(current_group)
        search_idx += 1
    return trials_groups

def _contain_standalone_op(trial, logical_plan, counted_nodes):
    _standalone_ops = ['breakpoint']
    for node in trial.hidden_nodes:
        if node.name in _standalone_ops:
            if logical_plan:
                for logical_node in logical_plan.graph.hidden_nodes:
                    if isinstance(logical_node, DedupNode):
                        node_water_mark = node.name + '!' + node.graph.name
                        if node_water_mark in logical_node.water_mark:
                            if logical_node.name not in counted_nodes:
                                counted_nodes.add(logical_node.name)
                                return True
                            else:
                                return False
            return True
    return False

def _greedy_pack(all_trials, 
                trial_perf, 
                logical_plan=None, 
                max_mem_util = 1.01, 
                max_trial_per_gpu = 1, 
                disbale_standalone = False):
    all_trials.sort(key=lambda x:trial_perf[x.name]['mem_util'])
    sum_mem = 0
    trials_to_pack = []
    trials_grouped_by_one_gpu = []
    all_trials.append(None)
    counted_nodes = set()
    for trial in all_trials:
        if trial:
            has_standalone_op = _contain_standalone_op(trial, logical_plan, counted_nodes)
        if trial == None or \
            sum_mem+trial_perf[trial.name]['mem_util'] >= max_mem_util or \
            len(trials_to_pack) == max_trial_per_gpu or \
            (len(trials_to_pack) > 0 and has_standalone_op \
                and disbale_standalone==False):
            
            if len(trials_to_pack) > 0:
                trials_grouped_by_one_gpu.append([ _ for _ in trials_to_pack])
            sum_mem = 0
            trials_to_pack = []
            if trial == None:
                break
        sum_mem += trial_perf[trial.name]['mem_util']
        trials_to_pack.append(trial)
        if has_standalone_op and disbale_standalone==False:
            trials_grouped_by_one_gpu.append([ _ for _ in trials_to_pack])
            sum_mem = 0
            trials_to_pack = []
    return trials_grouped_by_one_gpu

def _name_replace_logical_to_physical(graph):
    all_nodes = graph.input_nodes + graph.hidden_nodes + graph.output_nodes
    for node in all_nodes:
        if 'logical_g' in node.name:
            node.name = node.name.replace('logical_', 'physical_') 

def _logical_node_repalce(phy_graph, trials):
    has_optimized_hidden_nodes = False
    edge_mapping = {}
    edge_replace_idx = {}
    all_nodes = phy_graph.input_nodes + phy_graph.hidden_nodes + phy_graph.output_nodes
    for node in all_nodes:
        if node.node_type == NodeType.Logical:
            node_or_graph_to_replace = node.physical_replace(trials)
            if isinstance(node_or_graph_to_replace, Graph):
                assert(node.node_type == NodeType.Logical)
                graph_to_replace = node_or_graph_to_replace
                # Input/Output nodes of graph_to_replace will be added as hidden nodes of the physical 
                is_output = False
                if node in phy_graph.hidden_nodes:
                    phy_graph.hidden_nodes.remove(node)
                elif node in phy_graph.output_nodes:
                    phy_graph.output_nodes.remove(node)
                    is_output = True
                else:
                    raise ValueError('node to be replaced by batching should not be InputNode')
                edge_mapping[node.name] = []
                for input_node in graph_to_replace.input_nodes:
                    phy_graph.hidden_nodes.append(input_node)
                    edge_mapping[node.name].append(input_node)
                edge_replace_idx[node.name] = 0
                
                
                for hidden_node in graph_to_replace.hidden_nodes:
                    phy_graph.hidden_nodes.append(hidden_node)
                
                assert(len(graph_to_replace.output_nodes) <= 1) # FIXME: support MIMO
                if len(graph_to_replace.output_nodes) == 1:
                    output_node = graph_to_replace.output_nodes[0]
                    if is_output:
                        phy_graph.output_nodes.append(output_node)    
                    else:
                        phy_graph.hidden_nodes.append(output_node)
                    edge_mapping[node.name].append(output_node)

                for edge in graph_to_replace.edges:
                    phy_graph.edges.append(edge)
            else:
                node_to_replace = node_or_graph_to_replace
                if node_to_replace.node_type == NodeType.Input:
                    phy_graph.input_nodes.remove(node)
                    phy_graph.input_nodes.append(node_to_replace)
                elif node_to_replace.node_type == NodeType.Hidden:
                    phy_graph.hidden_nodes.remove(node)
                    phy_graph.hidden_nodes.append(node_to_replace)
                    has_optimized_hidden_nodes = True
                elif node_to_replace.node_type == NodeType.Output:
                    phy_graph.output_nodes.remove(node)
                    phy_graph.output_nodes.append(node_to_replace)
                edge_mapping[node.name] = [node_to_replace]
                edge_replace_idx[node.name] = 0
    
    unique_edges = []
    edge_set = set()
    for edge in phy_graph.edges:
        if edge.head.name in edge_mapping:
            edge.head = edge_mapping[edge.head.name][-1]
        if edge.tail.name in edge_mapping:
            tail_name = edge.tail.name
            edge.tail = edge_mapping[tail_name][edge_replace_idx[tail_name]]
            edge_replace_idx[tail_name] += 1
        named_edge = (edge.head.name, edge.tail.name)
        if named_edge not in edge_set:
            edge_set.add((edge.head.name, edge.tail.name))
            unique_edges.append(edge)
    phy_graph.edges = unique_edges
    return has_optimized_hidden_nodes

def _partition_breakpoint(trials, phy_graph):
    breakpoint_operations = set(['breakpoint']) #set(["BertEmbedding"])
    
    final_phy_graphs, graph_rank, original_graph = \
        _partition_one_graph_per_process(trials, phy_graph)
    if len(final_phy_graphs) == 1:
        return final_phy_graphs, graph_rank, original_graph

    lastest_partitioned_graph = {}
    for g_name in final_phy_graphs:
        lastest_partitioned_graph[g_name] = g_name
        
    sorted_nodes = phy_graph.topo_sort()
    sorted_nodes.extend(phy_graph.output_nodes)
    for node in sorted_nodes:
        if node.graph.name not in final_phy_graphs:
            continue
        node.partition = lastest_partitioned_graph[node.graph.name]
        if node.operation and node.name in breakpoint_operations:
            new_graph_name = node.graph.name+f"_after_{node.name}"
            original_graph[new_graph_name] = node.graph
            graph_rank[new_graph_name] = len(final_phy_graphs)
            new_graph = Graph()
            new_graph.name = new_graph_name
            final_phy_graphs[new_graph_name] = new_graph
            lastest_partitioned_graph[node.graph.name] = new_graph_name
           
    return final_phy_graphs, graph_rank, original_graph

def _partition_one_graph_per_process(trials, phy_graph):
    # each orginal graph is in one process
    graph_names = [_.name for _ in trials]
    original_graph = {}
    graph_rank = {}

    for idx, g in enumerate(trials):
        original_graph[g.name] = g
        graph_rank[g.name] = idx
    
    final_phy_graphs = {}
    for g_name in graph_names:
        g = Graph()
        g.name = g_name
        final_phy_graphs[g_name] = g
    
    all_nodes = phy_graph.input_nodes + phy_graph.hidden_nodes + phy_graph.output_nodes
    for node in all_nodes:
        node.partition = node.graph.name
    return final_phy_graphs, graph_rank, original_graph

def _generate_final_graphs(phy_graph, final_phy_graphs, graph_rank):
    added_send_nodes = {}
    send_recv_pair = {}
    added_received_nodes = {}
    for g in final_phy_graphs:
        added_received_nodes[g] = {}
    use_distributed = False
            
    all_nodes = phy_graph.input_nodes + phy_graph.hidden_nodes + phy_graph.output_nodes 
    # generate final phy_graphs according to the partition
    for node in all_nodes:
        if node.graph.name not in final_phy_graphs:
            continue
        if node.node_type == NodeType.Input:
            final_phy_graphs[node.partition].input_nodes.append(node)
        elif node.node_type == NodeType.Hidden:
            final_phy_graphs[node.partition].hidden_nodes.append(node)
        elif node.node_type == NodeType.Output:
            final_phy_graphs[node.partition].output_nodes.append(node)
    for edge in phy_graph.edges:
        if edge.head.partition not in final_phy_graphs:
            continue
        if edge.tail.partition not in final_phy_graphs:
            continue
        if edge.head.partition == edge.tail.partition:
            final_phy_graphs[edge.head.partition].edges.append(edge)
        else:
            # an edge exists in two processes
            # use distributed communication to transfer cross-process data
            use_distributed = True
            src_graph = edge.head.partition
            dst_graph = edge.tail.partition
            
            if edge.head.name in added_received_nodes[dst_graph]:
                # Already received the data
                dst_recv_node = added_received_nodes[dst_graph][edge.head.name]
                dst_recv_edge = Edge(dst_recv_node, edge.tail)
                final_phy_graphs[dst_graph].edges.append(dst_recv_edge)
            else:
                if edge.head.name in added_send_nodes:
                    src_send_node = added_send_nodes[edge.head.name]
                else:
                    src_send_node = Node(final_phy_graphs[src_graph], NodeType.Hidden, 
                        f'send_{edge.head.name}')
                    src_send_node.set_operation(Operation.load(
                        {'type': 'Broadcast', 
                        'rank':graph_rank[src_graph], 
                        'dtype':edge.head.get_attribute('dtype'), 
                        'size':edge.head.get_attribute('shape'), 
                        'is_src': True, 'device':'\"cuda\"'})
                    )
                    added_send_nodes[edge.head.name] = src_send_node
                    src_send_edge = Edge(edge.head, src_send_node)
                    final_phy_graphs[src_graph].hidden_nodes.append(src_send_node)
                    final_phy_graphs[src_graph].edges.append(src_send_edge)

                dst_recv_node = Node(final_phy_graphs[src_graph], NodeType.Hidden, 
                    f'recv_{edge.head.name}_{edge.tail.name}')
                dst_recv_node.set_operation(Operation.load(
                    {'type': 'Broadcast', 
                    'rank': graph_rank[src_graph], 
                    'dtype':edge.head.get_attribute('dtype'), 
                    'size':edge.head.get_attribute('shape'), 
                    'is_src': False, 
                    'device':'\"cuda\"'})
                )
                
                dst_recv_edge = Edge(dst_recv_node, edge.tail)
                send_recv_pair[src_send_node.name] = dst_recv_node
                added_received_nodes[dst_graph][edge.head.name] = dst_recv_node

                final_phy_graphs[dst_graph].hidden_nodes.append(dst_recv_node)
                final_phy_graphs[dst_graph].edges.append(dst_recv_edge)
    return use_distributed

def _add_distributed_config(final_phy_graphs, graph_rank):
    for g in final_phy_graphs:
        final_phy_graphs[g].configs['env'] = {}
        final_phy_graphs[g].configs['env']['distributed_backend'] = 'nccl'
        final_phy_graphs[g].configs['env']['rank'] = graph_rank[g]
        final_phy_graphs[g].configs['env']['world_size'] = len(final_phy_graphs)


def _add_device_placement(final_phy_graphs, config, disbale_standalone=False):
    phy_graphs = []
    phy_graph_perf = {}
    for phy_graph_name in final_phy_graphs:
        g = final_phy_graphs[phy_graph_name]
        phy_graphs.append(g)
        phy_graph_perf[phy_graph_name] = profiler.profile(g)
    
    phy_graphs_in_one_gpu = _greedy_pack(phy_graphs, 
                                        phy_graph_perf, 
                                        max_mem_util=config['max_mem_util'], 
                                        max_trial_per_gpu=config['max_trial_per_gpu'],
                                        disbale_standalone=disbale_standalone)
    for gpu_id, graphs in enumerate(phy_graphs_in_one_gpu):
        for phy_graph in graphs:
            phy_graph.configs['env']['DEVICE_ID'] = gpu_id

        
        #for t in optimized_trials:
        #    print([ g[0].name + '-GPU'+ str(g[0].configs['env']['DEVICE_ID']) for g in t])
    