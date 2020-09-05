from .base_rule import BaseRule, BaseLogicalNode
from ..graph import node_hash
from ..graph import NodeType, Node, Edge, Graph
import copy
from ..operations import *

BATCHABLE_OPS = ['Conv2d','BatchNorm2d','Linear','aten::relu', 'aten::avg_pool2d', 'aten::size', 'aten::Int', 'aten::view', 'aten::add', 'aten::relu']
class BatchNode(BaseLogicalNode):
    def __init__(self, dup_ops : "List[Node]", graph, cfg):
        BaseLogicalNode.__init__(self,
            graph = graph,
            node_type = NodeType.Logical,
            name = f'batch_{dup_ops[0].name}')
        self.dup_ops = dup_ops.copy()
        self.cfg = cfg
        self.water_mark = {}
        for node in self.dup_ops:
            water_mark_name = '_'.join(node.name.split('_')[2:]) \
                                    + '!' + node.graph.name
            self.water_mark[water_mark_name] = node
        

    def physical_replace(self, graphs : "List[Graph]"):
        all_graph_names = [_.name for _ in graphs]
        for op in self.dup_ops:
            if op.graph.name in all_graph_names:
                graph_to_replace = Graph()
                phy_op = copy.copy(op)
                if 'logical_g' in phy_op.name:
                    phy_op.name = "_".join(phy_op.name.split('_')[2:])
                phy_op.set_attribute('batch', op.name)
                phy_op.graph = graph_to_replace
                graph_to_replace.hidden_nodes.append(phy_op)
                
                logical_preds = self.graph.get_predecessors(self)
                
                pre_view_added = False
                for pred in logical_preds:
                    input_node = Node(graph_to_replace, NodeType.Input, phy_op.name + '_input_'+ pred.name)
                    graph_to_replace.input_nodes.append(input_node)
                    if not isinstance(pred, BatchNode):
                        expand_node = Node(graph_to_replace, 
                                            NodeType.Hidden, 
                                            f"expand_{pred.name}", 
                                            Operation.new('Expand', num_copies=len(graphs))
                                        )
                        graph_to_replace.hidden_nodes.append(expand_node)
                        graph_to_replace.edges.append(Edge(input_node, expand_node)) # xs = torch.cat([x] * num_batch)
                        input_node = expand_node
                    if phy_op.node_type==NodeType.Output or phy_op.get_attribute('non-trainable'):
                        graph_to_replace.edges.append(Edge(input_node, phy_op))
                    else:
                        assert(pre_view_added == False) # FIXME: trainable node should be single input
                        pre_view_added = True
                        op_pre_view_node = Node(graph_to_replace, 
                                        NodeType.Hidden, 
                                        f'{phy_op.name}_BatchSizeView_pre', 
                                        Operation.new('BatchSizeView', batch_size=self.cfg['batch_size'])
                                        )
                        graph_to_replace.hidden_nodes.append(op_pre_view_node)
                        assert(phy_op.operation.type=='Conv2d') # only support batching conv2d for now
                        if phy_op.operation.type=='Conv2d':
                            phy_op.operation.params['in_channels'] *= len(graphs)
                            phy_op.operation.params['out_channels'] *= len(graphs)
                            phy_op.operation.params['groups'] = len(graphs)
                        graph_to_replace.edges.append(Edge(input_node, op_pre_view_node))
                        graph_to_replace.edges.append(Edge(op_pre_view_node, phy_op))
                output_node = Node(graph_to_replace, NodeType.Output, phy_op.name + '_output')
                graph_to_replace.output_nodes.append(output_node)
                if pre_view_added:
                    view_node = Node(graph_to_replace, 
                                    NodeType.Hidden, 
                                    f'{phy_op.name}_BatchSizeView', 
                                    Operation.new('BatchSizeView', 
                                        batch_size=self.cfg['batch_size']*len(graphs))
                                    )
                    graph_to_replace.hidden_nodes.append(view_node)
                    graph_to_replace.edges.append(Edge(phy_op, view_node))
                    graph_to_replace.edges.append(Edge(view_node, output_node))
                else:
                    graph_to_replace.edges.append(Edge(phy_op, output_node))
                
                return graph_to_replace
        assert(False)

class BatchRule(BaseRule):
    def __init__(self, cfg):
        self.batched_nodes = {}
        self.cfg = cfg

    def _type_check(self, op):
        # is_freezed = op.get_attribute('non-trainable')
        
        if op.node_type in [ NodeType.Input, NodeType.Logical]:
            return False
        
        if node_hash(op) in self.batched_nodes:
            return False
        
        if op.node_type == NodeType.Output:
            return True
        
        if op.operation.type not in BATCHABLE_OPS:
            return False
        
        return True 
    
    def _check_batchable(self, logical_plan : "LogicalPlan", 
                                  op : "Node") -> "Bool":
        
        if self._type_check(op) == False:
            return False
        else:
            return True
        # all_pred_deduplicated = True
        # for pred_op in logical_plan.graph.get_predecessors(op):
        #     #node_hash(pred_op) not in self.deduplicated_nodes:
        #     if not isinstance(pred_op, DedupNode): 
        #         all_pred_deduplicated = False
        #         break
        # if all_pred_deduplicated:
        #     return True
        # else:
        #     return False
        
    def logical_transform(self, logical_plan : "LogicalPlan"):
        all_nodes = []
        
        all_nodes.extend(logical_plan.graph.input_nodes)
        topo_sorted_hidden_nodes = logical_plan.graph.topo_sort()
        assert(len(topo_sorted_hidden_nodes) == len(logical_plan.graph.hidden_nodes)) 
        all_nodes.extend(topo_sorted_hidden_nodes)
        # all_nodes.extend(logical_plan.graph.hidden_nodes)
        all_nodes.extend(logical_plan.graph.output_nodes)
        for node in all_nodes:
            if self._check_batchable(logical_plan, node):
                node_pred_hash = \
                    set([ node_hash(_) \
                        for _ in logical_plan.graph.get_predecessors(node)])

                ops_with_same_hash = logical_plan.graph.find_multiple_nodes(
                    hashval = node_hash(node)
                )
                
                ops_to_batch = []
                for other_op in ops_with_same_hash:
                    # Trainable hidden nodes are not duplicable
                    
                    if self._type_check(other_op) == False:
                        continue
                    
                    other_op_preds = logical_plan.graph.get_predecessors(other_op)
                    check_failed = False
                    for other_pred in other_op_preds:
                        if node_hash(other_pred) not in node_pred_hash:
                            check_failed = True
                            break
                    if check_failed == False:
                        ops_to_batch.append(other_op)
                
                if len(ops_to_batch) > 1:
                    logical_op_batch = BatchNode(ops_to_batch, logical_plan.graph, self.cfg)
                    name_of_ops = set([_.name for _ in ops_to_batch])
                    added_edge = set()
                    edge_to_remove = []
                    for edge in logical_plan.graph.edges:
                        if edge.head.name in name_of_ops:
                            edge.head = logical_op_batch
                        if edge.tail.name in name_of_ops:
                            if (edge.head.name, logical_op_batch.name) in added_edge:
                                edge_to_remove.append(edge)
                            else:
                                edge.tail = logical_op_batch
                        added_edge.add((edge.head.name, edge.tail.name))
                    logical_plan.graph.edges = \
                        [_ for _ in logical_plan.graph.edges \
                            if _ not in edge_to_remove]
                    for _ in ops_to_batch:
                        self.batched_nodes[node_hash(_)] = logical_op_batch
                    if node.node_type == NodeType.Input:
                        for _ in ops_to_batch:
                            logical_plan.graph.input_nodes.remove(_)
                        logical_plan.graph.input_nodes.append(logical_op_batch)
                    elif node.node_type == NodeType.Hidden:
                        for _ in ops_to_batch:
                            logical_plan.graph.hidden_nodes.remove(_)
                        logical_plan.graph.hidden_nodes.append(logical_op_batch)
                    elif node.node_type == NodeType.Output:
                        for _ in ops_to_batch:
                            logical_plan.graph.output_nodes.remove(_)
                        logical_plan.graph.output_nodes.append(logical_op_batch)
                    for edge in logical_plan.graph.edges:
                        if edge.head in ops_to_batch:
                            edge.head = logical_op_batch
                        if edge.tail in ops_to_batch:
                            edge.tail = logical_op_batch
                    