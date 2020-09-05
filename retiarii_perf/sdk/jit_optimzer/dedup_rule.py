from .base_rule import BaseRule, BaseLogicalNode
from ..graph import node_hash
from ..graph import NodeType, Node
import copy

class DedupNode(BaseLogicalNode):
    def __init__(self, dup_ops : "List[Node]", graph):
        BaseLogicalNode.__init__(self,
            graph = graph,
            node_type = NodeType.Logical,
            name = f'dedup_{dup_ops[0].name}')
        self.dup_ops = dup_ops.copy()
        self.water_mark = {}
        for node in self.dup_ops:
            water_mark_name = '_'.join(node.name.split('_')[2:]) \
                                    + '!' + node.graph.name
            self.water_mark[water_mark_name] = node

    def physical_replace(self, graphs : "List[Graph]"):
        all_graph_names = [_.name for _ in graphs]
        for op in self.dup_ops:
            if op.graph.name in all_graph_names:
                phy_op = copy.copy(op)
                if 'logical_g' in phy_op.name:
                    phy_op.name = "_".join(phy_op.name.split('_')[2:])
                phy_op.set_attribute('dedup', op.name)
                return phy_op
        assert(False)

class DeduplicationRule(BaseRule):
    def __init__(self, cfg):
        self.deduplicated_nodes = {}
        self.cfg = cfg

    def _type_check(self, op):
        is_freezed = op.get_attribute('non-trainable')
        
        if op.node_type == NodeType.Hidden and is_freezed == None:
            return False
        
        if op.node_type == NodeType.Hidden and is_freezed == False:
            return False
        
        if op.node_type == NodeType.Output:
            return False
        
        if node_hash(op) in self.deduplicated_nodes:
            return False
        
        return True 
    def _check_deduplicable(self, logical_plan : "LogicalPlan", 
                                  op : "Node") -> "Bool":
        # Trainable hidden nodes are not duplicable
        # Already duplicated nodes are not duplicable
        
        if self._type_check(op) == False:
            return False
        
        all_pred_deduplicated = True
        for pred_op in logical_plan.graph.get_predecessors(op):
            #node_hash(pred_op) not in self.deduplicated_nodes:
            if not isinstance(pred_op, DedupNode): 
                all_pred_deduplicated = False
                break
        if all_pred_deduplicated:
            return True
        else:
            return False
        
    def logical_transform(self, logical_plan : "LogicalPlan"):
        all_nodes = []
        
        all_nodes.extend(logical_plan.graph.input_nodes)
        topo_sorted_hidden_nodes = logical_plan.graph.topo_sort()
        assert(len(topo_sorted_hidden_nodes) == len(logical_plan.graph.hidden_nodes)) 
        all_nodes.extend(topo_sorted_hidden_nodes)
        # all_nodes.extend(logical_plan.graph.hidden_nodes)
        all_nodes.extend(logical_plan.graph.output_nodes)
        
        for node in all_nodes:
            if self._check_deduplicable(logical_plan, node):
                node_pred_hash = \
                    set([ node_hash(_) \
                        for _ in logical_plan.graph.get_predecessors(node)])

                ops_with_same_hash = logical_plan.graph.find_multiple_nodes(
                    hashval = node_hash(node)
                )
                
                ops_to_dedup = []
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
                        ops_to_dedup.append(other_op)
                
                if len(ops_to_dedup) > 1:
                    logical_op_dedup = DedupNode(ops_to_dedup, logical_plan.graph)
                    name_of_ops = set([_.name for _ in ops_to_dedup])
                    added_edge = set()
                    edge_to_remove = []
                    for edge in logical_plan.graph.edges:
                        if edge.head.name in name_of_ops:
                            edge.head = logical_op_dedup
                        if edge.tail.name in name_of_ops:
                            if (edge.head.name, logical_op_dedup.name) in added_edge:
                                edge_to_remove.append(edge)
                            else:
                                edge.tail = logical_op_dedup
                        added_edge.add((edge.head.name, edge.tail.name))
                    logical_plan.graph.edges = \
                        [_ for _ in logical_plan.graph.edges \
                            if _ not in edge_to_remove]
                    for _ in ops_to_dedup:
                        self.deduplicated_nodes[node_hash(_)] = logical_op_dedup
                    if node.node_type == NodeType.Input:
                        for _ in ops_to_dedup:
                            logical_plan.graph.input_nodes.remove(_)
                        logical_plan.graph.input_nodes.append(logical_op_dedup)
                    elif node.node_type == NodeType.Hidden:
                        for _ in ops_to_dedup:
                            logical_plan.graph.hidden_nodes.remove(_)
                        logical_plan.graph.hidden_nodes.append(logical_op_dedup)
                    for edge in logical_plan.graph.edges:
                        if edge.head in ops_to_dedup:
                            edge.head = logical_op_dedup
                        if edge.tail in ops_to_dedup:
                            edge.tail = logical_op_dedup
                    