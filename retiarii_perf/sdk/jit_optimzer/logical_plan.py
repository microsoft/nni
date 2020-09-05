from ..graph import Graph
import copy

class LogicalPlan:
    def __init__(self):
        self.all_trials = []
        self.node_mapping = {} # graph_id, node_id -> logical_plan
        self.graph = Graph()


def logical_plan_generator(trial_batch : 'List[Graph]') -> 'LogicalPlan':
    raw_lp = LogicalPlan()
    for graph_idx, trial in enumerate(trial_batch):
        raw_lp.all_trials.append(trial)
        for node in trial.input_nodes:
            new_node = copy.copy(node)
            new_node.name = f"logical_g{graph_idx}_" + node.name
            raw_lp.graph.input_nodes.append(new_node)
        for node in trial.output_nodes:
            new_node = copy.copy(node)
            new_node.name = f"logical_g{graph_idx}_" + node.name
            raw_lp.graph.output_nodes.append(new_node)
        for node in trial.hidden_nodes:
            new_node = copy.copy(node)
            new_node.name = f"logical_g{graph_idx}_" + node.name
            raw_lp.graph.hidden_nodes.append(new_node)
        for edge in trial.edges:
            new_edge = copy.copy(edge)
            new_edge.head = \
                raw_lp.graph.find_node(f"logical_g{graph_idx}_" + edge.head.name)
            new_edge.tail = \
                raw_lp.graph.find_node(f"logical_g{graph_idx}_" + edge.tail.name)
            raw_lp.graph.edges.append(new_edge)
    return raw_lp