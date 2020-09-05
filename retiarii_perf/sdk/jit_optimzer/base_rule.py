from ..graph import Node, Graph

class BaseRule:
    def __init__(self):
        pass

    def logical_transform(self, logical_plan : "LogicalPlan"):
        pass

class BaseLogicalNode(Node):
    def __init__(self,
            graph: 'Graph',
            node_type: 'NodeType',
            name: 'str',
            operation: 'Optional[Operation]' = None
    ) -> None:
        Node.__init__(self, graph, node_type, name, operation= operation)

    def physical_replace(self, merged_graphs: 'List[Graph]'):
        pass
