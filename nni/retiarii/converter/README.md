# PyTorch Graph Converter

## Namespace for PyTorch Graph

We should have a concrete rule for specifying nodes in graph with namespace.

Each node has a name, either specified or generated. The nodes in the same hierarchy cannot have the same name.

* The name of module node natively follows this rule, because we use variable name for instantiated modules like what PyTorch graph does.

* For the nodes created in `forward` function, we use a global sequence number.

### Namespace for mutated (new) nodes

TBD

## Graph Simplification

TBD

## Node Types

We define concrete type string for each node type.

## Module's Input Arguments

We use wrapper to obtain the input arguments of modules. Users need to use our wrapped "nn" and wrapped "Module".

## Control Flow

### for loop

Currently, we only support `ModuleList` (`ModuleDict`) based for loop, which is automatically unfolded by TorchScript. That is to say, we do not support loop in TorchScript for now.

### if/else

For now, we only deal with the case that the condition is constant or attribute. In this case, only one branch is kept during generating the graph.