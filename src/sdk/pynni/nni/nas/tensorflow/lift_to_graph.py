import collections

from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity


UnliftableError = op_selector.UnliftableError


def _as_operation(op_or_tensor):
    return op_or_tensor.op if isinstance(op_or_tensor, ops.Tensor) else op_or_tensor

def _constant_inputs(op_or_tensor):
    for i in op_selector.graph_inputs(_as_operation(op_or_tensor)):
        if _as_operation(i).type != 'Const' or _as_operation(i).control_inputs:
            return False
    return True


_InputMutation = collections.namedtuple("_InputMutation", ["copied_op", "input_index", "old_graph_tensor"])
_ControlMutation = collections.namedtuple("_ControlMutation", ["copied_op", "old_graph_op"])


def _copy_non_source(op, graph, op_map, base_graph):
    input_mutations = []
    control_mutations = []
    copied_inputs = []
    for input_index, original_input in enumerate(op.inputs):
        copied_input = op_map.get(original_input, None)
        if copied_input is None:
            copied_input = array_ops.placeholder(name="unused_control_flow_input", shape=original_input.shape, dtype=original_input.dtype)
            input_mutations.append(_InputMutation(copied_op=None, input_index=input_index, old_graph_tensor=original_input))
        copied_inputs.append(copied_input)

    copied_control_inputs = []
    for original_control_input in op.control_inputs:
        copied_control_input = op_map.get(original_control_input, None)
        if copied_control_input is None:
            control_mutations.append(_ControlMutation(copied_op=None, old_graph_op=original_control_input))
        else:
            copied_control_inputs.append(copied_control_input)

    with ops.control_dependencies(copied_control_inputs), ops.device(op.device):
        f = base_graph._functions.get(op.type, None)
        if f is not None and compat.as_str(f.name) not in graph._functions:
            f.add_to_graph(graph)

        attrs = {k: v for k, v in op.node_def.attr.items() if not k.startswith("_class") and not k.startswith("_tpu_replicate")}
        copied_op = graph.create_op(op_type=op.type, inputs=copied_inputs, dtypes=[x.dtype for x in op.outputs], attrs=attrs, name=op.name)
    op_map[op] = copied_op
    for i, o in enumerate(op.outputs):
        op_map[o] = copied_op.outputs[i]

    inputs = [mutation._replace(copied_op=copied_op) for mutation in input_mutations]
    controls = [mutation._replace(copied_op=copied_op) for mutation in control_mutations]
    return inputs, controls


def _copy_source(s, graph, op_map, handle_captures, inverse_captures, base_graph):
    if handle_captures and s in inverse_captures:
        copied_placeholder = graph.capture(inverse_captures[s], name=s.op.name)
    elif s.op.type == "PlaceholderWithDefault" and _constant_inputs(s):
        default_value = s.op.inputs[0]
        unavailable_inputs, unavailable_control_inputs = _copy_non_source(
                op=default_value.op, graph=graph, op_map=op_map,
                base_graph=base_graph)
        if unavailable_inputs or unavailable_control_inputs:
            raise AssertionError("Could not copy source node {} because it has inputs.".format(default_value))

        with ops.device(s.op.device):
            copied_placeholder = array_ops.placeholder_with_default(input=op_map[default_value], shape=s.shape, name=s.op.name)
    else:
        with ops.device(s.op.device):
            copied_placeholder = array_ops.placeholder(dtype=s.dtype, shape=s.shape, name=s.op.name)

    base_handle = resource_variable_ops.get_resource_handle_data(s)
    if base_handle.shape_and_type:
        resource_variable_ops._set_handle_shapes_and_types(copied_placeholder, base_handle, graph_mode=True)

    op_map[s] = copied_placeholder
    op_map[s.op] = copied_placeholder.op


def lift_to_graph(tensors, graph, sources=None, op_map=None):
    variable_init_tensors = []
    init_tensors = []
    for tensor in tensors:
        if isinstance(tensor, resource_variable_ops.ResourceVariable):
            variable_init_tensors.append(tensor)
        else:
            init_tensors.append(tensor)
    base_graph = base_graph or init_tensors[0].graph
    op_map = op_map or object_identity.ObjectIdentityDictionary()

    sources = object_identity.ObjectIdentitySet(sources or [])
    visited_ops = set(x.op for x in sources)
    op_outputs = collections.defaultdict(set)

    for init_tensor in init_tensors:
        sources.update(op_selector.map_subgraph(
            init_tensor=init_tensor,
            sources=sources,
            disallowed_placeholders=disallowed_placeholders,
            visited_ops=visited_ops,
            op_outputs=op_outputs,
            add_sources=add_sources
        ))

    ops_to_copy = []
    marked_ops = set([])
    ops_to_visit = [_as_operation(t) for t in init_tensors
                                    if not op_outputs[_as_operation(t)]]
    unvisited_ops = set(ops_to_visit)
    while unvisited_ops:
        while ops_to_visit:
            op = ops_to_visit.pop()
            if op in marked_ops:
                continue
            marked_ops.add(op)
            ops_to_copy.append(op)
            for inp in op_selector.graph_inputs(op):
                if inp.name == "TPUReplicateMetadata":
                    continue
                unvisited_ops.add(inp)
                if (all(x in marked_ops for x in op_outputs[inp]) and inp not in sources):
                    ops_to_visit.append(inp)
        unvisited_ops.difference_update(marked_ops)
        if unvisited_ops:
            ops_to_visit.append(next(iter(unvisited_ops)))

    captures = []
    inverse_captures = object_identity.ObjectIdentityDictionary()
    internal_captures = []
    if (isinstance(base_graph, func_graph.FuncGraph) and isinstance(graph, func_graph.FuncGraph)):
        captures = base_graph.captures
        for external_capture, internal_capture in captures:
            inverse_captures[internal_capture] = external_capture
        internal_captures = base_graph.internal_captures

    with graph.as_default():
        for i in variable_init_tensors:
            op_map[i] = i
        source_ops = set()
        for s in internal_captures:
            if s in sources:
                sources.remove(s)
                source_ops.add(s.op)
                _copy_source(
                    s=s,
                    graph=graph,
                    op_map=op_map,
                    handle_captures=handle_captures,
                    inverse_captures=inverse_captures,
                    base_graph=base_graph
                )
        for s in sources:
            source_ops.add(s.op)
            _copy_source(
                s=s,
                graph=graph,
                op_map=op_map,
                handle_captures=handle_captures,
                inverse_captures=inverse_captures,
                base_graph=base_graph
            )

        input_mutations = []
        control_mutations = []
        for op in reversed(ops_to_copy):
            if op in source_ops or op in op_map:
                continue
            new_input_mutations, new_control_mutations = _copy_non_source(op=op, graph=graph, op_map=op_map, base_graph=base_graph)
            input_mutations.extend(new_input_mutations)
            control_mutations.extend(new_control_mutations)

        with graph._mutation_lock():
            for mutation in input_mutations:
                mutation.copied_op._update_input(mutation.input_index, op_map[mutation.old_graph_tensor])
            for mutation in control_mutations:
                if mutation.old_graph_op.name == "TPUReplicateMetadata":
                    continue
                mutation.copied_op._add_control_input(op_map[mutation.old_graph_op])

        return op_map
