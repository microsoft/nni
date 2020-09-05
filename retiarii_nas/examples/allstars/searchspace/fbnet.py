import torch

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet as fbnet_base
from mobile_cv.arch.fbnet_v2.modeldef_utils import e1, e3, e4, e6
from mobile_cv.arch.fbnet_v2.fbnet_modeldef_cls import MODEL_ARCH
from mobile_cv.arch.fbnet_v2.fbnet_builder import WrapperOp

from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph


__all__ = ['fbnet', 'WrapperOp']


MODEL_ARCH.register("fbnet_no_skip", {
    "input_size": 224,
    "blocks": [
        # [op, c, s, n, ...]
        # stage 0
        [["conv_k3", 16, 2, 1]],
        # stage 1
        [["ir_k3", 16, 1, 1, e1]],
        # stage 2
        [
            ["ir_k3", 24, 2, 1, e6],
            ["ir_k3", 24, 1, 1, e1],
            ["ir_k3", 24, 1, 1, e1],
            ["ir_k3", 24, 1, 1, e1],
        ],
        # stage 3
        [
            ["ir_k5", 32, 2, 1, e6],
            ["ir_k5", 32, 1, 1, e3],
            ["ir_k5", 32, 1, 1, e6],
            ["ir_k3", 32, 1, 1, e6],
        ],
        # stage 4
        [
            ["ir_k5", 64, 2, 1, e6],
            ["ir_k5", 64, 1, 1, e3],
            ["ir_k5", 64, 1, 1, e6],
            ["ir_k5", 64, 1, 1, e6],
            ["ir_k5", 112, 1, 1, e6],
            ["ir_k5", 112, 1, 1, e6],
            ["ir_k5", 112, 1, 1, e6],
            ["ir_k5", 112, 1, 1, e3],
        ],
        # stage 5
        [
            ["ir_k5", 184, 2, 1, e6],
            ["ir_k5", 184, 1, 1, e6],
            ["ir_k5", 184, 1, 1, e6],
            ["ir_k5", 184, 1, 1, e6],
            ["ir_k3", 352, 1, 1, e6],
        ],
        # stage 6
        [("conv_k1", 1984, 1, 1)],
    ]
})


class BlockMutator(Mutator):
    def __init__(self, target):
        self.target = target

    def mutate(self, graph):
        target_node = graph.find_node(self.target)
        basic_kwargs = target_node.operation.params['kwargs']
        allow_skip = ['skip'] if basic_kwargs['stride'] == 1 else []
        op_type = self.choice(['ir_k3_e1', 'ir_k3_g2_e1', 'ir_k3_e3', 'ir_k3_e6',
                               'ir_k5_e1', 'ir_k5_g2_e1', 'ir_k5_e3', 'ir_k5_e6'] + allow_skip)
        if any(op_type.endswith(e) for e in ['e1', 'e3', 'e4', 'e6']):
            kwargs = {'expansion': int(op_type[-1])}
            op_type = op_type[:-3]
        else:
            kwargs = {}
        target_node.update_operation(None, block_op=op_type, kwargs={**basic_kwargs, **kwargs})


def fbnet():
    model = fbnet_base("fbnet_no_skip", pretrained=False)  # choose this as it skips no ops
    blocks = []
    for name, module in model.named_modules():
        if isinstance(module, WrapperOp):
            blocks.append(name)
    blocks = blocks[1:-1]  # pop the first one and the last one
    model_graph = gen_pytorch_graph(model, dummy_input=(torch.randn(1, 3, 224, 224),), collapsed_nodes={
        name: 'WrapperOp' for name in blocks
    })

    mutators = [BlockMutator(block) for block in blocks]
    return model_graph, mutators
