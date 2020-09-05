# FBNet Model Builder

We provide a general model builder ([fbnet_v2/fbnet_builder.py](fbnet_v2/fbnet_builder.py)) to build a pytorch model from an architecture defined as a dict.

## Code Structure

* [fbnet_v2/fbnet_builder.py](fbnet_v2/fbnet_builder.py): FBNet model builder to build pytorch model from a given architecture defined as a dict.
* [fbnet_v2/blocks_factory.py](fbnet_v2/blocks_factory.py): Building block factory.
* fbnet_v2/fbnet_model_cls_*.py: Pre-defined model architectures.
  * [fbnet_v2/fbnet_model_cls.py](fbnet_v2/fbnet_model_cls.py): Model architectures for baseline and FBNet models
  * [fbnet_v2/fbnet_model_cls_dmasking.py](fbnet_v2/fbnet_model_cls_dmasking.py): Model architectures for FBNetV2 models (DMaskingNet)
  * [fbnet_v2/fbnet_model_cls_efficient_net.py](fbnet_v2/fbnet_model_cls_efficient_net.py): Model architectures for Efficient Net

## Model Architecture Definition

We define a model architecture as a dict like the following:

```python
e6 = {"expansion": 6}
no_bias = {"bias": False}
backbone = [
    # [op, c, s, n, ...]
    # stage 0
    [("conv_k3", 32, 2, 1)],
    # stage 1
    [
        ("ir_k3", 64, 2, 2, e6, no_bias),
        ("ir_k5", 96, 1, 1, e6)
    ],
]
```
Here `backbone` is the defined architecture with two `stages`, and each stage has one or more `building blocks`.

**Building blocks**

A building block `block` is represented as a tuple with four or more elements:
```python
    # [op, c, s, n, ...]
    block = ("ir_k3", 64, 2, 2, e6, no_bias)
```
where
  * `op` is the name of the block
  * `c` is the block output channel size,
  * `s` is the stride of the block,
  * `n` represents the number of repeats for this block.

Note that when `n` > 1, the stride `s` of the repeated blocks will be set to 1. For example, blocks `[("ir_k5", 96, 2, 2, e6)]` is equivalent to `[("ir_k5", 96, *2*, 1, e6), ("ir_k5", 96, *1*, 1, e6)]`.

Any additional arguments represent as dicts of argument pairs after `n` (like
`e6`, `no_bias` etc.) will be merged together in the order of appearance and pass
to the op's constructor.

All the supported building blocks are defined in [fbnet_v2/blocks_factory.py](fbnet_v2/blocks_factory.py) and additional blocks could be registered dynamically.

**Model architecture**

A list of building blocks represents a stage of the network
```python
    stage1 = [
        # block 0
        ("ir_k3", 64, 2, 2, e6, no_bias),
        # block 1
        ("ir_k5", 96, 1, 1, e6),
        ...
    ]
```
and a list of stages represent the architecture:
```python
    backbone = [
        # stage 0
        [("conv_k3", 32, 2, 1)],
        # stage 1
        [
            ("ir_k3", 64, 2, 2, e6, no_bias),
            ("ir_k5", 96, 1, 1, e6)
        ],
        ...
    ]
```
Note that the grouping of blocks to stages is only for convenience and does not provide additional information to the architecture definition. We usually group all
the blocks that apply on the same spatial resolution feature map into the same stage.

## Model Builder

We provide the following functions/classes to parse the above definition:

```python
from mobile_cv.arch.fbnet_v2 import fbnet_builder as mbuilder

e6 = {"expansion": 6}
bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
arch_def = {
    # global arguments that will be applied to every op in the arch
    basic_args = {
        "relu_args": "swish",
    },
    "blocks": [
        # [op, c, s, n, ...]
        # stage 0
        [
            ("conv_k3", 4, 2, 1, bn_args)
        ],
        # stage 1
        [
            ("ir_k3", 8, 2, 2, e6, bn_args),
            ("ir_k5_sehsig", 8, 1, 1, e6, bn_args)
        ],
    ],
}
# unify architecture definition
arch_def = mbuilder.unify_arch_def(arch_def, ["blocks"])
# create builder
builder = mbuilder.FBNetBuilder(1.0)
# add global arguments
builder.add_basic_args(basic_args)
# build `nn.Module` from `blocks`
model = builder.build_blocks(arch_def["blocks"], dim_in=3)
# evaluation mode
model.eval()
```
Here
* `mbuilder.unify_arch_def` converts the arch definition to a way easier to handle later
  (convert to list of blocks, expand the repeats etc.). Only the dicts specified in the second argument will be unified and the rest will be unchanged.
* `builder.build_blocks(unified_arch_def["blocks"])` to create a `nn.Module` that corresponds to the architecture defined in `blocks`.
* We support specifying global default arguments to the builder that will be later override by each op by using `FBNetBuilder.add_basic_args(basic_args)`. Some common global arguments like the batch norm type and width divisor could be passed from the `FBNetBuilder` constructor as well.
