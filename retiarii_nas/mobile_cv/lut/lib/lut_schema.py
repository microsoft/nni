#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import abc
import copy
import pickle
import typing
from collections import namedtuple


class OpBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, rhs):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    @classmethod
    def name(cls):
        return cls.__name__

    # num of params of the op
    def get_nparams(self):
        raise NotImplementedError

    # num of flops of the op
    def get_flops(self, input_shape):
        raise NotImplementedError


Device = namedtuple("Device", ["device", "backend", "binary"])
Device.__new__.__defaults__ = ("", "", "")


class TensorShape(object):
    """ Shape of a tensor, always store in NCHW format
    """

    def __init__(self, shape):
        if isinstance(shape, TensorShape):
            shape = shape.shape
        self.shape: typing.Tuple = tuple(shape)

    @property
    def N(self):
        assert len(self.shape) == 4
        return self.shape[0]

    @property
    def C(self):
        assert len(self.shape) == 4
        return self.shape[1]

    @property
    def H(self):
        assert len(self.shape) == 4
        return self.shape[-2]

    @property
    def W(self):
        assert len(self.shape) == 4
        return self.shape[-1]

    @property
    def NHWC(self):
        assert len(self.shape) == 4
        return (self.N, self.H, self.W, self.C)

    @property
    def NCHW(self):
        return self.shape

    def __getitem__(self, index):
        return self.shape[index]

    def __len__(self):
        return len(self.shape)

    def __hash__(self):
        return hash(self.shape)

    def __eq__(self, rhs):
        return self.shape == rhs.shape

    def __repr__(self):
        return f"TensorShape({self.shape})"


class OpInfo(object):
    def __init__(self, op: OpBase, input_shapes: typing.List[TensorShape]):
        self.op = op
        self.input_shapes = [TensorShape(x) for x in input_shapes]

    def __hash__(self):
        return hash((self.op, tuple(self.input_shapes)))

    def __eq__(self, rhs):
        return self.op == rhs.op and self.input_shapes == rhs.input_shapes

    def __repr__(self):
        return f"OpInfo(op={self.op}, input_shapes={self.input_shapes})"


class LUTBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def query_op(self, op, input_shape):
        pass

    def query(self, op_infos):
        assert isinstance(op_infos, list)
        ret = []
        for x in op_infos:
            ret.append(self.query_op(x.op, x.input_shapes))
        return ret

    def query_total(self, op_infos):
        ops_latency = self.query(op_infos)
        return sum(ops_latency)


class LutItem(object):
    """ Represents a row in db """

    def __init__(
        self,
        op: OpBase = None,
        input_shapes=None,
        latency=None,
        op_info: OpInfo = None,
    ):
        has_op = op is not None and input_shapes is not None
        has_op_info = op_info is not None
        assert int(has_op) + int(has_op_info) == 1
        if has_op:
            assert isinstance(op, OpBase)
            assert isinstance(input_shapes, (list, tuple))
            op_info = OpInfo(op, input_shapes)
        assert isinstance(op_info, OpInfo)

        self.op_info = op_info
        self.latency = latency

    def get_op_info(self):
        return self.op_info

    @property
    def op(self):
        return self.op_info.op

    @property
    def input_shapes(self):
        return self.op_info.input_shapes

    def __eq__(self, rhs):
        return self.op_info == rhs.op_info and self.latency == rhs.latency


def get_mirrored_bias_items(items):
    lut = {}
    for x in items:
        if x.op_info in lut:
            lut[x.op_info].append(x.latency)
        else:
            lut[x.op_info] = [x.latency]

    mirrs = []
    for item in items:
        assert isinstance(item, LutItem)
        mi = copy.deepcopy(item)
        if "bias" not in mi.op_info.op.info:
            continue
        mi.op_info.op.bias = not mi.op_info.op.bias
        if mi.op_info in lut:
            continue
        mirrs.append(mi)

    return mirrs


class LutTable(object):
    def __init__(self, items=None):
        self.items: typing.List[LutItem] = items or []
        assert all(isinstance(x, LutItem) for x in self.items)

    def append(self, item):
        assert isinstance(item, LutItem)
        self.items.append(item)
        return self

    def extend(self, items):
        assert all(isinstance(x, LutItem) for x in items)
        self.items.extend(items)
        return self

    def __len__(self):
        return len(self.items)

    @classmethod
    def Load(cls, file_name):
        with open(file_name, "rb") as f:
            ret = pickle.load(f)
        return ret

    def save(self, file_name):
        with open(file_name, "wb") as outfile:
            pickle.dump(self, outfile)

    def __eq__(self, rhs):
        return self.items == rhs.items


class LutQuery(LUTBase):
    def __init__(self, lut_items: typing.List[LutItem], mirror_bias=True):
        self._setup(lut_items, mirror_bias)

    def query_op(self, op, input_shapes):
        op_info = OpInfo(op, input_shapes)
        assert op_info in self.lut, f"Op does not existed {op_info}"
        lats = self.lut[op_info]
        return sum(lats) / len(lats)

    def _setup(self, lut_items, mirror_bias):
        if mirror_bias:
            mirr_items = get_mirrored_bias_items(lut_items)
            lut_items = lut_items + mirr_items
            print(
                f"Added {len(mirr_items)} mirrored items, total items {len(lut_items)}."
            )

        self.lut = {}
        for x in lut_items:
            if x.op_info in self.lut:
                self.lut[x.op_info].append(x.latency)
            else:
                self.lut[x.op_info] = [x.latency]
        if len(self.lut) != len(lut_items):
            print(f"{len(lut_items) - len(self.lut)} duplicated items found: ")
            for x, y in self.lut.items():
                if len(y) > 1:
                    print(f"{x}: {y}")
