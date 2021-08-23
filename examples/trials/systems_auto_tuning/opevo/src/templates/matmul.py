import numpy as np
import tvm
import logging
import sys, time, subprocess
from tvm import autotvm
import topi
import json
from topi.util import get_const_tuple
import os


op_attributes = {
  "N": int(os.environ['N']) if 'N' in os.environ else 1024,
  "K": int(os.environ['K']) if 'K' in os.environ else 64,
  "M": int(os.environ['M']) if 'M' in os.environ else 4096,
  "P": os.environ['P'] if 'P' in os.environ else "NN",
}

@autotvm.template
def get_template_op(**kargs):
    batch = op_attributes["N"]
    in_dim = op_attributes["K"]
    out_dim = op_attributes["M"]
    pose = op_attributes["P"]

    if pose == 'NN':
      A = tvm.placeholder((batch, in_dim), name='A', dtype="float32")
      B = tvm.placeholder((in_dim, out_dim), name='B', dtype="float32")
      k = tvm.reduce_axis((0, in_dim), name='k')
      C = tvm.compute((batch, out_dim), lambda i, j: tvm.sum(
          A[i, k] * B[k, j], axis=k), name='C')
    elif pose == 'NT':
      A = tvm.placeholder((batch, in_dim), name='A', dtype="float32")
      B = tvm.placeholder((out_dim, in_dim), name='B', dtype="float32")
      k = tvm.reduce_axis((0, in_dim), name='k')
      C = tvm.compute((batch, out_dim), lambda i, j: tvm.sum(
          A[i, k] * B[j, k], axis=k), name='C')
    elif pose == 'TN':
      A = tvm.placeholder((in_dim, batch), name='A', dtype="float32")
      B = tvm.placeholder((in_dim, out_dim), name='B', dtype="float32")
      k = tvm.reduce_axis((0, in_dim), name='k')
      C = tvm.compute((batch, out_dim), lambda i, j: tvm.sum(
          A[k, i] * B[k, j], axis=k), name='C')
    elif pose == 'TT':
      A = tvm.placeholder((in_dim, batch), name='A', dtype="float32")
      B = tvm.placeholder((out_dim, in_dim), name='B', dtype="float32")
      k = tvm.reduce_axis((0, in_dim), name='k')
      C = tvm.compute((batch, out_dim), lambda i, j: tvm.sum(
          A[k, i] * B[j, k], axis=k), name='C')
    else:
      raise

    cfg = autotvm.get_config()
    s = tvm.create_schedule(C.op)

    cfg.add_flop(batch * in_dim * out_dim * 2.0)

    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    y, x = C.op.axis
    k = CC.op.reduce_axis[0]

    cfg.define_split('K', cfg.axis(k), num_outputs=3)
    cfg.define_split('X', cfg.axis(y), num_outputs=4)
    cfg.define_split('Y', cfg.axis(x), num_outputs=4)

    ko, kt, ki = cfg['K'].apply(s, CC, k)

    block_x = tvm.thread_axis('blockIdx.x')
    block_y = tvm.thread_axis('blockIdx.y')
    thread_x = tvm.thread_axis('threadIdx.x')
    thread_y = tvm.thread_axis('threadIdx.y')

    by, tyz, ty, yi = cfg['X'].apply(s, C, y)
    bx, txz, tx, xi = cfg['Y'].apply(s, C, x)

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(tyz, tvm.thread_axis('vthread'))
    s[C].bind(txz, tvm.thread_axis('vthread'))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)

    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    s[CC].reorder(ko, kt, yo, xo, ki)
    s[CC].unroll(kt)

    for stage in [AL, BL]:
        s[stage].compute_at(s[CC], kt)

    for stage in [AA, BB]:
        s[stage].compute_at(s[CC], ko)

        fused = s[stage].fuse(*s[stage].op.axis)
        ty, tx = s[stage].split(fused, nparts=cfg['X'].size[2])
        tx, xi = s[stage].split(tx, nparts=cfg['Y'].size[2])
        _, xi = s[stage].split(xi, factor=4)

        s[stage].bind(ty, thread_y)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(xi)
        s[stage].double_buffer()

    return s, [A, B, C]
