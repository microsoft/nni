import torch
from sparta.common.sparsity import TeSA

class Algebra:
    """
    TeSA algebra for pruning
    True means non-pruned
    False means pruned
    """
    def __init__(self, val):
        self.val = bool(val)

    def __add__(self, ele):
        return Algebra(self.val or ele.val)

    def __mul__(self, ele):
        return Algebra(self.val and ele.val)


def algebra_matmul(tesa_in1: TeSA, tesa_in2: TeSA) -> TeSA:
    """
    tesa_out = tesa_in1 * tesa_in2
    (m,n) = (m,k) * (k,n)
    """
    m, k = tesa_in1.tesa.size()
    n = tesa_in2.tesa.size()[1]
    tesa_out = TeSA(torch.zeros(m, n))
    for i in range(m):
        for j in range(n):
            tmp = Algebra(tesa_out.tesa[i][j])
            for x in range(k):
                tmp += Algebra(tesa_in1.tesa[i][x]) * Algebra(tesa_in2.tesa[x][j])
            tesa_out.tesa[i][j] = tmp.val
    return tesa_out


def transpose(tesa: TeSA) -> TeSA:
    return TeSA(torch.transpose(tesa.tesa, 0, 1))

def merge_tesa(ta: TeSA, tb: TeSA) -> TeSA:
    """
    NOTE: inplace update ``ta``
    """
    m, n = ta.tesa.size()
    for i in range(m):
        for j in range(n):
            ta.tesa[i][j] = ta.tesa[i][j] and tb.tesa[i][j]
    return ta

def propagate_matmul(tesa_in1: TeSA, tesa_in2: TeSA, tesa_out: TeSA) -> tuple[TeSA, TeSA, TeSA]:
    """
    sparsity propagation on matmul, both forward and backward
    """
    # forward propagation
    prop_tesa_out = algebra_matmul(tesa_in1, tesa_in2)
    tesa_out = merge_tesa(tesa_out, prop_tesa_out)
    # backward propagation
    prop_tesa_in1 = algebra_matmul(tesa_out, transpose(tesa_in2))
    tesa_in1 = merge_tesa(tesa_in1, prop_tesa_in1)
    prop_tesa_in2 = algebra_matmul(transpose(tesa_in1), tesa_out)
    tesa_in2 = merge_tesa(tesa_in2, prop_tesa_in2)
    return tesa_in1, tesa_in2, tesa_out