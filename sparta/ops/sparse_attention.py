# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging
from torch.utils.cpp_extension import load as module_load
from .sparse_opbase import SparseOPBase
from .Template.SparseAttention import *
from sparta.common.utils import *

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
our_sparse_attention = None


class SparseAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        d_m_index,
        d_n_index,
        d_block_index,
        val,
        row_ptr,
        col_idx,
        val_mask,
        col_range_index,
        gradv_row_idx,
        gradv_col_idx,
        gradv_subblock_idx
    ):
        ctx.save_for_backward(
            Q,
            K,
            V,
            val,
            gradv_row_idx,
            gradv_col_idx,
            gradv_subblock_idx,
            d_m_index,
            d_n_index,
            d_block_index,
            col_range_index,
            row_ptr,
            col_idx
        )

        return our_sparse_attention.forward(
            Q,
            K,
            V,
            d_m_index,
            d_n_index,
            d_block_index,
            val,
            row_ptr,
            col_idx,
            val_mask,
            col_range_index,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            Q,
            K,
            V,
            val,
            gradv_row_idx,
            gradv_col_idx,
            gradv_subblock_idx,
            m_index,
            n_index,
            block_index,
            col_range_index,
            row_ptr,
            col_idx
        ) = ctx.saved_tensors
        assert len(grad_outputs) == 1
        grad_q, grad_k, grad_v, qxk_grad, val_grad = our_sparse_attention.backward(
            grad_outputs[0],
            Q,
            K,
            V,
            gradv_row_idx,
            gradv_col_idx,
            gradv_subblock_idx,
            val,
            m_index,
            n_index,
            block_index,
            col_range_index,
            row_ptr,
            col_idx
        )

        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None, None, None, None


class SparseAttention(SparseOPBase):
    """
    The Sparse Attention module.
    """

    def __init__(self, out_mask, HEAD_NUM, seq_len, hidden_dim):
        super(SparseAttention, self).__init__()
        assert isinstance(out_mask, torch.Tensor)
        out_mask
        self.HEAD_NUM = HEAD_NUM
        self.M = seq_len
        self.N = seq_len
        self.K = hidden_dim
        err_msg = 'Currently, seq_len and hidden_dim should be divisible by 32'
        assert seq_len % 32 == 0, err_msg
        assert hidden_dim % 32 == 0, err_msg
        # currently only support 32 x 64
        self.block_size_h = 32
        self.block_size_w = 64
        self.target_device = None
        # build the index used for the kernel
        self.specialize(out_mask)

    def specialize(self, out_mask):
        self.out_mask = out_mask
        _logger.info("Preprocess the index of the sparse pattern")
        with torch.no_grad():
            self.row_ptr, self.col_index, self.val_mask = convert_bcsr(
                self.out_mask, self.out_mask, self.block_size_h, self.block_size_w)

            self.val_mask = self.val_mask.to(torch.int32)
            self.val_size = self.val_mask.numel() + self.block_size_h * self.block_size_w
            # build the index for original csr format
            self.csr_index = {}
            for row_id in range(self.row_ptr.size(0)-1):
                self.csr_index[row_id] = {}
                _start = self.row_ptr[row_id]
                _end = self.row_ptr[row_id+1]
                for _pos in range(_start, _end):
                    col_id = self.col_index[_pos].item()
                    self.csr_index[row_id][col_id] = _pos
            self._m_index, self._n_index, self._block_index, self._col_range_index = self._build_forward_index()
            # following indexes are for backward function
            self._gradv_row_idx, self._gradv_col_idx, self._gradv_subblock_idx = self._build_backward_index()
            self.register_buffer('m_index', self._m_index)
            self.register_buffer('n_index', self._n_index)
            self.register_buffer('block_index', self._block_index)
            self.register_buffer('col_range_index', self._col_range_index)
            self.register_buffer('gradv_row_idx', self._gradv_row_idx)
            self.register_buffer('gradv_col_idx', self._gradv_col_idx)
            self.register_buffer('gradv_subblock_idx', self._gradv_subblock_idx)

        self.load_kernel_library()


    def forward(self, Q, K, V):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        """
        # need create val each time
        assert isinstance(Q, torch.Tensor)
        assert isinstance(K, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        if self.target_device != Q.device:
            self.target_device = Q.device
            self._move_index(self.target_device)
        batch_size = Q.size(0)
        val = torch.zeros(batch_size * self.HEAD_NUM * self.val_size,
                          dtype=torch.float32, device=self.target_device)
        result = SparseAttentionFunction.apply(Q, K, V,
                                               self.m_index,
                                               self.n_index,
                                               self.block_index,
                                               val,
                                               self.row_ptr,
                                               self.col_index,
                                               self.val_mask,
                                               self.col_range_index,
                                               # following are for backwards
                                               self.gradv_row_idx,
                                               self.gradv_col_idx,
                                               self.gradv_subblock_idx)

        return result

    def reference_forward(self, Q, K, V):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        add_mask = torch.zeros(self.out_mask.size()).to(self.target_device)
        add_mask[self.out_mask == 0] = float(-inf)
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        added = torch.add(dots, add_mask)
        attn = added.softmax(dim=-1)
        ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)

        return ref_out

    def _move_index(self, target_device):
        with torch.no_grad():
            # move the index tensors to the target device
            self.row_ptr, self.col_index, self.val_mask, self.m_index, self.n_index, self.block_index,\
                self.col_range_index, self.gradv_row_idx, self.gradv_col_idx, self.gradv_subblock_idx =\
                self.row_ptr.to(target_device), self.col_index.to(target_device), self.val_mask.to(target_device),\
                self.m_index.to(target_device), self.n_index.to(target_device), self.block_index.to(target_device),\
                self.col_range_index.to(target_device), self.gradv_row_idx.to(target_device), self.gradv_col_idx.to(target_device),\
                self.gradv_subblock_idx.to(target_device)


    def _build_forward_index(self):
        block_idx = []
        m_idx = []
        n_idx = []
        large_block_cnt = 0
        # dummy code copy from Quanlu
        M, N = self.out_mask.size()
        for m in range(M//self.block_size_h):
            for n in range(N//self.block_size_w):
                m_start = m * self.block_size_h
                m_end = m_start + self.block_size_h
                n_start = n * self.block_size_w
                n_end = n_start + self.block_size_w
                n_mid = n_start + self.block_size_w//2

                if torch.sum(self.out_mask[m_start:m_end, n_start:n_end]) > 0:
                    if torch.sum(self.out_mask[m_start:m_end, n_start:n_mid]) > 0:
                        m_idx.append(m)
                        n_idx.append(2*n)
                        block_idx.append(2*large_block_cnt)
                    if torch.sum(self.out_mask[m_start:m_end, n_mid:n_end]) > 0:
                        m_idx.append(m)
                        n_idx.append(2*n+1)
                        block_idx.append(2*large_block_cnt+1)
                    large_block_cnt += 1
        # TODO fix me, bug here when there is a blank line
        col_range_index = [0] * (M // self.block_size_h + 1)
        for i in range(1, len(block_idx)):
            if m_idx[i] != m_idx[i-1]:
                for row_id in range(m_idx[i-1]+1, m_idx[i]+1):
                    col_range_index[row_id] = i
        col_range_index[M//self.block_size_h] = len(block_idx)
        return torch.tensor(m_idx, dtype=torch.int32), torch.tensor(n_idx, dtype=torch.int32),  \
            torch.tensor(block_idx, dtype=torch.int32), torch.tensor(
                col_range_index, dtype=torch.int32)

    def _build_backward_index(self):
        """
        Build the index used to calculate the gradient of V.
        """
        # append a zero block at the end of the vals
        zero_idx = self.col_index.size(
            0) * self.block_size_h * self.block_size_w  # TODO point to the zeros at the end of the val
        self.zero_idx = zero_idx
        t_mask = self.out_mask.data.t()
        gradv_row_idx, gradv_col_idx, _ = convert_bcsr(
            t_mask, t_mask, block_h=self.block_size_h, block_w=self.block_size_w)
        subblock_idx = []
        for row_id in range(gradv_row_idx.size(0)-1):
            index_start = gradv_row_idx[row_id]
            index_end = gradv_row_idx[row_id+1]
            for _pos in range(index_start, index_end):
                # Note: there must be a subblock of zeros at the end of vals
                # the
                col_id = gradv_col_idx[_pos].item()
                i_start = row_id * self.block_size_h
                i_end = i_start + self.block_size_h
                j_start = col_id * self.block_size_w
                j_mid = j_start + self.block_size_w // 2
                j_end = j_start + self.block_size_w
                # print(i_start, i_end, j_start, j_mid, j_end)
                if torch.sum(t_mask[i_start:i_end, j_start:j_mid]) > 0:
                    # left subblock has values to be computed
                    # left upper corner(i_start, j_start)
                    subblock_i, subblock_j = j_start//self.block_size_h, i_start//self.block_size_w
                    subblock_pos = self.csr_index[subblock_i][subblock_j] * self.block_size_h * self.block_size_w + 32 * (
                        i_start % self.block_size_w != 0)
                    subblock_idx.append(subblock_pos)
                else:
                    subblock_idx.append(zero_idx)

                if torch.sum(t_mask[i_start:i_end, j_mid:j_end]) > 0:
                    # right subblock has values to be computed
                    subblock_i, subblock_j = j_mid//self.block_size_h, i_start//self.block_size_w
                    subblock_pos = self.csr_index[subblock_i][subblock_j] * self.block_size_h * self.block_size_w + 32 * (
                        i_start % self.block_size_w != 0)
                    subblock_idx.append(subblock_pos)
                else:
                    subblock_idx.append(zero_idx)
        gradv_subblock_idx = torch.tensor(subblock_idx, dtype=torch.int32)

        return gradv_row_idx, gradv_col_idx, gradv_subblock_idx

    def load_kernel_library(self):
        _logger.info('Building and loading the kernel library')
        global our_sparse_attention
        need_replace = {'_REPLACE_HEAD_NUM': self.HEAD_NUM,
                        '_REPLACE_GLOBAL_M': self.M,
                        '_REPLACE_GLOBAL_N': self.N,
                        '_REPLACE_GLOBAL_K': self.K,
                        '_REPLACE_SPARSE_VAL_SIZE': self.val_size,
                        '_REPLACE_SMALL_BLOCK_NUM': len(self.block_index)
                        }
        kernel_template = copy.deepcopy(sparse_attention_template)
        interface_template = copy.deepcopy(sparse_attention_interface)
        for k, v in need_replace.items():
            kernel_template = kernel_template.replace(k, str(v))
        prefix = tempfile.gettempdir()
        cu_f = os.path.join(prefix, 'sparse_attention_kernel.cu')
        interface_f = os.path.join(prefix, 'sparse_attention.cpp')
        with open(cu_f, 'w') as f:
            f.write(kernel_template)
        with open(interface_f, 'w') as f:
            f.write(interface_template)
        our_sparse_attention = module_load(name='our_sparse_attention', sources=[
                                                              cu_f, interface_f], extra_cflags=['-std=c++14', '-O3'], extra_cuda_cflags=['-lcusparse'])
        _logger.info('Kernel library loaded')

