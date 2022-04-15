import abc
import math
import torch
import torch.nn as nn
from itertools import product
from typing import List
from sparta.common import SparseModuleInfo, TeSA
from sparta.specialization import specialize_matmul
from sparta.transformation.weighted_block_cover import wbc_matmuls

class TransformedModule:
    def __init__(self, module_info: SparseModuleInfo, kernels: list, aggregate_type: str = None):
        self.module_info: SparseModuleInfo = module_info
        self.kernels: list = kernels
        self.aggregate_type: str = aggregate_type

class TransformPolicyBase(abc.ABC):
    @abc.abstractmethod
    def transform_module(self, module_sparsity: SparseModuleInfo):
        pass

class TransformPolicy(TransformPolicyBase):
    def __init__(self, device_info: str = None):
        self.device_info = device_info

    def hardware_bit_precisions(self) -> list:
        # TODO: refine a device info class
        # currently, specific for 2080ti
        # int4, int8, float16, float32
        return [4, 8, 16, 32]

    def transform_module(self, module_sparsity: SparseModuleInfo):
        """
        Enumerate possible tensor decomposition options.
        In the current implementation, we do not decompose activation tensor into
        multiple tensors. Only do this for weight tensors.

        Returns
        -------
        TransformedModule
        """
        kernels = None
        if isinstance(module_sparsity.module_obj, nn.Linear):
            kernels, aggr_type = self.transform_matmul(module_sparsity.input_tesa,
                                            module_sparsity.weight_tesa,
                                            module_sparsity.output_tesa)
        elif isinstance(module_sparsity.module_obj, nn.Conv2d):
            ...
        # support our sparse attention
        #elif isinstance(module_sparsity.module_obj, SparseAttention):
        #    ...
        else:
            ...
        return TransformedModule(module_sparsity, kernels, aggr_type)
    
    def transform_tesas(self, tesa: TeSA, n_candidates: int = 1, decompose: bool = True) -> List[tuple]:
        """
        Sparsity pattern matcher here for decomposing/splitting and covering

        Parameters
        ----------
        tesa : TeSA
            the sparsity attribute
        n_candidates : int
            the number of generated transformation options
        decompose : bool
            whether decompose a tensor into multiple tensors

        Returns
        -------
        list[tuple]
            a list of candidates, each candidate is a tuple of sub-tensors
        """
        # handling bit numbers
        if not decompose:
            # aligning to the maximum bit number
            ...
        else:
            ...

        return [(tesa,)]

    def _extract_bit_nums(self, tesa: TeSA) -> set:
        bit_set = set()
        flat_tesa = tesa.tesa.reshape(-1)
        for ele in flat_tesa:
            bit_set.add(ele)
        if 0 in bit_set:
            bit_set.remove(0)
        return bit_set

    def _convert_tesa_bit(self, tesa: TeSA, aligned_bit: int):
        flat_tesa = tesa.tesa.reshape(-1)
        for i, _ in enumerate(flat_tesa):
            if flat_tesa[i] > 0:
                flat_tesa[i] = aligned_bit

    def _convert_tesa_bits(self, tesa: TeSA, aligned_bits: set) -> List[tuple]:
        print('start converting tesa...')
        low_bit = min(aligned_bits)
        high_bit = max(aligned_bits)
        low_tesa = TeSA(torch.zeros_like(tesa.tesa))
        high_tesa = TeSA(torch.zeros_like(tesa.tesa))
        flat_tesa = tesa.tesa.reshape(-1)
        flat_low_tesa = low_tesa.tesa.reshape(-1)
        flat_high_tesa = high_tesa.tesa.reshape(-1)
        for i, _ in enumerate(flat_tesa):
            # TODO: consider TesaAttr carefully!
            if flat_tesa[i] <= 0:
                continue
            if flat_tesa[i] <= low_bit:
                flat_low_tesa[i] = low_bit
                flat_tesa[i] = high_bit
            elif flat_tesa[i] <= high_bit:
                flat_high_tesa[i] = high_bit
                flat_tesa[i] = high_bit
            else:
                raise
        low_tesa.n_bits = 1
        high_tesa.n_bits = 1
        print('converting tesa done.')
        return [(tesa,), (low_tesa, high_tesa)]

    def _align_to_hardware_bits(self, bits: set, bit_options: list) -> bool:
        # NOTE: here assume hardware supported bit number is 2^x
        aligned_bits = set()
        for bit in bits:
            if bit <= 0:
                continue
            aligned_ideal_bit = 2 ** math.ceil(math.log(bit,2))
            if aligned_ideal_bit in bit_options:
                aligned_bits.add(aligned_ideal_bit)
            else:
                aligned_bits.add(min([b for b in bit_options if b > aligned_ideal_bit]))
        return aligned_bits

    def bit_align(self, tesa: TeSA, n_candidates: int = 1, decompose: bool = True) -> List[tuple]:
        bit_options = self.hardware_bit_precisions()
        print('start extracting bit nums...')
        bits = self._extract_bit_nums(tesa)
        print('extracting bit nums done.')
        aligned_bits = self._align_to_hardware_bits(bits, bit_options)
        print('aligning bit nums...')
        if not decompose or len(aligned_bits) == 1:
            aligned_bit = max(bits)
            if aligned_bit not in bit_options:
                aligned_bit = min([bit for bit in bit_options if bit > aligned_bit])
            if len(aligned_bits) == 1:
                assert aligned_bit == aligned_bits.pop()
            self._convert_tesa_bit(tesa, aligned_bit)
            tesa.n_bits = 1
            return [(tesa,)]
        else:
            # NOTE: only handle at most 2 different bits
            assert len(aligned_bits) == 2
            return self._convert_tesa_bits(tesa, aligned_bits)

    def transform_matmul(self, in_tesa: TeSA, weight_tesa: TeSA, out_tesa: TeSA):
        best_latency = float('inf')
        best_kernels = None
        best_aggr_type = None

        # do bit alignment for quantization
        # only consider decomposing weight for simplicity
        print('weight bit aligning...')
        weight_tesas = self.bit_align(weight_tesa)
        print('input bit aligning...')
        in_tesas = self.bit_align(in_tesa, decompose=False)
        print('output bit aligning...')
        out_tesas = self.bit_align(out_tesa, decompose=False)

        for w_tesas in weight_tesas:
            # this is one transformation option.
            # do weighted block cover for pruned elements
            print('wbc_matmuls starting...')
            transform_options = wbc_matmuls(in_tesas[0], w_tesas, out_tesas[0])
            print('wbc_matmuls done.')
            '''
            # skip, waiting for more official end-to-end
            for (in_ts, w_ts, out_ts) in transform_options:
                # do not decompose in_tesa and out_tesa for simplicity
                assert(len(in_ts) == 1 and len(out_ts) == 1 and len(w_ts) >= 1)
                print("transformation done!")
                # then specialize kernels for this transformation option
                latency, kernels, aggr_type = specialize_matmul(in_ts, w_ts, out_ts)
                if latency < best_latency:
                    best_latency = latency
                    best_kernels = kernels
                    best_aggr_type = aggr_type
            '''
        return best_kernels, best_aggr_type

    def transform_conv(self, in_tesa: TeSA, weight_tesa: TeSA, out_tesa: TeSA):
        ...
