import math
import torch
from typing import List, Tuple, Dict
from sortedcontainers import SortedList
from sparta.common import TeSA

def available_covering_size() -> Dict[Tuple[int, int], float]:
    """
    the numbers below are the profiled cost of computing one block.
    the numbers highly depends on the underlying kernel optimizations,
    new optimization techniques leads adjustment of those numbers.
    there will be a cost database to indicate the cost of various
    sparsity patterns in future.
    """
    cover_sizes = {
        (1, 1): 0.0064,
        (32, 32): 6.25,
        (32, 64): 8.59,
        (64, 32): 25.2,
        (32, 128): 16.02,
        (64, 64): 50,
        (64, 128): 96.8
    }
    return cover_sizes

def extract_non_pruned_eles(universe: torch.Tensor, csize: tuple, row: int, col: int) -> set:
    non_pruned = set()
    for i in range(csize[0]):
        for j in range(csize[1]):
            if universe[row*csize[0]+i][col*csize[1]+j] > 0:
                non_pruned.add(universe[row*csize[0]+i][col*csize[1]+j])
    return non_pruned

def min_price_block(covered_eles: set, cover_set: list, cover_sizes: dict) -> tuple:
    min_price = float('inf')
    best_block = None
    for block in cover_set:
        effective_eles = block[2] - covered_eles
        if len(effective_eles) <= 0:
            continue
        price = cover_sizes[block[0]] / len(effective_eles)
        if price < min_price:
            min_price = price
            best_block = block
    return best_block

def weighted_block_cover(tesa: TeSA, cover_sizes: dict) -> Tuple[list, set]:
    """
    convert this weighted block cover to the classic weighted set cover problem
    """
    assert len(tesa.tesa.size()) == 2, 'the number of dimension should be 2'
    # numbering non-pruned elements in row major
    universe = torch.zeros_like(tesa.tesa, dtype=torch.int32)
    n_row, n_col = universe.size()
    seq_id = 1
    for i in range(n_row):
        for j in range(n_col):
            # TODO: consider negative values
            if tesa.tesa[i][j] > 0:
                universe[i][j] = seq_id
                seq_id += 1
    universe_count = seq_id - 1
    # prepare block sets and build ordered price dict
    ordered_pricing = SortedList(key=lambda x: x[0])
    for csize in cover_sizes:
        if csize == (1, 1):
            continue
        # traverse the blocks of this size
        for i in range(int(n_row/csize[0])):
            for j in range(int(n_col/csize[1])):
                non_pruned = extract_non_pruned_eles(universe, csize, i, j)
                if len(non_pruned) > 0:
                    covering_info = (csize, (i, j), non_pruned)
                    price = cover_sizes[csize] / len(non_pruned)
                    ordered_pricing.add((price, covering_info))
    # handle block size (1, 1) separately
    if cover_sizes[(1, 1)] < ordered_pricing[0][0]:
        # all use block size (1, 1)
        unique_block_sizes = set()
        unique_block_sizes.add((1, 1))
        return None, unique_block_sizes
    # algorithm for weighted set cover
    covered_eles = set()
    chosen_blocks = list()
    unique_block_sizes = set()
    while len(covered_eles) < universe_count:
        #print('number of covered elements: ', len(covered_eles))
        (price, covering_info) = ordered_pricing.pop(0)
        new_price = cover_sizes[covering_info[0]] / len(covering_info[2] - covered_eles)
        print(price, new_price, covering_info[0])
        if math.isclose(price, new_price):
            if price > cover_sizes[(1, 1)]:
                # use block size (1, 1) for all the left non-pruned elements
                unique_block_sizes.add((1, 1))
                return chosen_blocks, unique_block_sizes
            chosen_blocks.append(covering_info)
            unique_block_sizes.add(covering_info[0])
            covered_eles.update(covering_info[2])
        else:
            ordered_pricing.add((new_price, covering_info))

    return chosen_blocks, unique_block_sizes

def decompose_blocked_tesas(tesa: TeSA, chosen_blocks: list, unique_block_sizes: set):
    ...

def wbc_matmul(in_tesa: TeSA, w_tesa: TeSA, out_tesa: TeSA) -> List[tuple]:
    """
    currently, we only support several commonly used and easy-to-optimize block sizes
    currently, we only support covering weight tesa, covering other tesas can be supported by adding
    one more dimension (i.e., m dimension for IN(m, k)*W(k, n)=OUT(m, n)).
    """
    cover_sizes = available_covering_size()
    # weighted block cover
    chosen_blocks, unique_block_sizes = weighted_block_cover(w_tesa, cover_sizes)
    print('chosen block sizes: ', unique_block_sizes)
    if len(unique_block_sizes) == 1:
        w_tesa.block_size = unique_block_sizes.pop()
        return [(in_tesa, w_tesa, out_tesa)]
    else:
        # TODO: support more than one block size
        decompose_blocked_tesas(w_tesa, chosen_blocks, unique_block_sizes)
        raise

def wbc_matmuls(in_tesas: tuple, w_tesas: tuple, out_tesas: tuple) -> List[tuple]:
    """
    in the current version, only return one covering option

    Parameters
    ----------
    in_tesas: tuple
        one transformation option from bit alignment, only one tesa in it
    w_tesas: tuple
        one transformation option from bit alignment, one or two tesas in it
    out_tesas: tuple
        one transformation option from bit alignment, only one tesa in it
    """
    transform_options = []
    in_tesa = in_tesas[0]
    out_tesa = out_tesas[0]
    for w_tesa in w_tesas:
        transformed = wbc_matmul(in_tesa, w_tesa, out_tesa)
        transform_options.extend(transformed)
    return transform_options
