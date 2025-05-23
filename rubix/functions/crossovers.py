from typing import Any

import torch
from rubix.classes.cube import Rubix

# Mappings
def _get_slice_map(
    **kwargs
) -> torch.Tensor:
    
    _, _, W = Rubix.shape
    block_ranges = Rubix.block_ranges 
    num_pairs = kwargs['n_pairs']
    crossover_prob = kwargs['p_cross']

    slice_map = torch.zeros((num_pairs, W), dtype=torch.bool)

    for start, end in block_ranges:
        cross_flags = torch.rand(num_pairs) < crossover_prob
        for i, flag in enumerate(cross_flags):
            if flag:
                slice_map[i, start:end] = True

    return slice_map

def _get_cutoff_map(**kwargs):
    n_pairs = kwargs['n_pairs']
    p_cross = kwargs['p_cross']
    H = Rubix.shape[1]
    num_blocks = len(Rubix.block_ranges)

    flags = (torch.rand((n_pairs, num_blocks)) < p_cross).bool()
    cuts = torch.randint(0, H, (n_pairs, num_blocks))

    return flags, cuts

def block_crossover(    
    cube: torch.Tensor,
    mapping: torch.Tensor
) -> torch.Tensor:
    """
    Perform block-wise crossover on the provided population of parents.
    Expects cube of shape [num_pairs * 2, rows, cols]
    and mapping of shape [num_pairs, cols]
    Returns offspring of same shape.
    """

    parents = cube.clone()  # shape [2*num_pairs, rows, cols]
    num_pairs = mapping.size(0)

    for i in range(num_pairs):
        idx1, idx2 = 2 * i, 2 * i + 1
        block_mask = mapping[i]  # shape [cols]

        # Swap columns at block_mask across all rows
        tmp = parents[idx1, :, block_mask].clone()
        parents[idx1, :, block_mask] = parents[idx2, :, block_mask]
        parents[idx2, :, block_mask] = tmp

        offspring = parents

    return offspring

def cutoff_crossover(
    cube: torch.Tensor,
    mapping: tuple
) -> torch.Tensor:
    """
    Applies Order Crossover (OX) to ensure permutation validity.
    Input:
        cube: [2*num_pairs, N] (1D permutations)
        mapping: tuple of (flags, cuts)
            - flags: [num_pairs, num_blocks] (bool mask per block)
            - cuts: [num_pairs, num_blocks] (cutoff per block)
    Output:
        offspring: [2*num_pairs, N]
    """
    flags, cuts = mapping
    num_pairs = flags.size(0)
    num_blocks = flags.size(1)

    parents = cube.clone()
    N = parents.size(1)

    def ox(p1: torch.Tensor, p2: torch.Tensor, start: int, end: int) -> torch.Tensor:
        child = -torch.ones(N, dtype=torch.int64)
        child[start:end+1] = p1[start:end+1]
        fill = [x for x in p2.tolist() if x not in child]
        j = 0
        for i in range(N):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child

    for i in range(num_pairs):
        idx1, idx2 = 2 * i, 2 * i + 1
        for j in range(num_blocks):
            if flags[i, j]:
                cutoff = cuts[i, j].item()
                # Define crossover region as [0, cutoff]
                p1, p2 = parents[idx1], parents[idx2]
                parents[idx1] = ox(p1, p2, 0, cutoff)
                parents[idx2] = ox(p2, p1, 0, cutoff)

    return parents

# Operator Mapping and Dispatch
CROSSOVER_STRATEGIES = {
    'block_crossover': (block_crossover, _get_slice_map),
    'cutoff_crossover': (cutoff_crossover, _get_cutoff_map),
}

def apply_crossover(
    cube: torch.Tensor,
    **kwargs
) -> Rubix:
    
    """
    Dispatch crossover operation by strategy.
    Generates mapping using associated mapping function, then applies operator.
    Args:
        cube: Rubix cube tensor instance.
        strategy: key selecting crossover operator.
        **kwargs: Additional parameters for map generation.
    Returns:
        New Rubix cube instance after crossover.
    """
    
    operator, mapping_fn = CROSSOVER_STRATEGIES[kwargs['x_strategy']]
    mapping = mapping_fn(
        **kwargs   
    )
    
    return operator(cube, mapping)
    