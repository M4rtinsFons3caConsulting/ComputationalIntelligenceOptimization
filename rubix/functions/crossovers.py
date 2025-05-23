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

def _get_cutoff_map(**kwargs) -> torch.Tensor:

    """
    Generates a cutoff crossover map.
    Returns a tensor of shape [num_pairs, num_blocks, 2] where:
        - map[i, j, 0] is a boolean (True if crossover happens for pair i at block j),
        - map[i, j, 1] is the cutoff row (0 <= cutoff < H) for that block in that pair.
    """
    
    n_pairs = kwargs['n_pairs']
    p_cross = kwargs['p_cross']
    H = Rubix.shape[1]
    num_blocks = len(Rubix.block_ranges)

    # Keep flags as bool
    flags = torch.rand((n_pairs, num_blocks)) < p_cross        # bool
    cuts  = torch.randint(0, H, (n_pairs, num_blocks))         # int64

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
    mapping: tuple[torch.BoolTensor, torch.Tensor]
) -> torch.Tensor:
    """
    Perform cutoff crossover on cube pairs given block-wise cutoff mapping.

    Args:
        cube: Tensor of shape [2*n_pairs, H, W]
        mapping: A tuple (flags, cuts) where
            - flags is a BoolTensor of shape [n_pairs, n_blocks],
              flags[i, b] == True means “do crossover for pair i on block b”
            - cuts is an IntTensor of shape [n_pairs, n_blocks],
              cuts[i, b] is the cutoff row index (0 <= cuts < H)

    Returns:
        offspring tensor of the same shape as cube.
    """
    parents = cube.clone()
    flags, cuts = mapping

    n_pairs, _ = flags.shape
    for i in range(n_pairs):
        idx1, idx2 = 2 * i, 2 * i + 1
        for b, (start, end) in enumerate(Rubix.block_ranges):
            if flags[i, b]:
                # inclusive cutoff: rows [0 .. cutoff]
                cutoff = cuts[i, b].item()
                # swap the block region up to the cutoff row
                tmp = parents[idx1, : cutoff + 1, start:end].clone()
                parents[idx1, : cutoff + 1, start:end] = parents[idx2, : cutoff + 1, start:end]
                parents[idx2, : cutoff + 1, start:end] = tmp

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
    