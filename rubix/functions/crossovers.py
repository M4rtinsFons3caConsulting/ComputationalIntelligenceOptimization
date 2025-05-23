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

def _get_cube_map():
    """Calculates a crossover map for a population of cubes"""
    pass

def slice_crossover(    
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

def cube_crossover(
    indices,
    **kwargs
) -> Rubix:
    """Takes a cube, chooses slices and permutes the slices"""
    pass

# Operator Mapping and Dispatch
CROSSOVER_STRATEGIES = {
    'slice_crossover': (slice_crossover, _get_slice_map),
    'cube_crossover': (cube_crossover, _get_cube_map),
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
    