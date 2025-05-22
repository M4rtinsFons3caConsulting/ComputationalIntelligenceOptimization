from typing import Tuple

import torch
from rubix.classes.cube import Rubix

# Mappings
def _get_slice_map(
    **kwargs
) -> torch.Tensor:
    """
    Generate slice crossover maps for num_pairs.
    For each pair, for each block, randomly decide if crossover occurs.
    If yes, randomly select columns inside the block to crossover.

    Args:
        shape: (n, H, W) tensor shape; only W is needed here.
        block_ranges: list of (start, end) column index tuples for blocks.
        num_pairs: int, number of slice pairs.
        crossover_prob: probability to crossover a block.

    Returns:
        Tensor of shape (num_pairs, W) with bool mask of columns selected for crossover.
    """

    _, _, W = kwargs['shape']
    block_ranges = kwargs['block_ranges']
    num_pairs = kwargs['num_pairs']
    crossover_prob = kwargs['p_cross']

    # Initialize full mask (False)
    slice_map = torch.zeros((num_pairs, W), dtype=torch.bool)

    for _, (start, end) in enumerate(block_ranges):
        block_len = end - start

        # Decide per pair if this block crosses
        cross_flags = torch.rand(num_pairs) < crossover_prob

        if cross_flags.any():
            # For pairs that cross this block, randomly select columns inside block
            # Generate a mask of shape (num_pairs, block_len), True means selected
            block_selections = torch.rand(num_pairs, block_len) < 0.5

            # Zero out block selections for pairs that don't cross
            block_selections = block_selections & cross_flags.unsqueeze(1)

            # Assign to global slice_map at proper indices
            slice_map[:, start:end] |= block_selections

    return slice_map

def _get_cube_map():
    """Calculates a crossover map for a population of cubes"""
    pass

def slice_crossover(    
    cube: torch.Tensor,
    mapping: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Rubix:
    """Takes two cube slices as inputs. Determines  chooses blocks, and performs crossover on those blocks"""
    
def cube_crossover():
    """Takes a cube, chooses slices and permutes the slices"""

# Operator Mapping and Dispatch
CROSSOVER_STRATEGIES = {
    'slice_crossover': (slice_crossover, _get_slice_map),
    'cube_crossover': (cube_crossover, _get_cube_map),
}

def apply_crossover(
    cube: torch.Tensor,
    strategy: str,
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
    operator, mapping_fn = CROSSOVER_STRATEGIES[strategy]

    mapping = mapping_fn(**kwargs) if mapping_fn else None

    if mapping is not None:
        return operator(cube, mapping)
    else:
        return operator(cube)