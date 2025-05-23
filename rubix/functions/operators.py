# rubix/functions/operators.py

"""
# Rubix Cube Operators Module

This module defines mappings and operator functions for manipulating Rubix cube data 
structures using different strategies. It includes functions to generate mappings 
that control roll and swap operations on cube slices and operators that apply these 
mappings to cube instances. Operators are mapped to strategies and can be applied 
generically via the `apply_operator` function.

- Mappings: _get_greedy_map, _get_annealing_map, _get_rubix_map
- Operators: random_operator, greedy_operator, rubix_operator
- Strategy mapping and operator dispatching via `operator_mapping` and `apply_operator`
"""

import torch
from typing import Tuple
from collections import namedtuple
from rubix.classes.cube import Rubix
from rubix.functions.solver_strategies import STRATEGY_NAMES

# Mappings for the operators/functors
def _get_greedy_map(
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    n, H, W = kwargs['shape']
    block_ranges = kwargs['block_ranges']
    valid_swaps = kwargs['valid_swaps']

    roll_map = torch.zeros((W, 2), dtype=torch.int64)
    swap_map = torch.zeros((len(block_ranges), H, n), dtype=torch.bool)
    mode_map = torch.zeros((n,), dtype=torch.int64)

    for m in range(n):
        action = torch.randint(0, 2, (1,)).item()
        mode_map[m] = action

        if action == 0:
            # Roll only
            block_id = torch.randint(0, len(block_ranges), (1,)).item()
            start, end = block_ranges[block_id]
            c = torch.randint(start, end, (1,)).item()
            shift = torch.randint(1, H, (1,)).item()
            roll_map[c, 0] = 1
            roll_map[c, 1] = shift

        else:
            # Swap + Roll using valid block
            block_id = valid_swaps[torch.randint(0, len(valid_swaps), (1,)).item()]
            start, end = block_ranges[block_id]

            r = torch.randint(0, H, (1,)).item()
            swap_map[block_id, r, m] = True

            c1 = torch.randint(start, end - 1, (1,)).item()
            c2 = c1 + 1
            selected_col = c1 if torch.rand(1).item() < 0.5 else c2
            shift = torch.randint(1, H, (1,)).item()
            roll_map[selected_col, 0] = 1
            roll_map[selected_col, 1] = shift

    return roll_map, swap_map, mode_map

def _get_annealing_map(
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    """
    Generate an annealing-based mapping for roll and swap operations, probabilistically
    controlled by the temperature parameter.

    Args:
        shape: Tuple of (n, H, W) representing the cube dimensions.
        block_indices: List of (start, end) column index tuples representing swap blocks.
        temperature: Float controlling probability and intensity of transformations (0 to 1).

    Returns:
        roll_map: Tensor of shape (W, 2), with roll flags and shifts scaled by temperature.
        swap_map: Boolean tensor of shape (num_blocks, H, n) indicating swaps applied probabilistically.
        mode_map: Tensor of shape (n,) with randomized operation modes per slice.
    """

    n, H, W = kwargs['shape']
    block_indices = kwargs['block_indices']
    temperature = kwargs['temperature']

    roll_prob = temperature
    swap_prob = temperature

    roll_flags = torch.rand(W) < roll_prob
    
    max_shift = max(1, H - 1)
    scaled_shift = torch.tensor(temperature * max_shift).clamp(min=1).to(torch.int64)

    roll_shifts = torch.randint(1, scaled_shift + 1, size=(W,), dtype=torch.int64)
    roll_map = torch.zeros((W, 2), dtype=torch.int64)
    roll_map[:, 0] = roll_flags.to(torch.int64)
    roll_map[:, 1] = roll_shifts * roll_flags
    
    swap_map = torch.rand((len(block_indices), H, n)) < swap_prob
    
    mode_map = torch.randint(0, 4, size=(n,), dtype=torch.int64)
    
    return roll_map, swap_map, mode_map

def _get_rubix_map(
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    """
    Generate a random mapping of roll and swap operations uniformly applied across the cube.

    Args:
        shape: Tuple of (n, H, W) representing the cube dimensions.
        block_indices: List of (start, end) column index tuples representing swap blocks.

    Returns:
        roll_map: Tensor of shape (W, 2), randomly flagging and shifting columns.
        swap_map: Boolean tensor of shape (num_blocks, H, n), randomly indicating swap rows.
        mode_map: Tensor of shape (n,) randomly assigning modes (0-3) per slice.
    """

    n, H, W = kwargs['shape']
    block_indices = kwargs['block_indices']
    roll_prob = kwargs.get("p_roll")
    swap_prob = kwargs.get("p_swap")
    
    # --- Roll Mapping ---
    roll_flags = torch.rand(W) < roll_prob
    roll_shifts = torch.randint(1, H, size=(W,), dtype=torch.int64)
    roll_map = torch.zeros((W, 2), dtype=torch.int64)
    roll_map[:, 0] = roll_flags.to(torch.int64)
    roll_map[:, 1] = roll_shifts * roll_flags 

    # --- Swap Mapping ---
    swap_map = torch.rand((len(block_indices), H, n)) < swap_prob

    # --- Mode Mapping ---
    mode_map = torch.randint(0, 4, size=(n,), dtype=torch.int64)

    # Return the generated mappings
    return roll_map, swap_map, mode_map

def _get_roll_map(**kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

    _, H, W = kwargs['shape']
    roll_prob = kwargs.get("p_roll", 0.5)

    col_map = torch.arange(W)[torch.rand(W) < roll_prob]
    shifts = torch.randint(1, H, size=(len(col_map),), dtype=torch.int64)

    return col_map, shifts

def _get_permute_map(**kwargs) -> torch.Tensor:
    _, _, W = kwargs['shape']
    permute_prob = kwargs.get("p_permute", 0.5)

    col_map = torch.arange(W)[torch.rand(W) < permute_prob]
    return col_map

# Operators
def random_operator(
) -> Rubix:

    """
    Generate a randomly initialized Rubix cube instance.

    Returns:
        A new Rubix cube initialized randomly.
    """

    return Rubix.initialize()

def greedy_operator(
    cube: torch.Tensor,
    mapping: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Rubix:
    
    """
    Apply a greedy operator on a Rubix cube using given roll and swap mappings.

    Args:
        cube: Rubix cube instance to transform.
        mapping: Tuple containing roll_map, swap_map, and mode_map.

    Returns:
        A new Rubix cube instance after applying the greedy operations.
    """

    # Unpack the map
    roll_map, swap_map, mode_map = mapping

    new_cube = cube
    n, H, W = Rubix.shape
    block_ranges = Rubix.block_ranges

    for m in range(n):
        mode = mode_map[m].item()

        if mode == 0:
            # Roll-only: only one column affected
            for c in range(W):
                if roll_map[c, 0]:
                    shift = roll_map[c, 1].item()
                    new_cube[m, :, c] = torch.roll(new_cube[m, :, c], shifts=shift, dims=0)

        elif mode == 1:
            # Swap + roll: exactly one row & block is set in swap_map
            for b, (start, end) in enumerate(block_ranges):
                for r in range(H):
                    if swap_map[b, r, m]:
                        perm = torch.randperm(end - start)
                        new_cube[m, r, start:end] = new_cube[m, r, start:end][perm]
            
            for c in range(W):
                if roll_map[c, 0]:
                    shift = roll_map[c, 1].item()
                    new_cube[m, :, c] = torch.roll(new_cube[m, :, c], shifts=shift, dims=0)

    return Rubix(new_cube)

def permute_operator(
    cube: torch.Tensor, 
    col_map: torch.Tensor
) -> Rubix:
    
    new_cube = cube.clone()
    n, H, _ = Rubix.shape

    for c in col_map:
        for m in range(n):
            new_cube[m, :, c] = new_cube[m, :, c][torch.randperm(H)]

    return Rubix(new_cube)

def roll_operator(
    cube: torch.Tensor, 
    mapping: Tuple[torch.Tensor, torch.Tensor]
) -> Rubix:
    
    col_map, shifts = mapping
    new_cube = cube.clone()

    for c, s in zip(col_map, shifts):
        new_cube[:, :, c] = torch.roll(new_cube[:, :, c], shifts=s.item(), dims=1)

    return Rubix(new_cube)

def rubix_operator(
    cube: torch.Tensor,
    mapping: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Rubix:
    
    """
    Apply a Rubix operator that performs roll and swap operations on the cube slices
    according to mode and mappings.

    Args:
        cube: Rubix cube instance to transform.
        mapping: Tuple containing roll_map, swap_map, and mode_map.

    Returns:
        A new Rubix cube instance after applying the roll and swap operations.
    """
    
    # Unpack the map
    roll_map, swap_map, mode_map = mapping

    #Initialize a new cube
    new_cube = cube

    # Get class variables
    n, H, W = Rubix.shape
    block_ranges = Rubix.block_ranges
    
    # Iter the map over the tensor
    for m in range(n):
        mode = mode_map[m].item()
        if mode in {0, 2}:  # swap (only or before roll)
            for b, (start, end) in enumerate(block_ranges):
                for r in range(H):
                    if swap_map[b, r, m]:
                        perm = torch.randperm(end - start)
                        new_cube[m, r, start:end] = new_cube[m, r, start:end][perm]

        if mode in {1, 3}:  # roll (only or before swap)
            for c in range(W):
                if roll_map[c, 0]:
                    shift = roll_map[c, 1].item()
                    new_cube[m, :, c] = torch.roll(new_cube[m, :, c], shifts=shift, dims=0)
            
        if mode == 3:  # roll then swap
            for b, (start, end) in enumerate(block_ranges):
                for r in range(H):
                    if swap_map[b, r, m]:
                        perm = torch.randperm(end - start)
                        new_cube[m, r, start:end] = new_cube[m, r, start:end][perm]
            
    return Rubix(new_cube)

# Operator Mapping
OPERATOR_STRATEGIES = {
    'random': (random_operator, None),
    'greedy': (greedy_operator, _get_greedy_map),
    'permute': (permute_operator, _get_permute_map),
    'roll': (roll_operator, _get_roll_map),
    'anneal': (rubix_operator, _get_annealing_map),
    'rubix': (rubix_operator, _get_rubix_map)
}

# APply operator
def apply_operator(
    cube: torch.Tensor,
    **kwargs
) -> Rubix:
    """
    Apply a Rubix operator by strategy key.

    Args:
        cube: Rubix cube tensor instance.
        strategy: Operator strategy key.
        **kwargs: Additional parameters for the mapping function.

    Returns:
        A new Rubix instance after applying the operator.
    """
    kwargs['shape'] = Rubix.shape
    kwargs['block_indices'] = Rubix.block_indices
    kwargs['block_ranges'] = Rubix.block_ranges
    kwargs['valid_swaps'] = Rubix.valid_swaps

    operator, mapping_fn = OPERATOR_STRATEGIES[kwargs['operator_type']]
    
    if mapping_fn:
        mapping = mapping_fn(**kwargs)
        return operator(cube, mapping)
    else:
        return operator()
