# tests/test_rubix.py

import pytest
import torch
from rubix.classes.cube import Rubix

@pytest.fixture
def seed_matrix():
    """Provides a deterministic 1x4x4 seed matrix for consistent testing."""
    H, W = 4, 4
    matrix = torch.arange(H * W).view(1, H, W)
    return matrix

@pytest.fixture
def weights():
    """Provides a weight tensor with ones for both ability and cost."""
    abilities = torch.ones(16)
    costs = torch.ones(16)
    return torch.stack((abilities, costs), dim=1)

def test_initialize_generates_valid_solutions(seed_matrix, weights):
    """
    Verifies that `Rubix.initialize` generates a solution tensor of the expected shape
    and computes initial fitness without error.
    """
    Rubix.set_shape((1, 4, 4))
    Rubix.set_weights(weights)
    Rubix.set_constraints(torch.tensor([0, 2]))  # Two column blocks: [0:2], [2:4]

    rubix = Rubix.initialize(seed_matrix)
    
    assert rubix.solutions.shape == (1, 4, 4)
    assert isinstance(rubix.rubix_fitness, torch.Tensor)

def test_compute_fitness_calculates_validity(seed_matrix, weights):
    """
    Ensures that `Rubix.compute_fitness` computes valid fitness for a solution
    and respects the total cost constraint.
    """
    Rubix.set_shape((1, 4, 4))
    Rubix.set_weights(weights)
    Rubix.set_constraints(torch.tensor([0, 2]))

    rubix = Rubix(seed_matrix, torch.tensor(0.0), torch.tensor([0.0]))
    rubix.compute_fitness()

    assert rubix.rubix_fitness != float("-inf")
    assert rubix.slice_fitnesses.shape == (1,)

def test_get_mappings_shapes(seed_matrix):
    """
    Confirms that `get_mappings` returns roll, swap, and mode maps with expected shapes
    based on the current cube configuration.
    """
    Rubix.set_shape((1, 4, 4))
    Rubix.set_constraints(torch.tensor([0, 2]))

    rubix = Rubix(seed_matrix, torch.tensor(0.0), torch.tensor([0.0]))
    roll, swap, mode = rubix.get_mappings()

    assert roll.shape == (4, 2)
    assert swap.shape == (2, 4, 1)
    assert mode.shape == (1,)

def test_rubix_permute_modifies_input(seed_matrix):
    """
    Tests that `rubix_permute` produces a new tensor that differs from the input,
    validating the effect of roll and swap permutations.
    """
    Rubix.set_shape((1, 4, 4))
    Rubix.set_constraints(torch.tensor([0, 2]))

    rubix = Rubix(seed_matrix, torch.tensor(0.0), torch.tensor([0.0]))
    roll_map, swap_map, mode_map = rubix.get_mappings()

    out = rubix.rubix_permute(seed_matrix.clone(), roll_map, swap_map, mode_map)

    assert out.shape == seed_matrix.shape
    assert not torch.equal(out, seed_matrix)
