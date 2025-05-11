# tests/test_rubix.py

import pytest
import torch
from rubix.classes.cube import Rubix


# Fixture to set up Rubix class-level parameters 
@pytest.fixture(scope='module')
def rubix_class_setup(seed_matrix):
    """Fixture to set up Rubix class-level parameters once for all tests."""
    layout_params = {'rubix_shape': (1, 4, 4), 'block_indices': [0, 1], 'block_ranges': [(0, 2), (2, 4)]}
    cost_params = {'arrays': [torch.ones(16)], 'lookup': {}}
    
    # Use class_setup to initialize Rubix class-level parameters
    Rubix.class_setup(seed_matrix, cost_params, layout_params)


@pytest.fixture
def rubix_instance(seed_matrix):
    """Fixture to create an instance of Rubix."""
    # Create the Rubix instance for testing
    return Rubix(seed_matrix, torch.tensor(0.0), torch.tensor([0.0]))


def test_compute_fitness_calculates_validity(rubix_class_setup, rubix_instance):
    """
    Ensures that `Rubix.compute_fitness` computes valid fitness for a solution
    and respects the total cost constraint.
    """
    rubix_instance.compute_fitness()

    assert rubix_instance.rubix_fitness != float("-inf")
    assert rubix_instance.slice_fitnesses.shape == (1,)


def test_get_mappings_shapes(rubix_class_setup, rubix_instance):
    """
    Confirms that `get_mappings` returns roll, swap, and mode maps with expected shapes
    based on the current cube configuration.
    """
    roll, swap, mode = rubix_instance.get_mappings()

    assert roll.shape == (4, 2)
    assert swap.shape == (2, 4, 1)
    assert mode.shape == (1,)


def test_rubix_permute_modifies_input(rubix_class_setup, rubix_instance):
    """
    Tests that `rubix_permute` produces a new tensor that differs from the input,
    validating the effect of roll and swap permutations.
    """
    roll_map, swap_map, mode_map = rubix_instance.get_mappings()

    out = rubix_instance.rubix_permute(rubix_instance.solutions.clone(), roll_map, swap_map, mode_map)

    assert out.shape == rubix_instance.solutions.shape
    assert not torch.equal(out, rubix_instance.solutions)
