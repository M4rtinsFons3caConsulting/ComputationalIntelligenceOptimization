import pytest
import torch
from rubix.classes.solver import Solver
from rubix.classes.cube import Rubix


@pytest.fixture
def setup_solver():
    """Fixture to set up solver for testing."""
    # Example setup values
    seed = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    cost_params = {
        'arrays': [torch.tensor([1, 2, 3])],
        'lookup': {'key': torch.tensor([1])}
    }
    layout_params = {
        'rubix_shape': (5, 2, 2),  # n, H, W
        'block_indices': [0, 1],
        'block_ranges': [(0, 1), (1, 2)]
    }
    solver_params = {
        'epochs': 10,
        'n': 5
    }

    # Prime Rubix with necessary setup
    Rubix.class_setup(seed, cost_params, layout_params)

    solver = Solver(seed, cost_params, layout_params, solver_params)
    return solver


def test_solver_initialization(setup_solver):
    """Test that the Solver is initialized correctly."""
    solver = setup_solver

    # Verify solver params are initialized correctly
    assert solver.solver_params['epochs'] == 10
    assert solver.solver_params['n'] == 5


def test_solver_run(setup_solver):
    """Test the solver's solve method."""
    solver = setup_solver
    best_fitness = solver.solve()

    # Ensure the result is a float value
    assert isinstance(best_fitness, float)
    assert best_fitness >= 0  # Fitness should be non-negative


def test_solver_no_epochs(setup_solver):
    """Test solver behavior when epochs is set to 0."""
    solver = setup_solver
    solver.solver_params['epochs'] = 0  # Set epochs to 0

    best_fitness = solver.solve()
    assert best_fitness == 0  # Should return zero fitness if no epochs are run


def test_solver_update(setup_solver):
    """Test that the solver can update its parameters and return a new solution."""
    solver = setup_solver

    # Update solver parameters
    new_solver_params = {
        'epochs': 20,
        'n': 10
    }
    solver.solver_params = new_solver_params

    # Solve again with new parameters
    best_fitness = solver.solve()

    # Ensure the result is still valid
    assert isinstance(best_fitness, float)
    assert best_fitness >= 0  # Fitness should be non-negative
    assert solver.solver_params['epochs'] == 20


def test_solver_empty_seed():
    """Test solver behavior with an empty seed."""
    seed = torch.empty((0, 0), dtype=torch.float32)
    cost_params = {'arrays': [torch.tensor([1, 2])], 'lookup': {'key': torch.tensor([3])}}
    layout_params = {'rubix_shape': (0, 0, 0), 'block_indices': [], 'block_ranges': []}
    solver_params = {'epochs': 5, 'n': 5}

    # Prime Rubix with necessary setup
    Rubix.class_setup(seed, cost_params, layout_params)

    solver = Solver(seed, cost_params, layout_params, solver_params)
    
    # Since the seed is empty, the solve method might either fail or return no useful result.
    # Here we just check it doesn't throw an error
    try:
        best_fitness = solver.solve()
        assert isinstance(best_fitness, float)
    except Exception as e:
        pytest.fail(f"Solver failed with empty seed: {e}")


def test_solver_invalid_epochs():
    """Test solver behavior with invalid number of epochs."""
    solver = setup_solver

    solver.solver_params['epochs'] = -1  # Set an invalid number of epochs
    with pytest.raises(ValueError):
        solver.solve()


def test_solver_best_fitness_improvement(setup_solver):
    """Test that the best solution improves over time in the solver."""
    solver = setup_solver
    initial_fitness = solver.solve()

    # Assume the fitness should improve after running the solver
    new_best_fitness = solver.solve()
    assert new_best_fitness >= initial_fitness
