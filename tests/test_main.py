import pytest
from unittest import mock
from rubix.main import main
from rubix.classes.solver import Solver
from rubix.classes.dataset import DataSet

# ---------- Fixtures ----------

@pytest.fixture
def mock_load_data():
    """
    Mocks the `load_data` function in `rubix.main`, allowing tests to run without actual file I/O.
    """
    with mock.patch('rubix.main.load_data') as mock_load_data:
        yield mock_load_data

@pytest.fixture
def mock_process_data():
    """
    Mocks the `process_data` function in `rubix.main`, preventing real dataset processing during the test.
    """
    with mock.patch('rubix.main.process_data') as mock_process_data:
        yield mock_process_data

@pytest.fixture
def mock_solver():
    """
    Mocks the `Solver` class in `rubix.main` to avoid invoking real optimization logic.
    """
    with mock.patch('rubix.main.Solver') as mock_solver:
        yield mock_solver

# ---------- Test ----------

def test_main(
    mock_load_data, 
    mock_process_data, 
    mock_solver
):
    """
    Full integration test of `main` with all dependencies mocked.

    Verifies:
    - `load_data` is called with the correct arguments.
    - `process_data` is called with the dataset from `load_data`.
    - `Solver` is instantiated with the processed dataset.
    - `solve` is called on the Solver instance.
    - The solution is printed as expected.
    """
    # Mocks
    fake_dataset = mock.Mock(spec=DataSet)
    mock_load_data.return_value = fake_dataset
    mock_process_data.return_value = fake_dataset

    fake_solver = mock.Mock(spec=Solver)
    mock_solver.return_value = fake_solver
    fake_solver.solve.return_value = "Optimal Solution"

    # Act
    with mock.patch('builtins.print') as mock_print:
        main(path="path_to_data.xlsx", config_path="config.json")

    # Assert
    mock_load_data.assert_called_once_with(path="path_to_data.xlsx", config_path="config.json")
    mock_process_data.assert_called_once_with(fake_dataset)
    mock_solver.assert_called_once_with(fake_dataset)
    fake_solver.solve.assert_called_once_with(path="path_to_data.xlsx", config_path="config.json")
    mock_print.assert_called_once_with("Solution:", "Optimal Solution")
