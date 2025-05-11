# tests/test_main.py

import pytest
from unittest import mock
from rubix.main import main
from rubix.classes.solver import Solver
from rubix.classes.dataset import DataSet
from rubix.constants import DATA_V1

# ---------- Fixtures ----------

@pytest.fixture
def mock_load_data():
    with mock.patch('rubix.main.load_data') as mock_load:
        yield mock_load

@pytest.fixture
def mock_process_data():
    with mock.patch('rubix.main.process_data') as mock_process:
        yield mock_process

@pytest.fixture
def mock_solver_class():
    with mock.patch('rubix.main.Solver') as mock_solver:
        yield mock_solver

# ---------- Test ----------

def test_main(mock_load_data, mock_process_data, mock_solver_class):
    fake_dataset = mock.Mock(spec=DataSet)
    fake_dataset.matrix = "tensor_matrix"
    fake_dataset.cost_params = {"cost": 1}
    fake_dataset.layout_params = {"layout": 2}
    fake_dataset.solver_params = {"solver": 3}
    
    # Mock the return value of load_data and process_data
    mock_load_data.return_value = fake_dataset
    mock_process_data.return_value = fake_dataset

    # Create a fake solver object and mock its solve method
    fake_solver = mock.Mock(spec=Solver)
    fake_solver.solve.return_value = "Optimal Solution"
    mock_solver_class.return_value = fake_solver

    with mock.patch('builtins.print') as mock_print:
        main(path="some_path")

    mock_print.assert_called_once_with(fake_dataset, "Optimal Solution")

    # Debug prints AFTER the mock patch
    print("mock_load_data call args:", mock_load_data.call_args)
    print("mock_process_data call args:", mock_process_data.call_args)
    print("mock_solver_class call args:", mock_solver_class.call_args)
    print("fake_solver.solve call args:", fake_solver.solve.call_args)
    print("mock_print call count:", mock_print.call_count)
