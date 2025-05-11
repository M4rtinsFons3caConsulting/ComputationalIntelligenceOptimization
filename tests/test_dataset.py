# tests/test_dataset.py

import pytest
import torch
import pandas as pd
from rubix.classes.dataset import DataSet
from dataclasses import FrozenInstanceError

# Fixture for Raw/Ingested state (No matrix provided)
@pytest.fixture
def raw_data():
    """Fixture for Raw/Ingested state where matrix is not provided."""
    dataframe = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    matrix = None  # No matrix provided, making it Raw/Ingested
    cost_params = {'arrays': [torch.tensor([1, 2])], 'lookup': {'key': torch.tensor([3])}}
    layout_params = {'dim': 2}
    solver_params = {'param1': 0.5}
    constructors = {'constructor1': 'value'}
    
    return dataframe, matrix, cost_params, layout_params, solver_params, constructors


# Fixture for Processed state (With matrix provided)
@pytest.fixture
def processed_data():
    """Fixture for Processed state where matrix is provided."""
    dataframe = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Matrix provided, making it Processed
    cost_params = {'arrays': [torch.tensor([1, 2])], 'lookup': {'key': torch.tensor([3])}}
    layout_params = {'dim': 2}
    solver_params = {'param1': 0.5}
    constructors = {'constructor1': 'value'}
    
    return dataframe, matrix, cost_params, layout_params, solver_params, constructors


# Test: Raw/Ingested state representation
def test_repr_raw_stage(raw_data):
    dataframe, matrix, cost_params, layout_params, solver_params, constructors = raw_data
    dataset = DataSet(dataframe=dataframe, matrix=matrix)

    repr_str = repr(dataset)
    assert "DataSet(" in repr_str
    assert "Raw/Ingested" in repr_str  # Raw/Ingested because matrix is None
    assert "dataframe:" in repr_str


# Test: Processed state representation
def test_repr_processed_stage(processed_data):
    dataframe, matrix, cost_params, layout_params, solver_params, constructors = processed_data
    dataset = DataSet(dataframe=dataframe, matrix=matrix)

    repr_str = repr(dataset)
    assert "DataSet(" in repr_str
    assert "Processed" in repr_str  # Processed because matrix is not None
    assert "dataframe:" in repr_str


# Test: Dataset initialization
def test_dataset_initialization():
    """Test that DataSet is initialized correctly."""
    dataframe = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cost_params = {'arrays': [torch.tensor([1, 2])], 'lookup': {'key': torch.tensor([3])}}
    layout_params = {'dim': 2}
    solver_params = {'param1': 0.5}
    constructors = {'constructor1': 'value'}

    dataset = DataSet(
        dataframe=dataframe,
        matrix=matrix,
        cost_params=cost_params,
        layout_params=layout_params,
        solver_params=solver_params,
        constructors=constructors
    )

    assert dataset.dataframe.shape == (2, 2)
    assert dataset.matrix.shape == (2, 2)
    assert dataset.cost_params['arrays'][0].shape == (2,)
    assert dataset.layout_params['dim'] == 2
    assert dataset.solver_params['param1'] == 0.5
    assert dataset.constructors['constructor1'] == 'value'


# Test: Immutability of DataSet object
def test_dataset_immutability(dataset):
    """Test that the DataSet object is immutable."""
    with pytest.raises(FrozenInstanceError):
        dataset.dataframe = pd.DataFrame({'C': [5, 6]})


# Test: Dataset update method creates a new DataSet object
def test_update_method():
    """Test the update method to ensure it creates a new DataSet."""
    dataframe = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset = DataSet(dataframe=dataframe, matrix=matrix)

    # Update the dataset
    new_matrix = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    updated_dataset = dataset.update(matrix=new_matrix)

    # Ensure a new object is returned
    assert updated_dataset is not dataset
    assert updated_dataset.matrix is not dataset.matrix
    assert torch.equal(updated_dataset.matrix, new_matrix)


# Test: Empty dataframe handling
def test_empty_dataframe():
    """Test that the DataSet class handles an empty dataframe."""
    dataframe = pd.DataFrame()
    dataset = DataSet(dataframe=dataframe)

    assert dataset.dataframe.empty
    assert dataset.matrix is None


# Test: Dataset update with partial fields
def test_update_partial_fields():
    """Test the update method with partial fields."""
    dataframe = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset = DataSet(dataframe=dataframe, matrix=matrix)

    # Update only cost_params
    new_cost_params = {'arrays': [torch.tensor([1])], 'lookup': {'key': torch.tensor([3])}}
    updated_dataset = dataset.update(cost_params=new_cost_params)

    assert updated_dataset is not dataset
    assert updated_dataset.cost_params == new_cost_params
    assert updated_dataset.dataframe.equals(dataset.dataframe)
    assert updated_dataset.matrix is dataset.matrix
