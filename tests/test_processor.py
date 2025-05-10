import pytest
import numpy as np
import pandas as pd
import torch
from rubix.classes.dataset import DataSet
from rubix.processor import (
    compute_fitness_window_shape,
    distribute_labels,
    generate_seed_matrix,
    process_data
)

@pytest.fixture
def simple_dataset():
    """
    Creates a simple dataset with two labels ('A', 'B') and a feature column,
    used for testing processing logic.
    """
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'label': ['A', 'A', 'B', 'B', 'B']
    })
    weights = torch.ones(len(df), dtype=torch.float32)
    return DataSet(dataframe=df, constructors={"weights": weights})

def test_compute_fitness_window_shape_default():
    """
    Tests inferred window shape calculation based on label distribution and partition sizes.
    """
    labels = np.array(['A', 'A', 'B', 'B', 'B'])
    partitions = {"A": 1, "B": 2}
    shape = compute_fitness_window_shape(labels, partitions)
    assert shape == (1, 3)

def test_compute_fitness_window_shape_with_window_valid():
    """
    Tests that a valid, user-supplied window shape is returned correctly.
    """
    labels = np.array(['A'] * 3 + ['B'] * 6)
    partitions = {"A": 1, "B": 2}
    shape = compute_fitness_window_shape(labels, partitions, window=(3, 3))
    assert shape == (3, 3)

def test_compute_fitness_window_shape_with_window_invalid():
    """
    Tests that an invalid user-supplied window shape raises a ValueError.
    """
    labels = np.array(['A'] * 3 + ['B'] * 6)
    partitions = {"A": 1, "B": 2}
    with pytest.raises(ValueError):
        compute_fitness_window_shape(labels, partitions, window=(4, 5))

def test_distribute_labels_balanced():
    """
    Tests that labels are evenly distributed across the partition columns.
    """
    labels = np.array(['A', 'A', 'B', 'B', 'B'])
    partitions = {"A": 1, "B": 2}
    dist = distribute_labels(labels, partitions)
    assert dist == {'A': [2], 'B': [2, 1]}

def test_generate_seed_matrix_shape():
    """
    Tests that the generated seed matrix has the expected shape and is not entirely NaN.
    """
    labels = np.array(['A', 'A', 'B', 'B', 'B'])
    partitions = {"A": 1, "B": 2}
    dist = distribute_labels(labels, partitions)
    matrix = generate_seed_matrix(labels, dist)
    assert matrix.shape == (2, 3)
    assert not np.isnan(matrix).all()

def test_process_data_output(simple_dataset):
    """
    Tests that process_data returns an updated DataSet with required tensors and expected matrix shape.
    """
    partitions = {"A": 1, "B": 2}
    updated = process_data(simple_dataset, 'label', partitions)
    assert isinstance(updated, DataSet)
    assert "weights" in updated.constructors
    assert "constraints" in updated.constructors
    assert "window" in updated.constructors
    assert updated.matrix.shape == (2, 3)
