# test/test_processor.py

import pytest
import numpy as np
import pandas as pd
import torch
from rubix.classes.dataset import DataSet
from rubix.processor import generate_seed_matrix, process_data

# ---------- Fixtures ----------

@pytest.fixture
def simple_dataset():
    df = pd.DataFrame({
        'Skill': [80, 75, 90, 85, 70, 88, 65],
        'Salary (â‚¬M)': [10, 8, 12, 11, 7, 13, 6],
        'Position': ['GK', 'DEF', 'DEF', 'MID', 'MID', 'FWD', 'FWD']
    })
    weights = torch.ones(len(df), dtype=torch.float32)
    return DataSet(
        dataframe=df,
        constructors={
            "weights": weights,
            "constraints": {
                "label_col": "Position",
                "weights": ["Skill", "Salary (â‚¬M)"],
                "constraints": {
                    "GK": [1],
                    "DEF": [1, 1],
                    "MID": [1, 1],
                    "FWD": [1, 1]
                }
            },
            "window": None
        },
        solver_params={"n": 1}
    )

# ---------- Tests ----------

def test_generate_seed_matrix_shape(simple_dataset):
    labels = simple_dataset.dataframe['Position'].values
    dist = simple_dataset.constructors["constraints"]["constraints"]
    matrix = generate_seed_matrix(labels, dist)
    assert matrix.shape == (1, 7)
    assert not np.isnan(matrix).all()

def test_process_data_output(simple_dataset):
    updated = process_data(simple_dataset)
    assert isinstance(updated, DataSet)
    assert "weights" in updated.constructors
    assert "constraints" in updated.constructors
    assert "window" in updated.constructors
    assert updated.matrix.shape == (1, 7)

def test_process_data_with_invalid_window(simple_dataset):
    invalid_dataset = simple_dataset
    invalid_dataset.constructors["window"] = (5, 5)
    with pytest.raises(ValueError):
        process_data(invalid_dataset)
