import pytest
import pandas as pd
import json
from rubix.classes.dataset import DataSet
import rubix.exceptions as ce
from rubix.loader import load_data

# ---------- Fixtures ----------

@pytest.fixture
def sample_excel(tmp_path):
    """
    Creates a temporary Excel file with valid feature and label columns for testing.
    """
    df = pd.DataFrame({
        'Unnamed: 0': [0, 1, 2],
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [0.5, 1.5, 2.5],
        'label': ['A', 'B', 'A']
    }).set_index('Unnamed: 0')
    file = tmp_path / "sample.xlsx"
    df.to_excel(file)
    return file

@pytest.fixture
def sample_config(tmp_path):
    """
    Creates a temporary JSON config file defining label column, feature columns, and partitions.
    """
    config = {
        "problem_constraints": {
            "label_col": "label",
            "feature_cols": ["feature1", "feature2"],
            "partitions": {"A": 2, "B": 1}
        }
    }
    file = tmp_path / "config.json"
    file.write_text(json.dumps(config))
    return file

# ---------- Tests ----------

def test_load_data_success(sample_excel, sample_config):
    """
    Ensures successful loading of data and config into a DataSet object when both are valid.
    """
    ds = load_data(str(sample_excel), str(sample_config))
    assert isinstance(ds, DataSet)
    assert not ds.dataframe.empty

def test_invalid_label_column_error(sample_excel, sample_config):
    """
    Expects InvalidLabelColumnError when config points to a label column that doesn't exist in the data.
    """
    config = json.loads(sample_config.read_text())
    config["problem_constraints"]["label_col"] = "not_a_column"
    sample_config.write_text(json.dumps(config))

    with pytest.raises(ce.InvalidLabelColumnError):
        load_data(str(sample_excel), str(sample_config))

def test_missing_feature_column(sample_excel, sample_config):
    """
    Expects MissingFeatureColumnError when a configured feature column is not present in the data.
    """
    config = json.loads(sample_config.read_text())
    config["problem_constraints"]["feature_cols"].append("nonexistent")
    sample_config.write_text(json.dumps(config))

    with pytest.raises(ce.MissingFeatureColumnError):
        load_data(str(sample_excel), str(sample_config))

def test_non_numeric_feature_column(sample_excel, sample_config):
    """
    Expects InvalidFeatureColumnError when a feature column contains non-numeric values.
    """
    df = pd.read_excel(sample_excel)
    df['feature1'] = ['x', 'y', 'z']
    df.to_excel(sample_excel)

    with pytest.raises(ce.InvalidFeatureColumnError):
        load_data(str(sample_excel), str(sample_config))

def test_negative_feature_values(sample_excel, sample_config):
    """
    Expects InvalidFeatureColumnError when a feature column contains non-positive values.
    """
    df = pd.read_excel(sample_excel)
    df['feature2'] = [-1.0, 1.5, 2.5]
    df.to_excel(sample_excel)

    with pytest.raises(ce.InvalidFeatureColumnError):
        load_data(str(sample_excel), str(sample_config))

def test_injective_constraint_missing_label(sample_excel, sample_config):
    """
    Expects InjectiveConstraintsError when a label in the data is not present in the partition keys.
    """
    config = json.loads(sample_config.read_text())
    config["problem_constraints"]["partitions"].pop("B")
    sample_config.write_text(json.dumps(config))

    with pytest.raises(ce.InjectiveConstraintsError):
        load_data(str(sample_excel), str(sample_config))

def test_injective_constraint_count_mismatch(sample_excel, sample_config):
    """
    Expects InjectiveConstraintsError when the partition count for a label exceeds available samples.
    """
    config = json.loads(sample_config.read_text())
    config["problem_constraints"]["partitions"]["A"] = 3  # More than available
    sample_config.write_text(json.dumps(config))

    with pytest.raises(ce.InjectiveConstraintsError):
        load_data(str(sample_excel), str(sample_config))
