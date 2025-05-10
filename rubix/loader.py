import json
import logging
from typing import Dict, List
import pandas as pd
from rubix.classes.dataset import DataSet
import rubix.exceptions as ce

def load_config(
    config_path: str
) -> dict:
    """
    Loads configuration from a JSON file.
    
    Args:
        config_path: Path to the config JSON file.
    
    Returns:
        A dictionary with the configuration data.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def _validate_data(
    df: pd.DataFrame, 
    label_col: str, 
    feature_cols: List[str], 
    partitions: Dict[str, int]
) -> None:
    """
    Validates the loaded data ensuring it matches the expectations for labels, features, and partitions.
    """

    if label_col not in df.columns:
        raise ce.InvalidLabelColumnError(f"Label column '{label_col}' is missing in the data.")

    labels_in_data = set(df[label_col].unique())
    labels_in_partitions = set(partitions.keys())

    if not labels_in_data.issubset(labels_in_partitions):
        missing_labels = labels_in_data - labels_in_partitions
        raise ce.InjectiveConstraintsError(f"Labels {missing_labels} are missing from partitions.")
      
    for col in feature_cols:
        if col not in df.columns:
            raise ce.MissingFeatureColumnError(f"Feature column '{col}' is missing in the data.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ce.InvalidFeatureColumnError(f"Feature column '{col}' is not numeric.")
        if (df[col] <= 0).any():
            raise ce.InvalidFeatureColumnError(f"Feature column '{col}' contains non-positive values.")

    logging.info("Data validation passed successfully.")

def load_data(
    path: str,
    config_path: str
) -> DataSet:
    """
    Main function to load the dataset. It combines configuration loading, data loading,
    validation, and preparation all in one function.
    
    Args:
        path: Path to the data file.
        config_path: Path to the configuration file (JSON).
    
    Returns:
        DataSet: The processed DataSet object.
    """
    
    # Load the configuration
    config = load_config(config_path)
    constraints = config['problem_constraints']
    solver_args = config['solver_kwargs']

    # Extract necessary info from config
    label_col = constraints['label_col']
    feature_cols = constraints['weights']
    blocks = constraints['constraints']
    
    # Load raw data from Excel
    df = pd.read_excel(path, index_col='Unnamed: 0')
    
    # Validate the loaded data
    _validate_data(df, label_col, feature_cols, blocks)
    
    # Process label column and features into the DataFrame
    df[label_col] = pd.Categorical(df[label_col], categories=list(blocks.keys()), ordered=True)
    df = df.sort_values(by=label_col).reset_index(drop=True)

    # Create the initial DataSet object with raw data
    return DataSet(
        dataframe=df, 
        constructors=constraints,
        solver_params= solver_args
        )
