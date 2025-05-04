"""
preprocess_input.py

This module provides a structured pipeline for preparing input data used in solving
a constrained optimization problem, such as team formation or resource allocation.

Functions included handle loading, sorting, and preprocessing of structured tabular data 
(e.g., Excel files), inferring feasible solution shapes, distributing constrained labels,
and generating an initial seed matrix for optimization algorithms.

Intended to be used as part of a solver framework where the dataset contains
entity labels (e.g., player roles), associated features (e.g., cost, skill),
and partitioning constraints (e.g., number of roles per team).

Main Output:
    - seed_matrix: 2D array with player indices arranged per constraints
    - weights: Feature matrix
    - constraints: Array of expected counts per label
    - window_shape: Tuple of rows/columns describing feasible subspace

"""

import logging
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

import rubix.exceptions as ce 

def _validate_data(
    df: pd.DataFrame, 
    label_col: str, 
    feature_cols: List[str], 
    partitions: Dict[str, int]
) -> None:
    
    """
    Validates the data after it has been loaded to ensure it meets the necessary criteria.

    Args:
        df: The DataFrame containing the data.
        label_col: The label column name.
        feature_cols: List of feature column names.
        partitions: Dictionary defining expected categories and their counts.

    Raises:
        MissingColumnError: If any required column is missing.
        InvalidFeatureColumnError: If any feature column is invalid.
        InvalidLabelColumnError: If the label column is missing or invalid.
        InjectiveConstraintsError: If the label column is not injective with partitions keys.
    """

    # Check if label column exists in DataFrame
    if label_col not in df.columns:
        raise ce.MissingColumnError(f"Label column '{label_col}' is missing in the data.")
    
    # Ensure that each label in the label column has a corresponding key in the partitions dictionary
    labels_in_data = set(df[label_col].unique())
    labels_in_partitions = set(partitions.keys())
    
    if not labels_in_data.issubset(labels_in_partitions):
        missing_labels = labels_in_data - labels_in_partitions
        raise ce.InjectiveConstraintsError(f"The following labels in '{label_col}' are not in the partitions keys: {missing_labels}")
    
    # Check if feature columns exist and are numeric with values greater than zero
    for col in feature_cols:
        if col not in df.columns:
            raise ce.MissingColumnError(f"Feature column '{col}' is missing in the data.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ce.InvalidFeatureColumnError(f"Feature column '{col}' is not numeric.")
        if (df[col] <= 0).any():
            raise ce.InvalidFeatureColumnError(f"Feature column '{col}' contains non-positive values.")

    # Log the success of validation
    logging.info("Data validation passed successfully.")

def load_data(
    path: str, 
    label_col: str, 
    feature_cols: List[str], 
    partitions: Dict[str, int]
) -> np.ndarray:
    
    """
    Loads and preprocesses data from an Excel file into a structured NumPy array.
    
    Args:
        path: Path to the input Excel file.
        label_col: Name of the column used for labels.
        feature_cols: List of feature column names.
        partitions: Dictionary defining expected categories and their counts.

    Returns:
        Numpy array with label and feature data.
    
    Raises:
        DataValidationError: If the data does not pass validation.
    """

    df = pd.read_excel(path, index_col='Unnamed: 0')

    # Perform data validation
    try:
        _validate_data(df, label_col, feature_cols, partitions)
    except ce.DataValidationError as e:
        logging.error(f"Data validation failed: {e}")
        raise
    
    # Process label column and features
    df[label_col] = pd.Categorical(df[label_col], categories=list(partitions.keys()), ordered=True)
    df = df.sort_values(by=label_col).reset_index()
    
    # Return the data in a numpy array
    return df[[label_col] + feature_cols].reset_index().to_numpy()


def compute_fitness_window_shape(
    labels: np.ndarray, 
    partitions: Dict[str, int], 
    window: Tuple[int, int] = None
) -> Tuple[int, int]:
    
    """
    Determines the size of the solution window, either from input or by inferring from data.

    Args:
        labels: Array of class labels.
        partitions: Dictionary of role counts per class.
        window: Optional user-defined shape.

    Returns:
        A tuple indicating the inferred or validated shape of the fitness window.
    """

    # Obtain label counts
    values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(values, counts))

    # Calculate the smallest possible window assuming a column space equal to the sum of block widths. 
    min_set = np.inf
    for key, value in partitions.items():
        cur_min = label_counts.get(key, 0) / value
        min_set = min(min_set, cur_min)

    # If a window is provided, check if there is sufficient data for it 
    if window:
        if np.any(window < min_set):
            raise ValueError("Insufficient data provided for the desired window shape")
        return window
    return (int(min_set), sum(partitions.values()))


def distribute_labels(
    labels: np.ndarray, 
    partitions: Dict[str, int]
) -> Dict[str, List[int]]:
    
    """
    Distributes player counts per label to column slots as evenly as possible.

    Args:
        labels: Array of player labels.
        partitions: Dictionary of label-to-column mappings.

    Returns:
        A dictionary mapping each label to a list of counts per column.
    """
    # Obtain the label counts
    values, counts = np.unique(labels, return_counts=True)

    # Distribute entries by partition key, equally into each block distributing any remainder. 
    distribution = {}
    for key, value in zip(values, counts):
        num_cols = partitions[key]
        base = value // num_cols
        remainder = value % num_cols
        distribution[key] = [base + 1 if i < remainder else base for i in range(num_cols)]

    return {key: distribution[key] for key in partitions}


def generate_seed_matrix(
    labels: np.ndarray, 
    distribution: Dict[str, List[int]]
) -> np.ndarray:
    
    """
    Constructs a matrix filled with indices for use as a seed in the optimization process.

    Args:
        labels: Array of player labels.
        distribution: Dictionary of how many players go in each column per label.

    Returns:
        A seed matrix (2D NumPy array) with player indices placed appropriately.
    """

    # Initialize an empty matrix of the desired size
    total_columns = sum(len(cols) for cols in distribution.values())
    full_rows = max(max(heights) for heights in distribution.values())
    seed = np.full((full_rows, total_columns), fill_value=np.nan)

    # Fill with indices that point to players valid for a given position
    col_offset = 0
    for label, col_heights in distribution.items():
        row_indices = np.where(labels == label)[0]
        ptr = 0
        for i, height in enumerate(col_heights):
            seed[:height, col_offset + i] = row_indices[ptr:ptr + height]
            ptr += height
        col_offset += len(col_heights)

    return seed


def data_loader(
    path: str,
    label_col: str,
    partitions: Dict[str, int],
    feature_cols: List[str],
    window: Tuple[int, int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    
    """
    Main data preparation pipeline. Loads and processes data, builds seed matrix and solution window.

    Args:
        path: Path to the input file.
        label_col: Label column name.
        partitions: Dictionary with number of expected players per label.
        feature_cols: List of features to use.
        window: Optional window shape.

    Returns:
        Tuple containing the seed matrix, features matrix, constraints array, and fitness window shape.
    """
    print(locals()) 
    # Load the data
    data = load_data(path, label_col, feature_cols, partitions)

    # Slice the data
    labels = data[:, 1]
    weights = data[:, 2:]
    constraints = np.array(list(partitions.values()))

    # Get window shape
    window_shape = compute_fitness_window_shape(labels, partitions, window)
    
    # Get label distribution
    label_distribution = distribute_labels(labels, partitions)
    
    # Fill seed matrix
    seed_matrix = generate_seed_matrix(labels, label_distribution)

    return seed_matrix, weights, constraints, window_shape
