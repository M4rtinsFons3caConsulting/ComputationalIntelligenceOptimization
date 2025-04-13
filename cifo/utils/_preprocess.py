import pandas as pd
import numpy as np
from typing import Dict, Tuple

def prepare_data(
        path: str
        , label_col: str
        , partitions: Dict[str, int]
        , feature_cols: list[str]
        , window: Tuple[int, int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    
    # --- LOAD DATA ---
    #
    # Load data from .csv to a pandas DataFrame, and extract to a numpy array.
    df = pd.read_excel(path, index_col='Unnamed: 0')

    # Order and sort the dataset
    df[label_col] = pd.Categorical(
        df[label_col],
        categories=list(partitions.keys()),
        ordered=True
    )
    df = df.sort_values(by=label_col).reset_index()

    # Final data to np.ndarray
    data = df[[label_col] + feature_cols].reset_index().to_numpy()

    # --- EXTRACT LABELS AND FEATURES ---
    labels = data[:, 1] # player position
    weights = data[:, 2:] # player ability, cost

    # --- COMPUTE SOLUTION SHAPE ---
    values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(values, counts))

    min_set = np.inf
    for key, value in partitions.items():
        cur_min = label_counts.get(key, 0) / value
        min_set = min(min_set, cur_min)

    # Solution shape might be an argument
    if window:
        if np.any(window < min_set):
            raise ValueError("Insufficient data provided for the desired window shape")
        else:
            fitness_window_shape = window
    else:
        fitness_window_shape = (int(min_set), sum(partitions.values()))
  
    # --- LABELS PER COLUMN DISTRIBUTION ---
    # Equally distribute the players using their position labels to the available positions. 
    label_col_distribution = {}
    for key, value in zip(values, counts):
        num_cols = partitions[key]
        base = value // num_cols
        remainder = value % num_cols
        label_col_distribution[key] = [base + 1 if i < remainder else base for i in range(num_cols)]
    
    label_col_distribution = {key: label_col_distribution[key] for key in partitions}

    # --- GENERATE SEED MATRIX ---
    # Generate a seed matrix, filled column-wise by player identifier, using the label distribution information.
    total_columns = sum(partitions.values())
    full_rows = max(max(values) for values in label_col_distribution.values())
    full_shape = (full_rows, total_columns)

    seed = np.full(full_shape, fill_value=np.nan)

    col_offset = 0
    for label, col_heights in label_col_distribution.items():
        row_indices = np.where(labels == label)[0]
        ptr = 0
        for i, height in enumerate(col_heights):
            seed[:height, col_offset + i] = row_indices[ptr:ptr + height]
            ptr += height
        col_offset += len(col_heights)

    # --- RETURN OUTPUTS ---
    return seed, weights, fitness_window_shape
