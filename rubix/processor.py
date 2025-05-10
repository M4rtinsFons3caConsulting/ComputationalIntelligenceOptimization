import torch
import numpy as np
from typing import Dict, List
from itertools import accumulate
from collections import Counter
from rubix.classes.dataset import DataSet


def generate_seed_matrix(
    labels: np.ndarray,
    distribution: Dict[str, List[int]]
) -> np.ndarray:
    
    total_cols = sum(len(v) for v in distribution.values())
    max_rows = max(max(v) for v in distribution.values())
    seed = np.full((max_rows, total_cols), np.nan)

    col_idx = 0
    for label, heights in distribution.items():
        indices = np.where(labels == label)[0]
        ptr = 0
        for i, height in enumerate(heights):
            seed[:height, col_idx + i] = indices[ptr:ptr + height]
            ptr += height
        col_idx += len(heights)

    return seed

def process_data(
    dataset: DataSet
) -> DataSet:
    df = dataset.dataframe
    label_col = dataset.constructors['label_col']
    partitions = dataset.constructors["constraints"]
    window = dataset.constructors["window"]
    weights = dataset.constructors["weights"]

    label_counts = Counter(df[label_col])
    label_distribution = {
        key: [
            count // partitions[key] + (1 if i < count % partitions[key] else 0)
            for i in range(partitions[key])
        ]
        for key, count in label_counts.items()
    }

    max_label_distribution = max(max(dist) for dist in label_distribution.values())
    window_shape = (
        max_label_distribution,
        sum(partitions.values())
    )

    if window:
        if any(w > ws for w, ws in zip(window, window_shape)):
            raise ValueError("Insufficient data for the provided window shape.")
        window_shape = window

    block_indices = [0] + list(accumulate(partitions.values()))[:-1]
    block_ranges = [
        (block_indices[i], block_indices[i+1] if i+1 < len(block_indices) else window_shape[1])
        for i in range(len(block_indices))
    ]

    seed_matrix = generate_seed_matrix(df[label_col], label_distribution)

    cost_arrays = [torch.tensor(df[col].values, dtype=torch.float64) for col in weights]
    weight_lookup = {i: col for i, col in enumerate(weights)}

    return dataset.update(
        matrix=torch.tensor(seed_matrix, dtype=torch.int64),
        layout_params={
            "block_indices": block_indices,
            "block_ranges": block_ranges,
            "n_cols": window_shape[0],
            "n_rows": window_shape[1],
            "n_layers": dataset.solver_params['n'],
            "window": window_shape,
            "rubix_shape": (dataset.solver_params['n'], *window_shape)
        },
        cost_params={
            "arrays": cost_arrays,
            "lookup": weight_lookup
        }
    )
