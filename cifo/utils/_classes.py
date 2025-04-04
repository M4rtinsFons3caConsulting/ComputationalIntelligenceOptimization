from typing import Dict, Tuple
import pandas as pd
import numpy as np


class Preprocessor:

    @staticmethod
    def load_data(
        path: str,
        label_col: str,
        partitions: Dict[str, int],
        feature_cols: list[str]
    ) -> np.ndarray:
        """Load and return sorted label and feature columns as NumPy array."""
        df = pd.read_excel(path, index_col='Unnamed: 0')

        df[label_col] = pd.Categorical(
            df[label_col],
            categories=list(partitions.keys()),
            ordered=True
        )
        df = df.sort_values(by=label_col)

        # Reset the indices
        df.reset_index(inplace=True, drop=True)
       
        return df[[label_col] + feature_cols].reset_index().to_numpy()

    @staticmethod
    def get_shape(
        label_values: np.ndarray,
        partitions: Dict[str, int]
    ) -> Tuple[int, int]:
        """Compute solution shape based on label counts and partitions."""
        values, counts = np.unique(label_values, return_counts=True)
        label_counts = dict(zip(values, counts))

        min_set = np.inf
        for key, value in partitions.items():
            cur_min = label_counts.get(key, 0) / value
            min_set = min(min_set, cur_min)

        return int(min_set), sum(partitions.values())

    @classmethod
    def process_data(
        cls,
        path: str,
        label_col: str,
        partitions: Dict[str, int],
        feature_cols: list[str]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """End-to-end: load data, extract labels, and compute shape."""
        data = cls.load_data(path, label_col, partitions, feature_cols)
        label_values = data[:, 0]
        shape = cls.get_shape(label_values, partitions)

        return data, shape
