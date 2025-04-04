import pandas as pd
import numpy as np
from typing import Dict, Tuple


class Preprocessor:

    @staticmethod
    def load_data(path: str, on: str, weights: list[str]) -> np.ndarray:
        """Load and return sorted label and weights as NumPy array."""
        df = pd.read_excel(path, index_col='Unnamed: 0')
        df = df.sort_values(by=on)
        return df[[on] + weights].to_numpy()

    @staticmethod
    def get_shape(labels: np.ndarray, partitions: Dict[str, int]) -> Tuple[int, int]:
        """Compute solution shape based on label counts and partitions."""
        values, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(values, counts))

        min_set = np.inf
        for key, value in partitions.items():
            cur_min = label_counts.get(key, 0) / value
            min_set = min(min_set, cur_min)

        return int(min_set), sum(partitions.values())

    @classmethod
    def process_data(cls, path: str, on: str, partitions: Dict[str, int], weights: list[str]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """End-to-end: load data, extract labels, and compute shape."""
        data = cls.load_data(path, on, weights)
        labels = data[:, 0]
        shape = cls.get_shape(labels, partitions)
        return data, shape
