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
        df.reset_index(inplace=True)
       
        return df[[label_col] + feature_cols].reset_index().to_numpy()
    
    @staticmethod
    def get_fitness_window_shape(
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

    @staticmethod
    def expand_constraints(
        label_values: np.ndarray,
        partitions: Dict[str, int]
    ) -> Dict[str, list[int]]:
        """Expand column constraints to row targets per label group."""
        values, counts = np.unique(label_values, return_counts=True)
        expanded = {}
        for key, value in zip(values, counts):
            num_cols = partitions[key]
            base = value // num_cols
            remainder = value % num_cols
            expanded[key] = [base + 1 if i < remainder else base for i in range(num_cols)]
        return expanded

    @staticmethod
    def generate_seed(
        label_values: np.ndarray,
        expanded_constraints: Dict[str, list[int]],
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Fill a matrix column-wise based on expanded row counts per label group."""
        seed = np.full(shape, fill_value=np.nan)  

        col_offset = 0
        for label, col_heights in expanded_constraints.items():
            row_indices = np.where(label_values == label)[0]
            np.random.shuffle(row_indices)
            ptr = 0
            for i, height in enumerate(col_heights):
                seed[:height, col_offset + i] = row_indices[ptr:ptr + height]
                ptr += height
            col_offset += len(col_heights)

        return seed

    @classmethod
    def process_data(
        cls,
        path: str,
        label_col: str,
        partitions: Dict[str, int],
        feature_cols: list[str]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Load data, compute expanded constraints, and generate matrix."""
        
        data = cls.load_data(path, label_col, partitions, feature_cols)
        labels = data[:, 1]
        weights = data[:, 2:]

        fitness_window_shape = cls.get_fitness_window_shape(labels, partitions)
        expanded = cls.expand_constraints(labels, partitions)

        total_columns = sum(partitions.values())
        full_rows = max(sum(expanded[label]) for label in expanded)
        full_shape = (full_rows, total_columns)

        seed = cls.generate_seed(labels, expanded, full_shape)

        # print(seed, weights, fitness_window_shape)

        print(weights[np.asarray(seed[~np.isnan(seed)], dtype=int)])



        raise StopIteration("Test end")
        return seed, weights, fitness_window_shape

    
