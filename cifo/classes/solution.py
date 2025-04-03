from numpy import array, sum, empty, inf, unique
from typing import Dict
import pandas as pd


class Solution:
    @staticmethod 
    def find_shape(label: array, partition: Dict[str, int]) -> tuple:
        """
        Compute the shape of the solution space based on dataset constraints.

        Parameters:
        - label: The array of labels.
        - on: The target column within the dataset, used as the group-by variable.
        - partition: A dictionary specifying how to partition each group in `on`.

        Returns:
        - shape: A tuple representing the computed shape of the solution space.
        """
        
        # Count occurrences of each label using np.unique
        partition_count, _ = unique(label, return_counts=True)
        
        # Convert partition_count to a dictionary for easier access
        partition_count = dict(zip(partition_count, _))

        min_set = inf
        for key, value in partition.items():
            # Get the count of the current key, default to 0 if not found
            cur_min = partition_count.get(key, 0) / value
            if cur_min < min_set:
                min_set = cur_min

        shape = (min_set, sum(partition.values()))
        return shape
    

    def __init__(self, shape: tuple, data: pd.DataFrame) -> None:
        self.solution = empty(shape)

