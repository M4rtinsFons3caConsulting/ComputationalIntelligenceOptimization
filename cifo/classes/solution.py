from numpy import empty, inf, sum
from typing import Dict
import pandas as pd

class Solution:
    @staticmethod 
    def find_shape(data: pd.DataFrame, on: str, partition: Dict[str, int]) -> tuple:
        """
        Compute the shape of the solution space based on dataset constraints.

        Parameters:
        - data: The input dataset (DataFrame).
        - on: The target column within the dataset, used as the group-by variable.
        - partition: A dictionary specifying how to partition each group in `on`.

        Returns:
        - shape: A tuple representing the computed shape of the solution space.
        """
        
        data = data.copy()
        partition_count = data.groupby(by=on)[on].count()

        min_set = inf
        for key, value in partition.items():
            cur_min = partition_count.get(key, 0) / value
            if cur_min < min_set:
                min_set = cur_min

        shape = (min_set, sum(partition.values()))
        return shape
    
    def __init__(self, shape: tuple, data: pd.DataFrame) -> None:
        self.solution = empty(shape)
        self.data = data

    def assign_child(self) -> None:
        pass
