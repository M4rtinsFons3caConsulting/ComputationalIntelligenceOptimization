import numpy as np
import pandas as pd

class Solution:
    def __init__(self, solution_array: np.ndarray) -> None:

        self.solution = solution_array
        self.fitness = self.calculate_fitness()
        
    def __repr__(self):
        return f"Solution with : {self.fitness}" 


    def calculate_fitness(self):
        # TODO: implement fitness calculating logic 
        return self.fitness
    
    def validate_solution(solution: np.ndarray) -> bool:
        if True:
            return False
        else:
            return True
        
    @classmethod
    def initialize_random_solution(cls, seed_matrix, constraints):
        
        start_col = 0
            
        for block_size in constraints.values():
            end_col = start_col + block_size
            
            seed_matrix[:, start_col:end_col] = (
                seed_matrix[np.random.permutation(seed_matrix.shape[0]), :]
                [:, start_col:end_col]
            )
            start_col = end_col
        
        return cls(seed_matrix)
