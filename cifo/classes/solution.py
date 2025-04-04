import numpy as np
import pandas as pd

class Solution:
    def __init__(self, solution_array: np.ndarray, data: pd.DataFrame) -> None:

        self.solution = solution_array
        self.data = data
        self.valid_solution = True
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
    def initialize(cls, seed_matrix, constraints):
        
        start_col = 0
            
        for block_size in constraints.values():
            end_col = start_col + block_size
            
            seed_matrix[:, start_col:end_col] = (
                seed_matrix[np.random.permutation(seed_matrix.shape[0]), :]
                [:, start_col:end_col]
            )
            start_col = end_col
        
        return cls(seed_matrix)
    

    def get_cost(self):
        for row in self.solution:
            if sum(
                self.data[int(index), -1] for index in row
            ) > 750:
                self.valid_solution = False
                return 'Invalid solution'
            

    def get_league_std_dev(self):
        return np.std([
            np.mean(
                [self.data[int(index), -2] for index in row]
            ) for row in self.solution
        ])
