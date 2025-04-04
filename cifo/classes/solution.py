import numpy as np
import pandas as pd


class Solution:
    def __init__(self, solution_array: np.ndarray, data) -> None:
        self.solution = solution_array
        self.data = data
        self.valid_solution = True
        self.fitness = self.calculate_fitness()
        

    def __repr__(self):
        return f"Solution with : {self.fitness}" 


    def calculate_fitness(self):
        self.fitness = np.std([
            np.mean(
                [self.data[int(index), -2] for index in row]
            ) for row in self.solution
        ])

        return self.fitness
    

    def validate_solution(self) -> bool:
        for row in self.solution:
            if sum(
                self.data[int(index), -1] for index in row
            ) > 750:
                # If team cost > 750, invalid solution
                return False
        
        return True
        
    @classmethod
    def initialize(cls, seed_matrix, constraints):
        
        start_col = 0
            
        for block_size in constraints:
            end_col = start_col + block_size
            
            seed_matrix[:, start_col:end_col] = (
                seed_matrix[np.random.permutation(seed_matrix.shape[0]), :]
                [:, start_col:end_col]
            )
            start_col = end_col
        
        return seed_matrix
    

    @classmethod
    def reproduce(cls, parent_1, parent_2, constraints):
        pass 
        # manipulate parents
        indiv_matrix = None
        individual = cls(indiv_matrix)

        if individual.validate_solution():
            return individual
        else: 
            return None
