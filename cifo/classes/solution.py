import numpy as np
from typing import List
from numba import njit

# Numba Just in Time helpers
@njit
def permute_blocks(
      matrix: np.ndarray
    , constraint_values: np.ndarray
    ) -> np.ndarray:

    start_col = 0

    for block_size in constraint_values:
        end_col = start_col + block_size

        matrix[:, start_col:end_col] = matrix[np.random.permutation(matrix.shape[0]), start_col:end_col]
        start_col = end_col
    
    return matrix.astype(np.int32)
     

class Solution:   
    abilities_array = None
    costs_array = None
    
    def __init__(
              self
            , solution_array: np.ndarray
            ) -> None:

            self.solution = solution_array 


    def __repr__(
              self
            ) -> int:
        
        return f"Solution with : {self.fitness}" 


    def validate_solution(
         self
        ) -> bool:
        
        return np.all(np.sum(Solution.costs_array[self.solution], axis=1) <= 750)
       
       
    def calculate_fitness(
          self,
        ) -> int:

        self.fitness = np.mean(np.std(Solution.ability_array[self.solution].astype(np.float64), axis=1))

             
    @classmethod
    def set_weights(cls, weights):
        cls.ability_array, cls.costs_array = weights[:, 0], weights[:, 1]

    @classmethod
    def initialize(
        cls: type["Solution"],
        seed_matrix: np.ndarray,
        constraints: np.ndarray,
        n: int = 100
    ) -> List["Solution"]:
        
        solutions = []

        while len(solutions) < n:

            new_solution = cls(permute_blocks(seed_matrix, constraints))
            
            if new_solution.validate_solution():
                new_solution.calculate_fitness()
                solutions.append(new_solution)

        return solutions