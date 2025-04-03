import constants as c
import pandas as pd
import numpy as np
from utils import PrimeGen

# Each player is defined by the following attributes: 
#  ●   A name that uniquely identifies them 
#  ●   Skill rating: Represents the player's ability. 
#  ●   Cost: The player's salary. 
#  ●   Position:  One  of  four  roles:  Goalkeeper  (GK),  Defender  (DEF),  Midfielder  (MID),  or Forward (FWD).

# Algorithm

# First derive the largest possible number of teams given the available data

# assign each team a unique identifying prime number
# assign each player a unique identifying number (skip the prime numbers chosen for each team)


class Solver:
    def __init__():
        pass
    def check_sum():
        pass
    def check_fitness():
        pass
    def update_tabu():
        pass
    # PERhaps select a number from 1 to 35, and assign
    
class Solution:
    def __init__(self, space_shape, data):

        # Set the space as empty
        self.solution = np.empty(space_shape)
        
        prime_ids = PrimeGen()

        # Assign unique id's to each parent, child
        for i in range(space_shape[0]):
            self.solution[i][0], skip_prime = prime_ids.get_prime() # Associate a prime number to each team
            for j in range(1, space_shape[1]): # Skip the team column
                id_number = j
                if (id_number == skip_prime):
                    id_number += 1 # Associate a number to each spot on each team
                self.solution[i][j] = id_number
                
                

    def assign_child():
        pass  
    


def main():
    # Get data path
    data = pd.read_excel(c.DATA_SOURCE, index_col='Unnamed: 0')

    # Get optimizer arguments
    args = c.OPTIMIZER_ARGS

    # Set an initial solution
    solution = Solution(c.SOLUTION_SPACE_SHAPE, data)




if __name__ == "__main__":
    main()



# We can draft each position at a time,

# We need to have some random starting setup, that is valid
# Then we just move pair wise between any two teams, ofc easiset option
# To speed it up we should probably create a tabu system, that checks if ignoring the team numbers, a specific ordering had been done

# added this here for now
class Solver:
    def __init__(self, num_elements: int):
        if num_elements > 64:
            raise ValueError("Bitmask approach only works for up to 64 elements.")
        self.num_elements = num_elements
        self.seen_combinations = set()
    
    def encode(self, elements: list[int]) -> int:
        """Generate a unique bitmask for a given set of elements."""
        bitmask = 0
        for elem in elements:
            if elem < 0 or elem >= self.num_elements:
                raise ValueError(f"Element {elem} is out of range (0-{self.num_elements-1}).")
            bitmask |= (1 << elem)
        return bitmask
    
    def is_new_combination(self, elements: list[int]) -> bool:
        """Check if this combination is new, and store it if so."""
        bitmask = self.encode(elements)
        if bitmask in self.seen_combinations:
            return False
        self.seen_combinations.add(bitmask)
        return True

# Example usage:
solver = Solver(num_elements=35)
print(solver.is_new_combination([1, 3, 5]))  # True (new combination)
print(solver.is_new_combination([5, 3, 1]))  # False (same as previous)
print(solver.is_new_combination([2, 4, 6]))  # True (new combination)

# Id say the cash one is simple to detect just check if the cash any one team spent after
#  a mutation is larger than 750, if that happens just rerool the mutaiton
# 
# after getting a player, just subtract their cost from cash, if cash goes below zero, reroll?



