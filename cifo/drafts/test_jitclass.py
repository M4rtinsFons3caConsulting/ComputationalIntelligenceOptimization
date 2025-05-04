import time
from numba.experimental import jitclass
from numba import float64, int32
import numpy as np

# Define the class specification
spec = [
    ('solution', float64[:]),  # The solution array is a 1D array of floats
    ('fitness', int32),        # Fitness is an integer
    ('last_index', int32),     # The last index is an integer
]

@jitclass(spec)
class Solution:
    def __init__(self, solution):
        self.solution = solution
        self.last_index = len(self.solution) - 1  # Set last_index during initialization
        self.update_fitness()  # Update fitness when initializing
    
    def roll(self, shift):
        """Roll the solution array by 'shift' positions using np.roll, keeping fitness unchanged."""
        self.solution = np.roll(self.solution, shift) 
    
    def update_fitness(self):
        """Update fitness to the last value of the solution array."""
        self.fitness = self.solution[self.last_index]
    
    def __lt__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness < other.fitness

    def __le__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness <= other.fitness
    
    def __gt__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness >= other.fitness
    
    def __eq__(self, other):
        """Check if the fitness values between self and another Solution are equal."""
        return self.fitness == other.fitness

sol1 = Solution(np.array([1.0, 1.0, 1.0]))
sol2 = Solution(np.array([0.0, 2.0, 0.0]))

while True:

    # Check the minimum fitness value before rolling
    print("Solution with minimum fitness before rolling:", min(sol1, sol2).fitness)

    # Roll the rings
    sol1.roll(1)
    sol2.roll(1)

    # Update the fitness after the roll
    sol1.update_fitness()
    sol2.update_fitness()
    time.sleep(1)


  #  TODO: Continue implementing this into a solution manipulator for class
  #
  # This means a solution should be able to modify itself based on requirements
  # and as constraints are genes, then 
  # 
  # one column wide constraints can only roll, or shuffle
  # two column wide constraints can roll, shuffle or swap
  #
  # and these operations are the successfull mutations.
  # shuffle can be implemented as all of: shuffle gene, shuffle column, or shuffle subset all, shuffle subset column,
  # with varying degrees of probability 

  # 
  # crossover ought to happen on a gene by gene basis as:
  # 
  # select a subset of a gene from each two individuals
  # compare the subsets, for each position in which they differ, find the corresponding element
  # in the other individuals gene and replace with the differing element
  #
  # then replace the values at the indices of the subset with the subset values.
  # 
  # repeat for each gene 


  # MAP THE INDICES TO PRIMES, so that the numbers in the matrix are already the primes
  # Calculate the row value hash, that is, the particular combination of players for a team
  # by multiplying the values of the players (indices)
