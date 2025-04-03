from pandas import read_excel

from classes.solution import Solution
from classes.solver import Solver

from utils._classes import PrimeGen
from constants import PATH, ON, PARTITIONS, SOLVER_ARGS

def main():
    # Get data path
    data = read_excel(PATH, index_col='Unnamed: 0')

    # Get optimizer arguments
    args = SOLVER_ARGS

    # Find the shape of the data given the constraints on the partitions
    shape = Solution.find_shape(data, ON, PARTITIONS)
    
    # Set an initial solution
    initial_solution = Solution(shape, data)

    # Solve for the given arguments
    final_solution = Solver(data, initial_solution, SOLVER_ARGS)


if __name__ == "__main__":
    main()