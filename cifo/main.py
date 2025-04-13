"""
Authors' Note:

Despite the data provided for the resolution of the problem turning out to 
have all players distributed into all teams, we assumed that our program should be able to 
work in the general case—as was the case before the teacher corrected the data :) —
and that the shape of the largest solution possible (i.e., the largest number of teams 
for which a solution exists) was one less, due to the fact that one player would be
left out.

For this reason, the dimensions of the problem space are not defined beforehand but are inferred
from the dataset provided and the constraints imposed by the caller. This inference is done dynamically
based on the number of players and feasible team configurations.

Moreover, we believe this solution will differ significantly from others. Despite using
OOP, we have abstracted the league as a window within a matrix—called the fitness window—
which's size is determined either through programmatic inference or through user input. This fitness
window defines the operational subspace in which valid team configurations are evaluated.

"""

from constants import PATH, LABELS, CONSTRAINTS, WEIGHTS, SOLVER_KWARGS
from  utils._preprocess import prepare_data
from classes.solver import Solver


def get_args():
    print('Please implement me :(')


def main():
    """Main execution routine."""
    
    # Preprocess the data
    seed_matrix, weights_matrix, window_shape = prepare_data(
        path=PATH,
        label_col=LABELS,
        partitions=CONSTRAINTS,
        feature_cols=WEIGHTS
    )

    # Solve the optimization problem
    final_solution = Solver(
        seed_matrix
        , weights_matrix
        , window_shape
        , CONSTRAINTS
        , SOLVER_KWARGS
        )

    # Print final solution
    print("Solution:", final_solution)


if __name__ == "__main__":
    # if argpars in None: TODO: set this up
    #     while True:
    #         try:
    #             if input("DO you want to give args? y to yes, other to "):
    #                 get_args()
    #         Exception as e:
    #             print('Please provide')
    #             continue
    main()
