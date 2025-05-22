# rubix/classes/crossover.py

"""
crossover.py - handles the crossover logic for the rubix genetic algorithm.

In the rubix framework, crossover is done between populations,
"""

from torch import Tensor


def get_baseline(
    values: Tensor,
    p: int
    ) -> float:
    if p not in [0, 100]:
        baseline = values.
        
    return baseline, p 


def reset():

