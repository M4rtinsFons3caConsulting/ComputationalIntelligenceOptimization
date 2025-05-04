"""
constants.py - this module centralizes the project's constants.

It contains constants in the following categories:

PROJECT TREE - which are the constant values for the project tree, to facilitate imports and all things 
path related.

PROBLEM CONSTRAINTS -i.e. problem constraints of our chosen problem, in a way that is coherent with our choice
of solution arquitecture.

SOLVER_ARGUMENTS - which defines the valid solver argument namespace, as well as its defaults.
"""


from pathlib import Path


# ----------- PROJECT TREE -------------- #
# Stores path logic for the project, anchoring at ROOT.

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data" 
DATA_V1 =  DATA_DIR / "player_data.xlsx"
DATA_V2 =  DATA_DIR / "player_data(Copy).xlsx"


# ----------- PROBLEM CONSTRAINTS -------------- #
# Stores key values and constraints inherent to the problem space.

LABELS = 'Position'

CONSTRAINTS = {
       'GK' : 1
    , 'DEF' : 2
    , 'MID' : 2
    , 'FWD' : 2
}

WEIGHTS = [
    'Skill'
    , 'Salary (â‚¬M)'
]


# ----------- SOLVER ARGUMENTS -------------- #
# Stores default solver arguments.

SOLVER_KWARGS = {
      'n' : 10_000_000
    , 'epochs': 100
    , '0_m' : 0.05
    , '0_c' : .5
    , 'd_m' : 1e-5
    , 'tol' : 1e-7
}
