# DATA SOURCE
PATH =  "../data/player_data (Copy).xlsx"

# CONSTRAINT SET
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

# SOLVER ARGUMENTS
SOLVER_KWARGS = {
    'n' : 100
    , '0_m' : 0.05
    , '0_c' : .5
    , 'd_m' : 1e-5
    , 'tol' : 1e-7
}