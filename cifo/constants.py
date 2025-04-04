# DATA SOURCE
PATH =  "../data/player_data.xlsx"

# CONSTRAINT SET
ON = 'position'

PARTITIONS = {
    'GK' : 1
    , 'DEF' : 2
    , 'MID' : 2
    , 'FWD' : 2
}

WEIGHTS = [
    'ability'
    , 'cost'
]

SOLVER_KWARGS = {
    'n' : 100
    , '0_m' : 0.05
    , '0_c' : .5
    , 'd_m' : 1e-5
    , 'tol' : 1e-7
}