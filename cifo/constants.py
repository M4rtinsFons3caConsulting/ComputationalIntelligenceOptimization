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


SOLVER_ARGS = None
