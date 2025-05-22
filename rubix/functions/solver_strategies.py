# rubix/solver_strategies.py

STRATEGY_NAMES = [
    'random',
    'greedy',
    'annealing',
    'rubix_search',
    'genetic',
    'rubix_evolve'
]


import numpy as np
from rubix.classes.cube import Rubix
from rubix.functions.operators import apply_operator

def random_search(
    cube: Rubix,
    params: dict
) -> Rubix:

    # Initialize
    n_iter = params['n_iter']
    best_cube = cube

    print(f"Running for {n_iter} iterations.")

    # Iter
    for _ in range(n_iter):
        new_cube = apply_operator(
            None, 
            **params
        )

        # Update the best solution if fitness improves
        if new_cube < best_cube:
            best_cube = new_cube
            
    return best_cube

def hill_climber(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    n_iter = params.get('n_iter', 1e9)
    max_patience = params.get('patience', 100)
    
    best_cube = cube
    patience = max_patience

    for epoch in range(n_iter):
        if patience == 0:
            break

        print(f"Epoch {epoch + 1}/{n_iter}...")
        new_cube = apply_operator(
            best_cube.solutions, 
            **params
        )

        if new_cube < best_cube:
            best_cube = new_cube
            patience = max_patience
            print(best_cube)
        else:
            patience -= 1

    return best_cube

def annealer(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    temperature = params['temperature']
    base_decay = params['decay_rate']
    min_decay = params['min_decay']
    k = params['k']
    tol = params['tol']
    best_cube = cube
    
    k = 1
    improve_threshold = 0.5

    while temperature > tol:

        new_cube = apply_operator(
            best_cube.solutions, 
            **params
        )
        
        if new_cube < best_cube:
            improvement = (best_cube.rubix_fitness - new_cube.rubix_fitness) / best_cube.rubix_fitness

            logistic_decay = 1 / (
                1 + np.exp(
                    -k * (improvement - improve_threshold)
                    )
                )
            
            improve_threshold = improvement

            decay_rate = max(
                base_decay / (10 * (1 + logistic_decay)),
                min_decay
            )

            best_cube = new_cube

        else:
            decay_rate = base_decay

        temperature *= (1 - decay_rate)
        print(best_cube, temperature)

    return best_cube

def rubix_search(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    epochs = params['epochs']
    patience = params['patience']

    print(f"Running for {epochs} iterations.")

    # Initialize the population (cube)
    best_cube = cube
    
    for epoch in range(epochs):

        if patience != 0:
            print(f"Epoch {epoch + 1}/{epochs}...")
            
            new_cube = apply_operator(
                best_cube.solutions,
                **params
            )
           
            # Update the best solution if fitness improves
            if new_cube < best_cube:
                best_cube = new_cube
                patience = params['patience']

            else:
                patience -= 1
            
    return best_cube

def genetic_evolver():
    pass

def rubix_evolver():
    pass

STRATEGY = \
    dict(
        zip(
            STRATEGY_NAMES, 
            [
                random_search,
                hill_climber,
                annealer,
                rubix_search,
                genetic_evolver,
                rubix_evolver
            ]
        )
    )