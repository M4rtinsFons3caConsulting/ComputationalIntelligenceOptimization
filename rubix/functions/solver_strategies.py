# rubix/solver_strategies.py

STRATEGY_NAMES = [
    'random',
    'greedy',
    'annealing',
    'rubix_search',
    'genetic',
    'rubix_evolve'
]

import torch
import numpy as np
from rubix.classes.cube import Rubix
from rubix.functions.operators import apply_operator
from rubix.functions.crossovers import apply_crossover
from rubix.functions.selectors import apply_selector

def random_search(
    cube: Rubix,
    params: dict
) -> Rubix:

    # Initialize
    n_iter = params['n_iter']
    best_cube = cube
    Rubix.historic_fitness.append(best_cube.rubix_fitness)

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
            
        Rubix.historic_fitness.append(best_cube.rubix_fitness)
            
    return best_cube, Rubix.historic_fitness

def hill_climber(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    n_iter = params.get('n_iter', 1e9)
    max_patience = params.get('patience', 100)
    patience = max_patience

    best_cube = cube
    Rubix.historic_fitness.append(best_cube.rubix_fitness)

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
            
        else:
            patience -= 1

        Rubix.historic_fitness.append(best_cube.rubix_fitness)        
    
    return best_cube, Rubix.historic_fitness

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
    Rubix.historic_fitness.append(best_cube.rubix_fitness)
    
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
        
        Rubix.historic_fitness.append(best_cube.rubix_fitness)        

    return best_cube, Rubix.historic_fitness

def rubix_search(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    epochs = params['epochs']
    patience = params['patience']

    print(f"Running for {epochs} iterations.")

    # Initialize the population (cube)
    best_cube = cube
    Rubix.historic_fitness.append(best_cube.rubix_fitness)
    
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

        Rubix.historic_fitness.append(best_cube.rubix_fitness)
    
    return best_cube, Rubix.historic_fitness

def genetic_evolver(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    epochs = params['epochs']
    patience = params['patience']
    prob_mutation = params['p_mutation']

    
    if params['elitism']:
        elitism = params['elitism']
    else:
        elitism = False
        
    print(f"Running for {epochs} iterations.")

    # Initialize the population (cube)
    best_cube = cube
    Rubix.historic_fitness.append(best_cube.rubix_fitness)

    for epoch in range(epochs):

        if patience != 0:
            print(f"Epoch {epoch + 1}/{epochs}...")
            
            # Apply selection
            selected = apply_selector(
                best_cube.slice_fitnesses,
                **params
            )

            if elitism:
                # Get top `elitism` indices from full population
                elite_indices = torch.topk(best_cube.slice_fitnesses, elitism, largest=False).indices

                # Replace the worst `elitism` individuals in `selected`
                worst_indices = torch.topk(
                    best_cube.slice_fitnesses[selected], 1, largest=True
                ).indices

                for i, w_idx in enumerate(worst_indices):
                    selected[w_idx] = elite_indices[i]
                
            # Apply crossover
            new_slices = apply_crossover(
                best_cube.solutions[selected],
                **params
            )

            # Apply mutation
            if torch.rand(1).item() < prob_mutation:
                new_cube = apply_operator(
                    new_slices,
                    **params
                )
            else:
                new_cube = Rubix(new_slices)

            # Update the best solution if fitness improves
            if new_cube < best_cube:
                best_cube = new_cube
                patience = params['patience']

            else:
                patience -= 1
            
            Rubix.historic_fitness.append(best_cube.rubix_fitness)

    return best_cube, Rubix.historic_fitness

def rubix_evolver(
    cube: Rubix,
    params: dict
) -> Rubix:
    
    epochs = params['epochs']
    patience = params['patience']
    prob_mutation = params['p_mutation']
    
    print(f"Running for {epochs} iterations.")

    # Initialize the population (cube)
    best_cube = cube
    Rubix.historic_fitness.append(best_cube.rubix_fitness)
    cube_population = [best_cube]

    for epoch in range(epochs):
        for cube in cube_population:
            if patience != 0:
                print(f"Epoch {epoch + 1}/{epochs}...")
                
                # Apply  selection
                selected = apply_selector(
                    best_cube.slice_fitnesses,
                    **params
                )
                
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
                
            Rubix.historic_fitness.append(best_cube.rubix_fitness)       
             
    return best_cube, Rubix.historic_fitness

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