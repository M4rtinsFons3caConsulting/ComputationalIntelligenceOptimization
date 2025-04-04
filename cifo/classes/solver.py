from classes.solution import Solution


class Solver:
    def __init__(self, data, weights, **kwargs):
        # Labels, weights and args
        self.data = data
        self.weights = weights 
        self.kwargs = kwargs

        solutions = [
            Solution() for _ in range(self.kwargs['n'])
        ]
        
    def check_fitness():
        pass