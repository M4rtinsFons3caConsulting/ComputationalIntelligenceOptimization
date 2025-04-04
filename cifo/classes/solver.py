from classes.solution import Solution


class Solver:
    def __init__(self, data, weights, **kwargs):
        # Labels, weights and args
        self.weights = weights 
        self.kwargs = kwargs

        solutions = [
            Solution.initialize(data) for _ in range(self.kwargs['n'])
        ]
        print(solutions)