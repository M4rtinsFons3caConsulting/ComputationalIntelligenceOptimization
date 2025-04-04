from classes.solution import Solution

# FIXME: Solve mismatch in input output to Solution, i.e. Lovy fuck up

class Solver:
    def __init__(self, data, weights, kwargs):
        # Labels, weights and args
        self.weights = weights 
        self.kwargs = kwargs

        solutions = [
            Solution.initialize(data) for _ in range(self.kwargs['n'])
        ]

        print(solutions)


        # while n < 100:
        #     new_individual = transformation()
        #     if new_individual:
        #         n += 1
                