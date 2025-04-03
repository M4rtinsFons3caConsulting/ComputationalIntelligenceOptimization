from classes.solution import Solution

class Solver:
    def __init__(self, labels, weights, on, partition, **kwargs):
        # Labels, weights and args
        self.labels = labels
        self.weights = weights 
        self.kwargs = kwargs

        # Solution space shape
        self.shape = Solution.find_shape(labels, on, partition)
    
        solutions = [
            Solution(labels) for _ in range(self.kwargs['n'])
        ]
        

    def check_fitness():
        pass