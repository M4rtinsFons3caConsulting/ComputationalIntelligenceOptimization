from utils._classes import PrimeGenerator, IdGenerator

class Solver:
    @staticmethod
    def get_checksums(shape):
        
        group_id = [next(PrimeGenerator.get_prime()) for _ in range(shape[0])]

        elem_id = []
        for _ in range(shape[0], shape[1]):
            n = next(IdGenerator.get_id())
            if n in group_id:
                pass
            else:
                elem_id.append(n)

    def __init__():
        pass

    def check_sum():
        pass

    def check_fitness():
        pass

    def update_tabu():
        pass
        

def generate_elem_ids(start=0):
    """A generator that yields the next integer, starting from `start`."""
    num = start
    while True:
        yield num
        num += 1

# Example usage:
elem_gen = generate_elem_ids()

# Get the first 5 values
for _ in range(5):
    print(next(elem_gen))