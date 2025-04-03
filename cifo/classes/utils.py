# Small generator for efficient prime iteration, without poluting the namespace, and address with clutter.
class PrimeGen:
    from constants import PRIMES
    def __init__(self):
        """Initialize the prime generator"""
        # Precomputed list of primes up to 100
        self.primes = PRIMES
        self.index = 0  

    def get_prime(self):
        """Yield the next prime number up to the specified limit."""
        while self.index < len(self.primes):
            yield self.primes[self.index]
            self.index += 1

# Example usage
# primes = PrimeGen(limit=100)

# # Iterate through primes up to the limit
# for prime in primes.get_prime():
#     print(prime)
