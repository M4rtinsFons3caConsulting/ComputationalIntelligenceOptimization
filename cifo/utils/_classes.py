from numpy import sqrt

class PrimeGen:
    def __init__(self):
        """Initialize the prime generator."""
        self.current = 1  # Start checking from 2

    def is_prime(self, n):
        """Check if a number is prime using trial division."""
        if n <= 1:
            return False
        for i in range(2, int(sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def get_prime(self):
        """Yield the next prime dynamically (without precomputing)."""
        while True:
            self.current += 1
            if self.is_prime(self.current):
                yield self.current

# Example usage
# p = PrimeGen()

# # Print the first 10 primes dynamically
# primes = [next(p.get_prime()) for _ in range(10)]
