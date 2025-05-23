import torch
from functools import total_ordering

@total_ordering
class Rubix:
    """
    A Rubix object is a 3D tensor of 2D solutions. 
    """
    # Class variables
    seed = None
    shape = None
    block_indices = None
    block_ranges = None
    abilities_array = None 
    costs_array = None

    def __repr__(self) -> str:
        return (
            f"Rubix(\n"
            f"  Best index: {self.best_index}\n"
            f"  Best fitness: {self.rubix_fitness.item():.4f})\n"
            f"  Solution: \n"
            f"  {self.solutions[self.best_index]} "
            f"\n)"
        )
    
    # Instance constructor
    def __init__(
        self, 
        tensor: torch.Tensor
    ) -> None:  

        self.solutions = tensor
        self.compute_fitness()

    # __lt__ is defined to proc, total ordering
    def __lt__(self, other) -> bool:
        if isinstance(other, Rubix):
            return self.rubix_fitness < other.rubix_fitness
        try:
            return self.rubix_fitness < other
        except Exception:
            raise AttributeError("Error comparing rubix to {other}.")

    @classmethod
    def class_setup(cls, seed, costs, layout) -> None: 
        cls.seed = seed
        cls.shape = layout['rubix_shape']
        cls.block_indices = layout["block_indices"] 
        cls.block_ranges = layout["block_ranges"]
        cls.valid_swaps = [i for i, (start, end) in enumerate(cls.block_ranges) if (end - start) > 1]
        cls.cost_arrays = costs 
        cls.historic_fitness = []
     
    @classmethod
    def initialize(
        cls, 
    ) -> "Rubix":
        
        """
        Initializes a Cube object with a batch of n solutions by permuting blocks of the seed matrix.

        Args:
            seed_matrix (torch.Tensor): Seed matrix (H, W).
            n (int): Number of solutions to generate.

        Returns:
            Solution: Instance with all solutions and their fitness.
        """

        # The seed is itself a 2D tensor, of shape H, W
        device = cls.seed.device
        n, H, W = cls.shape

        # Initialize empty
        solutions = torch.empty((n, H, W), dtype=torch.long, device=device)

        # Create n slices
        for i in range(n):
            
            new_solution = torch.zeros((H, W), dtype=torch.long, device=device)
            
            for start, end in cls.block_ranges:
                block = cls.seed[:, start:end].flatten()
                permuted_block = block[torch.randperm(block.size(0), device=device)]
                new_solution[:, start:end] = permuted_block.view(-1, end - start)

            solutions[i] = new_solution
        
        return cls(solutions)

    # Instance methods
    def compute_fitness(self) -> None:

        """
        Computes fitness for each solution:
        - If total cost > 750 → fitness = -inf
        - Else → fitness = std(sum(row-wise mean abilities)) for valid solutions only.

        Stores:
            - self.fitnesses: (n,) tensor
            - self.fitness: scalar max fitness
            - self.best_index: index of max fitness
        """

        # Get abilities and costs for the batch (n, H, W)
        abilities = Rubix.cost_arrays['arrays'][0][self.solutions]  
        costs = Rubix.cost_arrays['arrays'][1][self.solutions]          

        # Compute total cost per solution
        total_costs = costs.sum(dim=2)
        row_means = abilities.mean(dim=2)
        
        # Initialize at -inf
        fitnesses = torch.full(
            (self.solutions.size(0),), 
            float('inf'), 
            device=self.solutions.device,
            dtype=row_means.dtype
        )
        
        # Fill the ones that meet the requirement
        fitnesses[(total_costs <= 750).all(dim=1)] = torch.std(
            row_means[(total_costs <= 750).all(dim=1)],
            dim=1,
            unbiased=True
        )

        # Store all fitnesses
        self.slice_fitnesses = fitnesses

        # Store best fitness
        self.rubix_fitness, self.best_index = torch.min(fitnesses, dim=0)
   