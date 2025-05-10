from typing import Tuple
from functools import total_ordering
import torch

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
        return self.rubix_fitness < other.rubix_fitness

    @classmethod
    def class_setup(cls, seed, costs, layout) -> None: 
        cls.seed = seed
        cls.shape = layout['rubix_shape']
        cls.block_indices = layout["block_indices"] 
        cls.block_ranges = layout["block_ranges"]
        cls.cost_arrays = costs 
     
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
        abilities = self.__class__.cost_arrays['arrays'][0][self.solutions]  
        costs = self.__class__.cost_arrays['arrays'][1][self.solutions]          

        # Compute total cost per solution
        total_costs = costs.sum(dim=2)
        row_means = abilities.mean(dim=2)
        
        # Compute the standard deviation of row means for valid solutions
        fitnesses = torch.std(
            row_means[(total_costs <= 750).all(dim=1)],
            dim=1,
            unbiased=True
        )

        # Store all fitnesses
        self.slice_fitnesses = fitnesses

        # Store best fitness
        self.rubix_fitness, self.best_index = torch.min(fitnesses, dim=0)
    
    def get_mappings(
            self,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        n, H, W = self.__class__.shape

        # --- Roll Mapping ---
        roll_flags = torch.rand(W) < 0.5 #TODO: consider making the roll chance a Rubix attribute
        roll_shifts = torch.randint(1, H, size=(W,), dtype=torch.int64)
        roll_map = torch.zeros((W, 2), dtype=torch.int64)
        roll_map[:, 0] = roll_flags.to(torch.int64)
        roll_map[:, 1] = roll_shifts * roll_flags 

        # --- Swap Mapping ---
        swap_map = torch.rand((len(self.__class__.block_indices), H, n)) < 0.5  #TODO: consider making the swap chance a Rubix attribute

        # --- Mode Mapping ---
        mode_map = torch.randint(0, 4, size=(n,), dtype=torch.int64)

        # Return the generated mappings
        return roll_map, swap_map, mode_map

    def rubix_permute(
        self,
        cube: torch.Tensor,             # shape: (n, H, W)
        roll_map: torch.Tensor,         # shape: (W, 2)
        swap_map: torch.Tensor,         # shape: (num_blocks, H, n)
        mode_map: torch.Tensor          # shape: (n,)
    ) -> torch.Tensor:
        
        n, H, W = self.__class__.shape
        block_ranges = self.__class__.block_ranges 

        new_cube = cube.clone()

        for m in range(n):
            mode = mode_map[m].item()
            if mode in {0, 2}:  # swap (only or before roll)
                for b, (start, end) in enumerate(block_ranges):
                    for r in range(H):
                        if swap_map[b, r, m]:
                            perm = torch.randperm(end - start)
                            new_cube[m, r, start:end] = new_cube[m, r, start:end][perm]
    
            if mode in {1, 3}:  # roll (only or before swap)
                for c in range(W):
                    if roll_map[c, 0]:
                        shift = roll_map[c, 1].item()
                        new_cube[m, :, c] = torch.roll(new_cube[m, :, c], shifts=shift, dims=0)
                
            if mode == 3:  # roll then swap
                for b, (start, end) in enumerate(block_ranges):
                    for r in range(H):
                        if swap_map[b, r, m]:
                            perm = torch.randperm(end - start)
                            new_cube[m, r, start:end] = new_cube[m, r, start:end][perm]
                
        return Rubix(new_cube)
    