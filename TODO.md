# TODO: Tasks for Solution, Solver, and Dataset Classes

## Solution Class

The goal is to implement a manipulator for solutions, which means a solution should be able to modify itself based on requirements. Constraints are treated as genes, and each constraint has specific operations that can be performed:

- **One-column wide constraints**: These can only undergo **roll** or **shuffle** operations.
- **Two-column wide constraints**: These can undergo **roll**, **shuffle**, or **swap** operations.

### Successful Mutations
- **Shuffle**: Can be implemented in various ways, such as:
  - Shuffle gene
  - Shuffle column
  - Shuffle subset (all or by column)
  
  Varying probabilities can be applied to these shuffling operations.

### Crossover
- Crossover occurs on a **gene-by-gene** basis:
  - Select a subset of a gene from each individual.
  - Compare the subsets.
  - For each position where the subsets differ, replace the corresponding element in the other individual's gene with the differing element.
  - Afterward, replace the values at the indices of the subset with the subset values.

### Integration
- **TODO**: Continue implementing this into a solution manipulator class. Review the commented code for pointers on the desired integration and refine the logic accordingly.

---

## Solver Class

The solver class is responsible for instantiating reproduction and societal concepts, which involve the following methods:

### Competition
- Define the process of competition between individuals in the population.

### Selection
- Implement the selection process to choose individuals for reproduction.

### Reproduction
- Define the process of reproduction, including crossover and mutation steps.

### Integration
- **TODO**: Implement the above methods to simulate the evolutionary process and integrate these steps within the solver class.

---

## Dataset Class

The Dataset class is responsible for managing the data pipeline and orchestrating data loading and manipulation tasks. It is not urgent but should be incorporated for improved readability and maintainability, especially if the data manipulation tasks expand.

### Data Pipeline
- **TODO**: Integrate the Dataset class via the loader into the main system, organizing the data pipeline for more efficient management.
  
---

