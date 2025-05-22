# RuBiX: A Primer

RuBiX is a PyTorch-based framework for solving combinatorial optimization problems involving multi-constraint, categorical partitioning into fixed lattice structures.

In essence, it finds the best way to group elements of a population—based on categorical labels—into a predefined shape, while optimizing for weighted objectives and satisfying hard constraints.

What is a Rubix?
A rubix is a 3D tensor, where each slice (a 2D matrix) represents a candidate solution: a specific way to partition the population. These solutions may repeat or differ across the tensor and are evaluated according to optimization criteria.


### Core Concepts

* **Seed Matrix**: The starting configuration of the population, typically a matrix where rows represent elements and columns represent group labels or positions in the lattice.

* **Cost Parameters**: Tensor-encoded objectives (e.g., group balance, distribution fairness) that quantify the desirability of a solution. Each is weighted according to importance.

* **Layout Parameters**: Metadata defining the lattice structure, such as group sizes, spatial layout, and constraints like exclusivity or adjacency.

* **Solver**: An iterative engine that evolves a population of solutions (rubix) using permutation operations and selects the best one based on cost function scores.

---

### Optimization Flow

1. **Ingestion**: Raw data is wrapped in a `DataSet` object, capturing the dataframe, configuration, and solver settings.
2. **Processing**: The `process_data()` function transforms raw input into a structured form, embedding tensor representations and layout metadata.
3. **Evolution**: A `Solver` initializes a Rubix and iteratively generates new solutions through controlled permutations—`roll`, `swap`, and `mode` mappings.
4. **Selection**: After each epoch, the best solution is retained. The final result is the lowest-cost configuration within the lattice constraints.

---

### 🧠 Real-World Analogy: The School Timetable

Imagine you're organizing students (data) into classrooms (lattice) across a fixed schedule (layout), where:

* Each classroom has capacity limits (layout constraints),
* Students belong to categories (e.g., skill levels, grades),
* Your goal is to distribute them such that each classroom is balanced and constraints are respected (cost minimization).

RuBiX helps find the *best arrangement* of students, while navigating constraints and priorities—like fairness, distribution, or exclusivity—using evolutionary optimization.

---

### ⚙️ Code Snippet: Initializing a Raw Dataset

```python
import pandas as pd
from rubix.classes.dataset import DataSet

raw_df = pd.read_csv("students.csv")
dataset = DataSet(
    dataframe=raw_df,
    constructors={"source": "students.csv"}, # FIXME: Correct this example.
    solver_params={"n": 40, "epochs": 50}
)
```

---

### ⚙️ Code Snippet: Processing Data

```python
from rubix.process import process_data

processed_dataset = process_data(dataset)
```

This embeds your raw tabular data into tensors and structures required by the solver.

---

### ⚙️ Code Snippet: Solving the Optimization

```python
from rubix.solver import Solver

solver = Solver(
    seed=processed_dataset.matrix,
    cost_params=processed_dataset.cost_params,
    layout_params=processed_dataset.layout_params,
    solver_params=processed_dataset.solver_params
)

best_solution = solver.solve()
```

---

### 🔁 Under the Hood

Each epoch in `Solver.solve()` applies transformations like:

```python
roll_map, swap_map, mode_map = cube.get_mappings()
new_cube = cube.rubix_permute(cube.solutions, roll_map, swap_map, mode_map)
```

These mimic genetic operators—reconfiguring the cube of solutions to search the space intelligently.

---

Would you like a diagram or sketch of the Rubix tensor structure to complement this?

