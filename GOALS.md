# IMPLEMENTATION

### 1. **Poisoning (Killing 50% of the slices)**
   - **Step 1.1:** Randomly select 50% of the slices from the current cube (the population of solutions). This will simulate the "DL50 poison" effect, where half of the current solutions are removed.
   - **Step 1.2:** Once these slices are identified, set them to some form of invalid state or discard them entirely (e.g., set them to a placeholder solution or simply ignore them in the next steps).
   
### 2. **Selecting Parent Cubes**
   - **Step 2.1:** From the remaining valid slices, randomly select two parent cubes. These cubes will be the sources for the next generation of solutions.
   - **Step 2.2:** Ensure that these cubes have a diverse range of solutions, so you avoid overly similar solutions and maintain diversity in the population.

### 3. **Distribute Individuals Between Parent Cubes**
   - **Step 3.1:** Perform a crossover operation where individuals (solutions) from the parent cubes are mixed.
   - **Step 3.2:** Decide on a method for selecting and distributing the individuals between the two parent cubes. One option could be:
     - Swap entire slices between the two cubes (i.e., entire rows or blocks of solutions).
     - Alternatively, you could perform a more granular mixing by randomly selecting individual solutions (rows, columns, or even smaller blocks) and placing them in the opposite parent.
   - **Step 3.3:** The mixed solutions are then used to form the offspring (the next generation).

### 4. **Evaluate Fitness**
   - **Step 4.1:** Once the new cube (offspring) has been generated, re-compute the fitness values for each of the new solutions.
   - **Step 4.2:** Update the population with the new solutions, ensuring that fitness-based selection will determine the best solutions for future generations.

### 5. **Repeat Optimization**
   - **Step 5.1:** Continue iterating the process for the desired number of epochs (as defined in the solver).
   - **Step 5.2:** Regularly check if the population's fitness improves and maintain diversity across generations to avoid premature convergence.

### Summary of Steps:
1. **Poisoning:** Kill off 50% of the current solutions.
2. **Selection:** Choose two parent cubes with valid solutions.
3. **Crossover:** Distribute solutions between the two parent cubes using slice/block-based mixing.
4. **Fitness Evaluation:** Recompute the fitness of the new offspring.
5. **Repeat:** Continue this process over several generations (epochs).



# REMARKS

### 1. **Crossover and Population Management**

* **Crossover in Evolutionary Algorithms:** In many genetic algorithms (GAs) or evolutionary strategies, crossover operations combine the genetic material (or solutions) of two parent individuals to create offspring. This is akin to what you're aiming for—mixing individuals between two cubes.
* **Selective Population Reduction:** Many evolutionary algorithms include a form of selective pressure, such as killing off individuals or enforcing elitism (where only the best individuals survive). This can be seen as a mechanism to maintain diversity while also pushing the population towards better solutions.
* **Random Removal or Poisoning:** In some evolutionary strategies, a percentage of individuals can be randomly removed or mutated in a "poisonous" manner to introduce randomness, which helps avoid local minima and maintain diversity. For example, "mutation poisoning" is a term used to describe adding noise or chaos to an individual's genome to explore different regions of the solution space.

### 2. **Diversity Preservation in Evolutionary Algorithms**

* **Diversity-driven Selection:** Some algorithms explicitly encourage diversity within the population by introducing strategies like random selection, niching, or crowding. In this context, "killing off" half the population could be a way to control the convergence rate, ensuring that the population doesn't become too homogeneous and that better solutions can emerge from exploration.
* **Generation Skipping or Resetting:** A concept that could be somewhat related is when some algorithms perform generation skipping or population resets to prevent premature convergence. By removing individuals (or perturbing them in some way), the algorithm encourages the exploration of new solution spaces.

### 3. **Poisoning in Evolutionary Computation**

* While "poisoning" in the sense you're describing isn't a common term in evolutionary algorithms, the idea of artificially degrading or removing part of the population for exploration purposes has been studied.
* **Dynamic Fitness Landscapes:** Some algorithms introduce dynamic changes to the fitness landscape, either by altering the cost or fitness functions or by removing individuals from the population (a "poisoning" step) to ensure that the population doesn't get stuck in local optima.
* **Killing Off Individuals:** In more extreme cases, algorithms like "Genetic Drift" or "Cultural Algorithms" might remove or heavily mutate individuals to force the population to adapt in novel ways, similar to your idea.

### Conclusion:

While your exact idea of a "DL50 poison" crossover might not be widely documented, the underlying principles—such as selectively removing individuals, promoting diversity, and then applying a crossover or distribution mechanism—are certainly present in many evolutionary algorithms and optimization strategies. Your approach would be a novel twist on these ideas, with the combination of poison and crossover making for an interesting exploration mechanism.



# ARCHITECTURE

In the context of the architecture you've provided, the methods that enforce the logic you're aiming for—particularly the crossover, poisoning, and population management logic—should be considered a blend of **both the solver logic and cube logic**, though the distinction is a bit nuanced.

Here's a breakdown of how responsibilities might be divided between the **Solver** and **Rubix (Cube)** classes:

### 1. **Solver Logic**

* **Overall Optimization and Evolutionary Strategy:** The `Solver` class manages the evolutionary process and optimization workflow. This includes the high-level orchestration of epochs, fitness evaluation, and managing the population of solutions.
* **Generation Control:** Tasks like selecting the parent cubes, applying the crossover algorithm, and potentially killing off 50% of the slices fall under the **Solver** class because these operations govern the evolution of the population over multiple generations.
* **Epoch Loop:** The loop that runs over several epochs to improve the population and track the best fitness score is part of the `Solver`. It coordinates the larger iterative process, and it will invoke the methods for crossover and population management at each step.

### 2. **Cube Logic (Rubix)**

* **Solution Representation and Fitness Evaluation:** The `Rubix` class is responsible for representing the "cube" or solution space, managing the individual solutions (slices), and computing their fitness. The cube logic will handle the structure of the individual solutions, including how they are represented, permuted, and how their fitness is calculated.
* **Mating Logic (Crossover within Cube):** The `Rubix` class could be responsible for the **low-level crossover logic**, such as how individuals (solutions) are mixed, sliced, or permuted within the parent cubes. In your case, distributing individuals between two cubes could be part of this—manipulating the individual "slices" of the cube to generate offspring.
* **Mapping and Permutation Operations:** The `Rubix` class already handles permutations, roll maps, and block-based operations, so it seems appropriate to keep these responsibilities here. Methods like `rubix_permute()` should likely reside in `Rubix` since they directly manipulate the cube's structure.

### Key Points of Interaction:

* **The Solver** orchestrates the overall optimization process, including crossover, and determines when and how the population should evolve (e.g., after applying crossover and fitness checks).
* **The Cube (Rubix)** represents the solutions themselves and has the internal logic for managing the structure of the individual solutions (like slicing, mixing, and permuting) and their fitness evaluation.

### What Should Go Where?

* **Poisoning / Killing Off 50% of the Population:** This could be handled by the `Solver` because it is part of the population management strategy and evolutionary control. It can modify the cube by either removing solutions entirely or marking them as invalid.

* **Crossover (Distributing Individuals Between Parent Cubes):** This can be a joint responsibility. The **Solver** might handle the high-level decision of selecting the parents and triggering the crossover operation, while the **Rubix** class will implement the crossover logic at the solution level, manipulating the slices, swapping them, or performing the distribution.

### High-Level Architecture Plan:

* **Solver Class:** Responsible for orchestrating the entire optimization process over multiple epochs, including:

  * Managing the population size (including killing off 50%).
  * Selecting parent cubes.
  * Triggering the crossover operation.
  * Evaluating and storing the best fitness scores.
* **Rubix Class:** Responsible for the actual data structure representing a solution (cube) and performing operations like:

  * Initializing solutions.
  * Permuting solutions.
  * Evaluating the fitness of the solutions.
  * Handling the internal crossover (distributing individuals between cubes).

### Conclusion:

* **Solver Logic**: Oversees the evolutionary process, including managing population changes, deciding on when to apply "poisoning," and applying the crossover between cubes.
* **Cube (Rubix) Logic**: Handles the internal logic of solutions, including the fitness evaluation and the low-level operations of permuting and mixing slices for the crossover.

This division ensures that the **high-level optimization** process and **low-level solution manipulation** are properly encapsulated in their respective classes.
