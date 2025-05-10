# rubix.configs

This folder contains configuration files used to set up and customize the data loading and solver process for the Rubix optimization framework. The configuration files are written in JSON format, which allows for easy data manipulation and setup.

## File Structure

The current configuration structure consists of two main sections:

### 1. `problem_constraints`

This section defines the problem-related data for the optimization solver. It includes the following fields:

- `label_col`: (String) The name of the column in the dataset that contains the label or category for each record (e.g., "position").
  
- `weights`: (Array of Strings) A list of feature columns (e.g., "Skill", "Salary") that will be used as weights for optimization.
  
- `constraints`: (Object) A dictionary that defines how many entites are required, for instance:
  - `"GK": 1`: Requires 1 goalkeeper.
  - `"DEF": 2`: Requires 2 defenders.

### Example:

```json
"problem_constraints": {
    "label_col": "position",
    "weights": [
        "Skill", 
        "Salary (â‚¬M)"
    ],
    "constraints": {
        "GK": 1,
        "DEF": 2,
        "MID": 2,
        "FWD": 2
    }
}
```

### 2. `solver_args`

This section defines the parameters for the optimization solver, which are passed into the solver function. It includes the following fields:

- `n`: (Integer) The total number of iterations or population size for the optimization.
epochs: (Integer) The number of epochs for the optimization solver to run.
- `0_m`: (Float) A parameter for the solver (possibly related to momentum or similar).
- `0_c`: (Float) Another parameter for the solver (potentially a constant or coefficient).
- `d_m`: (Float) A parameter for determining the solver's dynamics (perhaps related to decay or movement).
- `tol`: (Float) The tolerance or convergence threshold for the optimization.

### Example:
```json

"solver_args": {
    "n": 1000000,
    "epochs": 1,
    "0_m": 0.05,
    "0_c": 0.5,
    "d_m": 1e-5,
    "tol": 1e-7
}
```

## Usage

To use the configurations, simply reference the JSON files within your Rubix framework. These files will be loaded into the framework, and their values will guide the solver's behavior and the data loading process.


## Full configuration file example

Below is an example of a complete configuration JSON file:

```json
{
    "problem_constraints": {
        "label_col": "position",
        "weights": [
            "Skill", 
            "Salary (â‚¬M)"
        ],
        "constraints": {
            "GK": 1,
            "DEF": 2,
            "MID": 2,
            "FWD": 2
        }
    },
    "solver_args": {
        "n": 1000000,
        "epochs": 1,
        "0_m": 0.05,
        "0_c": 0.5,
        "d_m": 1e-5,
        "tol": 1e-7
    }
}
```

This JSON configuration will be parsed by the Rubix framework to initialize the dataset and solver with the specified parameters.

## Closing Remarks

The use of configuration files offer a flexible and user-friendly way to customize the Rubix optimization framework. This setup allows for easy updates to problem constraints and solver parameters, enabling adaptation to a wide range of use cases. It goes beyond the initial problem layout, solving for a general case with an undefined number of elements per sub-group and adjustable weights for any given label. At the same time, it supports efficient parameterization, storage, and future optimization of implementation hyperparameters. 