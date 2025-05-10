# rubix/classes/dataset.py

"""
This module defines the `DataSet` class, which serves as an immutable container for raw and processed 
data used in optimization tasks within the framework. The `DataSet` class is designed to manage tabular 
data and its transformation into tensor-based representations, facilitating the optimization process.

Key features include:
- Storing raw data in the form of a `pandas.DataFrame`.
- Storing transformed data as a `torch.Tensor` in the `matrix` attribute.
- Storing derived metadata such as cost parameters, constraints, and window dimensions in a dictionary.
- Providing methods for updating the data in an immutable manner while keeping the original object intact.
- Custom string representation for easy inspection of the dataset's state and contents.

The `DataSet` class is intended to be used for managing data through different stages of processing in the 
optimization pipeline, where raw data may be transformed into a structured matrix and additional metadata 
is added for use in various optimization algorithms.

Classes:
    DataSet: Immutable container for data used in optimization, supporting both raw and processed forms.
"""

import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, replace
from torch import Tensor

@dataclass(frozen=True)
class DataSet:
    """
    DataSet is an immutable container for raw and processed data used in the optimization framework.

    Attributes:
        dataframe (pd.DataFrame): The original or ingested tabular data.
        matrix (Optional[Tensor]): Tensor form of the seed matrix used in optimization.
        cost_params (Optional[Dict[str, Tensor]]): Parameters used in optimization (e.g., weights for cost function).
        layout_params (Optional[Dict[str, Any]]): Metadata for the structure and layout of the data.
        solver_params (Optional[Dict[str, Any]]): Additional solver arguments or settings for the optimization process.
        constructors (Optional[Dict[str, Any]]): Metadata and arguments used during the initial transformation or preprocessing.
    """
    
    # Raw
    dataframe: pd.DataFrame
    constructors: Optional[Dict[str, Any]] = field(default_factory=dict) 
    solver_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Processed
    matrix: Optional[Tensor] = None
    cost_params: Optional[Dict[str, Tensor]] = field(default_factory=dict)
    layout_params: Optional[Dict[str, Any]] = field(default_factory=dict)
            
    def __repr__(self) -> str:
        stage = "Processed" if self.matrix is not None else "Raw/Ingested"

        parts = [
            f"DataSet(",
            f"  stage: {stage},\n",
            f"  dataframe:\n{self.dataframe.head()}\n",
        ]

        if self.constructors:
            parts.append("  constructors:\n" + "\n".join(
                f"    {k}: {repr(v)}" for k, v in self.constructors.items()) + "\n")
        if self.solver_params:
            parts.append("  solver parameters:\n" + "\n".join(
                f"    {k}: {repr(v)}" for k, v in self.solver_params.items()) + "\n")

        if stage == "Processed":
            matrix_preview = self.matrix[:5] if self.matrix.ndim == 2 else self.matrix
            parts.append(f"  matrix:\n{repr(matrix_preview)}\n")

            if self.layout_params:
                parts.append("  layout parameters:\n" + "\n".join(
                    f"    {k}: {repr(v)}" for k, v in self.layout_params.items()) + "\n")

            if self.cost_params:
                arrays = self.cost_params.get("arrays", [])
                lookup = self.cost_params.get("lookup", {})
                cost_str = "  cost parameters:\n"
                if arrays:
                    cost_str += "    arrays:\n"
                    for i, v in enumerate(arrays):
                        cost_str += f"       {i}: {repr(v[:7])} ...\n"
                if lookup:
                    cost_str += "    lookup:\n"
                    for k, v in lookup.items():
                        cost_str += f"      {k}: {repr(v)}\n"
                parts.append(cost_str)

        parts.append(")\n")
        return "\n".join(parts)

    def update(
        self, 
        matrix: Optional[Tensor] = None, 
        cost_params: Optional[Dict[str, Tensor]] = None,
        layout_params: Optional[Dict[str, Any]] = None,
        solver_params: Optional[Dict[str, Any]] = None,
        constructors: Optional[Dict[str, Any]] = None  # Ensure constructors can be updated too
    ) -> "DataSet":
        """
        Creates a new DataSet with the specified changes while keeping the original data immutable.

        Args:
            matrix (Optional[Tensor]): The new matrix to update.
            cost_params (Optional[Dict[str, Tensor]]): The new cost parameters to update.
            layout_params (Optional[Dict[str, Any]]): The new layout parameters to update.
            solver_params (Optional[Dict[str, Any]]): The new solver parameters to update.
            constructors (Optional[Dict[str, Any]]): The new constructor data to update.

        Returns:
            DataSet: A new instance of DataSet with updated values.
        """
        return replace(
            self,
            matrix=matrix if matrix is not None else self.matrix,
            cost_params=cost_params if cost_params is not None else self.cost_params,
            layout_params=layout_params if layout_params is not None else self.layout_params,
            solver_params=solver_params if solver_params is not None else self.solver_params,
            constructors=constructors if constructors is not None else self.constructors
        )
