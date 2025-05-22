# rubix/classes/dataset.py

"""
This module defines the `DataSet` class, a core structure in the Rubix optimization framework.

The `DataSet` is an immutable container that represents either:
- **Raw data** (only the original DataFrame and basic metadata), or
- **Processed data** (fully transformed, with tensor matrix, cost parameters, and layout information).

Key features:
- Enforces immutability using `dataclass(frozen=True)`
- Validates construction context (processed data must be created by `rubix.process.process_data`)
- Ensures field constraints per stage (raw vs processed)
- Provides an `update()` method to create altered copies
- Custom `__repr__` for clear inspection

Raises custom exceptions from `rubix.exceptions` for robust validation enforcement.

Classes:
    DataSet: An immutable container representing raw or processed data within the optimization pipeline.
"""

import inspect
import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass, replace
from torch import Tensor
from rubix.exceptions import (
    InvalidRawDataSetError,
    InvalidProcessedDataSetError,
    UnauthorizedDataSetConstructionError,
    ProcessedDataSetUpdateError,  # Import the custom error
)

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
    constructors: Dict[str, Any]
    solver_params: Dict[str, Any]

    # Processed
    matrix: Optional[Tensor] = None
    cost_params: Optional[Dict[str, Tensor]] = None
    layout_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Ensures that DataSet is initialized correctly, based on whether it's raw or processed.
        - Processed DataSet requires matrix, cost_params, layout_params, constructors, and solver_params.
        - Raw DataSet must not have matrix, cost_params, or layout_params.
        """
        stack = inspect.stack()
        caller = next(
            (
                f for f in stack
                if f.function == "process_data"
                and "rubix.process" in f.frame.f_globals.get("__name__", "")
            ),
            None
        )

        is_processed = self.matrix is not None

        if is_processed:
            if caller is None:
                raise UnauthorizedDataSetConstructionError(
                    "Processed DataSet instances must be created by `rubix.process.process_data()`."
                )
            if not all([
                self.matrix is not None,
                self.cost_params is not None,
                self.layout_params is not None,
                self.constructors is not None,
                self.solver_params is not None
            ]):
                raise InvalidProcessedDataSetError(
                    "Processed DataSet requires matrix, cost_params, layout_params, constructors, and solver_params."
                )
            if self.dataframe is None:
                raise InvalidProcessedDataSetError(
                    "Processed DataSet must include the original dataframe."
                )
        else:
            if any([self.matrix is not None, self.cost_params is not None, self.layout_params is not None]):
                raise InvalidRawDataSetError(
                    "Raw DataSet must only include dataframe, constructors, and solver_params."
                )

    def __repr__(self) -> str:
        """
        Provides a string representation of the DataSet, showing key attributes depending on its stage.

        Returns:
            str: String representation of the DataSet instance.
        """
        stage = "Processed" if self.matrix is not None else "Raw/Ingested"
        parts = [f"DataSet(", f"  stage: {stage},\n", f"  dataframe:\n{self.dataframe.head()}\n"]

        if self.constructors:
            parts.append("  constructors:\n" + "\n".join(f"    {k}: {repr(v)}" for k, v in self.constructors.items()) + "\n")
        if self.solver_params:
            parts.append("  solver parameters:\n" + "\n".join(f"    {k}: {repr(v)}" for k, v in self.solver_params.items()) + "\n")

        if stage == "Processed":
            matrix_preview = self.matrix[:5] if self.matrix.ndim == 2 else self.matrix
            parts.append(f"  matrix:\n{repr(matrix_preview)}\n")
            if self.layout_params:
                parts.append("  layout parameters:\n" + "\n".join(f"    {k}: {repr(v)}" for k, v in self.layout_params.items()) + "\n")
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
        constructors: Optional[Dict[str, Any]] = None
    ) -> "DataSet":
        """
        Creates a new DataSet with the specified changes while keeping the original data immutable.
        Raises an error if the DataSet is already processed.

        Args:
            matrix (Optional[Tensor]): The new matrix to update.
            cost_params (Optional[Dict[str, Tensor]]): The new cost parameters to update.
            layout_params (Optional[Dict[str, Any]]): The new layout parameters to update.
            solver_params (Optional[Dict[str, Any]]): The new solver parameters to update.
            constructors (Optional[Dict[str, Any]]): The new constructor data to update.

        Returns:
            DataSet: A new instance of DataSet with updated values.

        Raises:
            ProcessedDataSetUpdateError: If an attempt is made to update a processed DataSet.
        """
        if self.matrix is not None:  # Check if it's a processed dataset
            raise ProcessedDataSetUpdateError("Processed DataSet cannot be updated.")
        
        return replace(
            self,
            matrix=matrix if matrix is not None else self.matrix,
            cost_params=cost_params if cost_params is not None else self.cost_params,
            layout_params=layout_params if layout_params is not None else self.layout_params,
            solver_params=solver_params if solver_params is not None else self.solver_params,
            constructors=constructors if constructors is not None else self.constructors
        )
