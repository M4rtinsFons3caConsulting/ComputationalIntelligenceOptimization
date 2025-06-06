# exceptions.py

# Data Processing Exceptions

class DataValidationError(Exception):
    """Base class for data validation errors."""
    pass

class MissingFeatureColumnError(DataValidationError):
    """Raised when a required column is missing."""
    pass

class InvalidFeatureColumnError(DataValidationError):
    """Raised when a feature column is invalid."""
    pass

class InvalidLabelColumnError(DataValidationError):
    """Raised when the label column is missing or invalid."""
    pass

class InjectiveConstraintsError(DataValidationError):
    """Raised when constraints are not injective (label column is not in partitions keys)."""
    pass


# DataSet Construction Exceptions

class DataSetConstructionError(Exception):
    """Base class for errors raised during DataSet construction."""
    pass

class InvalidRawDataSetError(DataSetConstructionError):
    """Raised when a raw DataSet is initialized with forbidden fields."""
    pass

class InvalidProcessedDataSetError(DataSetConstructionError):
    """Raised when a processed DataSet is missing required fields."""
    pass

class UnauthorizedDataSetConstructionError(DataSetConstructionError):
    """Raised when a processed DataSet is created outside of the authorized processor."""
    pass

class ProcessedDataSetUpdateError(Exception):
    """Raised when an attempt is made to update a processed DataSet."""
    pass