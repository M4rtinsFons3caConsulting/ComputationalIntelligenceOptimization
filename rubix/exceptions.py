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
