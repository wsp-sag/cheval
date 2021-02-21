"""Common constants & exception classes for LDF"""
from enum import Enum


class LinkageSpecificationError(ValueError):
    """Exception raised when a specified linkage cannot be made"""
    pass


class LinkAggregationRequired(Enum):
    YES = True
    NO = False
