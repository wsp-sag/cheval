"""Common constants & exception classes for LDF"""
from enum import Flag


class LinkageSpecificationError(ValueError):
    """Exception raised when a specified linkage cannot be made"""
    pass


class LinkAggregationRequired(Flag):
    YES = True
    NO = False
