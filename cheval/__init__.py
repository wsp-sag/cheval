from ._version import get_versions
from .exceptions import ModelNotReadyError, UnsupportedSyntaxError
from .ldf import (LinkageSpecificationError, LinkAggregationRequired,
                  LinkedDataFrame)
from .model import ChoiceModel

__version__ = get_versions()['version']
del get_versions
