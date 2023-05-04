"""Classes for managing missing data handling"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd

# Numpy dtypes are constructed on the fly, so the only way to test if a dtype is a 32-bit integer is to check using
# strict equality (==). Initially, the collections below were sets, but then Python checks for containment by hashing
# the dtype, which is not guaranteed to be consistent in all cases. So, these are lists instead which forces Python to
# check for equality.
_INT_TYPES = [np.int64, np.int32, int, np.int_, np.int8, np.int16]
_UINT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
_FLOAT_TYPES = [np.float64,  np.float32, float, np.float16, np.float32]


class PandasDtype(Enum):

    INT_NAME = 'int'
    UINT_NAME = 'uint'
    FLOAT_NAME = 'float'
    BOOL_NAME = 'bool'
    TEXT_NAME = 'text'
    OBJ_NAME = 'object'
    CAT_NAME = 'category'
    TIME_NAME = 'datetime'


def infer_dtype(s: pd.Series) -> PandasDtype:
    """Returns a simple name for the dtype of a Series. Currently doesn't handle TimeDelta or Time dtypes"""
    if hasattr(s, 'cat'):
        return PandasDtype.CAT_NAME
    if hasattr(s, 'str'):
        return PandasDtype.TEXT_NAME
    if hasattr(s, 'dt'):
        return PandasDtype.TIME_NAME

    type_to_check = s.dtype
    if type_to_check == np.bool_:
        return PandasDtype.BOOL_NAME
    if type_to_check in _INT_TYPES:
        return PandasDtype.INT_NAME
    if type_to_check in _UINT_TYPES:
        return PandasDtype.UINT_NAME
    if type_to_check in _FLOAT_TYPES:
        return PandasDtype.FLOAT_NAME

    return PandasDtype.OBJ_NAME


_default_fills = {
    PandasDtype.INT_NAME: 0, PandasDtype.UINT_NAME: 0, PandasDtype.FLOAT_NAME: np.nan, PandasDtype.BOOL_NAME: False,
    PandasDtype.TEXT_NAME: '', PandasDtype.CAT_NAME: np.nan, PandasDtype.TIME_NAME: np.datetime64('nat'),
    PandasDtype.OBJ_NAME: None
}


class SeriesFillManager:

    _fill_values: Dict[PandasDtype, Any]

    def __init__(self, **kwargs):
        self._fill_values = _default_fills.copy()

        for type_name, default_value in kwargs.items():
            enum_ = PandasDtype[type_name]
            self._fill_values[enum_] = default_value

    def get_fill(self, s: pd.Series):
        dtype = infer_dtype(s)
        return self._fill_values[dtype]

    def set_fill_defaults(self, **kwargs):
        for type_name, default_value in kwargs.items():
            enum_ = PandasDtype[type_name]
            self._fill_values[enum_] = default_value

    @contextmanager
    def temporary_fill_defaults(self, **kwargs):
        previous_values = {}

        try:
            for type_name, default_value in kwargs.items():
                enum_ = PandasDtype[type_name]
                previous_values[enum_] = self._fill_values[enum_]
                self._fill_values[enum_] = default_value
        finally:
            for enum_, original_value in previous_values.items():
                self._fill_values[enum_] = original_value

    def reset_fill_defaults(self):
        self._fill_values = _default_fills.copy()
