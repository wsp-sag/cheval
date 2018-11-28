from typing import Dict, Any
import enum
from contextlib import contextmanager

import pandas as pd
import numpy as np

_INT_TYPES = {int, np.int_, np.int8, np.int16, np.int32, np.int64}
_UINT_TYPES = (np.uint8, np.uint16, np.uint32, np.uint64)
_FLOAT_TYPES = {float, np.float16, np.float32, np.float64}


class PandasDtype(enum.Enum):

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
    if hasattr(s, 'cat'): return PandasDtype.CAT_NAME
    if hasattr(s, 'str'): return PandasDtype.TEXT_NAME
    if hasattr(s, 'dt'): return PandasDtype.TIME_NAME

    type_to_check = s.dtype.type
    if type_to_check == np.bool_: return PandasDtype.BOOL_NAME
    if type_to_check in _INT_TYPES: return PandasDtype.INT_NAME
    if type_to_check in _UINT_TYPES: return PandasDtype.UINT_NAME
    if type_to_check in _FLOAT_TYPES: return PandasDtype.FLOAT_NAME

    return PandasDtype.OBJ_NAME


_default_fills = {
    PandasDtype.INT_NAME: 0, PandasDtype.UINT_NAME: 0, PandasDtype.FLOAT_NAME: np.nan, PandasDtype.BOOL_NAME: False,
    PandasDtype.TEXT_NAME: "", PandasDtype.CAT_NAME: np.nan, PandasDtype.TIME_NAME: np.datetime64('nat'),
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
