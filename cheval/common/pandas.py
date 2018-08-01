import enum

import pandas as pd
import numpy as np

_INT_TYPES = {int, np.int_, np.int8, np.int16, np.int32, np.int64, np.int128}
_UINT_TYPES = (np.uint8, np.uint16, np.uint32, np.uint64, np.uint128)
_FLOAT_TYPES = {float, np.float16, np.float32, np.float64, np.float128}


class PandasDtype(enum.Enum):

    INT_NAME = 'int'
    UINT_NAME = 'uint'
    FLOAT_NAME = 'float'
    BOOL_NAME = 'bool'
    TEXT_NAME = 'text'
    OBJ_NAME = 'object'
    CAT_NAME = 'category'


def infer_dtype(s: pd.Series) -> PandasDtype:
    """Returns a simple name for the dtype of a Series. Currently doesn't handle TimeDelta or Time dtypes"""
    if hasattr(s, 'cat'): return PandasDtype.CAT_NAME
    if hasattr(s, 'str'): return PandasDtype.TEXT_NAME

    type_to_check = s.dtype.type
    if type_to_check == np.bool_: return PandasDtype.BOOL_NAME
    if type_to_check in _INT_TYPES: return PandasDtype.INT_NAME
    if type_to_check in _UINT_TYPES: return PandasDtype.UINT_NAME
    if type_to_check in _FLOAT_TYPES: return PandasDtype.FLOAT_NAME

    return PandasDtype.OBJ_NAME
