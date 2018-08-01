import pandas as pd
import numpy as np

INT_TYPES = {int, np.int_, np.int8, np.int16, np.int32, np.int64,np.int128}
UINT_TYPES = (np.uint8, np.uint16, np.uint32, np.uint64, np.uint128)
FLOAT_TYPES = {float, np.float16, np.float32, np.float64, np.float128}

INT_NAME = 'int'
UINT_NAME = 'uint'
FLOAT_NAME = 'float'
BOOL_NAME = 'bool'
TEXT_NAME = 'text'
OBJ_NAME = 'object'
CAT_NAME = 'category'


def infer_dtype(s: pd.Series) -> str:
    """Returns a simple name for the dtype of a Series. Currently doesn't handle TimeDelta or Time dtypes"""
    if hasattr(s, 'cat'): return CAT_NAME
    if hasattr(s, 'str'): return TEXT_NAME

    type_to_check = s.dtype.type
    if type_to_check == np.bool_: return BOOL_NAME
    if type_to_check in INT_TYPES: return INT_NAME
    if type_to_check in UINT_TYPES: return UINT_NAME
    if type_to_check in FLOAT_TYPES: return FLOAT_NAME

    return OBJ_NAME
