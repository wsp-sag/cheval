import pandas as pd
import numpy as np

def convert_series(s: pd.Series, allow_raw=False) -> np.ndarray:
    dtype = s.dtype

    if dtype.name == 'category':
        # Categorical column
        categorical = s.values

        category_index = categorical.categories
        if category_index.dtype.name == 'object':
            max_len = category_index.str.len().max()
            typename = 'S%s' % max_len
        else:
            typename = category_index.dtype

        return categorical.astype(typename)
    elif dtype.name == 'object':
        # Object or text column
        # This is much slower than other dtypes, but it can't be helped. For now, users should just use Categoricals
        max_length = s.str.len().max()
        if np.isnan(max_length):
            raise TypeError("Could not get max string length")

        return s.values.astype("S%s" % max_length)
    elif np.issubdtype(dtype, np.datetime64):
        raise TypeError("Datetime columns are not supported")
    elif np.issubdtype(dtype, np.timedelta64):
        raise TypeError("Timedelta columns are not supported")
    try:
        return s.values[...]
    except AttributeError:
        if allow_raw: return s
        raise
