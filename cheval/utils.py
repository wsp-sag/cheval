import pandas as pd
from pandas.api.types import is_integer_dtype
import numpy as np
from typing import Union

_USE_TO_NUMPY = hasattr(pd.Series, 'to_numpy')


def to_numpy(frame_or_series: Union[pd.DataFrame, pd.Series, pd.Index], ignore_check: bool = False) -> np.ndarray:
    """A helper function compatible with all versions of pandas to access numpy arrays. Set `ignore_check=True` to save
    the computational cost of confirming that `.to_numpy()` does not produce a copy of `frame_or_series` values."""
    arr = frame_or_series.to_numpy(copy=False) if _USE_TO_NUMPY else frame_or_series.values
    if _USE_TO_NUMPY and not ignore_check:  # only perform the check if we are using .to_numpy()
        if not np.shares_memory(frame_or_series, arr):
            arr = frame_or_series.values  # Fallback to using .values if we find that .to_numpy() is a copy
    return arr


def convert_series(s: pd.Series, allow_raw: bool = False) -> np.ndarray:
    dtype = s.dtype

    if is_integer_dtype(dtype) and _USE_TO_NUMPY:
        # need to deal with NA values in integer series in pandas >= 0.24
        # https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.0.0.html#arrays-integerarray-now-uses-pandas-na
        if np.all(~s.isna()):  # no NA values found
            return s.to_numpy()[...]
        else:  # if NA values are present, yet pandas keeps the series as an IntegerArray instead of float
            try:
                return np.asarray(s, dtype='float')[...]
            except ValueError:
                return s.to_numpy(dtype='float', na_value=np.nan)[...]  # pandas >= 1.0.0
    elif dtype.name == 'category':
        # Categorical column
        categorical = s.values

        category_index = categorical.categories
        if category_index.dtype.name == 'object':
            max_len = category_index.str.len().max()
            typename = f'S{max_len}'
        else:
            typename = category_index.dtype

        return categorical.astype(typename)
    elif dtype.name == 'object':
        # Object or text column
        # This is much slower than other dtypes, but it can't be helped. For now, users should just use Categoricals
        max_length = s.str.len().max()
        if np.isnan(max_length):
            raise TypeError('Could not get max string length')

        return s.to_numpy(dtype=f'S{max_length}') if _USE_TO_NUMPY else s.values.astype(f'S{max_length}')
    elif np.issubdtype(dtype, np.datetime64):
        raise TypeError('Datetime columns are not supported')
    elif np.issubdtype(dtype, np.timedelta64):
        raise TypeError('Timedelta columns are not supported')
    try:
        return to_numpy(s)[...]
    except AttributeError:
        if allow_raw:
            return s
        raise
