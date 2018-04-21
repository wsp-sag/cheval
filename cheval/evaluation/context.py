from typing import Set, Any, Union, Dict
import abc

import pandas as pd
import numpy as np


class AbstractSymbol(object, metaclass=abc.ABCMeta):

    def __init__(self, parent: 'EvaluationContext', name: str):
        self._parent = parent
        self._name = name

    @abc.abstractmethod
    def fill(self, data): pass

    @abc.abstractmethod
    def get(self) -> Union[float, np.ndarray]: pass


class NumberSymbol(AbstractSymbol):
    def __init__(self, parent: 'EvaluationContext', name: str):
        super().__init__(parent, name)
        self._val = None

    def fill(self, data):
        self._val = float(data)

    def get(self):
        assert self._val is not None
        return self._val


class VectorSymbol(AbstractSymbol):

    def __init__(self, parent: 'EvaluationContext', name: str, orientation: int):
        super().__init__(parent, name)

        assert orientation in {0, 1}
        self._orientation = orientation
        self._raw_array: np.ndarray = None

    def fill(self, data):
        index_to_check = self._parent.cols if self._orientation else self._parent.rows

        if isinstance(data, pd.Series):
            assert index_to_check.equals(data.index), "Series does not match context rows or columns"
            vector = data.values
        elif isinstance(data, np.ndarray):
            assert len(data.shape) == 1, "Only 1D arrays are permitted"
            assert len(data) == len(index_to_check), "Array length does not match length of rows or columns"
            vector = data
        else:
            raise TypeError(type(data))

        self._raw_array = vector[...]  # Shallow copy
        n = len(index_to_check)

        if self._orientation: self._raw_array.shape = 1, n
        else: self._raw_array.shape = n, 1

    def get(self): return self._raw_array


class TableSymbol(AbstractSymbol):

    def __init__(self, parent: 'EvaluationContext', name: str, orientation: int, mandatory_attributes: Set[str]=None,
                 allow_links: bool=True):
        super().__init__(parent, name)
        assert orientation in {0, 1}
        self._orientation = orientation

        if mandatory_attributes is None: mandatory_attributes = set()
        self._mandatory_attributes = mandatory_attributes
        self._allow_links = bool(allow_links)

    def fill(self, data):
        raise NotImplementedError()

    def get(self):
        return NotImplementedError()


class MatrixSymbol(AbstractSymbol):

    def __init__(self, parent: 'EvaluationContext', name: str, allow_transpose: bool=True):
        super().__init__(parent, name)
        self._allow_transpose = bool(allow_transpose)
        self._matrix: np.ndarray = None

    def fill(self, data):
        rows = self._parent.rows
        cols = self._parent.cols

        if isinstance(data, pd.DataFrame):
            if rows.equals(data.index) and cols.equals(data.columns):
                # Orientation matches
                array = data.values.astype('f8')
            elif rows.equals(data.columns) and cols.equals(data.index):
                assert self._allow_transpose
                array = data.transpose().values.astype('f8')
            else:
                raise TypeError("Shapes do not match")
        elif isinstance(data, np.ndarray):
            assert len(data.shape) == 2
            r, c = data.shape
            if r == len(rows) and c == len(cols):
                array = data.astype('f8')
            elif r == len(cols) and c == len(rows):
                assert self._allow_transpose
                array = data.transpose().astype('f8')
            else:
                raise TypeError("Shapes do not match")
        else:
            raise TypeError(type(data))

        self._matrix = array

    def get(self): return self._matrix


class EvaluationContext(object):

    def __init__(self):
        self._row_index: pd.Index = None
        self._col_index: pd.Index = None
        self._symbols: Dict[str, AbstractSymbol] = {}

    @property
    def rows(self): return self._row_index

    @property
    def cols(self): return self._col_index

    def define_rows(self, index: pd.Index):
        pass

    def define_columns(self, index: pd.Index):
        pass

    def declare_number(self, name: str):
        pass

    def declare_vector(self, name: str, orientation: int):
        pass

    def declare_table(self, name: str, orientation: int, mandatory_attributes: Set[str]=None,
                      allow_links=True):
        pass

    def declare_matrix(self, name: str, allow_transpose=True):
        pass

    def define_symbol(self, name: str, data):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, item):
        pass
