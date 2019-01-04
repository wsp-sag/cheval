from typing import Dict, Union, Set, TYPE_CHECKING, List, Tuple, Generator
import abc
import ast
import astor

import pandas as pd
import numpy as np
import attr

from .parsing.expr_items import ChainTuple, ChainedSymbol
from .parsing.ast_transformer import ExpressionParser
from .ldf import LinkedDataFrame
from .exceptions import ModelNotReadyError
if TYPE_CHECKING:
    from .model import ChoiceModel


class ChoiceNode(object):

    def __init__(self, name: str, parent: 'ChoiceNode'=None, logsum_scale: float=1.0,
                 level: int=0):
        assert name != ".", 'Choice node name cannot be "."'
        assert 0.0 < logsum_scale <= 1.0, "Logsum scale must be in hte interval (0, 1], got %s" % logsum_scale

        self._name: str = str(name)
        self._parent = parent
        self._logsum_scale = None
        self.logsum_scale = logsum_scale
        self._level = level
        self._children: Dict[str, 'ChoiceNode'] = {}

    def __str__(self): return self.name

    def __repr__(self): return f"ChoiceNode({self.name})"

    @property
    def logsum_scale(self) -> float: return self._logsum_scale

    @logsum_scale.setter
    def logsum_scale(self, value):
        assert 0.0 < value <= 1.0, "Logsum scale must be in hte interval (0, 1], got %s" % value
        self._logsum_scale = float(value)

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def level(self):
        return self._level

    @property
    def is_parent(self):
        return len(self._children) > 0

    def children(self):
        yield from self._children.values()

    def max_level(self):
        max_level = self._level

        for c in self.children():
            max_level = max(max_level, c.max_level())

        return max_level

    def _nested_id(self, max_level: int):
        retval = ['.'] * max_level
        if self._parent is None:
            retval[0] = self._name
        else:
            cutoff = self._level + 1
            retval[: cutoff] = self._parent._nested_id(max_level)[: cutoff]
            retval[cutoff - 2] = self.name
        return tuple(retval)

    def nested_ids(self, max_level: int):
        retval = [self._nested_id(max_level)]
        for c in self._children.values():
            retval += c.nested_ids(max_level)
        return retval

    def add_choice(self, name: str, logsum_scale: float=1.0) -> 'ChoiceNode':
        node = ChoiceNode(name, self, logsum_scale, self.level + 1)
        self._children[name] = node
        return node

    def clear(self):
        for c in self._children.values(): c.clear()
        self._children.clear()


@attr.s
class _Expression(object):
    """Simple data class for utility expressions"""

    raw: str = attr.ib()
    transformed: str = attr.ib()
    chains: Dict[str, ChainedSymbol] = attr.ib()
    dict_literals: Dict[str, pd.Series] = attr.ib()

    @staticmethod
    def parse(e: str, prior_simple: Set[str]=None, prior_chained: Set[str]=None) -> '_Expression':
        tree = ast.parse(e, mode='eval').body
        transformer = ExpressionParser(prior_simple, prior_chained)
        new_tree = transformer.visit(tree)

        new_e = _Expression(e, astor.to_source(new_tree), transformer.chained_symbols, transformer.dict_literals)
        return new_e


class ExpressionGroup(object):

    def __init__(self, root: 'ChoiceModel'):
        self.root: 'ChoiceModel' = root
        self._expressions: List[_Expression] = []
        self._simple_symbols: Set[str] = set()
        self._chained_symbols: Set[str] = set()

    def append(self, e: str):
        # Parse the expression and look for invalid syntax and inconsistent usage. self._simple_sybols and
        # self._chained_symbols are modified in-place during parsing.
        expr = _Expression.parse(e, self._simple_symbols, self._chained_symbols)
        self._expressions.append(expr)

    def clear(self):
        self._expressions.clear()
        self._simple_symbols.clear()
        self._chained_symbols.clear()

    def itersimple(self):
        yield from self._simple_symbols

    def __iter__(self) -> Generator[_Expression, None, None]:
        yield from self._expressions


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
    except:
        if allow_raw: return s
        raise


class AbstractSymbol(object, metaclass=abc.ABCMeta):

    def __init__(self, parent: 'ChoiceModel', name: str):
        self._parent = parent
        self._name = name

    @abc.abstractmethod
    def assign(self, data): pass

    @abc.abstractmethod
    def _get(self, **kwargs) -> Union[float, np.ndarray]: pass

    @abc.abstractclassmethod
    def empty(self): pass


class NumberSymbol(AbstractSymbol):
    def __init__(self, parent: 'ChoiceModel', name: str):
        super().__init__(parent, name)
        self._val = None

    def assign(self, data):
        self._val = float(data)

    def _get(self):
        if self._val is None:
            raise ModelNotReadyError()
        return self._val

    def empty(self): self._val = None


class VectorSymbol(AbstractSymbol):

    def __init__(self, parent: 'ChoiceModel', name: str, orientation: int):
        super().__init__(parent, name)

        assert orientation in {0, 1}
        self._orientation = orientation
        self._raw_array: np.ndarray = None

    def assign(self, data):
        index_to_check = self._parent.decision_units if self._orientation else self._parent.choices

        if isinstance(data, pd.Series):
            assert index_to_check.equals(data.index), "Series does not match context rows or columns"
            vector = convert_series(data, allow_raw=False)  # Convert Categorical/Text right away
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

    def _get(self):
        if self._raw_array is None:
            raise ModelNotReadyError
        return self._raw_array

    def empty(self): self._raw_array = None


class TableSymbol(AbstractSymbol):

    def __init__(self, parent: 'ChoiceModel', name: str, orientation: int, mandatory_attributes: Set[str]=None,
                 allow_links: bool=True):
        super().__init__(parent, name)
        assert orientation in {0, 1}
        self._orientation = orientation

        if mandatory_attributes is None: mandatory_attributes = set()
        self._mandatory_attributes = mandatory_attributes
        self._allow_links = bool(allow_links)
        self._table: pd.DataFrame = None

    def assign(self, data):
        assert isinstance(data, pd.DataFrame)
        index_to_check = self._parent.decision_units if self._orientation else self._parent.choices
        assert data.index.equals(index_to_check), "DataFrame index does not match context rows or columns"

        for column in self._mandatory_attributes:
            assert column in data, f"Mandatory attribute {column} not found in DataFrame"

        if not self._allow_links and not isinstance(data, LinkedDataFrame):
            raise TypeError(f"LinkedDataFrames not allowed for symbol {self._name}")

        self._table = data

    def _get(self, chain_info: ChainTuple=None):
        assert chain_info is not None

        chained = len(chain_info.chain) > 1

        if chained:
            assert isinstance(self._table, LinkedDataFrame)
            item = self._table
            for item_name in chain_info.chain:
                item = item[item_name]

            if chain_info.withfunc:
                series = getattr(item, chain_info.func)(chain_info.args)
            else:
                series = item
        else:
            attribute_name = chain_info.chain[0]
            series = self._table[attribute_name]

        vector = convert_series(series, allow_raw=False)

        n = len(vector)
        new_shape = (n, 1) if self._orientation == 0 else (1, n)
        vector.shape = new_shape
        return vector

    def empty(self):
        self._table = None


class MatrixSymbol(AbstractSymbol):

    def __init__(self, parent: 'ChoiceModel', name: str, allow_transpose: bool=True):
        super().__init__(parent, name)
        self._allow_transpose = bool(allow_transpose)
        self._matrix: np.ndarray = None

    def assign(self, data):
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

    def _get(self): return self._matrix

    def empty(self): self._matrix = None

