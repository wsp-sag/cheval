from typing import Dict, Union, Set, TYPE_CHECKING, List, Generator, Hashable
import abc

import pandas as pd
import numpy as np
import attr

from .parsing.expr_items import ChainTuple
from .parsing.expressions import Expression
from .ldf import LinkedDataFrame
from .exceptions import ModelNotReadyError
from .misc import convert_series
if TYPE_CHECKING:
    from .model import ChoiceModel


class ChoiceNode(object):

    def __init__(self, name: str, parent: 'ChoiceNode'=None, logsum_scale: float=1.0,
                 level: int=0):
        assert '.' not in name, 'Choice node name cannot contain "."'
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
    def is_parent(self) -> bool:
        return len(self._children) > 0

    @property
    def n_children(self) -> int:
        return len(self._children)

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
class ExpressionSubGroup:
    name: Hashable = attr.ib()
    simple_symbols: Set[str] = attr.ib(default=attr.Factory(set))
    chained_symbols: Set[str] = attr.ib(default=attr.Factory(set))
    expressions: List[Expression] = attr.ib(default=attr.Factory(list))

    def append(self, e: Expression):
        self.expressions.append(e)
        self.simple_symbols |= e.symbols
        for chain_name in e.chains.keys(): self.chained_symbols.add(chain_name)

    def __add__(self, other: 'ExpressionSubGroup') -> 'ExpressionSubGroup':
        new = ExpressionSubGroup(self.name)
        for e in self.expressions: new.append(e)
        for e in other.expressions: new.append(e)
        return new

    def itersimple(self):
        yield from self._simple_symbols

    def iterchained(self):
        yield from self._chained_symbols

    def __iter__(self):
        yield from self.expressions


class ExpressionGroup(object):

    def __init__(self):
        self._ungrouped_expressions: List[Expression] = []
        self._simple_symbols: Set[str] = set()
        self._chained_symbols: Set[str] = set()
        self._subgroups: Dict[Hashable, ExpressionSubGroup] = {}

    def append(self, e: str, group: Hashable = None):
        # Parse the expression and look for invalid syntax and inconsistent usage. self._simple_symbols and
        # self._chained_symbols are modified in-place during parsing.
        expr = Expression.parse(e, self._simple_symbols, self._chained_symbols)
        if group is not None:
            if group not in self._subgroups:
                subgroup = ExpressionSubGroup(group)
                self._subgroups[group] = subgroup
            else:
                subgroup = self._subgroups[group]

            subgroup.append(expr)
        else:
            self._ungrouped_expressions.append(expr)

    def clear(self):
        self._ungrouped_expressions.clear()
        self._subgroups.clear()
        self._simple_symbols.clear()
        self._chained_symbols.clear()

    def itersimple(self):
        yield from self._simple_symbols
        for subgroup in self._subgroups.values():
            yield from subgroup.itersimple()

    def iterchained(self):
        yield from self._chained_symbols
        for subgroup in self._subgroups.values():
            yield from subgroup.iterchained()

    def __iter__(self) -> Generator[Expression, None, None]:
        yield from self._ungrouped_expressions
        for subgroup in self._subgroups.values():
            yield from subgroup

    def __add__(self, other: 'ExpressionGroup') -> 'ExpressionGroup':
        new = ExpressionGroup()

        new._simple_symbols = self._simple_symbols | other._simple_symbols
        new._chained_symbols = self._chained_symbols | other._chained_symbols
        new._ungrouped_expressions = self._ungrouped_expressions + other._ungrouped_expressions

        new._subgroups = {}
        keys = set(self._subgroups.keys()) | set(other._subgroups.keys())
        for key in keys:
            key_in_self, key_in_other = key in self._subgroups, key in other._subgroups
            if key_in_self and key_in_other:
                new._subgroups[key] = self._subgroups[key] + other._subgroups[key]
            elif key_in_self:
                new._subgroups[key] = self._subgroups[key]
            elif key_in_other:
                new._subgroups[key] = other._subgroups[key]

        return new

    def tolist(self, raw=True):
        if raw: return [e.raw for e in self._ungrouped_expressions]
        return [e for e in self._ungrouped_expressions]

    def get_group(self, name: Hashable) -> ExpressionSubGroup:
        return self._subgroups[name]

    def drop_group(self, name: Hashable):
        del self._subgroups[name]

    def copy(self):
        new = ExpressionGroup()
        for e in self._ungrouped_expressions: new._ungrouped_expressions.append(e)
        for name, subgroup in self._subgroups.items():
            new_sg = ExpressionSubGroup(name)
            for e in subgroup: new_sg.append(e)
            new._subgroups[name] = new_sg


class AbstractSymbol(object, metaclass=abc.ABCMeta):
    def __init__(self, parent: 'ChoiceModel', name: str):
        self._parent = parent
        self._name = name

    @abc.abstractmethod
    def assign(self, data): pass

    @abc.abstractmethod
    def _get(self, **kwargs) -> Union[float, np.ndarray]: pass

    @abc.abstractmethod
    def empty(self): pass

    @abc.abstractmethod
    def copy(self, new_parent: 'ChoiceModel', copy_data) -> 'AbstractSymbol': pass


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

    def copy(self, new_parent: 'ChoiceModel', copy_data):
        new = NumberSymbol(new_parent, self._name)
        if copy_data:
            new._val = self._val

        return new


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

    def copy(self, new_parent: 'ChoiceModel', copy_data):
        new = VectorSymbol(new_parent, self._name, self._orientation)
        if copy_data:
            new._raw_array = self._raw_array

        return new


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
        index_to_check = self._parent.decision_units if self._orientation == 0 else self._parent.choices
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
            for item_name in reversed(chain_info.chain):
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

    def copy(self, new_parent: 'ChoiceModel', copy_data):
        new = TableSymbol(new_parent, self._name, self._orientation, self._mandatory_attributes, self._allow_links)
        if copy_data: new._table = self._table
        return new


class MatrixSymbol(AbstractSymbol):

    def __init__(self, parent: 'ChoiceModel', name: str, allow_transpose: bool=True, allow_row_reindex=True,
                 allow_col_reindex=True):
        super().__init__(parent, name)
        self._allow_transpose = bool(allow_transpose)
        self._matrix: np.ndarray = None

    def assign(self, data):
        rows = self._parent.decision_units
        cols = self._parent.choices

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

    def copy(self, new_parent: 'ChoiceModel', copy_data):
        new = MatrixSymbol(new_parent, self._name, self._allow_transpose)
        if copy_data:
            new._matrix = self._matrix
        return new
