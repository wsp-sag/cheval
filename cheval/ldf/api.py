from typing import List, Dict, Any, Callable, Union, Iterator, Tuple, Deque
from collections import deque, Hashable

import pandas as pd
from pandas import Index, DataFrame, Series
import numpy as np
import attr

from .exceptions import LinkageSpecificationError
from cheval.common.pandas import PandasDtype, infer_dtype

# region LinkedDataFrame class
_label = Union[str, List[str]]


class LinkedDataFrame(DataFrame):

    # region Static Readers

    @staticmethod
    def read_csv(*args, **kwargs): return LinkedDataFrame(pd.read_csv(*args, **kwargs))

    @staticmethod
    def read_table(*args, **kwargs): return LinkedDataFrame(pd.read_table(*args, **kwargs))

    @staticmethod
    def read_clipboard(*args, **kwargs): return LinkedDataFrame(pd.read_clipboard(*args, **kwargs))

    @staticmethod
    def read_excel(*args, **kwargs): return LinkedDataFrame(pd.read_excel(*args, **kwargs))

    @staticmethod
    def read_fwf(*args, **kwargs): return LinkedDataFrame(pd.read_fwf(*args, **kwargs))

    @staticmethod
    def read_(name, *args, **kwargs):
        func = getattr(pd, f"read_{name}")
        return LinkedDataFrame(func(*args, **kwargs))

    # endregion
    # region DataFrame Boilerplate

    @property
    def _constructor(self):
        return LinkedDataFrame

    def __init__(self, *args, **kwargs):
        super(LinkedDataFrame, self).__init__(*args, **kwargs)
        self.__links: Dict[str, '_LinkEntry'] = {}
        self.__identifier_links = set()

    def __finalize__(self, original_ldf: 'LinkedDataFrame', method=None, **kwargs):
        pd.DataFrame.__finalize__(self, original_ldf, method=method, **kwargs)

        # Sometimes the original frame is not necessarily a LinkedDataFrame
        if not isinstance(original_ldf, LinkedDataFrame): return self

        # Copy the link meta
        self.__links = {}
        for alias, entry in original_ldf.__links.items():
            try:
                new_entry = entry.copy(self)
                self.__links[alias] = new_entry
                if alias.isidentifier(): self.__identifier_links.add(alias)
            except (KeyError, AttributeError):
                # This happens often when columns are omitted from the new copy (slicing by the columns)
                # So skip the copied link.
                pass
        return self

    # endregion

    def link_names(self) -> Iterator[str]:
        yield from self.__links.keys()

    def link_to(self, other: DataFrame, alias: str, on: _label=None, on_self: _label=None, on_other: _label=None,
                levels: _label=None, self_levels: _label=None, other_levels: _label=None):

        on_not_none = on is not None
        levels_not_none = levels is not None

        if on_not_none and levels_not_none:
            raise LinkageSpecificationError("Can only specify one of 'on=' or 'levels='")

        if on_not_none:
            on_self = on
            on_other = on

        if levels_not_none:
            self_levels = levels
            other_levels = levels

        if on_self is not None and self_levels is not None:
            raise LinkageSpecificationError()
        elif self_levels is not None:
            self_indexer = _IndexerMeta(self_levels, from_row_labels=True)
        else:
            self_indexer = _IndexerMeta(on_self, from_row_labels=False)
        self_indexer.make(self)

        if on_other is not None and other_levels is not None:
            raise LinkageSpecificationError()
        elif other_levels is not None:
            other_indexer = _IndexerMeta(other_levels, from_row_labels=True)
        else:
            other_indexer = _IndexerMeta(on_other, from_row_labels=False)
        other_indexer.make(other)

        # TODO: Fill value options to store on the entry
        aggregation_required = _is_aggregation_required(self_indexer.indexer, other_indexer.indexer)
        entry = _LinkEntry(self_indexer, other_indexer, other, aggregation_required)
        self.__links[alias] = entry
        if alias.isidentifier(): self.__identifier_links.add(alias)

    def __getitem__(self, item):
        if isinstance(item, Hashable) and item in self.__links:
            return self._get_link(item)
        return super()[item]

    def _get_link(self, item):
        # Needs to return a _LinkageNode
        entry = self.__links[item]

        other_frame = entry.other_table
        if isinstance(other_frame, LinkedDataFrame):
            # Return a Node

            node = _LinkageNode(entry.other_table, entry.self_indexer.indexer)

            raise NotImplementedError()
        elif entry.aggregation_required:
            # Return an Aggregator Leaf
            raise NotImplementedError()
        else:
            # Return a simple Leaf
            raise NotImplementedError

        # history = deque()
        # history.append(entry)
        # raise NotImplementedError()

# endregion
# region Internal Storage


def _is_aggregation_required(self_indexer: Index, other_indexer: Index) -> bool:
    """
    Optimized code to test if Linkage relationship is many-to-one or one-to-one.

    Args:
        self_indexer:
        other_indexer:

    Returns: bool

    """
    # If the right indexer is 100% unique, then no aggregation is required
    if other_indexer.is_unique:
        return False

    # Otherwise, aggregation is only required if at least one duplicate value
    # is in the left indexer.
    dupes = other_indexer.get_duplicates()
    for dupe in dupes:
        # Eager loop through the index. For this to be slow, a large number of duplicates must be missing
        # in self_indexer, which is practically never the case.
        if dupe in self_indexer:
            return True
    return False


@attr.s
class _IndexerMeta(object):
    labels: List[str] = attr.ib(convert=lambda x: [x] if isinstance(x, str) else list(x))
    from_row_labels: bool = attr.ib()
    indexer: Index = attr.ib(default=None)

    def make(self, frame: DataFrame):
        self.indexer = self._make(frame)

    def _make(self, frame: DataFrame) -> Index:
        if self.labels is None: return frame.index

        arrays = []
        if self.from_row_labels:
            if len(self.labels) > frame.index.nlevels:
                raise LinkageSpecificationError("Cannot specify more levels than in the index")

            for label in self.labels:
                # `get_level_values` works on both Index and MultiIndex objects and accepts both
                # integer levels AND level names
                try:
                    arrays.append(frame.index.get_level_values(label))
                except KeyError:
                    raise LinkageSpecificationError(f"Level '{label}' not in the index")
        else:
            for label in self.labels:
                if label not in frame:
                    raise LinkageSpecificationError(f"Column '{label}' not in the columns")
                arrays.append(frame[label].values)

        if len(arrays) == 1:
            name = self.labels[0]
            return Index(arrays[0], name=name)
        return pd.MultiIndex.from_arrays(arrays, names=self.labels)

    def copy(self, frame) -> '_IndexerMeta':
        new_item = _IndexerMeta(self.labels, self.from_row_labels)
        new_item.make(frame)
        return new_item


def _default_fill(column: Series) -> Any:
    pandas_type = infer_dtype(column)
    if pandas_type == PandasDtype.FLOAT_NAME: return np.nan
    if pandas_type == PandasDtype.INT_NAME: return 0
    if pandas_type == PandasDtype.BOOL_NAME: return False
    if pandas_type == PandasDtype.TEXT_NAME: return ""
    if pandas_type == PandasDtype.CAT_NAME:
        return column.cat.categories[0]
    if pandas_type == PandasDtype.UINT_NAME: return 0
    return None


@attr.s
class _LinkEntry(object):
    """Internal storage of linkage data"""

    # TODO: Optionally precompute the 'simple', offset-based indexer ([0, 1, 2...] instead of [A, B, C]).
    # This is significantly faster for indexing operations, especially where MultiIndex objects are concerned

    self_indexer: _IndexerMeta = attr.ib()
    other_indexer: _IndexerMeta = attr.ib()
    other_table: Union[DataFrame, LinkedDataFrame] = attr.ib()
    aggregation_required: bool = attr.ib()

    fill_function: Callable[Any, [Series]] = attr.ib(default=_default_fill)

    def copy(self, new_frame):
        new_s_indexer = self.self_indexer.copy(new_frame)
        new_o_indexer = self.other_indexer.copy(self.other_table)
        new_entry = _LinkEntry(new_s_indexer, new_o_indexer, self.other_table, self.aggregation_required,
                               self.fill_function)
        return new_entry

# endregion
# region Front-facing links


@attr.s
class _HistoryItem(object):
    self_indexer: Index = attr.ib()
    other_indexer: Index = attr.ib()
    fill_function: Callable[Any, [Series]] = attr.ib()


class _LinkLeaf(object):
    """Front-facing object for links made to DataFrames"""
    pass


class _LinkLeafAggr(object):
    """Front-facing object for links that require aggregation afterwards"""
    pass


@attr.s
class _LinkageNode(object):
    """Front-facing object for links made to other LinkedDataFrames"""
    # _history: List[_HistoryItem] = attr.ib()
    # _history: List[_LinkEntry] = attr.ib()
    # _history = Deque[_LinkEntry] = attr.ib()
    _frame: LinkedDataFrame = attr.ib()
    _root_index: Index = attr.ib()
    _history = Deque['_LinkageNode'] = attr.ib(default=attr.Factory(deque))

    def __dir__(self):
        df_columns = [col for col in self._frame if col.is_identifier()]
        return df_columns + list(self._frame.link_names())

    def __getitem__(self, item):
        value = self._frame[item]

        if isinstance(value, (_LinkageNode, _LinkLeaf, _LinkLeafAggr)):
        # if isinstance(value, _LinkEntry):
            # Item refers to another link

            raise NotImplementedError()

            # return value
        else:
            assert isinstance(value, Series)
            # TODO: Resolve history

            raise NotImplementedError()

# endregion
