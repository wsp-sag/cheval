from typing import List, Dict, Union, Deque, Tuple, Callable, Any, Optional, Set
from collections import deque

from pandas import DataFrame, Series, Index, MultiIndex
import pandas as pd
from numpy import ndarray
import numpy as np
import attr

from .constants import LinkageSpecificationError, LinkAggregationRequired
from .missing_data import SeriesFillManager, infer_dtype, PandasDtype

_FillFunctionType = Callable[[Series], Union[int, float, bool, str]]
_NUMERIC_AGGREGATIONS = {'max', 'min', 'mean', 'median', 'prod', 'std', 'sum', 'var', 'quantile'}
_NON_NUMERIC_AGGREGATIONS = {'count', 'first', 'last', 'nth'}
_SUPPORTED_AGGREGATIONS = _NUMERIC_AGGREGATIONS | _NON_NUMERIC_AGGREGATIONS
_NUMERIC_TYPES = {PandasDtype.INT_NAME, PandasDtype.UINT_NAME, PandasDtype.FLOAT_NAME, PandasDtype.BOOL_NAME}


class _IndexMeta:
    labels: List[str] = attr.ib(convert=lambda x: [x] if isinstance(x, str) else list(x))
    from_row_labels: bool = attr.ib()

    def __init__(self, labels=None, from_row_labels=True):
        if isinstance(labels, str):
            labels = [labels]
        elif labels is None:
            from_row_labels = True
        self.labels = labels
        self.from_row_labels = from_row_labels

    def validate(self, frame: DataFrame):
        if self.labels is None: return # Use the index, which is always available
        frame_items = set(frame.index.levels) if self.from_row_labels else set(frame.columns)
        item_name = "index" if self.from_row_labels else "columns"

        for name in self.labels:
            assert name in frame_items, f"Could not find '{name}' in the {item_name}"

    def get_indexer(self, frame: DataFrame) -> Index:
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
        return MultiIndex.from_arrays(arrays, names=self.labels)

    def __str__(self):
        if self.from_row_labels:
            return f"From index: {self.labels}"
        return f"From columns: {self.labels}"

    def nlevels(self, frame: DataFrame) -> int:
        if self.labels is None: return frame.index.nlevels
        return len(self.labels)


class _LinkMeta:
    owner: 'LinkedDataFrame'
    other: Union['LinkedDataFrame', DataFrame]
    _other_has_links: bool
    aggregation_required: bool
    self_meta: _IndexMeta
    other_meta: _IndexMeta
    flat_indexer: Optional[ndarray]
    other_grouper: Optional[ndarray]

    @staticmethod
    def create(owner,  other, self_labels: Union[List[str], str], self_from_row_labels: bool,
               other_labels: Union[List[str], str], other_from_row_labels: bool, precompute: bool=True) -> '_LinkMeta':
        self_meta = _IndexMeta(self_labels, self_from_row_labels)
        other_meta = _IndexMeta(other_labels, other_from_row_labels)

        assert self_meta.nlevels(owner) == other_meta.nlevels(other)

        other_has_links = isinstance(other, LinkedDataFrame)

        link = _LinkMeta(owner, other, self_meta, other_meta, other_has_links )
        link._determine_aggregation(precompute)

        return link

    def __init__(self, owner, other, self_meta, other_meta, other_has_links):
        self.owner = owner
        self.other = other
        self.self_meta = self_meta
        self.other_meta = other_meta
        self._other_has_links = other_has_links
        self.aggregation_required = False
        self.flat_indexer = None
        self.other_grouper = None

    def _determine_aggregation(self, precompute):
        self_indexer = self.self_meta.get_indexer(self.owner)
        other_indexer = self.other_meta.get_indexer(self.other)

        self_unique = self_indexer.is_unique
        other_unique = other_indexer.is_unique

        if self_unique and other_unique:
            flag = False
        elif self_unique:  # Other is not unique
            flag = True
        elif other_unique:
            flag = False
        else:
            raise RuntimeError("Many-to-many links are not permitted")
        self.aggregation_required = flag

        if precompute:
            self._make_indexer(self_indexer, other_indexer)

    def _get_indexers(self) -> Tuple[Index, Index]:
        return self.self_meta.get_indexer(self.owner), self.other_meta.get_indexer(self.other)

    def _make_indexer(self, self_indexer: Index, other_indexer: Index):
        if self.aggregation_required:
            flat_grouper, group_labels = other_indexer.factorize()
            self.other_grouper = flat_grouper
            self.flat_indexer = group_labels.get_indexer_for(self_indexer)
        else:
            self.flat_indexer = other_indexer.get_indexer(self_indexer)

    @property
    def indexer(self) -> ndarray:
        if self.flat_indexer is None:
            self.precompute()
        return self.flat_indexer

    @property
    def chained(self) -> bool:
        return self._other_has_links

    def precompute(self):
        """Top-level method to precompute the indexer"""
        self._make_indexer(*self._get_indexers())

    def copy(self, indices=None) -> '_LinkMeta':
        copied = _LinkMeta(self.owner, self.other, self.self_meta, self.other_meta, self._other_has_links)
        copied.aggregation_required = self.aggregation_required

        if indices is not None and self.flat_indexer is not None:
            copied.flat_indexer = self.flat_indexer[indices]

        if self.other_grouper is not None:
            copied.other_grouper = self.other_grouper

        return copied


_LabelType = Union[str, List[str]]


class LinkedDataFrame(DataFrame):

    __links: Dict[str, _LinkMeta]
    __identified_links: Set[str]
    __class_filler: SeriesFillManager = SeriesFillManager()
    __instance_filler: SeriesFillManager
    __column_fills: dict

    # region Static Readers

    @staticmethod
    def read_csv(*args, **kwargs):
        return LinkedDataFrame(pd.read_csv(*args, **kwargs))

    @staticmethod
    def read_table(*args, **kwargs):
        return LinkedDataFrame(pd.read_table(*args, **kwargs))

    @staticmethod
    def read_clipboard(*args, **kwargs):
        return LinkedDataFrame(pd.read_clipboard(*args, **kwargs))

    @staticmethod
    def read_excel(*args, **kwargs):
        return LinkedDataFrame(pd.read_excel(*args, **kwargs))

    @staticmethod
    def read_fwf(*args, **kwargs):
        return LinkedDataFrame(pd.read_fwf(*args, **kwargs))

    @staticmethod
    def read_(name, *args, **kwargs):
        func = getattr(pd, f"read_{name}")
        return LinkedDataFrame(func(*args, **kwargs))

    # endregion

    @property
    def _constructor(self):
        return LinkedDataFrame

    # _internal_names = ['__links']
    # _internal_names_set = set(_internal_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pandas gives me grief for trying to add sub-attributes, so I'm doing the this hard way
        object.__setattr__(self, '_LinkedDataFrame__links', {})
        object.__setattr__(self, '_LinkedDataFrame__identified_links', set())
        object.__setattr__(self, '_LinkedDataFrame__instance_filler', SeriesFillManager())
        object.__setattr__(self, '_LinkedDataFrame__column_fills', {})

    def link_to(self, other: DataFrame, name: str, *, on: _LabelType=None, levels: _LabelType=None,
                on_self: _LabelType=None, on_other: _LabelType=None, self_levels: _LabelType=None,
                other_levels: _LabelType=None, precompute: bool=True):
        """

        Args:
            other:
            name:
            on:
            levels:
            on_self:
            on_other:
            self_levels:
            other_levels:
            precompute:

        Returns:

        """
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
            self_labels, self_from_index = self_levels, True
        else:
            self_labels, self_from_index = on_self, False

        if on_other is not None and other_levels is not None:
            raise LinkageSpecificationError()
        elif other_levels is not None:
            other_labels, other_from_index = other_levels, True
        else:
            other_labels, other_from_index = on_other, False

        link_data = _LinkMeta.create(self, other, self_labels, self_from_index, other_labels, other_from_index,
                                     precompute)
        self.__links[name] = link_data
        if name.isidentifier():
            self.__identified_links.add(name)

        return LinkAggregationRequired.YES if link_data.aggregation_required else LinkAggregationRequired.NO

    def __dir__(self):
        """ Override dir() to show links as valid attributes """
        return super().__dir__() + sorted(self.__identified_links)

    # region Fill value management

    def set_column_fill(self, column, value):
        if column not in self.columns:
            raise KeyError(column)
        self.__column_fills[column] = value

    '''
    The pattern "self=None" allows these methods to have different behaviour depending on whether called from
    the class (statically) or from an instance.
    '''

    def set_fill_defaults(self=None, **kwargs):
        filler = self.__instance_filler if self is not None else LinkedDataFrame.__class_filler
        filler.set_fill_defaults(**kwargs)

    def temporary_fill_defaults(self=None, **kwargs):
        filler = self.__instance_filler if self is not None else LinkedDataFrame.__class_filler
        yield from filler.temporary_fill_defaults(**kwargs)

    def reset_fill_defaults(self):
        filler = self.__instance_filler if self is not None else LinkedDataFrame.__class_filler
        filler.reset_fill_defaults()

    @classmethod
    def _get_class_fill(cls, series: Series) -> Any:
        return cls.__class_filler.get_fill(series)

    def _get_fill(self, series: Series, series_name) -> Any:
        if series_name in self.__column_fills:
            return self.__column_fills[series_name]
        return self.__instance_filler.get_fill(series)

    # endregion

    # region Link lookups

    def __getitem__(self, item):
        if item in self.__links:
            link = self.__links[item]
            history = deque([link])

            if link.aggregation_required:
                return self._LeafAggregation(self.index, history)
            elif link.chained:
                return self._LinkNode(self.index, history)
            else:
                return self._LinkLeaf(self.index, history)
        return super().__getitem__(item)

    def __getattr__(self, item):
        return self[item]

    def _init_link_history(self, item):
        meta = self.__links[item]
        history = deque([meta])
        node = self._LinkNode(self.index, history)
        return node

    def _link_names(self):
        return list(self.__links.keys())

    def _has_link(self, item):
        return item in self.__links

    def _get_link(self, item) -> _LinkMeta:
        return self.__links[item]

    # endregion

    # region Link Node Classes

    class _BaseNode:

        _root_index: Index
        _history: Deque[_LinkMeta]

        def __init__(self, root_index: Index, link_entries: Deque[_LinkMeta]):
            self._root_index: Index = root_index
            self._history: Deque[_LinkMeta] = link_entries

        @property
        def _top(self) -> _LinkMeta:
            return self._history[0]

        def _resolve_history(self, series: ndarray, fill_value, skip_first=False):
            for meta in self._history:
                if skip_first:
                    skip_first = False
                    continue
                meta: _LinkMeta = meta
                indexer = meta.indexer
                series = series[indexer]
                series[indexer < 0] = fill_value
            return Series(series, index=self._root_index)

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError:
                raise AttributeError(item)

    class _LinkLeaf(_BaseNode):
        """One-to-one or one-to-many node for vanilla DataFrames"""

        def __dir__(self):
            df: DataFrame = self._top.other
            return [str(col) for col in df.columns if str(col).isidentifier()]

        def __getitem__(self, item):
            top = self._top
            df = top.other
            series = df[item]
            fill_value = LinkedDataFrame._get_class_fill(series)
            return self._resolve_history(series.values, fill_value)

    class _LinkNode(_BaseNode):
        """One-to-one or one-to-many node for LinkedDataFrames"""

        def __dir__(self):
            df: 'LinkedDataFrame' = self._top.other
            return [str(col) for col in df.columns if str(col).isidentifier()] + df._link_names()

        def __getitem__(self, item):
            top = self._top
            df: 'LinkedDataFrame' = top.other

            if df._has_link(item):
                history_copy = self._history.copy()
                link_item = df._get_link(item)
                history_copy.appendleft(link_item)

                if link_item.aggregation_required:
                    return LinkedDataFrame._LeafAggregation(self._root_index, history_copy)
                elif link_item.chained:
                    return LinkedDataFrame._LinkNode(self._root_index, history_copy)
                else:
                    return LinkedDataFrame._LinkLeaf(self._root_index, history_copy)

            series = df[item]
            fill_value = df._get_fill(series, item)
            return self._resolve_history(series.values, fill_value)

    class _LeafAggregation(_BaseNode):
        def __dir__(self):
            return _SUPPORTED_AGGREGATIONS[:]

        def __getattr__(self, item):
            if item not in _SUPPORTED_AGGREGATIONS: raise AttributeError(item)
            return self._Aggregator(self, item)

        class _Aggregator:
            def __init__(self, owner: 'LinkedDataFrame._LeafAggregation', func_name: str):
                self._func_name = func_name
                self._owner = owner
                self._allow_nonnumeric = func_name in _NON_NUMERIC_AGGREGATIONS

            def __repr__(self):
                return f"<LinkAggregator[{self._func_name}]>"

            def __call__(self, expr="1", *, int_fill=-1, **kwargs):
                top = self._owner._top
                df = top.other
                grouper = top.other_grouper
                evaluation = df.eval(expr)
                if not isinstance(evaluation, Series):
                    evaluation = pd.Series(evaluation, index=df.index)

                series_type = infer_dtype(evaluation)
                if not self._allow_nonnumeric and series_type not in _NUMERIC_TYPES:
                    raise RuntimeError(f"Results of evaluation '{expr}' is non-numeric, which is not allowed for "
                                       f"aggregation function '{self._func_name}'")

                grouped = evaluation.groupby(grouper)
                array = getattr(grouped, self._func_name)(**kwargs).values

                # A fill value of NaN is only disallowed for integer types
                fill_value = np.nan
                if series_type == PandasDtype.INT_NAME:
                    fill_value = int_fill
                elif series_type == PandasDtype.TIME_NAME:
                    raise NotImplementedError("Haven't found a way to instantiate NaT filler")
                return self._owner._resolve_history(array, fill_value)

    # endregion

    # region Slicing functions
    '''
    Slicing a LinkedDataFrame returns another LinkedDataFrame with its links intact in most cases. It is safe to assume
    that, when slicing along the rows (axis 0), link data is being preserved. When slicing along the columns (axis 1),
    this is not always the case, particularly when excluding columns that are needed for linkages. In such cases, the
    dependent links are removed silently - no warning is given, as such operations are routinely performed by various
    Pandas functions. For example, DataFrame.pivot_table() will often drop columns. 
    
    A note on performance:
    At the time of writing, the design of the DataFrame class makes it very difficult to have both fast slicing and
    fast link-based indexing. When LinkedDataFrame "precomputes" a linkage, it stores an array of positional indices
    which make for fast lookups. This can be an expensive operation the first time, but then subsequent calls are very
    fast. When slicing along the 0-axis, Pandas computes a similar indexer and applies it to the the underlying data. It
    would be ideal if this indexer could be subsequently applied to the link-specific indexer, but unfortunately the 
    design of the __finalize__() method (which is used to copy over the link data) excludes this information. If the
    links were to be re-computed, this would slow down ALL slicing considerably - 1000x times at least. Conversely,
    dropping the linkage indexer negates the advantage of precomputing an indexer in the first place. 
    
    Short of changing the Pandas internals to pass an indexer to the __finalize__() method (a monumental task), the
    solution implemented here is to support a subset of 0-axis indexing operations by overriding the relevant methods
    that call __finalize__(). For these common operations, slicing and link-indexing will both be fast - the best of
    both worlds. The cost is that such operations will need to be tested carefully to catch any changes to the Pandas
    internals in the future.  
    
    Supported operations:
     - .loc[[True, False, True, True, False...]] (Boolean indexing)
     - .loc[["A", "B", "A", "C", ...]] (Label based indexing)
     - .loc[1001: 1999] (Label based slicing)
     - .iloc[0:5] (Positional based slicing)
     - .head()
     - .tail()
     - .groupby() iteration (for group_id, frame_subset in frame.groupby(...)). This should also capture .get_group()
    
    Peter Kucirek November 22 2018
    '''

    def __finalize__(self, other: 'LinkedDataFrame', method=None, **kwargs):
        # Other is the parent LDF, self is the slice
        super().__finalize__(other, method=method, **kwargs)

        # Copy the link metadata
        for link_name, link in other.__links.items():
            self.__links[link_name] = link.copy()

        self.__identified_links = other.__identified_links.copy()

        return self

    def __subset_links(self, target: 'LinkedDataFrame', indexer: ndarray):
        for link_name, link_entry in self.__links.items():
            target.__links[link_name] = link_entry.copy(indexer)

    def _take(self, indices, axis=0, is_copy=True):
        sliced: 'LinkedDataFrame' = super()._take(indices, axis=axis, is_copy=is_copy)
        if axis == 0:
            self.__subset_links(sliced, indices)
        return sliced

    def _reindex_with_indexers(self, reindexers, fill_value=None, copy=False,
                               allow_dups=False):
        new_frame = super()._reindex_with_indexers(reindexers, fill_value=fill_value, copy=copy,
                                                   allow_dups=allow_dups)

        if isinstance(new_frame, LinkedDataFrame) and 0 in reindexers:
            _, indexer = reindexers[0]
            self.__subset_links(new_frame, indexer)
        return new_frame

    def _slice(self, slobj, axis=0, kind=None):
        new_frame = super()._slice(slobj, axis=axis, kind=kind)
        if axis == 0 and isinstance(new_frame, LinkedDataFrame):
            # slobj is not actually an ndarray, but a builtin <slice> Python object. Fortunately, ndarray.__getitem__()
            # supports <slice> objects just as easily as integer indexers, so it just works.
            self.__subset_links(new_frame, slobj)
        return new_frame

    # endregion

    def link_summary(self) -> DataFrame:
        summary_table = {'name': [], 'target_shape': [], 'on_self': [], 'on_other': [], 'chained': [],
                         'aggregation': [], 'preindexed': []}
        for name, entry in self.__links.items():
            summary_table['name'].append(name)
            summary_table['target_shape'].append(str(entry.other.shape))
            summary_table['on_self'].append(str(entry.self_meta))
            summary_table['on_other'].append(str(entry.other_meta))
            summary_table['chained'].append(entry.chained)
            summary_table['aggregation'].append(entry.aggregation_required)
            summary_table['preindexed'].append(entry.flat_indexer is not None)

        df = DataFrame(summary_table)
        df.set_index('name', inplace=True)
        return df
