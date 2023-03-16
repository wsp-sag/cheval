"""Top-level API, including the main LinkedDataFrame class"""
import attr
from collections import deque
from deprecated import deprecated
import numexpr as ne
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index, MultiIndex
from typing import Any, Dict, Deque, Hashable, List, Optional, Set, Tuple, Type, Union

from .constants import LinkageSpecificationError, LinkAggregationRequired
from .missing_data import SeriesFillManager, infer_dtype, PandasDtype
from ..parsing.constants import NAN_STR, NAN_VAL, NEG_INF_STR, NEG_INF_VAL
from ..parsing.expressions import Expression
from ..parsing.expr_items import EvaluationMode
from ..utils import convert_series, to_numpy

_LabelType = Union[str, List[str]]

_NUMERIC_AGGREGATIONS = {'max', 'min', 'mean', 'median', 'prod', 'std', 'sum', 'var', 'quantile'}
_NON_NUMERIC_AGGREGATIONS = {'count', 'first', 'last', 'nth'}
_SUPPORTED_AGGREGATIONS = sorted(_NUMERIC_AGGREGATIONS | _NON_NUMERIC_AGGREGATIONS)
_NUMERIC_TYPES = {PandasDtype.INT_NAME, PandasDtype.UINT_NAME, PandasDtype.FLOAT_NAME, PandasDtype.BOOL_NAME}


class _IndexMeta:

    labels: List[str] = attr.ib(converter=lambda x: [x] if isinstance(x, str) else list(x))
    from_row_labels: bool = attr.ib()

    def __init__(self, labels: Union[str, List[str]] = None, from_row_labels: bool = True):
        if isinstance(labels, str):
            labels = [labels]
        elif labels is None:
            from_row_labels = True
        self.labels = labels
        self.from_row_labels = from_row_labels

    def __str__(self):
        if self.from_row_labels:
            return f'From index: {self.labels}'
        return f'From columns: {self.labels}'

    def get_indexer(self, frame: DataFrame) -> Union[Index, MultiIndex]:
        if self.labels is None:
            return frame.index

        arrays = []
        if self.from_row_labels:
            if len(self.labels) > frame.index.nlevels:
                raise LinkageSpecificationError('Cannot specify more levels than in the index')

            for label in self.labels:
                # `get_level_values` works on both Index and MultiIndex objects and accepts both
                # integer levels AND level names
                try:
                    arrays.append(frame.index.get_level_values(label))
                except KeyError:
                    raise LinkageSpecificationError(f'Level `{label}` not in the index')
        else:
            for label in self.labels:
                if label not in frame:
                    raise LinkageSpecificationError(f'Column `{label}` not in the columns')
                arr = to_numpy(frame[label])
                arrays.append(arr)

        if len(arrays) == 1:
            name = self.labels[0]
            return Index(arrays[0], name=name)
        return MultiIndex.from_arrays(arrays, names=self.labels)

    def nlevels(self, frame: DataFrame) -> int:
        if self.labels is None:
            return frame.index.nlevels
        return len(self.labels)

    def validate(self, frame: DataFrame):
        if self.labels is None:
            return  # Use the index, which is always available
        frame_items = set(frame.index.levels) if self.from_row_labels else set(frame.columns)
        item_name = 'index' if self.from_row_labels else 'columns'

        for name in self.labels:
            assert name in frame_items, f'Could not find `{name}` in the {item_name}'


class _LinkMeta:
    owner: 'LinkedDataFrame'
    other: Union['LinkedDataFrame', DataFrame]
    _other_has_links: bool
    aggregation_required: bool
    self_meta: _IndexMeta
    other_meta: _IndexMeta

    flat_indexer: Optional[np.ndarray]
    other_grouper: Optional[np.ndarray]
    missing_indices: Optional[Union[np.ndarray, List]]

    @staticmethod
    def create(owner: 'LinkedDataFrame', other: Union['LinkedDataFrame', DataFrame], self_labels: Union[List[str], str],
               self_from_row_labels: bool, other_labels: Union[List[str], str], other_from_row_labels: bool,
               precompute: bool = True) -> '_LinkMeta':
        self_meta = _IndexMeta(self_labels, self_from_row_labels)
        other_meta = _IndexMeta(other_labels, other_from_row_labels)

        assert self_meta.nlevels(owner) == other_meta.nlevels(other)

        other_has_links = isinstance(other, LinkedDataFrame)

        link = _LinkMeta(owner, other, self_meta, other_meta, other_has_links)
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
        self.missing_indices = None

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
            raise RuntimeError('Many-to-many links are not permitted')
        self.aggregation_required = flag

        if precompute:
            self._make_indexer(self_indexer, other_indexer)

    def _get_indexers(self) -> Tuple[Index, Index]:
        return self.self_meta.get_indexer(self.owner), self.other_meta.get_indexer(self.other)

    def _make_indexer(self, self_indexer: Index, other_indexer: Index):
        if self.aggregation_required:
            group_ints, group_order = other_indexer.factorize()
            self.other_grouper = group_ints
            self.flat_indexer, self.missing_indices = group_order.get_indexer_non_unique(self_indexer)
        else:  # Performance-tuned fast paths for constructing indexers
            if self_indexer.equals(other_indexer):  # Indexers are identical
                self.flat_indexer = np.arange(len(other_indexer))
                self.missing_indices = np.array([], dtype=int)
            else:
                # Originally, different logic was used if the self indexer didn't map cleanly onto the other indexer
                # (the other was missing values, or the self had NaNs).
                # Those were combined into a single case for performance purposes
                self.flat_indexer = other_indexer.get_indexer(self_indexer)
                self.missing_indices = np.nonzero(self.flat_indexer == -1)[0]

    @property
    def chained(self) -> bool:
        return self._other_has_links

    @property
    def indexer_and_missing(self) -> Tuple[np.ndarray, np.ndarray]:
        if (self.flat_indexer is None) or (self.missing_indices is None):
            self.precompute()
        return self.flat_indexer, self.missing_indices

    def precompute(self):
        """Top-level method to precompute the indexer"""
        self._make_indexer(*self._get_indexers())

    def copy_meta(self) -> '_LinkMeta':
        copied = _LinkMeta(self.owner, self.other, self.self_meta, self.other_meta, self._other_has_links)
        copied.aggregation_required = self.aggregation_required
        return copied

    def copy(self, indices=None) -> '_LinkMeta':
        copied = self.copy_meta()

        if self.flat_indexer is not None:
            copied.flat_indexer = self.flat_indexer[indices] if indices is not None else self.flat_indexer

        if isinstance(self.missing_indices, list):
            copied.missing_indices = []
        elif isinstance(self.missing_indices, np.ndarray):
            if (indices is not None) and (len(self.missing_indices) > 0):
                mask = np.isin(self.missing_indices, indices)
                copied.missing_indices = self.missing_indices[mask]
            else:
                copied.missing_indices = self.missing_indices[:]

        if self.other_grouper is not None:
            copied.other_grouper = self.other_grouper

        return copied


class LinkedDataFrame(DataFrame):

    _links: Dict[str, _LinkMeta]
    _identified_links: Set[str]
    _class_filler: SeriesFillManager = SeriesFillManager()
    _instance_filler: SeriesFillManager
    _column_fills: dict

    # temporary properties
    _internal_names = DataFrame._internal_names + ['_links', '_identified_links', '_instance_filler', '_column_fills']
    _internal_names_set = set(_internal_names)

    # normal properties
    _metadata = []

    # region Static Readers

    @staticmethod
    def read_csv(*args, **kwargs) -> 'LinkedDataFrame':
        """Wrapper for pd.read_csv() that returns a LinkedDataFrame instead"""
        return LinkedDataFrame(pd.read_csv(*args, **kwargs))

    @staticmethod
    def read_table(*args, **kwargs) -> 'LinkedDataFrame':
        """Wrapper for pd.read_table() that returns a LinkedDataFrame instead"""
        return LinkedDataFrame(pd.read_table(*args, **kwargs))

    @staticmethod
    def read_clipboard(*args, **kwargs) -> 'LinkedDataFrame':
        """Wrapper for pd.read_clipboard() that returns a LinkedDataFrame instead"""
        return LinkedDataFrame(pd.read_clipboard(*args, **kwargs))

    @staticmethod
    def read_excel(*args, **kwargs) -> 'LinkedDataFrame':
        """Wrapper for pd.read_excel() that returns a LinkedDataFrame instead"""
        return LinkedDataFrame(pd.read_excel(*args, **kwargs))

    @staticmethod
    def read_fwf(*args, **kwargs) -> 'LinkedDataFrame':
        """Wrapper for pd.read_fwf() that returns a LinkedDataFrame instead"""
        return LinkedDataFrame(pd.read_fwf(*args, **kwargs))

    @staticmethod
    def read_(name, *args, **kwargs) -> 'LinkedDataFrame':
        func = getattr(pd, f'read_{name}')
        return LinkedDataFrame(func(*args, **kwargs))

    # endregion

    @property
    def _constructor(self) -> Type['LinkedDataFrame']:
        # see https://pandas.pydata.org/docs/development/extending.html#override-constructor-properties
        return LinkedDataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        object.__setattr__(self, '_links', {})
        object.__setattr__(self, '_identified_links', set())
        object.__setattr__(self, '_instance_filler', SeriesFillManager())
        object.__setattr__(self, '_column_fills', {})

    def link_to(self, other: DataFrame, name: str, *, on: _LabelType = None, levels: _LabelType = None,
                on_self: _LabelType = None, on_other: _LabelType = None, self_levels: _LabelType = None,
                other_levels: _LabelType = None, precompute: bool = True) -> LinkAggregationRequired:
        """Creates a new link from this DataFrame to another, assigning it to the given name.

        The relationship between the left-hand-side (this DataFrame itself) and the right-hand-side (the other
        DataFrame) must be pre-specified to create the link. The relationship can be based on the index (or a subset of
        it levels in a MultiIndex) OR based on columns in either DataFrame.

        By default, if both the "levels" and "on" args of one side are None, then the join will be made on ALL levels
        of the side's index.

        Regardless of whether the join is based on an index or columns, the same number of levels must be given. For
        example, if the left-hand indexer uses two levels from the index, then the right-hand indexer must also use two
        levels in the index or two columns.

        When the link is established, it is tested whether the relationship is one-to-one or many-to-one (the latter
        indicates that aggregation is required). The result of this test is returned by this method.

        Args:
            other: The table to join.
            name: The alias (symbolic name) of the new link. If Pythonic, this will show up as an attribute; otherwise
                the link will need to be accessed using [].
            on: If given, the join will be made on the provided **column(s)** in both this and the other DataFrame. This
                arg cannot be used with `levels` and will override `on_self` and `on_other`.
            on_self: If provided, the left-hand side of the join will be made on the column(s) in this DataFrame. This
                arg cannot be used with `self_levels`.
            on_other: If provided, th right-hand-side of the join will be made on the column(s) in the other DataFrame.
                This arg cannot be used with `other_levels`.
            levels: If provided, the join will be made on the given **level(s)** in both this and the other DataFrame's
                index. It can be specified as an integer or a string, if both indexes have the same level names. This
                arg cannot be used with `on` and will override `self_levels` and `other_levels`.
            self_levels: If provided, the left-hand-side of the join will be made on the level(s) in this DataFrame.
                This arg cannot be used with `on_self`.
            other_levels: If provided, the right-hand-side of the join will be made on the level(s) in the other
                DataFrame. This arg cannot be used with `on_other`.
            precompute: Link items store indexer arrays which allow for very fast lookup, but can take a significant
                amount of time to compute (especially when two or more columns are used for the join). When
                precompute is set to True, this indexing operation occurs within the link_to() method. Otherwise,
                indexing is done when first a link is requested, or manually using LinkedDataFrame.compute_indexer().

        Returns:
            LinkAggregationRequired: True if aggregation is required for this link. False otherwise.

        Raises:
            LinkageSpecificationError: For mis-specified linkages.
            KeyError: For linkages using columns or level not in this or the other DataFrame.
        """
        if other.columns.nlevels != 1:
            raise NotImplementedError('Cannot link to tables with multi-level columns')

        on_not_none = on is not None
        levels_not_none = levels is not None

        if on_not_none and levels_not_none:
            raise LinkageSpecificationError('Can only specify one of `on=` or `levels=`')

        if on_not_none:
            on_self = on
            on_other = on

        if levels_not_none:
            self_levels = levels
            other_levels = levels

        if (on_self is not None) and (self_levels is not None):
            raise LinkageSpecificationError()
        elif self_levels is not None:
            self_labels, self_from_index = self_levels, True
        else:
            self_labels, self_from_index = on_self, False

        if (on_other is not None) and (other_levels is not None):
            raise LinkageSpecificationError()
        elif other_levels is not None:
            other_labels, other_from_index = other_levels, True
        else:
            other_labels, other_from_index = on_other, False

        link_data = _LinkMeta.create(self, other, self_labels, self_from_index, other_labels, other_from_index,
                                     precompute)
        self._links[name] = link_data
        if name.isidentifier():
            self._identified_links.add(name)

        return LinkAggregationRequired.YES if link_data.aggregation_required else LinkAggregationRequired.NO

    def __dir__(self):
        """Override dir() to show links as valid attributes"""
        return super().__dir__() + sorted(self._identified_links)

    # region Fill value management

    def set_column_fill(self, column, value):
        """Set a specific fill value for a column in this LinkedDataFrame. This is only used when this frame is the
        TARGET of a link (e.g. some_other_table.link_to(this_table,...).

        Args:
            column: The column name. Must currently exist in this frame, otherwise KeyError gets raised.
            value: The value to be returned if accessed through a partial link.

        Raises:
            KeyError: if the column doesn't exist.
        """
        if column not in self.columns:
            raise KeyError(column)
        self._column_fills[column] = value

    # The pattern "self=None" allows these methods to have different behaviour depending on whether called from the
    # class (statically) or from an instance.

    def set_fill_defaults(self=None, **kwargs):
        filler = self._instance_filler if self is not None else LinkedDataFrame._class_filler
        filler.set_fill_defaults(**kwargs)

    def temporary_fill_defaults(self=None, **kwargs):
        filler = self._instance_filler if self is not None else LinkedDataFrame._class_filler
        yield from filler.temporary_fill_defaults(**kwargs)

    def reset_fill_defaults(self=None):
        filler = self._instance_filler if self is not None else LinkedDataFrame._class_filler
        filler.reset_fill_defaults()

    @classmethod
    def _get_class_fill(cls, series: Series) -> Any:
        return cls._class_filler.get_fill(series)

    def _get_fill(self, series: Series, series_name) -> Any:
        if series_name in self._column_fills:
            return self._column_fills[series_name]
        return self._instance_filler.get_fill(series)

    # endregion

    # region Link lookups

    def __getitem__(self, item):
        if isinstance(item, Hashable) and item in self._links:
            return self._init_link(item)
        return super().__getitem__(item)

    def __getattr__(self, item):
        if isinstance(item, Hashable) and item in self._links:
            return self._init_link(item)
        return super().__getattr__(item)

    def _init_link(self, item):
        link = self._links[item]
        history = deque([link])

        if link.aggregation_required:
            return self._LeafAggregation(self.index, history)
        elif link.chained:
            return self._LinkNode(self.index, history)
        else:
            return self._LinkLeaf(self.index, history)

    def _init_link_history(self, item):
        meta = self._links[item]
        history = deque([meta])
        node = self._LinkNode(self.index, history)
        return node

    def _link_names(self):
        return list(self._links.keys())

    def _has_link(self, item):
        return item in self._links

    def _get_link(self, item) -> _LinkMeta:
        return self._links[item]

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

        def _resolve_history(self, series: np.ndarray, fill_value):
            for meta in self._history:
                # Get the indexing data
                indexer, missing = meta.indexer_and_missing

                series = series[indexer]  # Reorder/duplicate/drop the raw array
                series[missing] = fill_value  # Fill in missing values
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
            arr = to_numpy(series)
            return self._resolve_history(arr, fill_value)

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
            arr = to_numpy(series)
            return self._resolve_history(arr, fill_value)

    class _LeafAggregation(_BaseNode):
        def __dir__(self):
            return _SUPPORTED_AGGREGATIONS[:]

        def __getattr__(self, item):
            if item not in _SUPPORTED_AGGREGATIONS:
                raise AttributeError(item)
            return self._Aggregator(self, item)

        class _Aggregator:
            def __init__(self, owner: 'LinkedDataFrame._LeafAggregation', func_name: str):
                self._func_name = func_name
                self._owner = owner
                self._allow_nonnumeric = func_name in _NON_NUMERIC_AGGREGATIONS

            def __repr__(self):
                return f'<LinkAggregator[{self._func_name}]>'

            def __call__(self, expr='1', *, int_fill=-1, **kwargs):
                top = self._owner._top
                df = top.other

                if expr in df.columns:  # Shortcut if the expression just refers to a column name
                    evaluation = df[expr]
                else:
                    try:
                        evaluation = df.evaluate(expr)
                    except AttributeError:
                        evaluation = df.eval(expr)
                if not isinstance(evaluation, Series):
                    evaluation = pd.Series(evaluation, index=df.index)

                series_type = infer_dtype(evaluation)
                if not self._allow_nonnumeric and (series_type not in _NUMERIC_TYPES):
                    raise RuntimeError(f'Results of evaluation `{expr}` is non-numeric type {series_type}, which is not'
                                       f' allowed for aggregation function `{self._func_name}`')

                # A fill value of NaN is only disallowed for integer types
                fill_value = np.nan
                if series_type in {PandasDtype.INT_NAME, PandasDtype.UINT_NAME, PandasDtype.BOOL_NAME}:
                    fill_value = int_fill
                elif series_type == PandasDtype.TIME_NAME:
                    raise NotImplementedError("Haven't found a way to instantiate NaT filler")

                # Aggregate and align the result
                grouper = top.other_grouper
                grouped_result = to_numpy(getattr(evaluation.groupby(grouper), self._func_name)(**kwargs))

                return self._owner._resolve_history(grouped_result, fill_value)

    # endregion

    # region Slicing functions
    """
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

    Short of changing the Pandas internals to pass an indexer to the __finalize__() method (a monumental, if not
    impossible, task), the solution implemented here is to support a subset of 0-axis indexing operations by overriding
    the relevant methods that call __finalize__(). For these common operations, slicing and link-indexing will both be
    fast - the best of both worlds. The cost is that such operations will need to be tested carefully to catch any
    changes to the Pandas internals in the future.

    Supported operations:
     - .loc[[True, False, True, True, False...]] (Boolean indexing)
     - .loc[["A", "B", "A", "C", ...]] (Label based indexing)
     - .loc[1001: 1999] (Label based slicing)
     - .iloc[0:5] (Positional based slicing)
     - .head()
     - .tail()
     - .groupby() iteration (for group_id, frame_subset in frame.groupby(...)). This should also capture .get_group()

    Peter Kucirek November 22 2018

    ---

    Note: the following function in this section are intended to override the original functions in pandas. Our
    functions are intended to inject additional LinkedDataFrame operations. Depending on your version of pandas, type
    hinting may raise issues as these functions may be decorated with "@final"... we are just going to ignore that...
    """

    def __finalize__(self, other: 'LinkedDataFrame', method=None, **kwargs):
        # Other is the parent LDF, self is the slice
        super().__finalize__(other, method=method, **kwargs)

        # Sometimes, other is not a LDF (e.g. _Concatenator)
        if not isinstance(other, LinkedDataFrame):
            return self

        # Copy the link metadata from the parent dataframe
        # Note that as of pandas 1.5, the names of the methods is not considered stable
        # and may change in future versions.
        if method == "groupby":
            self._copy_links(other, reindex=True)
        else:
            for link_name, link in other._links.items():
                self._links[link_name] = link.copy_meta()
            self._identified_links = other._identified_links.copy()

        return self

    def _copy_links(self, parent: 'LinkedDataFrame', reindex: bool = False):
        """Copies linkage information from a parent dataframe

        If `reindex` is True, computes an indexer based on the parent DataFrame's and
        this DataFrame's indexes and passes it to `_LinkMeta.copy` to modify the linkage.
        """
        if reindex:
            indexer = parent.index.get_indexer(self.index)
        else:
            indexer = None
        link_names = set()
        for link_name, link in parent._links.items():
            self._links[link_name] = link.copy(indexer)
            link_names.add(link_name)
        self._identified_links = link_names

    def __subset_links(self, target: 'LinkedDataFrame', indexer: np.ndarray):
        for link_name, link_entry in self._links.items():
            target._links[link_name] = link_entry.copy(indexer)

    def _take_with_is_copy(self, indices, axis=0):  # pandas >= 1.0.x
        sliced: 'LinkedDataFrame' = super()._take_with_is_copy(indices, axis=axis)
        if axis == 0:
            self.__subset_links(sliced, indices)
        return sliced

    def _take(self, indices, axis=0, is_copy=True):  # pandas < 1.0.x
        sliced: 'LinkedDataFrame' = super()._take(indices, axis=axis, is_copy=is_copy)
        if axis == 0:
            self.__subset_links(sliced, indices)
        return sliced

    def _reindex_with_indexers(self, reindexers, fill_value=None, copy=False, allow_dups=False):
        new_frame = super()._reindex_with_indexers(reindexers, fill_value=fill_value, copy=copy, allow_dups=allow_dups)

        if isinstance(new_frame, LinkedDataFrame) and (0 in reindexers):
            _, indexer = reindexers[0]
            self.__subset_links(new_frame, indexer)
        return new_frame

    def _slice(self, slobj, axis=0, kind=None):
        new_frame = super()._slice(slobj, axis=axis)
        if axis == 0 and isinstance(new_frame, LinkedDataFrame):
            # slobj is not actually an ndarray, but a builtin <slice> Python object. Fortunately, ndarray.__getitem__()
            # supports <slice> objects just as easily as integer indexers, so it just works.
            self.__subset_links(new_frame, slobj)
        return new_frame

    def copy(self, deep=False) -> 'LinkedDataFrame':
        copied = super().copy(deep=deep)

        # Copy the link data
        for link_name, link_entry in self._links.items():
            copied._links[link_name] = link_entry.copy()

        return copied

    # endregion

    # region Expression evaluation

    def evaluate(self, expr, local_dict: Dict[str, Any] = None, out: Series = None, allow_casting=True,
                 ignore_check=False):
        """Evaluates a mathematical expression over all the rows in this frame. Very similar to vanilla .eval() in
        concept, but supports Cheval-style expression syntax. For example, DataFrame.eval() doesn't support the
        where() function, but LinkedDataFrame.evaluate does. Also supports accessing links (including aggregation)
        inside the expression.

        Args:
            expr: The expression string to evaluate. Supports a subset of Cheval syntax, excluding dict literals,
                and "@ <choice_name>" referencing which are particular to the ChoiceModel object.
            local_dict: A dictionary that replaces the local operands in current frame.
            out: An existing Series where the outcome is going to be stored. Useful for avoiding unnecessary new array
                allocations.
            allow_casting: A flag to allow for casts within a kind, like float64 to float32.
            ignore_check: A flag to ignore checking

        Returns:

        """
        new_expr = Expression.parse(expr, mode=EvaluationMode.DATAFRAME)

        ld = {} if local_dict is None else local_dict.copy()
        ld[NEG_INF_STR] = NEG_INF_VAL
        ld[NAN_STR] = NAN_VAL

        for column in new_expr.symbols:
            try:
                ld[column] = convert_series(self[column], allow_raw=False)
            except KeyError:
                if column not in ld:
                    raise

        for link_name, chain_symbol in new_expr.chains.items():
            assert self._has_link(link_name), f'Link `{link_name}` does not exist'

            for substitution, chain_tuple in chain_symbol.items():
                node = self[link_name]
                for attribute_name in reversed(chain_tuple.chain):
                    node = node[attribute_name]

                if chain_tuple.withfunc:
                    series = getattr(node, chain_tuple.func)(chain_tuple.args)
                else:
                    series = node

                ld[substitution] = convert_series(series)

        if out is not None:
            out = to_numpy(out)

        casting_rule = 'same_kind' if allow_casting else 'safe'
        vector = ne.evaluate(new_expr.transformed, local_dict=ld, out=out, casting=casting_rule)
        return Series(vector, index=self.index)

    @deprecated(reason='Use `LinkedDataFrame.evaluate()` instead, to avoid confusion over NumExpr semantics')
    def eval(self, expr, *args, **kwargs):
        return self.evaluate(expr, *args, **kwargs)

    # endregion

    # region Other methods

    def link_summary(self) -> DataFrame:
        """Produces a table summarizing all outgoing links from this frame.

        Returns:
            DataFrame: Has the following columns:
                - name: The assigned name of the link
                - target_shape: The shape of the "other" frame
                - on_self: String indicating which columns or levels in THIS frame used for the join
                - on_other: String indicating which columns or levels in the OTHER frame used for the join
                - chained: Flag indicating if the target frame also supports links
                - aggregation: Flag indicating if the relationship must be aggregated
                - preindexed: Flag indicating if the relationship has already been indexed.
        """
        summary_table = {'name': [], 'target_shape': [], 'on_self': [], 'on_other': [], 'chained': [],
                         'aggregation': [], 'preindexed': []}
        for name, entry in self._links.items():
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

    def compute_indexers(self, refresh: bool = True):
        """For all outgoing links, compute any missing indexers (precompute=False when calling link_to()), or refresh
        already-computed indexers.

        Args:
            refresh: If False, indexers will only be computed for links without them. Otherwise, this forces indexers
                to be re-computed.
        """
        for entry in self._links.values():
            if refresh or entry.flat_indexer is None:
                entry.precompute()

    def pivot_table(self, values=None, index=None, columns=None, **kwargs):
        # Construct a new DataFrame from any of the columns or lookups specified
        # in values, index, or columns, and then pivot that
        new_columns = {}
        value_cols = self._resolve_columns(values)
        new_columns.update(value_cols)
        index_cols = self._resolve_columns(index)
        new_columns.update(index_cols)
        column_cols = self._resolve_columns(columns)
        new_columns.update(column_cols)
        pivot_df = pd.DataFrame(new_columns, index=self.index)

        return pivot_df.pivot_table(index=index, columns=columns, values=values, **kwargs)

    def _resolve_columns(self, lookup_items) -> Dict[str, pd.Series]:
        """Resolves potential column names or lookups to a dict of Series

        Helper for pivot_table to pull any relevant columns or linkage lookups
        into a clean DataFrame.
        Because pivot_table can take a wide variety of inputs as index, columns, and values,
        this attempts to be very flexible with whether and how it looks up provided items.

        Any items which get resolved as columns or linkage lookups are returned as a
        Series in a key: value pair in the returned dict.
        """
        new_columns = {}
        if isinstance(lookup_items, str):
            if lookup_items in self.columns:
                new_columns[lookup_items] = self[lookup_items]
            else:
                new_columns[lookup_items] = self.evaluate(lookup_items)
        elif (lookup_items is not None) and not isinstance(lookup_items, (pd.Series, np.ndarray)):
            for item in lookup_items:
                if lookup_items in self.columns:
                    new_columns[item] = self[item]
                else:
                    new_columns[item] = self.evaluate(item)
        return new_columns

    def get_link_target(self, name) -> Union[DataFrame, 'LinkedDataFrame']:
        """Gets the referenced target ("other") of an outbound link. This is useful when the original reference to the
        targeted frame is no longer available, or to to modify the fill management of that target.

        Args:
            name: The name of the link alias.

        Returns:
            DataFrame: The targeted "other" frame of the link, if the link is not chained.
            LinkedDataFrame: The targeted "other" frame of the link, if the link IS chained

        Raises:
            KeyError: When the specified name is not a link.
        """
        return self._links[name].other

    # endregion
