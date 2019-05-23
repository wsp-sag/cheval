"""Top-level API, including the main LinkedDataFrame class"""
from typing import List, Dict, Union, Deque, Tuple, Callable, Any, Optional, Set
from collections import deque, Hashable

from pandas import DataFrame, Series, Index, MultiIndex
import pandas as pd
from numpy import ndarray
import numpy as np
import attr
import numexpr as ne

from .constants import LinkageSpecificationError, LinkAggregationRequired
from .missing_data import SeriesFillManager, infer_dtype, PandasDtype
from ..parsing.expressions import Expression
from ..parsing.expr_items import EvaluationMode
from cheval.misc import convert_series

_FillFunctionType = Callable[[Series], Union[int, float, bool, str]]
_NUMERIC_AGGREGATIONS = {'max', 'min', 'mean', 'median', 'prod', 'std', 'sum', 'var', 'quantile'}
_NON_NUMERIC_AGGREGATIONS = {'count', 'first', 'last', 'nth'}
_SUPPORTED_AGGREGATIONS = sorted(_NUMERIC_AGGREGATIONS | _NON_NUMERIC_AGGREGATIONS)
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
        if self.labels is None: return  # Use the index, which is always available
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
            if self_indexer.equals(other_indexer):
                flat_indexer = np.arange(len(other_indexer))
            else:
                flat_indexer = other_indexer.get_indexer(self_indexer)
            self.flat_indexer = flat_indexer

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
    def read_csv(*args, **kwargs) -> 'LinkedDataFrame':
        return LinkedDataFrame(pd.read_csv(*args, **kwargs))

    @staticmethod
    def read_table(*args, **kwargs) -> 'LinkedDataFrame':
        return LinkedDataFrame(pd.read_table(*args, **kwargs))

    @staticmethod
    def read_clipboard(*args, **kwargs) -> 'LinkedDataFrame':
        return LinkedDataFrame(pd.read_clipboard(*args, **kwargs))

    @staticmethod
    def read_excel(*args, **kwargs) -> 'LinkedDataFrame':
        return LinkedDataFrame(pd.read_excel(*args, **kwargs))

    @staticmethod
    def read_fwf(*args, **kwargs) -> 'LinkedDataFrame':
        return LinkedDataFrame(pd.read_fwf(*args, **kwargs))

    @staticmethod
    def read_(name, *args, **kwargs) -> 'LinkedDataFrame':
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
                other_levels: _LabelType=None, precompute: bool=True) -> LinkAggregationRequired:
        """
        Creates a new link from this DataFrame to another, assigning it to the given name.

        The relationship between the left-hand-side (this DataFrame itself) and the right-hand-side (the other
        DataFrame) must be pre-specified to create the link. The relationship can be based on the index (or a subset of
        it levels in a MultiIndex) OR based on columns in either DataFrame.

        By default, if both the "levels" and "on" args of one side are None, then the join will be made on ALL levels
        of the side's index.

        Regardless of whether the join is based on an index or columns, the same number of levels must be given. For
        example, if the left-hand indexer uses two levels from the index, then the right-hand indexer must also use
        two levels in the index or two columns.s

        When the link is established, it is tested whether the relationship is one-to-one or many-to-one (the latter
        indicates that aggregation is required). The result of this test is returned by this method.

        Args:
            other: The table to join.
            name:  The alias (symbolic name) of the new link. If Pythonic, this will show up as an
                attribute; otherwise the link will need to be accessed using [].
            on: If given, the join will be made on the provided **column(s)** in both this
                and the other DataFrame. This arg cannot be used with `levels` and will override `on_self` and
                `on_other`.
            on_self: If provided, the left-hand side of the join will be made on the
                column(s) in this DataFrame. This arg cannot be used with `self_levels`.
            on_other: If provided, th right-hand-side of the join will be made on the
                column(s) in the other DataFrame. This arg cannot be used with `other_levels`.
            levels: If provided, the join will be made on the given **level(s)**
                in both this and the other DataFrame's index. It can be specified as an integer or a string,
                if both indexes have the same level names. This arg cannot be used with `on` and will override
                `self_levels` and `other_levels`.
            self_levels: If provided, the left-hand-side of the join will be made on the
                level(s) in this DataFrame. This arg cannot be used with `on_self`.
            other_levels: If provided, the right-hand-side of the join will be made on the
                level(s) in the other DataFrame. This arg cannot be used with `on_other`.
            precompute: Link items store indexer arrays which allow for very fast lookup, but can take a
                significant amount of time to compute (especially when two or more columns are used for the join). When
                precompute is set to True, this indexing operation occurs within the link_to() method. Otherwise,
                indexing is done when first a link is requested, or manually using LinkedDataFrame.compute_indexer().

        Returns:
            LinkAggregationRequired: True if aggregation is required for this link. False otherwise.

        Raises:
            LinkageSpecificationError: For mis-specified linkages.
            KeyError: For linkages using columns or level not in this or the other DataFrame.

        """
        if other.columns.nlevels != 1:
            raise NotImplementedError("Cannot link to tables with multi-level columns")

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

    def reset_fill_defaults(self=None):
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
        if isinstance(item, Hashable) and item in self.__links:
            return self._init_link(item)
        return super().__getitem__(item)

    def __getattr__(self, item):
        if isinstance(item, Hashable) and item in self.__links:
            return self._init_link(item)
        return super().__getattr__(item)

    def _init_link(self, item):
        link = self.__links[item]
        history = deque([link])

        if link.aggregation_required:
            return self._LeafAggregation(self.index, history)
        elif link.chained:
            return self._LinkNode(self.index, history)
        else:
            return self._LinkLeaf(self.index, history)

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
                    raise RuntimeError(f"Results of evaluation '{expr}' is non-numeric type {series_type}, which is not"
                                       f" allowed for aggregation function '{self._func_name}'")

                # A fill value of NaN is only disallowed for integer types
                fill_value = np.nan
                if series_type == PandasDtype.INT_NAME:
                    fill_value = int_fill
                elif series_type == PandasDtype.TIME_NAME:
                    raise NotImplementedError("Haven't found a way to instantiate NaT filler")

                grouped = evaluation.groupby(grouper)
                series = getattr(grouped, self._func_name)(**kwargs)

                if not series.index.equals(grouper):
                    # In some cases (like nth), the returned array is missing some values
                    series = series.reindex(grouper, fill_value=fill_value)

                return self._owner._resolve_history(series.values, fill_value)

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

        # Sometimes, other is not a LDF (e.g. _Concatenator)
        if not isinstance(other, LinkedDataFrame): return self

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

    # region Expression evaluation

    def eval(self, expr, *args, **kwargs):
        """
        Overload of DataFrame.eval(), which adds string and categorical support, but removes some of the flexibility
        (such as the inplace arg, and the ability to change engines). Also supports links inside the expression.

        Args:
            expr:

        Returns:

        """

        new_expr = Expression.parse(expr, mode=EvaluationMode.DATAFRAME)

        ld = {} if 'local_dict' not in kwargs else kwargs['local_dict'].copy()
        for column in new_expr.symbols:
            try:
                ld[column] = convert_series(self[column], allow_raw=False)
            except TypeError:
                pass
            except KeyError:
                if column not in ld: raise

        for link_name, chain_symbol in new_expr.chains.items():
            assert self._has_link(link_name), f"Link '{link_name}' does not exist"

            for substitution, chain_tuple in chain_symbol.items():
                node = self[link_name]
                for attribute_name in reversed(chain_tuple.chain):
                    node = node[attribute_name]

                if chain_tuple.withfunc:
                    series = getattr(node, chain_tuple.func)(chain_tuple.args)
                else:
                    series = node

                ld[substitution] = convert_series(series)

        vector = ne.evaluate(new_expr.transformed, local_dict=ld)
        return Series(vector, index=self.index)

    # endregion

    # region Other methods

    def link_summary(self) -> DataFrame:
        """
        Produces a table summarizing all outgoing links from this frame.

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

    def compute_indexers(self, refresh: bool = True):
        """
        For all outoing links, compute any missing indexers (precompute=False when calling link_to()), or refresh
        already-computed indexers.

        Args:
            refresh: If False, indexers will only be computed for links without them. Otherwise, this forces indexers
                to be re-computed.
        """

        for entry in self.__links.values():
            if refresh or entry.flat_indexer is None:
                entry.precompute()

    def pivot_table(self, values=None, index=None, columns=None,
                    aggfunc='mean', fill_value=None, margins=False,
                    dropna=True, margins_name='All'):
        temp_columns = []
        try:
            new_index, temp_flags = self._make_temp_col(index)
            if new_index[0] is None:
                new_index = None
            else:
                for col_name, is_temp in zip(new_index, temp_flags):
                    if is_temp: temp_columns.append(col_name)
                if len(new_index) == 1: new_index = new_index[0]

            new_columns, temp_flags = self._make_temp_col(columns)
            if new_columns[0] is None:
                new_columns = None
            else:
                for col_name, is_temp in zip(new_columns, temp_flags):
                    if is_temp: temp_columns.append(col_name)
                if len(new_columns) == 1: new_columns = new_columns[0]

            new_values, temp_flags = self._make_temp_col(values)
            if new_values[0] is None:
                new_values = None
            else:
                for col_name, is_temp in zip(new_values, temp_flags):
                    if is_temp: temp_columns.append(col_name)
                if len(new_values) == 1: new_values = new_values[0]

            return super().pivot_table(index=new_index, columns=new_columns, values=new_values, aggfunc=aggfunc,
                                       fill_value=fill_value, margins=margins, dropna=dropna,
                                       margins_name=margins_name)
        finally:
            for c in temp_columns:
                del self[c]

    def _make_temp_col(self, item: Union[str, List[str]]) -> Tuple[List[Optional[str]], List[bool]]:
        if item is None: return [None], [False]

        item_is_single = isinstance(item, str)
        items = [item] if item_is_single else item

        new_columns, flags = [], []
        for sub_item in items:
            if sub_item in self:
                new_columns.append(sub_item)
                flags.append(False)
            else:
                new_column_values = self.eval(sub_item)  # Evaluate the column to find links
                counter = 0
                new_column_name = f"temp_{counter}"
                while new_column_name in self:
                    counter += 1
                    new_column_name = f"temp_{counter}"
                self[new_column_name] = new_column_values
                new_columns.append(new_column_name)
                flags.append(True)
        return new_columns, flags

    def get_link_target(self, name) -> Union[DataFrame, 'LinkedDataFrame']:
        """
        Gets the referenced target ("other") of an outbound link. This is useful when the original
        reference to the targeted frame is no longer available, or to to modify the fill management of that
        target.

        Args:
            name: The name of the link alias.

        Returns:
            DataFrame: The targeted "other" frame of the link, if the link is not chained.
            LinkedDataFrame: The targeted "other" frame of the link, if the link IS chained

        Raises:
            KeyError: When the specified name is not a link.

        """
        return self.__links[name].other

    # endregion
