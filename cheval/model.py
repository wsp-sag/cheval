from typing import Iterable, List, Dict, Union, Tuple, Iterator, Set
from collections import deque
from itertools import chain as iter_chain
from multiprocessing import cpu_count
from logging import Logger

from pandas import Series, DataFrame, Index, MultiIndex
import pandas as pd
from numpy import ndarray
import numpy as np
import numexpr as ne
import numba as nb

from .api import AbstractSymbol, ExpressionGroup, ChoiceNode, NumberSymbol, VectorSymbol, TableSymbol, MatrixSymbol
from .exceptions import ModelNotReadyError
from .parsing import NAN_STR as NAN_STR
from .core import (worker_nested_probabilities, worker_nested_sample, worker_multinomial_probabilities,
                   worker_multinomial_sample)


OUT_STR = "__OUT"
NEG_INF_STR = "NEG_INF"
NEG_INF = -np.inf


class ChoiceModel(object):

    def __init__(self):
        self._max_level: int = 0
        self._top_children: Dict[str, ChoiceNode] = {}
        self._expressions: ExpressionGroup = ExpressionGroup()
        self._scope: Dict[str, AbstractSymbol] = {}
        self._decision_units: Index = None
        self._cached_cols: Index = None

    # region Tree operations

    def add_choice(self, name: str, logsum_scale: float=1.0) -> ChoiceNode:
        """
        Create and add a new discrete choice to the model, at the top level. Returns a node object which can also add
        nested choices, and so on. Choice names must only be unique within a given nest, although for clarity it is
        recommended that choice names are unique across all nests (especially when sampling afterwards)

        Args:
            name: The name of the choice to be added. The name will also appear in the returned Series or DataFrame when
                the model is run in discrete mode.
            logsum_scale: The "theta" parameter, commonly referred to as the logsum scale. Must be in the interval
                (0, 1.0].

        Returns:
            ChoiceNode: The added choice node, which also has an "add_choice" method for constructing nested models.

        """
        if self._cached_cols is not None: self._cached_cols = None

        node = ChoiceNode(name, logsum_scale=logsum_scale, level=1)
        self._top_children[name] = node
        return node

    def add_choices(self, names: Iterable[str], logsum_scales: Iterable[float]=None
                    ) -> Dict[str, ChoiceNode]:
        """
        Convenience function for batch-adding several choices at once (for a multinomial logit model). See add_choice()
        for more details

        Args:
            names:
            logsum_scales:

        Returns:
            dict: Mapping of name: ChoiceNode for the added nodes

        """
        if self._cached_cols is not None: self._cached_cols = None

        if logsum_scales is None:
            logsum_scales = [1.0 for _ in names]
        retval = {}
        for name, logsum_scale in zip(names, logsum_scales):
            node = ChoiceNode(name, logsum_scale=logsum_scale, level=1)
            retval[name] = node
            self._top_children[name] = node
        return retval

    @property
    def choices(self) -> Index:
        """Pandas Index representing the choices in the model"""
        self.validate_tree()
        max_level = self.depth

        if max_level == 1:
            return Index(sorted(self._top_children.keys()))
        else:
            node_ids = self._nested_tuples(max_level)

            level_names = ['root']
            for i in range(1, max_level): level_names.append(f'nest_{i + 1}')

            return MultiIndex.from_tuples(node_ids, names=level_names)

    @property
    def depth(self) -> int:
        return max(c.max_level() for c in self._top_children.values())

    def _nested_tuples(self, max_level):
        node_ids = []
        for c in self._top_children.values():
            node_ids += c.nested_ids(max_level)

        return node_ids

    def _all_children(self) -> Iterator[ChoiceNode]:
        q = deque()
        for c in self._top_children.values(): q.append(c)
        while len(q) > 0:
            c = q.popleft()
            yield c
            for c2 in c.children(): q.append(c2)

    def _flatten(self) -> Tuple[ndarray, ndarray, ndarray]:
        """Converts nested structure to arrays for Numba-based processing"""
        max_level = self.depth
        assert max_level > 1
        node_ids = self._nested_tuples(max_level)
        node_positions = {name: i for i, name in enumerate(node_ids)}

        hierarchy = np.full(len(node_ids), -1, dtype='i8')
        levels = np.zeros(len(node_ids), dtype='i8')
        logsum_scales = np.ones(len(node_ids), dtype='f8')

        for node in self._all_children():
            position = node_positions[node._nested_id(max_level)]
            levels[position] = node.level - 1  # Internal levels start at 1.

            if node.parent is not None:
                parent_position = node_positions[node.parent._nested_id(max_level)]
                hierarchy[position] = parent_position

            if node.is_parent:
                logsum_scales[position] = node.logsum_scale

        return hierarchy, levels, logsum_scales

    # endregion
    # region Expressions and scope operations

    @property
    def decision_units(self) -> Index:
        if self._decision_units is None: raise ModelNotReadyError("No decision units defined")
        return self._decision_units

    @decision_units.setter
    def decision_units(self, item):

        # If there are any assigned symbols, clear them so as not to conflict with the new decision units
        for symbol in self._scope.values():
            symbol.empty()

        if isinstance(item, Index):
            self._decision_units = item
        else:
            self._decision_units = Index(item)

    def declare_number(self, name: str):
        """Declares a simple scalar variable, of number or text type"""
        symbol = NumberSymbol(self, name)
        self._scope[name] = symbol

    def declare_vector(self, name: str, orientation: int):
        """
        Declares a vector variable. Vectors can be aligned with the decision units (rows, orientation=0) or choices (
        columns, orientation=1). Supports NumPy arrays or Pandas Series objects.

        Args:
            name: Name of the variable to declare
            orientation: 0 if oriented to the decision units/rows, 1 if oriented to the choices/columns

        """
        self._scope[name] = VectorSymbol(self, name, orientation)

    def declare_table(self, name: str, orientation: int, mandatory_attributes: Set[str]=None,
                      allow_links=True):
        """
        Declares a table variable. Similar to vectors, tables can align with either the decision units (rows,
        orientation=0) or choices (columns, orientation=1), but allow for more complex attribute lookups. For ideal
        usage, all columns in the specified table should be valid Python variable names, as otherwise "dotted" access
        will not work in utility computation. LinkedDataFrames are fully supported (and even encouraged).

        Args:
            name:
            orientation:
            mandatory_attributes:
            allow_links:

        """
        self._scope[name] = TableSymbol(self, name, orientation, mandatory_attributes, allow_links)

    def declare_matrix(self, name: str, allow_transpose=True):
        """Declares a 2D variable which aligns with both the decision units (rows) and choices (columns). Of limited
        use."""
        self._scope[name] = MatrixSymbol(self, name, allow_transpose)

    def __getitem__(self, item) -> AbstractSymbol:
        """Gets a declared symbol to be assigned"""
        return self._scope[item]

    def clear_scope(self):
        self._scope.clear()

    @property
    def expressions(self) -> ExpressionGroup:
        return self._expressions

    @expressions.setter
    def expressions(self, item):
        for expr in item:
            self._expressions.append(expr)

    # endregion
    # region Run methods

    def validate_tree(self):
        if len(self._top_children) <= 1:
            raise ModelNotReadyError("At least two or more choices must be defined")

        for c in self._all_children():
            n_children = c.n_children
            if n_children == 1:
                raise ModelNotReadyError("In a nested model, all sub-choices must have 0, or 2 or more children")

    def validate_scope(self):
        for symbol_name in iter_chain(self._expressions.itersimple(), self._expressions.iterchained()):
            if symbol_name not in self._scope:
                raise KeyError(f"Symbol {symbol_name} not declared")

        # TODO: Check that symbols are also assigned

    def run_discrete(self, *, random_seed: int=None, n_draws: int=1,
                     astype: Union[str, np.dtype]='category', squeeze: bool=True, n_threads: int=1,
                     clear_scope: bool=True, precision: int=8, result_name: str=None, logger: Logger=None
                     ) -> Tuple[Union[DataFrame, Series], Series]:
        """
        For each decision unit, discretely sample one or more times (with replacement) from the probability
        distribution.

        Args:
            random_seed: The random seed for drawing uniform samples from the Monte Carlo.
            n_draws: The number of times to draw (with replacement) for each record. Must be >= 1. Run time is
                proportional to the number of draws.
            astype: The dtype of the return array; the result will be cast to the
                given dtype. The special value 'category' returns a Categorical Series (or a DataFrame for n_draws > 1).
                The special value 'index' returns the positional index in the sorted array of node names.
            squeeze: Only used when n_draws == 1. If True, then a Series will be returned, otherwise a DataFrame
                with one column will be returned.
            n_threads: The number of threads to uses in the computation. Must be >= 1
            clear_scope: If True and override_utilities not provided, data stored in the scope for
                utility computation will be released, freeing up memory. Turning this off is of limited use.
            precision: The number of bytes to store for each cell in the utility array; one of 1, 2, 4, or 8. More
                precision requires more memory.
            result_name: Name for the result Series or name of the columns of the result DataFrame. Purely aesthetic.

        Returns:
            Tuple[DataFrame or Series, Series]: The first item returned is always the results of the model evaluation,
                representing the choice(s) made by each decision unit. If n_draws > 1, the result is a DataFrame, with
                n_draws columns, otherwise a Series. The second item is the top-level logsum term from the logit model,
                for each decision unit. This is always a Series, as its value doesn't change with the number of draws.

        """
        self.validate_tree()
        self.validate_scope()
        if random_seed is None:
            random_seed = np.random.randint(1, 1000)

        assert n_draws >= 1

        # Utility computations
        utility_table = self._evaluate_utilities(precision=precision, n_threads=n_threads, logger=logger).values
        if clear_scope: self.clear_scope()

        # Compute probabilities and sample
        nb.config.NUMBA_NUM_THREADS = n_threads  # Set the number of threads for parallel execution
        nested = self.depth > 1
        if nested:
            hierarchy, levels, logsum_scales = self._flatten()
            raw_result, logsum = worker_nested_sample(utility_table, hierarchy, levels, logsum_scales, n_draws,
                                                      random_seed)
        else:
            raw_result, logsum = worker_multinomial_sample(utility_table, n_draws, random_seed)

        # Finalize results
        logsum = Series(logsum, index=self.decision_units)
        result = self._convert_result(raw_result, astype, squeeze, result_name)
        return result, logsum

    def _make_column_mask(self, filter_: str) -> Union[int, None]:
        if filter_ is None: return None
        col_index = self.choices

        column_depth = col_index.nlevels
        filter_parts = filter_.split('.')
        assert len(filter_parts) <= column_depth
        index_item = tuple(filter_parts + ['.'] * (column_depth - len(filter_parts)))
        return col_index.get_loc(index_item)  # Get the column number for the selected choice

    def _evaluate_utilities(self, precision=8, n_threads: int=None, logger: Logger=None) -> DataFrame:
        if self._decision_units is None:
            raise ModelNotReadyError("Decision units must be set before evaluating utility expressions")
        if n_threads is None:
            n_threads = cpu_count()
        if logger is not None: logger.debug("Allocating utility table")
        row_index = self._decision_units
        col_index = self.choices
        r, c = len(row_index), len(col_index)

        dtype_str = "f%s" % precision
        utilities = np.zeros([r, c], dtype=dtype_str)

        if logger is not None: logger.debug("Building shared locals")
        # Prepare locals, including scalar, vector, and matrix variables that don't need any further processing.
        shared_locals = {NAN_STR: np.nan, OUT_STR: utilities}
        for name in self._expressions.itersimple():
            symbol = self._scope[name]
            shared_locals[name] = symbol._get()

        ne.set_num_threads(n_threads)

        for expr in self._expressions:
            if logger is not None: logger.debug(f"Evaluating expression '{expr.raw}'")
            # TODO: Add error handling
            # TODO: Add support for watching particular rows and logging the results

            choice_mask = self._make_column_mask(expr.filter_)

            local_dict = shared_locals.copy()  # Make a shallow copy of the shared symbols

            # Add in any dict literals, expanding them to cover all choices
            for substitution, series in expr.dict_literals.items():
                local_dict[substitution] = series.reindex(col_index, fill_value=0)

            # Evaluate any chains on-the-fly
            for symbol_name, usages in expr.chains.items():
                symbol = self._scope[symbol_name]
                for substitution, chain_info in usages.items():
                    data = symbol._get(chain_info=chain_info)
                    local_dict[substitution] = data

            self._kernel_eval(expr.transformed, local_dict, utilities, choice_mask)

        return DataFrame(utilities, index=row_index, columns=col_index)

    @staticmethod
    def _kernel_eval(transformed_expr: str, local_dict: Dict[str, np.ndarray], out: np.ndarray, column_index):
        if column_index is not None:
            for key, val in local_dict.items():
                if hasattr(val, 'shape') and val.shape[1] > 1:
                    local_dict[key] = val[:, column_index]
            out = out[:, column_index]

        expr_to_run = f"{OUT_STR} + ({transformed_expr})"
        ne.evaluate(expr_to_run, local_dict=local_dict, out=out)

    def _convert_result(self, raw_result: ndarray, astype, squeeze: bool, result_name: str) -> Union[Series, DataFrame]:
        n_draws = raw_result.shape[1]
        column_index = pd.RangeIndex(n_draws, name=result_name)
        record_index = self.decision_units

        if astype == 'index':
            if squeeze and n_draws == 1:
                return pd.Series(raw_result[:, 0], index=record_index, name=result_name)
            return pd.DataFrame(raw_result, index=record_index, columns=column_index)
        elif astype == 'category':
            lookup_table = pd.Categorical(self.choices)
        else:
            lookup_table = self.choices.astype(astype)

        retval = []
        for col in range(n_draws):
            indices = raw_result[:, col]
            retval.append(Series(lookup_table.take(indices), index=record_index))
        retval = pd.concat(retval, axis=1)
        retval.columns = column_index

        if n_draws == 1 and squeeze:
            retval = retval.iloc[:, 0]
            retval.name = result_name
        return retval

    def run_stochastic(self, n_threads: int=1, clear_scope: bool=True, precision: int=8, logger: Logger=None
                       ) -> Tuple[DataFrame, Series]:
        """
        For each record, compute the probability distribution of the logit model. A DataFrame will be returned whose
        columns match the sorted list of node names (alternatives) in the model. Probabilities over all alternatives for
        each record will sum to 1.0.

        Args:
            n_threads: The number of threads to be used in the computation. Must be >= 1.
            clear_scope: If True and override_utilities not provided, data stored in the scope for
                utility computation will be released, freeing up memory. Turning this off is of limited use.
            precision: The number of bytes to store for each cell in the utility array; one of 1, 2, 4, or 8. More
                precision requires more memory.

        Returns:
            Tuple[DataFrame, Series]: The first item returned is always the results of the model evaluation,
                representing the probabilities of each decision unit picking each choice. The columns of the result
                table represent choices in this model; if this is a multinomial logit model then this will be a simple
                string index. Nested logit models, however, will have a MultiIndex columns, with a number of levels
                equal to the max depth of nesting. The second item is the top-level logsum term from the logit model,
                for each decision unit.
        """
        self.validate_tree()
        self.validate_scope()

        # Utility computations
        utility_table = self._evaluate_utilities(precision=precision, n_threads=n_threads, logger=logger).values
        if clear_scope: self.clear_scope()

        # Compute probabilities
        nb.config.NUMBA_NUM_THREADS = n_threads  # Set the number of threads for parallel execution
        nested = self.depth > 1
        if nested:
            hierarchy, levels, logsum_scales = self._flatten()
            raw_result, logsum = worker_nested_probabilities(utility_table, hierarchy, levels, logsum_scales)
        else:
            raw_result, logsum = worker_multinomial_probabilities(utility_table)

        result_frame = DataFrame(raw_result, index=self.decision_units, columns=self.choices)
        logsum = Series(logsum, index=self.decision_units)

        return result_frame, logsum

    # endregion