from typing import Iterable, Dict, Union, Tuple, Set, Hashable
from itertools import chain as iter_chain
from multiprocessing import cpu_count
from logging import Logger

from pandas import Series, DataFrame, Index, MultiIndex
import pandas as pd
from numpy import ndarray
import numpy as np
import numexpr as ne
import numba as nb

from .api import AbstractSymbol, ExpressionGroup, ChoiceNode, NumberSymbol, VectorSymbol, TableSymbol, MatrixSymbol, \
    ExpressionSubGroup
from .exceptions import ModelNotReadyError
from .core import (worker_nested_probabilities, worker_nested_sample, worker_multinomial_probabilities,
                   worker_multinomial_sample, fast_indexed_add, UtilityBoundsError)
from .parsing.constants import *


class ChoiceModel(object):

    def __init__(self, *, precision: int = 8, debug_id=None):

        # Tree data
        self._max_level: int = 0
        self._all_nodes: Dict[str, ChoiceNode] = {}
        self._top_nodes: Set[ChoiceNode] = set()

        # Scope and expressions
        self._expressions: ExpressionGroup = ExpressionGroup()
        self._scope: Dict[str, AbstractSymbol] = {}

        # Index objects
        self._decision_units: Index = None

        # Cached items
        self._cached_cols: Index = None
        self._cached_utils: DataFrame = None

        # Other
        self._precision: int = 0
        self.precision = precision
        
        self.debug_id = debug_id  # note that debug_id needs to be a valid label that can used to search a Pandas index
        self.debug_results: DataFrame = None

    @property
    def precision(self) -> int:
        """The number of bytes used to store floating-point utilities. Can only be 4 or 8"""
        return self._precision

    @precision.setter
    def precision(self, i: int):
        assert i in {4, 8}, f"Only precision values of 4 or 8 are allowed (got {i})"
        self._precision = i

    @property
    def _partial_utilities(self) -> DataFrame:
        self.validate(expressions=False, assignment=False)
        if self._cached_utils is None:
            dtype = np.dtype(f"f{self._precision}")
            matrix = np.zeros(shape=[len(self.decision_units), len(self.choices)], dtype=dtype)
            table = DataFrame(matrix, index=self.decision_units, columns=self.choices)
            self._cached_utils = table
        else: table = self._cached_utils
        return table

    # region Tree operations

    def _create_node(self, name: str, logsum_scale: float, parent: ChoiceNode = None) -> ChoiceNode:
        expected_namespace = name
        if parent is None and name in self._all_nodes:
            old_node = self._all_nodes.pop(name)  # Remove from model dictionary
            self._top_nodes.remove(old_node)  # Remove from top-level choices
        elif parent is not None:
            expected_namespace = parent.full_name + '.' + name
            if expected_namespace in self._all_nodes:
                del self._all_nodes[expected_namespace]  # Remove from model dictionary

        level = 1 if parent is None else (parent.level + 1)
        node = ChoiceNode(self, name, parent=parent, logsum_scale=logsum_scale, level=level)
        self._all_nodes[expected_namespace] = node

        return node

    def add_choice(self, name: str, logsum_scale: float = 1.0) -> ChoiceNode:
        """
        Create and add a new discrete choice to the model, at the top level. Returns a node object which can also add
        nested choices, and so on. Choice names must only be unique within a given nest, although for clarity it is
        recommended that choice names are unique across all nests (especially when sampling afterwards).

        The model preserves the order of insertion of choices.

        Args:
            name: The name of the choice to be added. The name will also appear in the returned Series or DataFrame when
                the model is run in discrete mode.
            logsum_scale: The "theta" parameter, commonly referred to as the logsum scale. Must be in the interval
                (0, 1.0].

        Returns:
            ChoiceNode: The added choice node, which also has an "add_choice" method for constructing nested models.

        """
        if self._cached_cols is not None: self._cached_cols = None
        if self._cached_utils is not None: self._cached_utils = None

        node = self._create_node(name, logsum_scale)
        self._top_nodes.add(node)
        return node

    def add_choices(self, names: Iterable[str], logsum_scales: Iterable[float] = None
                    ) -> Dict[str, ChoiceNode]:
        """
        Convenience function for batch-adding several choices at once (for a multinomial logit model). See add_choice()
        for more details

        The model preserves the order of insertion of choices.

        Args:
            names: Iterable of string names of choices.
            logsum_scales: Iterable of logsum scale parameters (see add_choice). Must be the same length as `names`, if
                provided

        Returns:
            dict: Mapping of name: ChoiceNode for the added nodes

        """
        if self._cached_cols is not None: self._cached_cols = None
        if self._cached_utils is not None: self._cached_utils = None

        if logsum_scales is None:
            logsum_scales = [1.0 for _ in names]
        retval = {}
        for name, logsum_scale in zip(names, logsum_scales):
            node = self._create_node(name, logsum_scale)
            retval[name] = node
            self._top_nodes.add(node)
        return retval

    @property
    def choices(self) -> Index:
        """Pandas Index representing the choices in the model"""
        if self._cached_cols is not None: return self._cached_cols

        self.validate(decision_units=False, expressions=False, assignment=False)
        max_level = self.depth

        if max_level == 1:
            return Index(self._all_nodes.keys())
        else:
            nested_tuples = [node.nested_id(max_level) for node in self._all_nodes.values()]

            level_names = ['root']
            for i in range(1, max_level): level_names.append(f'nest_{i + 1}')

            return MultiIndex.from_tuples(nested_tuples, names=level_names)

    @property
    def elemental_choices(self) -> Index:
        """For a nested model, return the Index of 'elemental' choices without children that are available to be
        chosen."""
        max_level = self.depth

        if max_level == 1: return self.choices

        elemental_tuples = []
        for node in self._all_nodes.values():
            if node.is_parent: continue
            elemental_tuples.append(node.nested_id(max_level))

        return MultiIndex.from_tuples(elemental_tuples)

    @property
    def depth(self) -> int:
        """The maximum number of levels in a nested logit model. By definition, multinomial models have a depth of 1"""
        return max(node.level for node in self._all_nodes.values())

    def _flatten(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Converts nested structure to arrays for Numba-based processing"""
        max_level = self.depth
        assert max_level > 1
        n_nodes = len(self._all_nodes)

        hierarchy = np.full(n_nodes, -1, dtype='i8')
        levels = np.zeros(n_nodes, dtype='i8')
        logsum_scales = np.ones(n_nodes, dtype='f8')
        bottom_flags = np.full(n_nodes, True, dtype='?')

        node_positions = {node.full_name: i for i, node in enumerate(self._all_nodes.values())}

        for node in self._all_nodes.values():
            position = node_positions[node.full_name]
            levels[position] = node.level - 1  # Internal levels start at 1.

            if node.parent is not None:
                parent_position = node_positions[node.parent.full_name]
                hierarchy[position] = parent_position

            if node.is_parent:
                logsum_scales[position] = node.logsum_scale
                bottom_flags[position] = False

        return hierarchy, levels, logsum_scales, bottom_flags

    # endregion
    # region Expressions and scope operations

    @property
    def decision_units(self) -> Index:
        """
        The units or agents or OD pairs over which choices are to be evaluated. MUST BE SET before symbols can be
        assigned, or utilities calculated; otherwise ModelNotReadyError will be raised
        """
        if self._decision_units is None: raise ModelNotReadyError("No decision units defined")
        return self._decision_units

    @decision_units.setter
    def decision_units(self, item):
        """
        The units or agents or OD pairs over which choices are to be evaluated. MUST BE SET before symbols can be
        assigned, or utilities calculated; otherwise ModelNotReadyError will be raised.
        """

        # If there are any assigned symbols, clear them so as not to conflict with the new decision units
        for symbol in self._scope.values():
            if isinstance(symbol, NumberSymbol): continue  # Don't empty symbols that don't depend on the DU.
            symbol.empty()

        if isinstance(item, Index):
            self._decision_units = item
        else:
            self._decision_units = Index(item)

    @staticmethod
    def _check_symbol_name(name: str):
        # TODO: Check function names from NumExpr
        if name in RESERVED_WORDS:
            raise SyntaxError(f"Symbol name '{name}' cannot be used as it is a reserved keyword.")

    def declare_number(self, name: str):
        """Declares a simple scalar variable, of number, boolean, or text type"""
        self._check_symbol_name(name)
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
        self._check_symbol_name(name)
        self._scope[name] = VectorSymbol(self, name, orientation)

    def declare_table(self, name: str, orientation: int, mandatory_attributes: Set[str] = None,
                      allow_links=True):
        """
        Declares a table variable. Similar to vectors, tables can align with either the decision units (rows,
        orientation=0) or choices (columns, orientation=1), but allow for more complex attribute lookups. For ideal
        usage, all columns in the specified table should be valid Python variable names, as otherwise "dotted" access
        will not work in utility computation. LinkedDataFrames are fully supported (and even encouraged).

        Args:
            name: Name of the variable to declare
            orientation: 0 if oriented to the decision units/rows, 1 if oriented to the choices/columns
            mandatory_attributes:
            allow_links:

        """
        self._check_symbol_name(name)
        self._scope[name] = TableSymbol(self, name, orientation, mandatory_attributes, allow_links)

    def declare_matrix(self, name: str, orientation: int = 0, reindex_cols=True, reindex_rows=True):
        """
        Declares a matrix that fully or partially aligns with the rows or columns. This is useful when manual control
        is needed over both the decision units and the choices. Only DataFrames are supported.

        Args:
            name: Name of the variable to declare
            orientation: 0 if the index/columns are oriented to the decision units/choices, 1 if oriented to the
                choices/decision units.
            reindex_cols: If True, allows the model to expand the assigned matrix over the decision units, filling any
                missing values with 0
            reindex_rows: If True, allows the model to expand the assigned matrix over the choices, filling any
                missing values with 0

        """
        self._check_symbol_name(name)
        self._scope[name] = MatrixSymbol(self, name, orientation, reindex_cols, reindex_rows)

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

    def validate(self, *, tree=True, decision_units=True, expressions=True, assignment=True, group=None):
        """
        Checks that the model components are self-consistent and that the model is ready to run. Optionally, some
        components can be skipped, in order to partially validate a model under construction.

        Also gets called internally by the model at various stages

        Args:
            tree: Checks that all nested nodes have two or more children.
            decision_units: Checks that the decision units have been assigned.
            expressions: Checks that expressions use declared symbols.
            assignment: Checks that used and declared symbols have been assigned
            group: If not ``None``,  checks the expressions for only the specified group. This also applies to the
                assignment check. Otherwise, all expressions and symbols will be checked.

        Raises:
            ModelNotReadyError: if any check fails.

        """

        def assert_valid(condition, message):
            if not condition: raise ModelNotReadyError(message)

        if tree:
            assert_valid(len(self._top_nodes) >= 2, "At least two or more choices must be defined")
            for c in self._all_nodes.values():
                n_children = c.n_children
                assert_valid(n_children != 1, f"Nested choice '{c.full_name}' cannot have exactly one child node")

        if decision_units:
            assert_valid(self.decision_units is not None, "Decision units must be defined.")

        if not expressions and not assignment: return

        expr_container = self._expressions if group is None else self._expressions.get_group(group)
        symbols_to_check = list(expr_container.itersimple()) + list(expr_container.iterchained())

        for name in symbols_to_check:
            if name in RESERVED_WORDS: continue  # These gets added in manually later.
            assert_valid(name in self._scope, f"Symbol '{name}' used in expressions but has not been declared")
            if assignment: assert_valid(self._scope[name].filled, f"Symbol '{name}' is declared but never assigned")

    def run_discrete(self, *, random_seed: int = None, n_draws: int = 1,
                     astype: Union[str, np.dtype] = 'category', squeeze: bool = True, n_threads: int = 1,
                     clear_scope: bool = True, result_name: str = None, logger: Logger = None, scale_utilities=True
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
            result_name: Name for the result Series or name of the columns of the result DataFrame. Purely aesthetic.
            logger: Optional Logger instance which reports expressions being evaluated
            scale_utilities: For a nested model, if True then lower-level utilities will be divided by the logsum scale
                of the parent nest. If False, no scaling is performed. This is entirely dependant on the reported form
                of estimated model parameters.

        Returns:
            Tuple[DataFrame or Series, Series]: The first item returned is always the results of the model evaluation,
                representing the choice(s) made by each decision unit. If n_draws > 1, the result is a DataFrame, with
                n_draws columns, otherwise a Series. The second item is the top-level logsum term from the logit model,
                for each decision unit. This is always a Series, as its value doesn't change with the number of draws.

        """
        self.validate()
        if random_seed is None:
            random_seed = np.random.randint(1, 1000)

        assert n_draws >= 1

        # Utility computations
        utility_table = self._evaluate_utilities(self._expressions, n_threads=n_threads, logger=logger).values
        if clear_scope: self.clear_scope()

        # Compute probabilities and sample
        nb.config.NUMBA_NUM_THREADS = n_threads  # Set the number of threads for parallel execution
        nested = self.depth > 1
        if nested:
            hierarchy, levels, logsum_scales, bottom_flags = self._flatten()
            raw_result, logsum = worker_nested_sample(utility_table, hierarchy, levels, logsum_scales, bottom_flags,
                                                      n_draws, random_seed, scale_utilities=scale_utilities)
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

    def _evaluate_utilities(self, expressions: Union[ExpressionGroup, ExpressionSubGroup],
                            n_threads: int = None, logger: Logger = None, allow_casting=True) -> DataFrame:
        if self._decision_units is None:
            raise ModelNotReadyError("Decision units must be set before evaluating utility expressions")
        if n_threads is None:
            n_threads = cpu_count()
        row_index = self._decision_units
        col_index = self.choices

        # if debug, get index location of corresponding id
        if self.debug_id:
            debug_label = row_index.get_loc(self.debug_id)
            debug_expr = []
            debug_results = []

        utilities = self._partial_utilities.values

        # Prepare locals, including scalar, vector, and matrix variables that don't need any further processing.
        shared_locals = {NAN_STR: np.nan, OUT_STR: utilities, NEG_INF_STR: NEG_INF_VAL}
        for name in expressions.itersimple():
            if name in shared_locals: continue
            symbol = self._scope[name]
            shared_locals[name] = symbol._get()

        ne.set_num_threads(n_threads)
        casting_rule = 'same_kind' if allow_casting else 'safe'

        for expr in expressions:
            if logger is not None: logger.debug(f"Evaluating expression '{expr.raw}'")
            # TODO: Add error handling

            choice_mask = self._make_column_mask(expr.filter_)

            local_dict = shared_locals.copy()  # Make a shallow copy of the shared symbols

            # Add in any dict literals, expanding them to cover all choices
            expr._prepare_dict_literals(col_index, local_dict)

            # Evaluate any chains on-the-fly
            for symbol_name, usages in expr.chains.items():
                symbol = self._scope[symbol_name]
                for substitution, chain_info in usages.items():
                    data = symbol._get(chain_info=chain_info)
                    local_dict[substitution] = data

            self._kernel_eval(expr.transformed, local_dict, utilities, choice_mask, casting_rule=casting_rule)

            # save each expression and values for a specific od pair
            if self.debug_id:
                debug_expr.append(expr.raw)
                debug_results.append(utilities[debug_label].copy())

        nans = np.isnan(utilities)
        n_nans = nans.sum()
        if n_nans > 0:
            raise UtilityBoundsError(f"Found {n_nans} cells in utility table with NaN")

        if self.debug_id:
            self.debug_results = DataFrame(debug_results, index=debug_expr, columns=col_index)  # expressions.tolist() doesn't work...

        return DataFrame(utilities, index=row_index, columns=col_index)

    @staticmethod
    def _kernel_eval(transformed_expr: str, local_dict: Dict[str, np.ndarray], out: np.ndarray, column_index,
                     casting_rule='same_kind'):
        if column_index is not None:
            for key, val in local_dict.items():
                if hasattr(val, 'shape'):
                    if val.shape[1] > 1:
                        local_dict[key] = val[:, column_index]
                    elif val.shape[1] == 1:
                        local_dict[key] = val[:, 0]
            out = out[:, column_index]

        expr_to_run = f"{OUT_STR} + ({transformed_expr})"
        ne.evaluate(expr_to_run, local_dict=local_dict, out=out, casting=casting_rule)

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

    def run_stochastic(self, n_threads: int = 1, clear_scope: bool = True, logger: Logger = None,
                       group: str = None, scale_utilities=True) -> Tuple[DataFrame, Series]:
        """
        For each record, compute the probability distribution of the logit model. A DataFrame will be returned whose
        columns match the sorted list of node names (alternatives) in the model. Probabilities over all alternatives for
        each record will sum to 1.0.

        Args:
            n_threads: The number of threads to be used in the computation. Must be >= 1.
            clear_scope: If True data stored in the scope for utility computation will be released, freeing up memory.
                Turning this off is of limited use.
            logger: Optional Logger instance which reports expressions being evaluated
            group: Evaluate only the specified utility group. Raises KeyError if the group name is not defined.
            scale_utilities: For a nested model, if True then lower-level utilities will be divided by the logsum scale
                of the parent nest. If False, no scaling is performed. This is entirely dependant on the reported form
                of estimated model parameters.

        Returns:
            Tuple[DataFrame, Series]: The first item returned is always the results of the model evaluation,
                representing the probabilities of each decision unit picking each choice. The columns of the result
                table represent choices in this model; if this is a multinomial logit model then this will be a simple
                string index. Nested logit models, however, will have a MultiIndex columns, with a number of levels
                equal to the max depth of nesting. The second item is the top-level logsum term from the logit model,
                for each decision unit.
        """
        self.validate()

        # Utility computations
        expressions = self._expressions if group is None else self._expressions.get_group(group)
        utility_table = self._evaluate_utilities(expressions, n_threads=n_threads, logger=logger).values
        if clear_scope: self.clear_scope()

        # Compute probabilities
        nb.config.NUMBA_NUM_THREADS = n_threads  # Set the number of threads for parallel execution
        nested = self.depth > 1
        if nested:
            hierarchy, levels, logsum_scales, bottom_flags = self._flatten()
            raw_result, logsum = worker_nested_probabilities(utility_table, hierarchy, levels, logsum_scales,
                                                             bottom_flags, scale_utilities=scale_utilities)
            result_frame = self._build_nested_stochastic_frame(raw_result)
        else:
            raw_result, logsum = worker_multinomial_probabilities(utility_table)
            result_frame = DataFrame(raw_result, index=self.decision_units, columns=self.choices)
        logsum = Series(logsum, index=self.decision_units)

        return result_frame, logsum

    def _build_nested_stochastic_frame(self, raw_result: ndarray) -> DataFrame:
        elemental_index = self.elemental_choices
        choice_index = self.choices
        filter_array = choice_index.isin(elemental_index)

        return DataFrame(raw_result[:, filter_array].copy(), index=self.decision_units,
                         columns=choice_index[filter_array])

    # endregion

    # region Advanced functions

    def preval(self, group: Hashable, n_threads: int = None, logger: Logger = None,
               drop_group=True, cleanup_scope=True):
        """
        When using expression groups, "pre-evaluate" the utility expressions for a specified group, caching the utility
        table on the ChoiceModel.

        This is an advanced modelling technique to facilitate complex segmented stochastic models, where segments share
        decision units and some common utility expressions. Call preval() for the group of common expressions, and then
        copy() or copy_subset() the ChoiceModel to fill other symbols with segment-specific values.

        Discrete (micorsimulated) models don't need this because the decision units of each segment shouldn't overlap.
        So there's no downside to double-computing common expressions.

        Args:
            group: The name of the group to pre-compute.
            n_threads: Number of threads to use to evaluate the expressions
            logger: Optional logger for debugging
            drop_group: If True, the selected group will be "popped" from the set of groups, to avoid re-computing.
            cleanup_scope: If True, symbols unique to this group will be dropped from the scope. This can clean up
                memory by de-referencing objects and arrays that are no longer required.

        """
        self.validate(group=group)
        subgroup = self._expressions.get_group(group)
        utilities = self._evaluate_utilities(subgroup, n_threads=n_threads, logger=logger)
        self._cached_utils = utilities

        if cleanup_scope:
            # Need to get a set of only those symbols unique to this group, using set math
            all_symbols = set(iter_chain(self.expressions.itersimple(), self.expressions.itersimple()))
            ungrouped_symbols = set(
                iter_chain(self._expressions.iterchained(groups=False), self._expressions.itersimple(groups=False)))
            group_symbols = set(iter_chain(subgroup.iterchained(), subgroup.itersimple()))
            other_symbols = all_symbols - ungrouped_symbols - group_symbols

            symbols_to_clear = group_symbols - ungrouped_symbols - other_symbols

            for name in symbols_to_clear:
                del self._scope[name]

        if drop_group: self._expressions.drop_group(group)

    def copy(self, *, decision_units=True, scope_declared=True, scope_assigned=True, expressions=True, utilities=True
             ) -> 'ChoiceModel':
        """
        Makes a shallow or deep copy of this ChoiceModel. The tree of choices is always copied.

        Args:
            decision_units: Copy over the decision units, if already set.
            scope_declared: Copy over all declared symbols, but not their assigned values
            scope_assigned: Copy over all declared and their assigned values
            expressions: Copy over all expressions
            utilities: Copy over the cached utility table from preval(), if it exists.

        Returns: A copy of this ChoiceModel

        """

        new = ChoiceModel(precision=self.precision, debug_id=self.debug_id)
        new._max_level = self._max_level

        # ChoiceNode refs will be the same, but that's ok because users shouldn't be changing these at this point
        new._top_nodes = self._top_nodes.copy()
        new._all_nodes = self._all_nodes.copy()
        new._cached_cols = self._cached_cols

        # Force the DU to be copied if the assigned scope is also being copied
        if scope_assigned: decision_units = True

        if scope_declared or scope_assigned:
            for name, symbol in self._scope.items():
                new._scope[name] = symbol.copy(new, copy_data=scope_assigned, row_mask=None)

        if self._decision_units is not None and decision_units:
            new._decision_units = self._decision_units

        if expressions:
            # A new ExpressionGroup instance is important to allow copies to drop expressions and groups
            new._expressions = self._expressions.copy()

        if self._cached_utils is not None and utilities:
            # Make a deep copy of the partial utilities to avoid updating this instance's partial utils later when
            # using +=
            new._cached_utils = self._cached_utils.copy(deep=True)

        return new

    def copy_subset(self, mask: Series) -> 'ChoiceModel':
        """
        Similar to copy(), except applies a boolean filter Series to the decision units and stored data. This includes:
         - Decision units Index
         - Partial utilities computed through .preval()
         - Assigned vector, table, or matrix symbols (masked along the 0 axis)

        This should be a relatively fast operation.

        Args:
            mask: Series with a boolean dtype, whose index MUST match the current decision units.

        Returns:
            A copy of the current model, except with fewer decision units.

        """

        assert mask.index.equals(self.decision_units), "Mask Series must match decision units"
        subset_index = self.decision_units[mask]

        new = ChoiceModel(precision=self.precision, debug_id=self.debug_id)
        new._max_level = self._max_level

        # ChoiceNode refs will be the same, but that's ok because users shouldn't be changing these at this point
        new._top_nodes = self._top_nodes.copy()
        new._all_nodes = self._all_nodes.copy()

        new._cached_cols = self._cached_cols

        new.decision_units = subset_index

        for name, symbol in self._scope.items():
            new._scope[name] = symbol.copy(new, copy_data=True, row_mask=mask)

        new._expressions = self._expressions.copy()
        if self._cached_utils is not None:
            new._cached_utils = self._cached_utils.loc[mask].copy(deep=True)

        return new

    def add_partial_utilities(self, table: DataFrame, reindex_rows=False, reindex_columns=True):
        """
        Optimized function for adding partial utilities to the cached table from an external source. Faster than
        declaring a matrix symbol.

        Args:
            table: Partial utilities to be added. The index must align with the decision units, and the columns with the
                choices, or be a subset of them. The dtype MUST be np.float32 or np.float64
            reindex_rows: Allows expanding the table to cover all decision units, if its index is a subset. Missing
                values get filled with 0.
            reindex_columns: Allows expanding the table to cover all choices, if its index is a subset. Missing values
                get filled with 0.

        """
        self.validate(expressions=False, assignment=False)

        row_indexer = None
        if not self.decision_units.equals(table.index):
            if not reindex_rows:
                raise KeyError("Partial utility table index must match model decision units when reindex_rows=False")
            row_indexer = self.decision_units.get_indexer(table.index)

        col_indexer = None
        if not self.choices.equals(table.columns):
            if not reindex_columns:
                raise KeyError("Partial utility table columns must match model choices when reindex_columns=False")
            col_indexer = self.choices.get_indexer(table.columns)

        target_table = self._partial_utilities.values
        fast_indexed_add(target_table, table.values, row_indexer, col_indexer)

    # endregion
