from typing import Dict, Set, Tuple, Optional
import ast

import attr
import pandas as pd
import numpy as np
import astor

from .expr_items import ChainedSymbol
from .ast_transformer import ExpressionParser
from .exceptions import UnsupportedSyntaxError


def _split_filter(e: str) -> Tuple[str, Optional[str]]:
    parts = e.split('@')
    if len(parts) == 1:
        return e, None

    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    raise UnsupportedSyntaxError("Only one '@' symbol is allowed in expressions")


@attr.s
class Expression(object):
    """Simple data class for utility expressions"""

    raw: str = attr.ib()
    transformed: str = attr.ib()
    chains: Dict[str, ChainedSymbol] = attr.ib()
    dict_literals: Dict[str, dict] = attr.ib()
    filter_: Optional[str] = attr.ib()
    symbols: Set[str] = attr.ib()

    @staticmethod
    def parse(e: str, prior_simple: Set[str] = None, prior_chained: Set[str] = None, mode='cheval') -> 'Expression':
        split_e, filter_ = _split_filter(e)

        tree = ast.parse(split_e, mode='eval').body
        transformer = ExpressionParser(prior_simple, prior_chained, mode=mode)
        new_tree = transformer.visit(tree)

        new_e = Expression(e, astor.to_source(new_tree), transformer.chained_symbols, transformer.dict_literals,
                           filter_, transformer.visited_simple)
        return new_e

    @property
    def all_symbols(self) -> Set[str]:
        return self.symbols | set(self.chains.keys())

    def _prepare_dict_literals(self, choice_index: pd.Index, local_dict: dict):
        # Prepares dict literals for use in evaluation, applying rules for special key names
        for substitution, raw_literal in self.dict_literals.items():
            new_array = np.zeros((1, len(choice_index)), dtype='f8')

            for key, val in raw_literal.items():
                self._insert_dict_val(key, choice_index, val, new_array)

            local_dict[substitution] = new_array

    @staticmethod
    def _insert_dict_val(key: tuple, choice_index: pd.Index, val, new_array):
        max_levels = choice_index.nlevels

        if max_levels == 1:
            # Dotted dict keys are not supported for multinomial models
            assert len(key) == 1

            loc = choice_index.get_loc(key[0])
            new_array[0, loc] = val
        else:
            '''
            Dotted dict literals require care to ensure explicit reindexing with nested logit models. For example,
            specifying the key "A.B" in a 3-level model (where "A.B.C" exists and is meaningful) there is need to
            disambiguate between the node with the unique ID of ('A', 'B', '.') (i.e. the parent node of "A.B.C"),
            versus applying a value to ALL children of that parent node. Pandas during reindexing will apply value
            to ALL nodes with the "A.B" pattern e.g. the parent and all of its children.

            To avoid this, some special naming conventions are enforced here (assuming max_level = 3):
                "A.B" refers to the parent node ONLY. The output new key is ('A', 'B', '.')
                "A.B._" refers to ALL children of parent node A.B. The output new key is ('A', 'B')

            The edge case "A._.B" (where an underscore occurs in the middle of a name) is technically meaningless,
            so the code below assumes that "A._" was meant instead
            '''
            new_key = []
            partial_key = False
            for sub_key in key:
                if sub_key == '_':
                    partial_key = True
                    break
                new_key.append(str(sub_key))

            delta = ['.'] * (max_levels - len(new_key))

            if partial_key:
                locs = choice_index.get_loc(tuple(new_key))
                parent_loc = choice_index.get_loc(tuple(new_key + delta))

                new_array[0, locs] = val
                new_array[0, parent_loc] = 0
            else:
                loc = choice_index.get_loc(tuple(new_key + delta))
                new_array[0, loc] = val
