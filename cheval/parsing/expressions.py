from typing import Dict, Set, Tuple, Optional, List
import ast

import attr
import pandas as pd
import astor

from .expr_items import ChainedSymbol
from .ast_transformer import ExpressionParser
from .exceptions import UnsupportedSyntaxError


def _split_filter(e: str) -> Tuple[str, Optional[str]]:
    parts = e.split('@')
    if len(parts) == 1: return e, None

    if len(parts) == 2: return parts[0].strip(), parts[1].strip()

    raise UnsupportedSyntaxError("Only one '@' symbol is allowed in expressions")


@attr.s
class Expression(object):
    """Simple data class for utility expressions"""

    raw: str = attr.ib()
    transformed: str = attr.ib()
    chains: Dict[str, ChainedSymbol] = attr.ib()
    dict_literals: Dict[str, pd.Series] = attr.ib()
    filter_: Optional[str] = attr.ib()
    symbols: Set[str] = attr.ib()

    @staticmethod
    def parse(e: str, prior_simple: Set[str] = None, prior_chained: Set[str] = None, mode='cheval') -> 'Expression':
        split_e, filter_ = _split_filter(e)

        tree = ast.parse(split_e, mode='eval').body
        transformer = ExpressionParser(prior_simple, prior_chained, mode=mode)
        new_tree = transformer.visit(tree)

        new_e = Expression(e, astor.to_source(new_tree), transformer.chained_symbols, transformer.dict_literals,
                           filter_, transformer.simple_symbols)
        return new_e

    @property
    def all_symbols(self) -> Set[str]:
        return self.symbols | set(self.chains.keys())
