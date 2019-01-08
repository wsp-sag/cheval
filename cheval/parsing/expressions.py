from typing import Dict, Set
import ast

import attr
import pandas as pd
import astor

from .expr_items import ChainedSymbol
from .ast_transformer import ExpressionParser


@attr.s
class Expression(object):
    """Simple data class for utility expressions"""

    raw: str = attr.ib()
    transformed: str = attr.ib()
    chains: Dict[str, ChainedSymbol] = attr.ib()
    dict_literals: Dict[str, pd.Series] = attr.ib()

    @staticmethod
    def parse(e: str, prior_simple: Set[str]=None, prior_chained: Set[str]=None) -> 'Expression':
        tree = ast.parse(e, mode='eval').body
        transformer = ExpressionParser(prior_simple, prior_chained)
        new_tree = transformer.visit(tree)

        new_e = Expression(e, astor.to_source(new_tree), transformer.chained_symbols, transformer.dict_literals)
        return new_e
