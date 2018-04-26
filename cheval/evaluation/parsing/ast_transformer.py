from typing import Dict, Tuple, Set

import ast
from collections import deque

import astor
import numpy as np
import pandas as pd
from numexpr import expressions as nee
import six

from .exceptions import UnsupportedSyntaxError
from .expr_items import ChainedSymbol

NAN_STR = '__NAN'  # This needs to be recognized by other modules

# Only nodes used in expressions are included, due to the limited parsing
_UNSUPPORTED_NODES: list = [
    ast.Load, ast.Store, ast.Del, ast.IfExp, ast.Subscript, ast.ListComp, ast.DictComp
]
if six.PY3:
    _UNSUPPORTED_NODES.append(ast.Starred)
_UNSUPPORTED_NODES: Tuple[type] = tuple(_UNSUPPORTED_NODES)
_NAN_REPRESENTATIONS = {'none', 'nan'}
_NUMEXPR_FUNCTIONS = set(nee.functions.keys())
_SUPPORTED_AGGREGATIONS = {
    'count', 'first', 'last', 'max', 'min', 'mean', 'median', 'prod', 'std', 'sum', 'var'
}


class ExpressionParser(ast.NodeTransformer):

    def __init__(self, prior_simple: Set[str]=None, prior_chained: Dict[str, ChainedSymbol]=None):
        self.dict_literals: Dict[str, pd.Series] = {}

        # Optionally, use an ongoing collection of simple and chained symbols to enforce consistent usage
        # across a group of expressions
        self.simple_symbols: Set[str] = set() if prior_simple is None else prior_simple
        self.chained_symbols: Dict[str, ChainedSymbol] = {} if prior_chained is None else prior_chained

    def visit(self, node):
        return self.__get_visitor(node)(node)

    def __get_visitor(self, node):
        if isinstance(node, _UNSUPPORTED_NODES):
            raise UnsupportedSyntaxError(node.__class__.__name__)
        name = "visit_" + node.__class__.__name__.lower()
        return getattr(self, name) if hasattr(self, name) else self.generic_visit

    # region Required transformations for NumExpr

    def visit_str(self, node):
        # Converts text-strings to NumExpr-supported byte-strings
        if six.PY3:
            return ast.Bytes(node.s.encode())
        return node

    def visit_unaryop(self, node):
        # Converts 'not' into '~' which NumExpr supports
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return node
        raise NotImplementedError(type(node.op))

    def visit_boolop(self, node):
        # Converts 'and' and 'or' into '&' and '|' which NumExpr supports
        # BoolOp objects have a list of values but need to be converted into a tree of BinOps

        if isinstance(node.op, ast.And):
            new_op = ast.BitAnd()
        elif isinstance(node.op, ast.Or):
            new_op = ast.BitOr()
        else:
            raise NotImplementedError(type(node.op))

        values = node.values
        left = self.visit(values[0])
        i = 1
        while i < len(values):
            right = self.visit(values[i])
            left = ast.BinOp(left=left, right=right, op=new_op)
            i += 1
        return left

    def visit_call(self, node):
        func_node = node.func

        if isinstance(func_node, ast.Name):
            # Top-level function
            return self.__visit_toplevel_func(node, func_node)
        elif isinstance(func_node, ast.Attribute):
            # Method of an object
            return self.__visit_method(node, func_node)
        else:
            return self.generic_visit(node)

    def __visit_toplevel_func(self, node, func_node):
        func_name = func_node.id
        if func_name not in _NUMEXPR_FUNCTIONS:
            raise UnsupportedSyntaxError("Function '%s' not supported." % func_name)

        node.args = [self.__get_visitor(arg)(arg) for arg in node.args]
        node.starargs = None
        if not hasattr(node, 'kwargs'):
            node.kwargs = None

        return node

    # endregion

    # region Dict literals

    @staticmethod
    def __get_dict_key(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Str):
            return node.s
        raise UnsupportedSyntaxError("Dict key of type '%s' unsupported" % node)

    def visit_dict(self, node):
        substitution = '__dict%s' % len(self.dict_literals)
        new_node = ast.Name(substitution, ast.Load())

        try:
            values = []
            for val in node.values:
                if isinstance(val, ast.UnaryOp):
                    assert isinstance(val.operand, ast.Num)
                    assert isinstance(val.op, ast.USub)
                    values.append(np.float32(-val.operand.n))
                elif isinstance(val, ast.Num):
                    values.append(np.float32(val.n))
                else:
                    raise ValueError()

            keys = [self.__get_dict_key(key) for key in node.keys]
        except (ValueError, AssertionError):
            raise UnsupportedSyntaxError("Dict literals are supported for numeric values only")

        s = pd.Series(values, index=keys)
        self.dict_literals[substitution] = s

        return new_node

    # endregion

    # region Simple symbols

    def visit_name(self, node):
        symbol_name = node.id

        if symbol_name.lower() in _NAN_REPRESENTATIONS:
            # Allow None or NaN or nan to mean 'null'
            node.id = NAN_STR
        elif symbol_name in self.chained_symbols:
            raise UnsupportedSyntaxError("Inconsistent use for symbol '%s'" % symbol_name)
        else:
            self.simple_symbols.add(symbol_name)
        return node

    # endregion

    # region Chained symbols

    def visit_attribute(self, node):
        name, chain = self.__get_name_from_attribute(node)

        if name in self.simple_symbols:
            raise UnsupportedSyntaxError("Inconsistent usage of symbol '%s'" % name)

        if name in self.chained_symbols:
            container = self.chained_symbols[name]
        else:
            container = ChainedSymbol(name)
            self.chained_symbols[name] = container
        substitution = container.add_chain(chain)

        return ast.Name(substitution, ast.Load())

    @staticmethod
    def __get_name_from_attribute(node):
        current_node = node
        stack = deque()
        while not isinstance(current_node, ast.Name):
            if not isinstance(current_node, ast.Attribute):
                raise UnsupportedSyntaxError()
            stack.append(current_node.attr)
            current_node = current_node.value

        return current_node.id, stack

    def __visit_method(self, call_node, func_node):
        name, chain = self.__get_name_from_attribute(func_node)
        func_name = chain.popleft()

        if func_name not in _SUPPORTED_AGGREGATIONS:
            raise UnsupportedSyntaxError("Aggregation method '%s' is not supported." % func_name)

        if not hasattr(call_node, 'starargs'): call_node.starargs = None
        if not hasattr(call_node, 'kwargs'): call_node.kwargs = None

        if len(call_node.keywords) > 0:
            raise UnsupportedSyntaxError("Keyword args are not supported inside aggregations")
        if call_node.starargs is not None or call_node.kwargs is not None:
            raise UnsupportedSyntaxError("Star-args or star-kwargs are not supported inside aggregations")
        arg_expression = astor.to_source(call_node.args[0])

        if name in self.simple_symbols:
            raise UnsupportedSyntaxError("Inconsistent usage of symbol '%s'" % name)

        if name in self.chained_symbols:
            container = self.chained_symbols[name]
        else:
            container = ChainedSymbol(name)
            self.chained_symbols[name] = container

        substitution = container.add_chain(chain, func_name, arg_expression)

        new_node = ast.Name(substitution, ast.Load())
        return new_node

    # endregion
