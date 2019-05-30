"""Module for AST Transformer subclass which specially-parses utility expressions"""
from typing import Dict, Tuple, Set

import ast
from collections import deque

import astor
import numpy as np
import pandas as pd
import numexpr3 as ne

from .exceptions import UnsupportedSyntaxError
from .expr_items import ChainedSymbol, EvaluationMode
from .constants import *

# Only nodes used in expressions are included, due to the limited parsing
_UNSUPPORTED_NODES: Tuple[type] = (ast.Load, ast.Store, ast.Del, ast.IfExp, ast.Subscript, ast.ListComp, ast.DictComp,
                                   ast.Starred)
_NAN_REPRESENTATIONS = {'none', 'nan'}
_NUMEXPR_FUNCTIONS = {t[0] for t in ne.OPTABLE.keys() if isinstance(t[0], str) and not t[0].startswith('_')}
_NUMEXPR_FUNCTIONS -= {''}  # The OPTABLE has a few keys without a function
_SUPPORTED_AGGREGATIONS = {
    'count', 'first', 'last', 'max', 'min', 'mean', 'median', 'prod', 'std', 'sum', 'var'
}


class ExpressionParser(ast.NodeTransformer):

    def __init__(self, prior_simple: Set[str]=None, prior_chained: Set[str]=None, mode=EvaluationMode.UTILITIES):
        self.mode: EvaluationMode = mode

        self.dict_literals: Dict[str, pd.Series] = {}

        # Optionally, use an ongoing collection of simple and chained symbols to enforce consistent usage
        # across a group of expressions
        self.simple_symbols: Set[str] = set() if prior_simple is None else prior_simple
        self.all_chained_symbols = prior_chained if prior_chained is not None else set()
        self.chained_symbols: Dict[str, ChainedSymbol] = {}
        self.visited_simple: Set[str] = set()

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
        return ast.Bytes(node.s.encode())

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
            return node.id, 1
        if isinstance(node, ast.Str):
            return node.s, 1
        if isinstance(node, ast.Attribute):
            keylist = deque()
            while not isinstance(node, ast.Name):
                keylist.appendleft(node.attr)
                node = node.value
            keylist.appendleft(node.id)
            return tuple(keylist), len(keylist)
        raise UnsupportedSyntaxError("Dict key of type '%s' unsupported" % node)

    @staticmethod
    def __resolve_key_levels(keys: list, max_level: int):
        assert max_level >= 1
        resovled_keys = []
        for key in keys:
            # Convert to list to pad if needed
            converted = list(key) if isinstance(key, tuple) else [key]
            length = len(converted)

            if max_level == 1:
                if length != 1:
                    raise UnsupportedSyntaxError("Inconsistent usage of multi-item keys")
                resovled_keys.append(converted[0])  # Convert to singleton for consistency
            elif length <= max_level:
                # Applies to top-level
                for _ in range(max_level - length): converted.append('.')
                resovled_keys.append(tuple(converted))
            else:
                raise NotImplementedError("This should never happen. Length=%s Max length=%s" % (length, max_level))
        return resovled_keys

    def visit_dict(self, node):
        if not self.mode != EvaluationMode.UTILITIES:
            raise UnsupportedSyntaxError("Dict literals not allowed in this context")

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

            keys, max_level = [], 0
            for key_node in node.keys:
                key, level = self.__get_dict_key(key_node)
                max_level = max(max_level, level)
                keys.append(key)
            resolved = self.__resolve_key_levels(keys, max_level)
            if max_level == 1:
                index = pd.Index(resolved)
            else:
                index = pd.MultiIndex.from_tuples(resolved)

            s = pd.Series(values, index=index)
            self.dict_literals[substitution] = s

            return new_node

        except (ValueError, AssertionError):
            # Catch simple errors and emit them as syntax errors
            raise UnsupportedSyntaxError("Dict literals are supported for numeric values only")

    # endregion

    # region Simple symbols

    def visit_name(self, node):
        symbol_name = node.id

        if symbol_name.lower() in _NAN_REPRESENTATIONS:
            # Allow None or NaN or nan to mean 'null'
            node.id = NAN_STR
        elif symbol_name in self.all_chained_symbols:
            raise UnsupportedSyntaxError("Inconsistent use for symbol '%s'" % symbol_name)
        else:
            self.simple_symbols.add(symbol_name)
            self.visited_simple.add(symbol_name)
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
        self.all_chained_symbols.add(name)
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
        self.all_chained_symbols.add(name)
        substitution = container.add_chain(chain, func_name, arg_expression)

        new_node = ast.Name(substitution, ast.Load())
        return new_node

    # endregion
