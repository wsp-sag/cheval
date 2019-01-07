"""Collection of common classes to store more complex substitutions (e.g. LinkedDataFrame lookup)"""
from collections import deque, namedtuple
from typing import Dict, Tuple

ChainTuple = namedtuple("ChainTuple", ['chain', 'func', 'args', 'withfunc'])


class ChainedSymbol(object):

    def __init__(self, name: str):
        self._name: str = name
        self._lookups: Dict[str, ChainTuple] = {}

    def add_chain(self, chain: deque, func: str=None, args: str=None, *, prepend_at=False) -> str:
        sub_name = self._make_sub(prepend_at=prepend_at)
        self._lookups[sub_name] = self._build_chain(chain, func, args)
        return sub_name

    @staticmethod
    def _build_chain(chain: deque, func: str=None, args: str=None) -> ChainTuple:
        if func is None or args is None:
            return ChainTuple(chain, None, None, False)
        return ChainTuple(chain, func, args, True)

    def _make_sub(self, *, prepend_at=False) -> str:
        s1 = "__sub_%s%s" % (self._name, len(self._lookups))
        if prepend_at: return '@' + s1
        return s1

    def items(self) -> Tuple[str, ChainTuple]:
        yield from self._lookups.items()
        # for item in iteritems(self._lookups):
        #     yield item
