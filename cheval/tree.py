from typing import Dict, Iterator, Tuple
import pandas as pd
import numpy as np
from collections import deque


class _ChoiceNode(object):

    def __init__(self, name: str, root: 'ChoiceTree', parent: '_ChoiceNode'=None, logsum_scale: float=1.0,
                 level: int=0):
        assert name != ".", 'Choice node name cannot "."'
        assert 0.0 < logsum_scale <= 1.0, "Logsum scale must be in hte interval (0, 1], got %s" % logsum_scale

        self._name: str = str(name)
        self._parent = parent
        self._root = root
        self._logsum_scale = None
        self.logsum_scale = logsum_scale
        self._level = level
        self._children: dict[str, '_ChoiceNode'] = {}

    def __str__(self): return self.name

    def __repr__(self): return f"ChoiceNode({self.name})"

    @property
    def logsum_scale(self) -> float: return self._logsum_scale

    @logsum_scale.setter
    def logsum_scale(self, value):
        assert 0.0 < value <= 1.0, "Logsum scale must be in hte interval (0, 1], got %s" % value
        self._logsum_scale = float(value)

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def level(self):
        return self._level

    @property
    def is_parent(self):
        return len(self._children) > 0

    def children(self):
        yield from self._children.values()

    def max_level(self):
        max_level = self._level

        for c in self.children():
            max_level = max(max_level, c.max_level())

        return max_level

    def _nested_id(self, max_level: int):
        retval = ['.'] * max_level
        if self._parent is None:
            retval[0] = self._name
        else:
            cutoff = self._level + 1
            retval[: cutoff] = self._parent._nested_id(max_level)[: cutoff]
            retval[cutoff - 2] = self.name
        return tuple(retval)

    def nested_ids(self, max_level: int):
        retval = [self._nested_id(max_level)]
        for c in self._children.values():
            retval += c.nested_ids(max_level)
        return retval

    def add(self, name: str, logsum_scale: float=1.0) -> '_ChoiceNode':
        node = _ChoiceNode(name, self._root, self, logsum_scale, self.level + 1)
        self._children[name] = node
        return node

    def clear(self):
        for c in self._children.values(): c.clear()
        self._children.clear()


class ChoiceTree(object):

    def __init__(self):
        self._max_level = 0
        self._top_children: Dict[str, _ChoiceNode] = {}

    def add(self, name: str, logsum_scale: float=1.0) -> _ChoiceNode:
        node = _ChoiceNode(name, self, None, logsum_scale, 1)
        self._top_children[name] = node
        return node

    def depth(self) -> int:
        return max(c.max_level() for c in self._top_children.values())

    def _nested_tuples(self, max_level):
        node_ids = []
        for c in self._top_children.values():
            node_ids += c.nested_ids(max_level)

        return node_ids

    def node_index(self) -> pd.Index:
        assert self._top_children, "No nodes defined"
        max_level = self.depth()

        if max_level == 1:
            return pd.Index(sorted(self._top_children.keys()))
        else:
            node_ids = self._nested_tuples(max_level)

            level_names = ['root']
            for i in range(1, max_level): level_names.append(f'nest_{i + 1}')

            return pd.MultiIndex.from_tuples(node_ids, names=level_names)

    def clear(self):
        for c in self._top_children.values(): c.clear()
        self._top_children.clear()

    def all_children(self) -> Iterator[_ChoiceNode]:
        q = deque()
        for c in self._top_children.values(): q.append(c)
        while len(q) > 0:
            c = q.popleft()
            yield c
            for c2 in c.children(): q.append(c2)

    def flatten(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_level = self.depth()
        assert max_level > 1
        node_ids = self._nested_tuples(max_level)
        node_positions = {name: i for i, name in enumerate(node_ids)}

        hierarchy = np.full(len(node_ids), -1, dtype='i8')
        levels = np.zeros(len(node_ids), dtype='i8')
        logsum_scales = np.ones(len(node_ids), dtype='f8')

        for node in self.all_children():
            position = node_positions[node._nested_id(max_level)]
            levels[position] = node.level - 1  # Internal levels start at 1.

            if node.parent is not None:
                parent_position = node_positions[node.parent._nested_id(max_level)]
                hierarchy[position] = parent_position

            if node.is_parent:
                logsum_scales[position] = node.logsum_scale

        return hierarchy, levels, logsum_scales

