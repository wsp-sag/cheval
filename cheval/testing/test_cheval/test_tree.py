import unittest
import pandas as pd
import numpy as np

from ...tree import ChoiceTree


class TestTree(unittest.TestCase):

    def test_flat_index(self):
        choices = sorted(list("abcdefg"))
        assert len(choices) > 1

        expected_result = pd.Index(choices)

        tree = ChoiceTree()
        for choice in choices: tree.add(choice)
        test_result = tree.node_index()

        assert expected_result.equals(test_result)

    def _build_nested_tree(self):
        tree = ChoiceTree()
        tree.add('a')
        b = tree.add('b', logsum_scale=0.3)

        b.add('c')
        d = b.add('d', logsum_scale=0.8)

        d.add('e')
        d.add('f')

        return tree

    def test_nested_index(self):
        tree = self._build_nested_tree()
        test_result = tree.node_index()

        # Build the expected result
        tuples = [tuple(s) for s in [
            "a..", "b..", "bc.", "bd.", "bde", "bdf"
        ]]
        expected_result = pd. MultiIndex.from_tuples(tuples)

        assert expected_result.equals(test_result)

    def test_flatten(self):
        tree = self._build_nested_tree()
        test_hierarchy, test_levels, test_scales = tree.flatten()

        expected_hierarchy = np.int64([-1, -1, 1, 1, 3, 3])
        expected_levels = np.int64([0, 0, 1, 1, 2, 2])
        expected_scales = np.float64([1, 0.3, 1, 0.8, 1, 1])

        assert np.all(expected_hierarchy == test_hierarchy)
        assert np.all(expected_levels == test_levels)
        assert np.allclose(expected_scales, test_scales)
