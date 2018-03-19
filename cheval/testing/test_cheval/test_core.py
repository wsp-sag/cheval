import unittest
from bisect import bisect_right

import numpy as np
from numpy.testing import assert_allclose

from ...core import sample_once, sample_multi, logarithmic_search, multinomial_probabilities, nested_probabilities
from ...tree import ChoiceTree


class TestChevalCore(unittest.TestCase):

    def _get_utility_row(self):
        ret = np.array(
            [1.678, 1.689, 1.348, 0.903, 1.845, 0.877, 0.704, 0.482],
            dtype=np.float64
        )

        return ret

    def test_mulitnomial_logit(self):
        utilities = self._get_utility_row()

        expected_result = np.float64(
            [0.181775432, 0.183785999, 0.130682672, 0.083744629, 0.214813892, 0.08159533, 0.068632901, 0.054969145]
        )
        test_result = multinomial_probabilities(utilities)

        assert_allclose(test_result, expected_result)

    def _build_nested_tree(self):
        tree = ChoiceTree()
        a = tree.add('a', 0.3)
        b = tree.add('b', 0.6)

        a.add('c')
        a.add('d')

        b.add('e')
        f = b.add('f', 0.5)

        f.add('g')
        f.add('h')

        return tree.flatten()

    def test_nested_logit(self):
        utilities = self._get_utility_row()
        tree_info = self._build_nested_tree()
        test_result = nested_probabilities(utilities, tree_info)

        expected_result = np.float64([0, 0.06527, 0.17893, 0.47448, 0.23791, 0.04341, 0, 0])

        assert np.allclose(expected_result, test_result)

    def test_saple_once(self):
        pass

    def test_sample_multi(self):
        pass

    def test_logarithmic_search(self):
        pass

    def test_simple_sample(self):
        pass

    def test_simple_multisample(self):
        pass

    def test_multinomial_sample(self):
        pass

    def test_multinomial_multisample(self):
        pass

    def test_nested_sample(self):
        pass

    def test_nested_multisample(self):
        pass


