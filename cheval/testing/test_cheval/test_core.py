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
        expected_ls = 29.4585222

        test_result, test_ls = multinomial_probabilities(utilities)

        assert_allclose(test_result, expected_result, rtol=0.000001)
        assert abs(expected_ls - test_ls) < 0.000001

    def _build_nested_tree(self):
        tree = ChoiceTree()

        auto = tree.add('auto', logsum_scale=0.7)
        auto.add('carpool')
        auto.add('drive')

        transit = tree.add('transit', logsum_scale=0.7)
        transit.add('bus')
        train = transit.add('train', logsum_scale=0.3)

        train.add('drive')
        train.add('walk')

        return tree.flatten()

    def test_nested_logit(self):
        utilities = np.float64([-0.001, -1.5, -0.5, -0.005, -1, -0.075, -0.3, -0.9])
        tree_info = self._build_nested_tree()
        test_result = nested_probabilities(utilities, tree_info)

        # Auto, Auto-carpool, auto-drive, transit, transit-bus, transit-train, train-bus, train-walk
        expected_result = np.float64([0, 0.0852, 0.3555, 0, 0.1563, 0, 0.3549, 0.0480])

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


