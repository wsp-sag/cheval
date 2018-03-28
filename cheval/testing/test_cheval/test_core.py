import unittest
from bisect import bisect_right

import numpy as np
from numpy.testing import assert_allclose

from ...core import (sample_once, sample_multi, logarithmic_search, multinomial_probabilities, nested_probabilities,
                     simple_probabilities, MIN_RANDOM_VALUE)
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
        test_result, test_ls = nested_probabilities(utilities, *tree_info)

        # Auto, Auto-carpool, auto-drive, transit, transit-bus, transit-train, train-bus, train-walk
        expected_result = np.float64([0, 0.085207, 0.355547, 0, 0.156274, 0, 0.354937, 0.048035])
        expected_ls = 1.597835

        assert_allclose(test_result, expected_result, rtol=0.00001)
        assert abs(expected_ls - test_ls) < 0.000001

    def test_sample_once(self):
        p = np.float64([0, 0, 0.25, 0.00, 0.25, 0.15, 0.35, 0.0])
        #              [0, 0, 0.25, 0.25, 0.50, 0.65, 1.00, 1.0]
        #              [0  1    2    3     4     5      6    7]

        expected_samples = [
            (MIN_RANDOM_VALUE, 2),  # 0
            (0.1, 2),               # 1
            (0.3, 4),               # 2
            (0.6, 5),               # 3
            (0.7, 6),               # 4
            (1.0, 6)                # 5
        ]

        for i, (random_draw, expected_result) in enumerate(expected_samples):
            test_result = sample_once.py_func(p, random_draw)
            assert test_result == expected_result, f"Test={i} Expected={expected_result}, Actual={test_result}"

    def test_sample_multi(self):
        p = np.float64([0, 0, .25, .25, 0, .5, 0])
        p2 = p.copy()  # Sample multi mutates the probability array after running, so make a copy now.
        seed = 12345
        n = 1000

        test_result = sample_multi.py_func(p, n, seed, None)

        np.random.seed(seed)
        expected_result = np.zeros(n, np.int64)
        for i in range(n):
            r = np.random.uniform(MIN_RANDOM_VALUE, 1, 1)[0]
            expected_result[i] = sample_once.py_func(p2, r)

        assert np.all(test_result == expected_result)

    def test_logarithmic_search(self):
        cumsums = np.array([0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float64)

        expected_samples = [
            (0.0, 2),
            (0.2, 2),
            (0.4, 7),
            (0.6, 8),
            (0.8, 9),
            (0.99, 9)
        ]

        for random_draw, expected_index in expected_samples:
            test_result = logarithmic_search(np.float64(random_draw), cumsums)
            assert test_result == expected_index

            standard_result = bisect_right(cumsums, random_draw)
            assert test_result == standard_result

    def test_simple_probabilities(self):
        weights = self._get_utility_row()

        expected_result = weights / weights.sum()

        test_result = simple_probabilities(weights)

        assert_allclose(test_result, expected_result)

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

    def test_multithreaded_stability(self):
        pass

