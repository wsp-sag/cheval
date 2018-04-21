import unittest
from bisect import bisect_right

import numpy as np
from numpy.testing import assert_allclose

from core import (
    MIN_RANDOM_VALUE, sample_once, sample_multi, logarithmic_search,
    simple_probabilities, simple_sample, simple_multisample, worker_weighted_sample,
    multinomial_probabilities, multinomial_sample, multinomial_multisample, worker_multinomial_sample,
        worker_multinomial_probabilities,
    nested_probabilities, nested_sample, nested_multisample, worker_nested_sample, worker_nested_probabilities
)
from tree import ChoiceTree


def _cp_midpoints(p_array):
    cps = np.cumsum(p_array)
    mask = np.where(p_array > 0)[0]

    mcps = cps[mask]

    out = np.full(len(mcps) + 1, fill_value=MIN_RANDOM_VALUE)
    out[1] = mcps[0] * 0.5
    out[2:] = (mcps[:-1] + mcps[1:]) * 0.5

    result_indices = [mask[0]] + list(mask)
    return out, result_indices


def _randomize(n, seed):
    randomizer = np.random.RandomState(seed)
    return randomizer.uniform(MIN_RANDOM_VALUE, 1.0, n)


class TestSharedCore(unittest.TestCase):

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


class TestWeighredSampling(unittest.TestCase):

    def test_simple_probabilities(self):
        weights = np.float64([1.678, 1.689, 1.348, 0.903, 1.845, 0.877, 0.704, 0.482])

        expected_result = weights / weights.sum()

        test_result = simple_probabilities(weights)

        assert_allclose(test_result, expected_result)

    def test_simple_sample(self):
        weights = np.float64([2, 4, 1, 1])  # [.25, .5, .125, .125] -> [.25, .75, .875, 1.]

        expected_results = [
            (MIN_RANDOM_VALUE, 0),
            (.2, 0),
            (.4, 1),
            (.8, 2),
            (.9, 3),
            (1., 3)
        ]

        for i, (r, expected_index) in enumerate(expected_results):
            test_result = simple_sample(weights, r)
            assert expected_index == test_result, f"Test={i} Expected={expected_index} Actual={test_result}"


class TestMultinomialCore(unittest.TestCase):

    def test_probabilities(self):
        utilities = np.float64([1.678, 1.689, 1.348, 0.903, 1.845, 0.877, 0.704, 0.482])

        expected_result = np.float64(
            [0.181775432, 0.183785999, 0.130682672, 0.083744629, 0.214813892, 0.08159533, 0.068632901, 0.054969145]
        )
        expected_ls = 29.4585222

        test_result, test_ls = multinomial_probabilities(utilities)

        assert_allclose(test_result, expected_result, rtol=0.000001)
        assert abs(expected_ls - test_ls) < 0.000001

    def test_sample_once(self):
        utilities = np.float64([1.678, 1.689, 1.348, 0.903, 1.845, 0.877, 0.704, 0.482])
        probabilities, _ = multinomial_probabilities(utilities)

        draws, expected_indices = _cp_midpoints(probabilities)

        for n, (draw, expected_index) in enumerate(zip(draws, expected_indices)):
            test_result, _ = multinomial_sample(utilities, draw)
            assert test_result == expected_index, f"{n}: Expected={expected_index} Actual={test_result}"

    def test_multisample(self):
        utilities = np.float64([1.678, 1.689, 1.348, 0.903, 1.845, 0.877, 0.704, 0.482])
        probabilities, _ = multinomial_probabilities(utilities)

        n_draws, seed = 100, 1

        test_results, _ = multinomial_multisample(utilities, n_draws, seed, None)

        draws = _randomize(n_draws, seed)
        for r, test_result in zip(draws, test_results):
            expected_result, _ = multinomial_sample(utilities, r)
            assert expected_result == test_result

    def test_worker_sampling(self):
        n_rows, n_cols, util_seed, sample_seed = 5, 6, 7, 8
        utilities = -_randomize((n_rows, n_cols), seed=util_seed)

        test_results, _ = worker_multinomial_sample(utilities, 1, seed=sample_seed)

        draws = _randomize(n_rows, sample_seed)
        for row in range(n_rows):
            util_row = utilities[row]
            r = draws[row]
            expected_result, _ = multinomial_sample(util_row, r)

            assert test_results[row] == expected_result

    def test_worker_probabilities(self):
        n_rows, n_cols, util_seed, sample_seed = 5, 6, 7, 8
        utilities = -_randomize((n_rows, n_cols), seed=util_seed)

        test_results, _ = worker_multinomial_probabilities(utilities)

        for row in range(n_rows):
            util_row = utilities[row]
            expected_result, _ = multinomial_probabilities(util_row)
            assert_allclose(test_results[row], expected_result)


class TestChevalCore(unittest.TestCase):

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

    def test_probabilities(self):
        utilities = np.float64([-0.001, -1.5, -0.5, -0.005, -1, -0.075, -0.3, -0.9])
        tree_info = self._build_nested_tree()
        test_result, test_ls = nested_probabilities(utilities, *tree_info)

        # Auto, Auto-carpool, auto-drive, transit, transit-bus, transit-train, train-bus, train-walk
        expected_result = np.float64([0, 0.085207, 0.355547, 0, 0.156274, 0, 0.354937, 0.048035])
        expected_ls = 1.597835

        assert_allclose(test_result, expected_result, rtol=0.00001)
        assert abs(expected_ls - test_ls) < 0.000001

    def test_sample_once(self):
        pass

    def test_multisample(self):
        pass

    def test_worker_sampling(self):
        pass

    def test_worker_probabilities(self):
        pass

