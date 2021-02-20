# import unittest

# import pandas as pd
# from pandas.testing import assert_series_equal
# import numpy as np

# from ..cheval.parsing import Expression


# class GreenTest(unittest.TestCase):

#     def test_simple_symbol(self):
#         e = Expression("a + b")

#         symbols = {s for s in e.itersymbols()}

#         assert 'a' in symbols
#         assert 'b' in symbols

#     def test_dict_literal(self):
#         tests = [
#             ("{a: 1, b: 2}", {'a': 1, 'b': 2}),  # Simple keys, positive numbers
#             ("{a: -0.5, b: -0.3}", {'a': -0.5, 'b': -0.3}),  # Simple keys, negative numbers
#             ("{'choice 1': 5}", {"choice 1": 5}),  # Spaced keys
#             ("{transit.bus.walk: 0.6}", {('transit', 'bus', 'walk'): 0.6}),  # Dotted keys
#             ("{transit.bus: 0.6, transit.bus.walk: 0.5}",  # Dotted keys with padding
#              {('transit', 'bus', '.'): 0.6, ('transit', 'bus', 'walk'): 0.5})
#         ]

#         for text, dict_ in tests:
#             e = Expression(text)
#             parsed_dicts = [val for key, val in e.iterdicts()]
#             assert len(parsed_dicts) == 1

#             expected = pd.Series(dict_, dtype=np.float64)
#             actual = parsed_dicts[0]

#             assert_series_equal(expected, actual, check_names=False)

#     def test_str(self):
#         pass

#     def test_unaryop(self):
#         tests = [
#             ("-b", "(-b)"),
#             ("not b", "(~b)")
#         ]

#         for expr, desired_result in tests:
#             e = Expression(expr)
#             actual_result = e.transformed.strip()

#             assert actual_result == desired_result, f"Desired='{desired_result}' Actual='{actual_result}'"

#     def test_bool_op(self):

#         # Use variables that contain boolean text to protect against replacement:
#         # 'and' in 'sand'
#         # 'or' in 'door'
#         # 'not' in 'knot'

#         tests = [
#             ("sand and door", "(sand & door)"),
#             ("sand and door and knot", "(sand & door & knot)"),
#             ("sand or door", "(sand | door)"),
#             ("sand or door or knot", "(sand | door | knot)")
#         ]

#         for expr, desired_result in tests:
#             e = Expression(expr)
#             actual_result = e.transformed.strip()

#             assert actual_result == desired_result, f"Desired='{desired_result}' Actual='{actual_result}'"

#     def test_call(self):
#         tests = [
#             "sum("
#         ]

#     def test_method(self):
#         pass

#     def test_name(self):
#         pass

#     def test_attribute(self):
#         pass

# # TODO: "Red Test" cases
