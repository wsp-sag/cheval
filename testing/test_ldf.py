import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from ..cheval import LinkedDataFrame

households_data = {
    'household_id': [14, 63, 61, 60, 33, 56, 58, 64, 10, 24],
    'income_class': [4, 3, 1, 1, 4, 6, 2, 1, 4, 6]
}
persons_list = {
    'household_id': [60, 33, 58, 64, 63, 61, 61, 14, 56, 56, 56, 10, 24, 56, 61, 61],
    'person_id': [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 4, 3, 4]
}


def test_link_to():
    hh_df = LinkedDataFrame(households_data)
    pers_df = LinkedDataFrame(persons_list)
    pers_df.link_to(hh_df, 'household', on='household_id')

    test_result = pers_df.household.income_class

    expected_result = pd.Series(
        {0: 1, 1: 4, 2: 2, 3: 1, 4: 3, 5: 1, 6: 1, 7: 4, 8: 6, 9: 6, 10: 6, 11: 4, 12: 6, 13: 6, 14: 1, 15: 1}
    )

    assert_series_equal(test_result, expected_result)


def test_slicing():
    hh_df = LinkedDataFrame(households_data)
    pers_df = LinkedDataFrame(persons_list)
    pers_df.link_to(hh_df, 'household', on='household_id')

    mask = pers_df['household_id'] == 56
    pers_df_subset = pers_df.loc[mask].copy()
    pers_df_subset['household_income_class'] = pers_df_subset.household.income_class

    test_result = pers_df_subset['household_income_class']

    expected_result = pd.Series({8: 6, 9: 6, 10: 6, 13: 6}, name='household_income_class')

    assert_series_equal(test_result, expected_result)


def test_evaluate():
    hh_df = LinkedDataFrame(households_data)
    pers_df = LinkedDataFrame(persons_list)
    pers_df.link_to(hh_df, 'household', on='household_id')
    pers_df['hhinc_upper'] = pers_df.evaluate('where(household.income_class >= 4, True, False)')

    test_result = pers_df['hhinc_upper']

    expected_result = pd.Series({
        0: False, 1: True, 2: False, 3: False,  4: False, 5: False, 6: False, 7: True, 8: True, 9: True, 10: True,
        11: True, 12: True, 13: True, 14: False, 15: False
    }, name='hhinc_upper')

    assert_series_equal(test_result, expected_result)


def test_choice_model_evaluate():
    assert False  # TODO


def test_link_summary():
    hh_df = LinkedDataFrame(households_data)
    pers_df = LinkedDataFrame(persons_list)
    pers_df.link_to(hh_df, 'household', on='household_id')

    test_result = pers_df.link_summary()

    expected_result = pd.DataFrame({
        'target_shape': {'household': '(10, 2)'}, 'on_self': {'household': "From columns: ['household_id']"},
        'on_other': {'household': "From columns: ['household_id']"}, 'chained': {'household': True},
        'aggregation': {'household': False}, 'preindexed': {'household': True}
    })
    expected_result.index.name = 'name'

    assert_frame_equal(test_result, expected_result)


def test_pivot_table():
    hh_df = LinkedDataFrame(households_data)
    pers_df = LinkedDataFrame(persons_list)
    pers_df.link_to(hh_df, 'household', on='household_id')
    pers_df['weight'] = 1

    test_result = pers_df.pivot_table(values='weight', index='household_id', columns='household.income_class',
                                      aggfunc='sum', fill_value=0)  # TODO: function fails if margins=True, need to fix

    expected_result = pd.DataFrame({
        1: {10: 0, 14: 0, 24: 0, 33: 0, 56: 0, 58: 0, 60: 1, 61: 4, 63: 0, 64: 1},
        2: {10: 0, 14: 0, 24: 0, 33: 0, 56: 0, 58: 1, 60: 0, 61: 0, 63: 0, 64: 0},
        3: {10: 0, 14: 0, 24: 0, 33: 0, 56: 0, 58: 0, 60: 0, 61: 0, 63: 1, 64: 0},
        4: {10: 1, 14: 1, 24: 0, 33: 1, 56: 0, 58: 0, 60: 0, 61: 0, 63: 0, 64: 0},
        6: {10: 0, 14: 0, 24: 1, 33: 0, 56: 4, 58: 0, 60: 0, 61: 0, 63: 0, 64: 0}
    })
    expected_result.index.name = 'household_id'
    expected_result.columns.name = 'temp_0'

    assert_frame_equal(test_result, expected_result)
