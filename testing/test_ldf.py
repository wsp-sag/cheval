import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from ..cheval import LinkedDataFrame

vehicles_data = {
    'household_id': [0, 0, 1, 2, 3],
    'vehicle_id': [0, 1, 0, 0, 0],
    'manufacturer': ['Honda', 'Ford', 'Ford', 'Toyota', 'Honda'],
    'model_year': [2009, 2005, 2015, 2011, 2013],
    'km_travelled': [103236, 134981, 19015, 75795, 54573]
}

households_data = {
    'household_id': [0, 1, 2, 3],
    'dwelling_type': ['house', 'apartment', 'house', 'house'],
    'drivers': [4, 1, 2, 3]
}


def test_link_to():
    vehicles = LinkedDataFrame(vehicles_data)
    households = LinkedDataFrame(households_data)

    vehicles.link_to(households, 'household', on='household_id')
    households.link_to(vehicles, 'vehicles', on='household_id')

    test_result = households.vehicles.sum("km_travelled")

    expected_result = pd.Series({0: 238217, 1: 19015, 2: 75795, 3: 54573})

    assert_series_equal(test_result, expected_result)


def test_slicing():
    vehicles = LinkedDataFrame(vehicles_data)
    households = LinkedDataFrame(households_data)

    vehicles.link_to(households, 'household', on='household_id')
    households.link_to(vehicles, 'vehicles', on='household_id')

    mask = vehicles['household_id'] == 0
    vehicles_subset = vehicles.loc[mask].copy()
    vehicles_subset['dwelling_type'] = vehicles_subset.household.dwelling_type

    test_result = vehicles_subset['dwelling_type']

    expected_result = pd.Series({0: 'house', 1: 'house'}, name='dwelling_type')

    assert_series_equal(test_result, expected_result)


def test_evaluate():
    vehicles = LinkedDataFrame(vehicles_data)
    households = LinkedDataFrame(households_data)

    vehicles.link_to(households, 'household', on='household_id')
    households.link_to(vehicles, 'vehicles', on='household_id')

    vehicles['multiple_drivers'] = False
    vehicles.evaluate('where(household.drivers > 1, True, False)', out=vehicles['multiple_drivers'])

    test_result = vehicles['multiple_drivers']

    expected_result = pd.Series({0: True, 1: True, 2: False, 3: True, 4: True}, name='multiple_drivers')

    assert_series_equal(test_result, expected_result)


def test_choice_model_evaluate():
    assert False  # TODO


def test_link_summary():
    vehicles = LinkedDataFrame(vehicles_data)
    households = LinkedDataFrame(households_data)

    vehicles.link_to(households, 'household', on='household_id')
    households.link_to(vehicles, 'vehicles', on='household_id')

    test_result = households.link_summary()

    expected_result = pd.DataFrame({
        'target_shape': {'vehicles': '(5, 5)'}, 'on_self': {'vehicles': "From columns: ['household_id']"},
        'on_other': {'vehicles': "From columns: ['household_id']"}, 'chained': {'vehicles': True},
        'aggregation': {'vehicles': True}, 'preindexed': {'vehicles': True}
    })
    expected_result.index.name = 'name'

    assert_frame_equal(test_result, expected_result)


def test_pivot_table():
    vehicles = LinkedDataFrame(vehicles_data)
    households = LinkedDataFrame(households_data)

    vehicles.link_to(households, 'household', on='household_id')
    households.link_to(vehicles, 'vehicles', on='household_id')

    test_result = vehicles.pivot_table(values='vehicle_id', index='manufacturer', columns='household.dwelling_type',
                                       aggfunc='count', fill_value=0)  # TODO: fails if margins=True, need to fix
    test_result.columns = test_result.columns.astype(str)

    expected_result = pd.DataFrame({
        'apartment': {'Ford': 1, 'Honda': 0, 'Toyota': 0}, 'house': {'Ford': 1, 'Honda': 2, 'Toyota': 1}
    })
    expected_result.index.name = 'manufacturer'
    expected_result.columns.name = 'household.dwelling_type'

    assert_frame_equal(test_result, expected_result)


def test_pivot_table_multiindex():
    """Test that pivot_table can correctly handle tables with a MultiIndex

    Creates LinkedDataFrames with MultiIndexes to ensure pivot_table is not getting
    tripped up trying to use the new pivoted index to reindex linkages which use
    the original table indexes.
    Generally, pivoting a LinkedDataFrame should produce a table which does not
    contain any linkages.

    This catches a specific error which arose in the GGHM demand model Python 3 conversion.
    """
    veh_index = pd.MultiIndex.from_arrays([vehicles_data["household_id"], vehicles_data["vehicle_id"], [0]*len(vehicles_data["vehicle_id"])])
    vehicles = LinkedDataFrame(vehicles_data, index=veh_index)

    hh_ids = households_data["household_id"]
    hh_index = pd.MultiIndex.from_arrays([hh_ids, [0]*len(hh_ids), [0]*len(hh_ids), [0]*len(hh_ids)])
    households = LinkedDataFrame(households_data, index=hh_index)

    vehicles["id_2"] = 0
    households["id_2"] = 0

    vehicles.link_to(households, 'household', on=['household_id', "id_2"])
    households.link_to(vehicles, 'vehicles', on=['household_id', "id_2"])

    test_result = vehicles.pivot_table(values='vehicle_id', index='manufacturer', columns='household.dwelling_type',
                                       aggfunc="sum", fill_value=0.0)
    test_result.columns = test_result.columns.astype(str)

    expected_result = pd.DataFrame({
        'apartment': {'Ford': 0, 'Honda': 0, 'Toyota': 0}, 'house': {'Ford': 1, 'Honda': 0, 'Toyota': 0}
    })
    expected_result.index.name = 'manufacturer'
    expected_result.columns.name = 'household.dwelling_type'

    assert_frame_equal(test_result, expected_result)


def test_groupby():
    """Verifies basic groupby functionality for LinkedDataFrames

    Most groupby functionality should be handled by pandas, but the internal
    implementation may differ version-to-version, which may impact the ability of
    LinkedDataFrame objects to correctly update the linkages.

    This tests that groupby can be run on LinkedDataFrames, and that the linked dataframe
    is accessible from each group and that it contains the correct data.
    """

    df1_dict = {
        "df2_id": [  0,   1,   2,   0,   1,   2,   0],
        "cats":   ["F", "M", "F", "M", "F", "M", "F"],
    }

    df2_dict = {
        "col1":   ["a", "b", "c"],
    }

    df1 = LinkedDataFrame(pd.DataFrame(df1_dict))
    df2 = LinkedDataFrame(pd.DataFrame(df2_dict))

    df1.link_to(df2, "df2", on_self="df2_id")

    groups = df1.groupby("cats")

    for cat, group_df in groups:
        if cat == "F":
            assert list(sorted(group_df.df2.col1)) == ["a", "a", "b", "c"]
        else:
            assert list(sorted(group_df.df2.col1)) == ["a", "b", "c"]
