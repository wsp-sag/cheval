import numpy as np
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
    # We want the indexes for each table to have different dimensions,
    # so that indexing logic errors are made explicit as raised exceptions
    # We're using the following MultiIndexes:
    # For the vehicles table: (hhid, veh_id, dummy)
    # For the households table: (hhid, dummy, dummy, dummy)
    # The output pivot table should have an index of (manufacturer)
    # A (manufacturer, household.dwelling_type) index may be created by pandas
    # as an internal implementation detail
    n_rows_veh = len(vehicles_data["household_id"])
    veh_index = pd.MultiIndex.from_arrays([vehicles_data["household_id"], vehicles_data["vehicle_id"],
                                           [0]*n_rows_veh
                                           ])
    vehicles = LinkedDataFrame(vehicles_data, index=veh_index)

    n_rows_hh = len(households_data["household_id"])
    hh_index = pd.MultiIndex.from_arrays([households_data["household_id"],
                                          [0]*n_rows_hh, [0]*n_rows_hh, [0]*n_rows_hh
                                          ])
    households = LinkedDataFrame(households_data, index=hh_index)

    vehicles.link_to(households, 'household', on=['household_id'])
    households.link_to(vehicles, 'vehicles', on=['household_id'])

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


def test_bool_columns():
    """Test that bool columns get correctly handled in aggregations on linkages

    Errors with bool columns originally came up when attempting to aggregate on
    a bool expression. It is hoped that this will catch similar errors with bool
    columns.
    """

    vehicles = LinkedDataFrame(vehicles_data)
    households = LinkedDataFrame(households_data)

    vehicles.link_to(households, 'household', on='household_id')
    households.link_to(vehicles, 'vehicles', on='household_id')

    expression = "vehicles.sum(manufacturer=='Honda')"
    test_result = households.evaluate(expression)

    expected_result = pd.Series([1, 0, 0, 1], index=households.index)

    assert_series_equal(test_result, expected_result)


def test_missing_linkage():
    """Test that linkages with missing values get handled correctly

    If a linkage is made where some rows in the linkage originator do not match
    with any rows in the linkage target, the LDF should link any rows which do
    match, and use a default value to fill any rows with "missing" lookups to
    the target.

    The test for slicing the linked dataframe is a regression test for an error
    where the linkage's missing rows information was not updated during slices
    or re-indexing. These test the invariant
    ```ldf[slice][linkage][column] == ldf[linkage][column][slice]```
    """
    df1 = {
        "df2_id": [0, 1, 2, 3]
    }
    df2 = {
        "idx": [0, 1, 2],
        "float_value": [0.0, 1.0, 2.0],
        "int_value": [0, 1, 2],
    }
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)

    df1 = LinkedDataFrame(df1)
    df2 = LinkedDataFrame(df2)

    df1.link_to(df2, "df2", on_self="df2_id", on_other="idx")

    # Test filling values for floats
    test_result = df1["df2"]["float_value"]
    expected_full_float = pd.Series([0.0, 1.0, 2.0, np.nan], index=df1.index)
    assert_series_equal(test_result, expected_full_float)

    # Test filling values for ints
    test_result = df1["df2"]["int_value"]
    expected_full_int = pd.Series([0, 1, 2, 0], index=df1.index)
    assert_series_equal(test_result, expected_full_int)

    # Test that the missing values are still handled correctly when we take a slice
    slice_indexes = [3, 2]
    slice_df = df1.loc[slice_indexes]

    # For floats
    test_result = slice_df["df2"]["float_value"]
    expected_slice_float = pd.Series([np.nan, 2.0], index=slice_df.index)
    assert_series_equal(expected_slice_float, expected_full_float.loc[slice_indexes])
    assert_series_equal(test_result, expected_slice_float)

    # For ints
    test_result = slice_df["df2"]["int_value"]
    expected_slice_int = pd.Series([0, 2], index=slice_df.index)
    assert_series_equal(expected_slice_int, expected_full_int.loc[slice_indexes])
    assert_series_equal(test_result, expected_slice_int)
