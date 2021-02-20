import numpy as np
import pandas as pd

from ..cheval import LinkedDataFrame


def test_link_to():
    households_data = {
        'household_id': [14, 63, 61, 60, 33, 56, 58, 64, 10, 24],
        'income_class': [4, 3, 1, 1, 4, 6, 2, 1, 4, 6]
    }
    persons_list = {
        'household_id': [60, 33, 58, 64, 63, 61, 61, 14, 56, 56, 56, 10, 24, 56, 61, 61],
        'person_id': [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 4, 3, 4]
    }

    hh_df = LinkedDataFrame(households_data)
    pers_df = LinkedDataFrame(persons_list)

    pers_df.link_to(hh_df, 'household', on='household_id')

    result = pd.Series(
        {0: 1, 1: 4, 2: 2, 3: 1, 4: 3, 5: 1, 6: 1, 7: 4, 8: 6, 9: 6, 10: 6, 11: 4, 12: 6, 13: 6, 14: 1, 15: 1}
    )

    assert np.all(pers_df.household.income_class == result)
