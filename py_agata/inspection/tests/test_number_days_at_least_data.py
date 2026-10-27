import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.inspection import number_days_at_least_data


def test_number_days_at_least_data():
    """
    Unit test of number_days_at_least_data function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    None
    """
    # Set test data
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 4, 0, 50, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))*np.nan
    glucose[0] = np.nan
    glucose[1:3] = [50, 50]
    glucose[3] = 80
    glucose[4:6] = [120, 120]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10:350] = 100
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Tests
    assert number_days_at_least_data(data) == 1
    assert number_days_at_least_data(data, 100) == 0

    # Set empty data
    d = {'t': [], 'glucose': []}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(number_days_at_least_data(data))
