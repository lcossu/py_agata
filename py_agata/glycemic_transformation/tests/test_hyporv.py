import os

import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.glycemic_transformation import hypo_rv, int_hypo
from py_agata.inspection import find_hypoglycemic_events
from py_agata.processing import retime_glucose


def test_hyporv():
    """
    Unit test of hyporv function.

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
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(data, timestep=5)
    data.glucose = data.glucose-10

    # Tests
    assert np.isnan(hypo_rv(data)) == False
    episodes_per_day = find_hypoglycemic_events(data)['events_per_week'] / 7
    assert episodes_per_day > 0
    assert np.isclose(hypo_rv(data), int_hypo(data) * episodes_per_day)

    # Set empty data
    t = np.arange(datetime(2000, 1, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = np.nan
    glucose[1:3] = [np.nan, np.nan]
    glucose[3] = np.nan
    glucose[4:6] = [np.nan, np.nan]
    glucose[6:8] = [np.nan, np.nan]
    glucose[8:10] = [np.nan, np.nan]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(hypo_rv(data))
