import os

import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.glycemic_transformation import int_hypo
from py_agata.processing import retime_glucose


def test_int_hypo():
    """
    Unit test of int_hypo function.

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
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [50, 50]
    glucose[3] = 80
    glucose[4:6] = [120, 120]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    glucose = [x if x > 0 else np.nan for x in glucose]

    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(int_hypo(data)) == False
    
    # 50 observed min: AUC below 70 = (30 + 20 + 20) * 5 = 350 mg/dl*min, 15 min below 70
    observed_days = 50 / 1440
    assert np.isclose(int_hypo(data), np.sqrt((350 / observed_days) ** 2 + (15 / observed_days) ** 2))

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
    assert np.isnan(int_hypo(data))
