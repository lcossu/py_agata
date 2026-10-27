import os

import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.glycemic_transformation import q_score
from py_agata.processing import retime_glucose


def test_q_score():
    """
    Unit test of q_score function.

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
    # Constant 7.8 mmol/l over two days: MBG at reference, range, MODD and times out of range are zero
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 3, 0, 0, 0), timedelta(minutes=5)).astype(
        datetime)
    data = pd.DataFrame(data={'t': t, 'glucose': np.full(t.shape[0], 7.8 * 18.018)})

    # Tests
    expected = 8 - 7.5 / 2.9 - 0.6 / 1.2 - 6.2 / 5.7 - 1.8 / 0.9
    assert np.isclose(q_score(data), expected)

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
    assert np.isnan(q_score(data))
