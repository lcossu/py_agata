import numpy as np
import pandas as pd

from py_agata.processing import select_time_window
from py_agata.time_in_ranges import time_in_target


def test_select_time_window():
    t = pd.date_range('2000-01-01', periods=288, freq='5min')
    glucose = np.where(t.hour < 6, 60., 120.)
    data = pd.DataFrame({'t': t, 'glucose': glucose})

    # Nighttime keeps 00:00-06:00 only
    night = select_time_window(data)
    assert night.glucose.notna().sum() == 72
    assert time_in_target(night) == 0

    # Daytime keeps 06:00-24:00 and leaves the input untouched
    day = select_time_window(data, 6, 24)
    assert day.glucose.notna().sum() == 216
    assert time_in_target(day) == 100
    assert data.glucose.notna().all()

    # Window across midnight
    late = select_time_window(data, 22, 2)
    assert late.glucose.notna().sum() == 48
