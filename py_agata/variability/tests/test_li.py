import numpy as np
import pandas as pd
import pytest
from py_agata.variability import li


def make_glucose_df(values):
    times = pd.date_range('2024-01-01', periods=len(values), freq='5min')
    return pd.DataFrame({'t': times, 'glucose': values})


def test_li():
    # A step of 18.018 mg/dl (1 mmol/l) every hour gives 1 (mmol/l)^2/h for each of the 168 hours of a week
    values = 100 + 18.018 * np.repeat(np.arange(4), 12)
    assert abs(li(make_glucose_df(values)) - 168) < 1e-9
    assert abs(li(make_glucose_df(100 + 18.018 * np.arange(10)), interval=5) - 168 * 144) < 1e-6
    assert li(make_glucose_df([120.] * 30)) == 0
    with pytest.raises(ValueError):
        li(make_glucose_df([120.] * 30), interval=7)


def test_li_empty():
    assert np.isnan(li(make_glucose_df([])))
    assert np.isnan(li(make_glucose_df([np.nan] * 30)))
