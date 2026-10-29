import numpy as np
import pandas as pd
import pytest

from py_agata.visualization import agp_percentiles


def _build_profile(start='2024-01-01', periods=288, offset=0):
    t = pd.date_range(start=start, periods=periods, freq='5min')
    glucose = 120 + offset + 25 * ((pd.Series(range(periods)) % 48) / 48)
    return pd.DataFrame({'t': t, 'glucose': glucose.astype(float)})


def test_agp_percentiles_covers_the_day_in_order():
    agp = agp_percentiles([_build_profile(), _build_profile(start='2024-01-02', offset=15)])

    assert list(agp.columns) == ['p05', 'p25', 'p50', 'p75', 'p95']
    assert agp.index[0] == 0 and agp.index[-1] == 1440
    assert np.all(agp['p05'] <= agp['p50']) and np.all(agp['p50'] <= agp['p95'])


def test_agp_percentiles_rejects_all_nan():
    data = _build_profile()
    data['glucose'] = np.nan
    with pytest.raises(ValueError):
        agp_percentiles(data)
