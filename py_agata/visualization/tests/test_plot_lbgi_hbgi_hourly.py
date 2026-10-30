import os

import pandas as pd

from py_agata.processing import retime_glucose
from py_agata.visualization import plot_lbgi_hbgi_hourly


def test_plot_lbgi_hbgi_hourly_multimeal():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(data, timestep=5)
    data['t'] = pd.to_datetime(data['t'])
    fig = plot_lbgi_hbgi_hourly(data)
    assert fig is not None


def test_lbgi_hbgi_hourly_matches_plot_and_averages_by_hour():
    import numpy as np
    from py_agata.visualization import lbgi_hbgi_hourly

    # Two days, low at 03:00 on both, high at 15:00 only on day one
    t = pd.date_range('2024-01-01', periods=2 * 288, freq='5min')
    glucose = np.full(len(t), 120.0)
    glucose[t.hour == 3] = 60.0
    glucose[(t.hour == 15) & (t.day == 1)] = 250.0
    data = pd.DataFrame({'t': t, 'glucose': glucose})

    blocks = lbgi_hbgi_hourly(data)
    assert len(blocks) == 48
    fig = plot_lbgi_hbgi_hourly(data)
    assert np.allclose(-np.array(fig.data[0].y), blocks['lbgi'])
    assert np.allclose(fig.data[1].y, blocks['hbgi'])

    by_hour = lbgi_hbgi_hourly(data, by_hour_of_day=True)
    assert list(by_hour.index) == list(range(24))
    assert by_hour['lbgi'].idxmax() == 3
    assert by_hour['hbgi'].idxmax() == 15
    assert np.isclose(by_hour.loc[15, 'hbgi'], blocks['hbgi'].iloc[[15, 39]].mean())
