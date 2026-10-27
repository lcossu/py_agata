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
