import os

import pandas as pd

from py_agata.processing import retime_glucose
from py_agata.utils import read_dexcom_data
from py_agata.visualization import plot_glucose_3d


def test_plot_3d_glucose():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))

    data = retime_glucose(data, timestep=5)
    fig = plot_glucose_3d(data)
    assert fig is None

    file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'dexcom_example.xlsx')
    data = read_dexcom_data(file)
    data = retime_glucose(data, timestep=5)
    fig = plot_glucose_3d(data)
    assert fig is not None
