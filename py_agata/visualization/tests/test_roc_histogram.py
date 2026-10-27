import os

import pandas as pd

from py_agata.processing import retime_glucose
from py_agata.visualization import plot_roc_histogram


def test_roc_histogram():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(data, timestep=5)
    plot_roc_histogram(data)