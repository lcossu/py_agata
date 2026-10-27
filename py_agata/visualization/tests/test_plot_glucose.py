import os

import pandas as pd

from py_agata.processing import retime_glucose, merge_additional_data
from py_agata.visualization import plot_glucose


def test_plot_glucose():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(data, timestep=5)
    plot_glucose(data)

def test_plot_other_data():
    origin = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(origin, timestep=5)
    data = merge_additional_data(data, origin, ['cho', 'bolus', 'basal'], 2.5)
    plot_glucose(data)