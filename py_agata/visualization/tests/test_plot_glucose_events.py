import os

import pandas as pd

from py_agata.inspection import find_hypoglycemic_events, find_hyperglycemic_events
from py_agata.processing import retime_glucose, merge_additional_data
from py_agata.visualization import plot_glucose, plot_glucose_events


def test_plot_glucose():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(data, timestep=5)
    hypo_events = find_hypoglycemic_events(data)
    hyper_events = find_hyperglycemic_events(data)
    # Combine events for visualization
    events = {'hypo': hypo_events, 'hyper': hyper_events}
    plot_glucose_events(data, events)
