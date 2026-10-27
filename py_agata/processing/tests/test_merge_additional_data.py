import os

import pandas as pd

from py_agata.processing import retime_glucose, merge_additional_data


def test_merge_additional_data():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    glucosedata = retime_glucose(data, timestep=5)
    merged_data = merge_additional_data(glucosedata, data, ['cho', 'bolus'], 2.5)
    assert 'cho' in merged_data
    assert 'bolus' in merged_data

    ##TODO: add delay in cho and bolus to verify correct parsing