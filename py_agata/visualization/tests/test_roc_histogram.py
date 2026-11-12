import os

import pandas as pd

from py_agata.processing import retime_glucose
from py_agata.visualization import plot_roc_histogram


def test_roc_histogram():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv'))
    data = retime_glucose(data, timestep=5)
    plot_roc_histogram(data)

def test_roc_histogram_matches_plot():
    import numpy as np
    import pandas as pd
    from py_agata.visualization import roc_histogram, plot_roc_histogram

    # Steps of +1, +1, -2 and +0.6 mg/dL/min, with one gap
    t = pd.date_range('2024-01-01', periods=6, freq='5min')
    data = pd.DataFrame({'t': t, 'glucose': [100, 105, 110, 100, 103, np.nan]})

    hist = roc_histogram(data)
    assert list(hist['bin_start']) == [-2.0, 0.5, 1.0]
    assert list(hist['count']) == [1, 1, 2]

    bar = plot_roc_histogram(data).data[0]
    assert list(bar.y) == list(hist['count'])


def test_roc_histogram_requires_homogeneous_grid():
    import pytest
    from py_agata.visualization import roc_histogram, plot_poincare

    t = pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:05', '2024-01-01 00:15'])
    data = pd.DataFrame({'t': t, 'glucose': [100., 105., 110.]})
    with pytest.raises(Exception, match='homogeneous'):
        roc_histogram(data)
    with pytest.raises(Exception, match='homogeneous'):
        plot_poincare(data)
