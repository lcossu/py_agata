import pandas as pd
import numpy as np
import pytest

from py_agata.processing import retime_glucose
from py_agata.visualization import plot_cvga


def test_plot_cvga_basic():
    # Create a simple DataFrame with glucose values
    data = pd.DataFrame({
        't': pd.date_range('2024-01-01', periods=10, freq='h'),
        'glucose': np.linspace(60, 200, 10)
    })
    fig = plot_cvga(data)
    assert fig is not None
    assert hasattr(fig, 'data')
    # Should have at least one scatter trace (the points)
    assert any(trace.mode == 'markers' or trace.mode == 'lines' for trace in fig.data)


def test_plot_cvga_handles_short_data():
    # Only two points
    data = pd.DataFrame({
        't': pd.date_range('2024-01-01', periods=2, freq='h'),
        'glucose': [100, 150]
    })
    fig = plot_cvga(data)
    assert fig is not None


def test_plot_cvga_with_real_data():
    import os
    # Load real example data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv')
    data = pd.read_csv(data_path)
    # Optionally retime if needed (as in other tests)
    if 't' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['t']):
        data['t'] = pd.to_datetime(data['t'])
    if 'glucose' not in data.columns:
        raise ValueError('multi-meal_example.csv must have a glucose column')
    data = retime_glucose(data, timestep=5)
    fig = plot_cvga(data)
    assert fig is not None


def test_plot_cvga_with_two_profiles():
    import pandas as pd
    from py_agata.visualization import plot_cvga
    # Profile 1
    time1 = pd.date_range('2000-01-01 00:00:00', periods=11, freq='5min')
    data1 = pd.DataFrame({'t': time1, 'glucose': [100, 75, 75, 80, 120, 120, 200, 200, 260, 260, float('nan')]})
    # Profile 2
    time2 = pd.date_range('2000-01-01 00:04:00', periods=11, freq='3min')
    data2 = pd.DataFrame({'t': time2, 'glucose': [80, float('nan'), float('nan'), 80, 120, 120, 200, 200, 220, 220, float('nan')]})
    # Plot
    plot_cvga([data1, data2])
