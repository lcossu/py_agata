import pandas as pd
import numpy as np
import os
from py_agata.visualization import plot_poincare

def test_plot_poincare_fake():
    # Create fake glucose data
    data = pd.DataFrame({
        't': pd.date_range('2024-01-01', periods=50, freq='5min'),
        'glucose': np.linspace(80, 180, 50) + np.random.normal(0, 10, 50)
    })
    fig = plot_poincare(data)
    assert fig is not None
    assert hasattr(fig, 'data')
    # Should have at least one scatter trace
    assert any(trace.mode == 'markers' for trace in fig.data)

def test_plot_poincare_real():
    # Use real data from multi-meal_example.csv
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv')
    data = pd.read_csv(data_path)
    if 't' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['t']):
        data['t'] = pd.to_datetime(data['t'])
    if 'glucose' not in data.columns:
        raise ValueError('multi-meal_example.csv must have a glucose column')
    # Optionally resample to 5 min grid if needed
    try:
        from py_agata.processing import retime_glucose
        data = retime_glucose(data, timestep=5)
    except Exception:
        pass
    fig = plot_poincare(data)
    assert fig is not None

