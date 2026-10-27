from py_agata.visualization import plot_gri


def test_gri_plot():
    import pandas as pd
    from py_agata.visualization import plot_cvga
    # Profile 1
    time1 = pd.date_range('2000-01-01 00:00:00', periods=11, freq='5min')
    data1 = pd.DataFrame({'t': time1, 'glucose': [100, 75, 75, 80, 120, 120, 200, 200, 260, 260, float('nan')]})
    # Profile 2
    time2 = pd.date_range('2000-01-01 00:04:00', periods=11, freq='3min')
    data2 = pd.DataFrame(
        {'t': time2, 'glucose': [80, float('nan'), float('nan'), 80, 120, 120, 200, 200, 220, 220, float('nan')]})
    # Plot
    plot_gri([data1, data2])


def test_gri_plot_with_real_data():
    import pandas as pd
    import os
    from py_agata.visualization import plot_gri
    # Load real example data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'multi-meal_example.csv')
    data = pd.read_csv(data_path)
    if 't' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['t']):
        data['t'] = pd.to_datetime(data['t'])
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'data.csv')
    data1 = pd.read_csv(data_path)
    data1['t'] = pd.to_datetime(data1['Time'])
    fig = plot_gri([data, data1])
    assert fig is not None
