import os

import pandas as pd
import pytest
from plotly.graph_objs import Figure

from py_agata.processing import retime_glucose
from py_agata.utils import read_dexcom_data
from py_agata.error import clarke
from py_agata.visualization import plot_clarke_error_grid


def test_plot_clarke_error_grid_basic():
    # simple dataset with known pairs
    data = pd.DataFrame({
        't': [1, 2, 3],
        'reference_glucose': [100, 200, 50],
        'glucose': [110, 190, 30],
    })
    fig, percentages = plot_clarke_error_grid(data)

    assert isinstance(fig, Figure)
    assert fig.layout.title.text == 'Clarke Error Grid'

    # the glucose pairs are the only marker trace
    scatter = [trace for trace in fig.data if trace.mode == 'markers']
    assert len(scatter) == 1
    assert list(scatter[0].x) == [100, 200, 50]
    assert list(scatter[0].y) == [110, 190, 30]

    # borders follow Clarke et al. 1987
    lines = {(tuple(trace.x), tuple(trace.y)) for trace in fig.data if trace.mode == 'lines'}
    assert ((70, 290), (180, 400)) in lines
    assert ((70, 70), (84, 400)) in lines
    assert ((130, 180), (0, 70)) in lines

    # percentages come from py_agata.error.clarke
    ref = pd.DataFrame({'t': data.t, 'glucose': data.reference_glucose.astype(float)})
    mon = pd.DataFrame({'t': data.t, 'glucose': data.glucose.astype(float)})
    assert percentages == clarke(ref, mon)

    labels = {ann.text for ann in fig.layout.annotations}
    assert labels == {f'<b>{z}</b>' for z in 'ABCDE'}


def test_plot_clarke_error_grid_missing_reference_column():
    # missing the default reference column
    data = pd.DataFrame({'glucose': [100, 120], 't': [1, 2]})
    with pytest.raises(ValueError) as exc:
        plot_clarke_error_grid(data)
    assert "Column 'reference_glucose' not found" in str(exc.value)


def test_plot_clarke_error_grid_missing_predicted_column():
    # missing the default predicted column
    data = pd.DataFrame({'reference_glucose': [100, 120], 't': [1, 2]})
    with pytest.raises(Exception, match='glucose'):
        plot_clarke_error_grid(data)


def test_plot_clarke_error_grid_dexcom():
    file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'data', 'dexcom_example.xlsx')
    data = read_dexcom_data(file)
    data = retime_glucose(data, timestep=5)
    # Add random variability to create reference values
    import numpy as np
    data['reference_glucose'] = data['glucose'] + np.random.randint(-30, 31, size=len(data))
    fig,_ = plot_clarke_error_grid(data)

    # returns a Plotly Figure
    assert isinstance(fig, Figure)
