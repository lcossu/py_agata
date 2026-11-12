import numpy as np
import pandas as pd
import pytest

from py_agata.visualization import plot_lasagna


def make_profile(days, offset=0.):
    t = pd.date_range('2024-01-01', periods=days * 288, freq='5min')
    return pd.DataFrame({'t': t, 'glucose': 120. + offset + 60 * np.sin(2 * np.pi * np.arange(t.size) / 288)})


def test_plot_lasagna_by_day():
    heatmap = plot_lasagna(make_profile(3)).data[0]
    assert np.shape(heatmap.z) == (3, 24)
    assert list(heatmap.y) == ['01 Jan', '02 Jan', '03 Jan']


def test_plot_lasagna_by_profile():
    heatmap = plot_lasagna([make_profile(2), make_profile(2, offset=40)], bin_minutes=30, labels=['A', 'B']).data[0]
    z = np.array(heatmap.z)
    assert z.shape == (2, 48)
    assert np.allclose(z[1] - z[0], 40)


def test_plot_lasagna_labels():
    with pytest.raises(ValueError):
        plot_lasagna([make_profile(1)], labels=['A', 'B'])


def test_plot_glucodensity():
    from py_agata.visualization import plot_glucodensity

    assert [tr.line.color for tr in plot_glucodensity(make_profile(2)).data] == ['steelblue']
    fig = plot_glucodensity([make_profile(2), make_profile(2, offset=40)], labels=['A', 'B'])
    assert [tr.line.color for tr in fig.data] == ['green', 'black']
    assert [tr.name for tr in fig.data] == ['A', 'B']
    with pytest.raises(ValueError):
        plot_glucodensity([make_profile(1)] * 3)
