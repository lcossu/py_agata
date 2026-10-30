import numpy as np
import pandas as pd

from py_agata.risk import dynamic_risk
from py_agata.visualization import plot_dynamic_risk


def test_plot_dynamic_risk_splits_risk_by_sign():
    t = pd.date_range('2024-01-01', periods=60, freq='5min')
    data = pd.DataFrame({'t': t, 'glucose': np.r_[np.linspace(60, 250, 30), np.linspace(250, 60, 30)]})

    fig = plot_dynamic_risk(data)
    _, hypo, hyper = fig.data
    risk = dynamic_risk(data)
    assert np.allclose(hypo.y, np.minimum(risk, 0))
    assert np.allclose(hyper.y, np.maximum(risk, 0))


def test_plot_dynamic_risk_adds_input_panels():
    t = pd.date_range('2024-01-01', periods=60, freq='5min')
    data = pd.DataFrame({'t': t, 'glucose': np.linspace(80, 200, 60), 'cho': 0.0, 'bolus': 0.0, 'basal': 0.01})
    data.loc[10, ['cho', 'bolus']] = [50, 5]

    fig = plot_dynamic_risk(data)
    titles = [fig.layout[f'yaxis{i}'].title.text for i in ['', 2, 3, 4, 5]]
    assert titles == ['Glucose [mg/dL]', 'Dynamic risk', 'CHO [g]', 'Bolus [U]', 'Basal [U/min]']


def test_input_panels_add_height():
    t = pd.date_range('2024-01-01', periods=60, freq='5min')
    data = pd.DataFrame({'t': t, 'glucose': np.linspace(80, 200, 60)})
    assert plot_dynamic_risk(data, height=600).layout.height == 600

    data['cho'], data['bolus'] = 0.0, 0.0
    assert plot_dynamic_risk(data, height=600).layout.height == 760
