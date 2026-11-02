import numpy as np
import pandas as pd
import pytest

from py_agata.events import meal_response_metrics
from py_agata.visualization import plot_meal_response


def _meal_day():
    t = pd.date_range('2024-01-01 06:00', periods=120, freq='5min')
    glucose = np.full(len(t), 100.0)
    glucose[30:42] = np.linspace(100, 180, 12)
    glucose[42:60] = np.linspace(180, 110, 18)
    cho = np.zeros(len(t))
    cho[30] = 50
    return pd.DataFrame({'t': t, 'glucose': glucose, 'cho': cho, 'bolus': cho / 10})


def test_plot_meal_response_marks_reported_peak_and_baseline():
    data = _meal_day()
    meal = meal_response_metrics(data)[0]

    fig = plot_meal_response(data, meal['event_time'])
    traces = {tr.name: tr for tr in fig.data}
    assert traces['Peak'].y[0] == 180
    assert traces['Baseline'].y[0] == 180 - meal['glucose_rise_to_peak']
    assert pd.Timestamp(traces['Glucose'].x[0]) == meal['event_time'] - pd.Timedelta(minutes=30)
    assert fig.layout.height == 350 + 2 * 80


def test_plot_meal_response_rejects_unknown_meal():
    with pytest.raises(ValueError):
        plot_meal_response(_meal_day(), '2024-01-01 07:00')
