import numpy as np
import pandas as pd

from py_agata.events import after_meal_auc, meal_response_metrics, tdi


def _day():
    t = pd.date_range('2000-01-01', periods=288, freq='5min')
    data = pd.DataFrame({'t': t, 'glucose': 100., 'cho': 0., 'bolus': 0., 'basal': 0.01})

    # Meal at 12:00, peak 200 at 12:30, back to 125 one hour after the peak
    meal = 144
    data.loc[meal, 'cho'] = 50.
    data.loc[meal, 'bolus'] = 5.
    data.loc[meal + 6, 'glucose'] = 200.
    data.loc[meal + 18:meal + 20, 'glucose'] = 125.
    return data


def test_tdi():
    # 5 U bolus + 0.01 U/min over 1440 min
    assert np.isclose(tdi(_day()), 5 + 14.4)


def test_meal_response_metrics():
    m = meal_response_metrics(_day())[0]
    assert m['glucose_rise_to_peak'] == 100.
    assert m['time_to_peak'] == 30.
    assert m['percent_baseline_recovery_one_hour_after_peak'] == 75.


def test_after_meal_auc():
    data = _day()
    data['glucose'] = 100.
    (_, auc), = after_meal_auc(data, t_span=120)
    assert np.isclose(auc, 100. * 120)
