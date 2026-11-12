import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from py_agata.variability import glucodensity, glucodensity_distance


def make_glucose_df(values):
    times = pd.date_range('2024-01-01', periods=len(values), freq='5min')
    return pd.DataFrame({'t': times, 'glucose': values})


def test_glucodensity_integrates_to_one():
    values = np.random.RandomState(0).normal(140, 30, 2000)
    gd = glucodensity(make_glucose_df(values))
    assert list(gd.columns) == ['glucose', 'density']
    assert abs(trapezoid(gd.density, gd.glucose) - 1) < 0.01
    assert abs(gd.glucose[gd.density.idxmax()] - 140) < 10


def test_glucodensity_constant():
    gd = glucodensity(make_glucose_df([120.] * 50))
    assert gd.density.isna().all()


def test_glucodensity_distance_shift():
    values = np.random.RandomState(1).normal(140, 30, 2000)
    assert glucodensity_distance(make_glucose_df(values), make_glucose_df(values)) == 0
    assert abs(glucodensity_distance(make_glucose_df(values), make_glucose_df(values + 20)) - 20) < 1e-9


def test_glucodensity_distance_nan():
    assert np.isnan(glucodensity_distance(make_glucose_df([np.nan] * 5), make_glucose_df([100., 110.])))

