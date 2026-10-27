import numpy as np
import pandas as pd
import pytest
from py_agata.variability import sample_entropy, multiscale_entropy


def make_glucose_df(values):
    times = pd.date_range('2024-01-01', periods=len(values), freq='5min')
    return pd.DataFrame({'t': times, 'glucose': values})


def test_sample_entropy_known():
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    se = sample_entropy(arr, m=2, r=0.2)
    assert not np.isfinite(se)


def test_sample_entropy_nan():
    arr = np.array([np.nan, np.nan, np.nan])
    se = sample_entropy(arr, m=2, r=0.2)
    assert np.isnan(se)


def test_sample_entropy_complex():
    # A periodic sequence should have lower entropy than a random one
    periodic = np.tile([1, 2, 3, 4], 25)
    random = np.random.RandomState(42).randn(100)
    se_periodic = sample_entropy(periodic, m=2, r=0.2)
    se_random = sample_entropy(random, m=2, r=0.2)
    assert se_periodic < se_random


def test_multiscale_entropy_simple():
    df = make_glucose_df(np.linspace(80, 180, 100))
    mse = multiscale_entropy(df, m=2, r=0.2, max_scale=5)
    assert np.isfinite(mse)


def test_multiscale_entropy_nan():
    df = make_glucose_df([np.nan] * 10)
    mse = multiscale_entropy(df, m=2, r=0.2, max_scale=5)
    assert np.isnan(mse)


def test_multiscale_entropy_short():
    df = make_glucose_df([100, 101])
    mse = multiscale_entropy(df, m=2, r=0.2, max_scale=5)
    assert np.isnan(mse)


def test_multiscale_entropy():
    from datetime import datetime
    from datetime import timedelta
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [50, 50]
    glucose[3] = 80
    glucose[4:6] = [120, 120]
    glucose[6:8] = [200, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)
    mse = multiscale_entropy(data, m=2, r=0.2, max_scale=5)
    assert (mse == 0.6931471805599453)
