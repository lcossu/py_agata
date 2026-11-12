import warnings

import numpy as np
import pandas as pd
import pytest

from py_agata.py_agata import Agata
from py_agata.visualization import plot_glucose_pentagon


@pytest.fixture
def one_sample():
    return pd.DataFrame({'t': pd.to_datetime(['2024-01-01']), 'glucose': [120.]})


def test_one_sample_profile_runs_through_agata(one_sample):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = Agata().analyze_glucose_profile(one_sample)
        Agata().compare_two_arms([one_sample] * 2, [one_sample] * 2, is_paired=False, alpha=0.05)

    # Level metrics still compute, and metrics needing a sampling time are nan
    assert results['variability']['mean_glucose'] == 120
    assert np.isnan(results['variability']['auc_glucose'])
    assert np.isnan(results['glycemic_transformation']['int_hypo'])
    assert results['events']['hypoglycemic_events']['l1']['time_start'].size == 0


def test_one_sample_pentagon_plot_raises(one_sample):
    with pytest.raises(ValueError, match='two samples'):
        plot_glucose_pentagon(one_sample)
