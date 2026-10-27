import numpy as np
import pytest
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, mannwhitneyu

from py_agata.py_agata import _compare_samples


def test_compare_samples():

    # Gaussian samples use the paired t-test when paired, the unpaired one otherwise
    rng = np.random.default_rng(0)
    r1 = rng.normal(100, 10, 20)
    r2 = r1 + rng.normal(2, 1, 20)
    assert np.isclose(_compare_samples(r1, r2, True, 0.05)['p'], ttest_rel(r1, r2).pvalue)
    assert np.isclose(_compare_samples(r1, r2, False, 0.05)['p'], ttest_ind(r1, r2).pvalue)
    assert _compare_samples(r1, r2, True, 0.05)['h'] == 1

    # Skewed samples use Wilcoxon signed-rank or Mann-Whitney U, ignoring nan
    r1 = np.array([1., 1, 1, 1, 1, 1, 1, 2, 50, 100, np.nan])
    r2 = np.array([1., 2, 3, 2, 3, 2, 3, 5, 90, 300, 7])
    assert np.isclose(_compare_samples(r1, r2, True, 0.05)['p'], wilcoxon(r1, r2, nan_policy='omit').pvalue)
    assert np.isclose(_compare_samples(r1, r2, False, 0.05)['p'], mannwhitneyu(r1, r2, nan_policy='omit').pvalue)

    # Identical paired skewed samples give p = 1
    assert _compare_samples(r1, r1.copy(), True, 0.05) == {'p': 1., 'h': 0}


def test_compare_two_arms_paired_length():
    import pandas as pd
    from py_agata.py_agata import Agata
    data = pd.DataFrame({'t': pd.date_range('2000-01-01', periods=10, freq='5min'), 'glucose': np.full(10, 100.)})
    with pytest.raises(ValueError):
        Agata().compare_two_arms([data, data], [data], is_paired=True, alpha=0.05)
