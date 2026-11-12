import numpy as np
import pandas as pd
import pytest

from py_agata.glycemic_transformation import (CGP_HEALTHY, cgp_axis_lengths, comprehensive_glucose_pentagon,
                                              int_hyper, int_hypo, pgr)
from py_agata.visualization import plot_comprehensive_glucose_pentagon


def make_glucose_df(values):
    times = pd.date_range('2024-01-01', periods=len(values), freq='5min')
    return pd.DataFrame({'t': times, 'glucose': np.asarray(values, dtype=float)})


def test_healthy_reference_area():
    # The paper reports 466 mm^2 for the healthy pentagon, 14 mm on every axis
    assert np.allclose(cgp_axis_lengths(CGP_HEALTHY), 14)
    assert abs(0.5 * np.sin(np.radians(72)) * 5 * 14 ** 2 - 466) < 0.5


def test_axis_scaling_formulae():
    # Vigersky et al. (2018): the axes reach about 76 mm at the end of their printed scales
    lengths = cgp_axis_lengths({'time_out_of_range': 1440., 'cv_glucose': 80., 'int_hypo': 7000.,
                                'int_hyper': 130000., 'mean_glucose': 310.})
    assert np.allclose(lengths, [(1440 * 0.00614) ** 1.581 + 14, 63 * 0.92 + 14, np.exp(7000 * 0.00057) + 13,
                                 (130000 * 0.000115) ** 1.51 + 14, (220 * 0.0217) ** 2.63 + 14])
    assert lengths[-1] == pytest.approx(75, abs=1)


def test_one_day_above_target():
    # 200 mg/dl all day: every minute is above 180 and out of range, by 20 mg/dl
    data = make_glucose_df(np.full(288, 200.))
    cgp = comprehensive_glucose_pentagon(data)
    assert cgp['time_out_of_range'] == 1440
    assert int_hyper(data) == pytest.approx(np.hypot(20 * 1440, 1440))
    assert int_hypo(data) == 0
    assert pgr(data) > 1


def test_thresholds_can_be_overridden():
    data = make_glucose_df(np.full(288, 200.))
    assert comprehensive_glucose_pentagon(data, th_l=70., th_h=250.)['time_out_of_range'] == 0
    assert int_hyper(data, th=250.) == 0


def test_cgp_plot():
    data = make_glucose_df(100 + 60 * np.sin(np.arange(576) / 40))
    fig = plot_comprehensive_glucose_pentagon(data)
    assert fig.layout.title.text.startswith('PGR = ')
    assert [tr.name for tr in fig.data] == ['Healthy reference', 'Data']
