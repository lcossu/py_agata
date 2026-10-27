import datetime

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from py_agata.input_validator import check_homogeneous_timegrid, check_data_columns, check_dataframe


def _validate_event(data: pd.DataFrame, event_name: str):
    if not event_name in data.columns:
        raise ValueError(
            f"Event column '{event_name}' not found in data. Use processing.merge_additional_data to add event columns.")


def _validate_meal(data: pd.DataFrame):
    _validate_event(data, 'cho')


def after_meal_auc(data: pd.DataFrame, t_span: int = 120) -> list[tuple[datetime.datetime, float]]:
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    _validate_meal(data)
    sample_time_minutes = data.t.diff().dt.total_seconds().dropna().iloc[0] / 60

    events = data.loc[data.cho > 0, 't']
    aucs = []
    for e in events:
        t_start = e
        t_end = e + pd.Timedelta(minutes=t_span)
        mask = (data.t >= t_start) & (data.t <= t_end)
        glucose_segment = data.loc[mask, 'glucose']
        auc = trapezoid(glucose_segment.values, dx=sample_time_minutes)
        aucs.append(auc)

    return list(zip(events.tolist(), aucs))


def meal_response_metrics(data: pd.DataFrame, pre_meal_window: int = 15, post_meal_window: int = 60) -> list[dict]:
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    _validate_meal(data)

    events = data.loc[data.cho > 0, 't']
    metrics = []
    for e in events:
        t_pre_start = e - pd.Timedelta(minutes=pre_meal_window)
        t_post_end = e + pd.Timedelta(minutes=post_meal_window)

        mask_pre = (data.t >= t_pre_start) & (data.t < e)
        baseline_glucose = data.loc[mask_pre, 'glucose'].mean()

        mask_post = (data.t >= e) & (data.t <= t_post_end)
        post_meal_glucose = data.loc[mask_post, 'glucose']

        if post_meal_glucose.empty:
            continue

        peak_glucose = post_meal_glucose.max()
        time_to_peak = (data.loc[post_meal_glucose.idxmax(), 't'] - e).total_seconds() / 60

        one_hour_after_peak_time = data.loc[post_meal_glucose.idxmax(), 't'] + pd.Timedelta(minutes=60)
        mask_one_hour_after = (data.t >= one_hour_after_peak_time) & (
                    data.t < one_hour_after_peak_time + pd.Timedelta(minutes=15))
        glucose_one_hour_after = data.loc[mask_one_hour_after, 'glucose'].mean()

        rise = peak_glucose - baseline_glucose
        percent_recovery = (peak_glucose - glucose_one_hour_after) / rise * 100 if rise != 0 else np.nan

        metrics.append({
            'event_time': e,
            'glucose_rise_to_peak': peak_glucose - baseline_glucose,
            'time_to_peak': time_to_peak,
            'percent_baseline_recovery_one_hour_after_peak': percent_recovery
        })

    return metrics


def tdi(data: pd.DataFrame) -> float:
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    _validate_event(data, 'bolus')
    _validate_event(data, 'basal')

    # Bolus is in U per sample, basal in U/min
    sample_time_minutes = data.t.diff().dt.total_seconds().dropna().iloc[0] / 60
    n_days = data.shape[0] * sample_time_minutes / 1440
    return (data['bolus'].sum() + data['basal'].sum() * sample_time_minutes) / n_days
