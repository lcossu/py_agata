import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.signal import find_peaks
from datetime import timedelta

from py_agata.inspection import find_hypoglycemic_events
from py_agata.time_in_ranges import time_in_target, time_in_hypoglycemia, time_in_hyperglycemia, \
    time_in_given_above_range, time_in_given_below_range, time_in_given_range
from py_agata.input_validator import *
from py_agata.utils import to_mmol_l
from py_agata.variability import mean_glucose, modd, std_glucose, cv_glucose, positive_auc_glucose


def mr_index(data, r=100):
    """
    Computes the mr value by Schlichtkrull (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)
    r: int, optional, default : 100
        Hyperparameter for mr value calculation

    Returns
    -------
    mr_index: float
        The mr value

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Schlichtkrull et al., "The M-value, an index of blood-sugar control in diabetics", Acta Medica Scandinavica, 1965,
    vol. 177, pp. 95-102. DOI: 10.1111/j.0954-6820.1965.tb01810.x
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_int_parameter(r)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Return the result
    trans_data = 1000 * abs(np.log10(values / r)) ** 3
    return np.mean(trans_data)


def hypo_index(data):
    """
    Computes the hypoglycemic index by Rodbard (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    hypo_index: float
        The hypo index

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Rodbard et al., "Interpretation of continuous glucose monitoring
    data: glycemic variability and quality of glycemic control", Diabetes
    %  Technology & Therapeutics, 2009, vol. 11, pp. S55-S67. DOI: 10.1089/dia.2008.0132.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    b = 2
    d = 30
    lltr = 70

    # Compute metric
    return np.sum((lltr - values[values < lltr]) ** b) / (values.size * d)


def hyper_index(data):
    """
    Computes the hyperglycemic index by Rodbard (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    hyper_index: float
        The hyper index

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Rodbard et al., "Interpretation of continuous glucose monitoring
    data: glycemic variability and quality of glycemic control", Diabetes
    %  Technology & Therapeutics, 2009, vol. 11, pp. S55-S67. DOI: 10.1089/dia.2008.0132.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    a = 1.1
    c = 30
    ultr = 180

    # Compute metric
    return np.sum((values[values > ultr] - ultr) ** a) / (values.size * c)


def igc(data):
    """
    Computes the index of glycemic control by Rodbard (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    igc: float
        The index of glycemic control

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Rodbard et al., "Interpretation of continuous glucose monitoring
    data: glycemic variability and quality of glycemic control", Diabetes
    %  Technology & Therapeutics, 2009, vol. 11, pp. S55-S67. DOI: 10.1089/dia.2008.0132.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Compute metric
    return hypo_index(data) + hyper_index(data)


def grade_hypo_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score in the hypoglycemic range (GRADEhypo)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_hypo_score: float
        The glycemic risk assessment diabetes equation score in the hypoglycemic range (GRADEhypo) (%).

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    grade = 425 * (np.log10(np.log10(values / 18)) + .16) ** 2
    g_tot = np.sum(grade)
    return 100 * np.sum(grade[values < 70]) / g_tot


def grade_hyper_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score in the hyperglycemic range (GRADEhyper)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_hyper_score: float
        The glycemic risk assessment diabetes equation score in the hyperglycemic range (GRADEhyper) (%).

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    grade = 425 * (np.log10(np.log10(values / 18)) + .16) ** 2
    g_tot = np.sum(grade)
    return 100 * np.sum(grade[values > 180]) / g_tot


def grade_eu_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score in the euglycemic range (GRADEeu)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_eu_score: float
        The glycemic risk assessment diabetes equation score in the euglycemic range (GRADEeu) (%).

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    return 100 - (grade_hypo_score(data) + grade_hyper_score(data))


def grade_score(data):
    """
    Computes the glycemic risk assessment diabetes equation score (GRADE)
    by Hill (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    grade_hyper_score: float
        The glycemic risk assessment diabetes equation score (GRADE) (%).

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None

    References
    ----------
    Hill et al., "A method for assessing quality of control from
    glucose profiles", Diabetic Medicine , 2007, vol. 24, pp. 753-758.
    DOI: 10.1111/j.1464-5491.2007.02119.x.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Set up the formula parameters
    grade = 425 * (np.log10(np.log10(values / 18)) + .16) ** 2
    return np.mean(grade)


def q_score(data):
    """
       Computes the Q-Score by Augstein (ignoring nan)

       Parameters
       ----------
       data: pd.DataFrame
           Pandas dataframe with a column `glucose` containing the glucose data
           to analyze (in mg/dl)

       Returns
       -------
       q-score: float
           The q-score evaluated from the data

       Raises
       ------
       None

       See Also
       --------
       None

       Examples
       --------
       None

       References
       ----------
       Augstein et al. "Q-Score: development of a new metric for continuous glucose monitoring
       that enables stratification of antihyperglycaemic therapies"
       BMC Endocr Disord. 2015;15:22. Published 2015 May 1. doi:10.1186/s12902-015-0019-0
   """

    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0:
        return np.nan

    # Augstein units: mmol/l for MBG, range and MODD; hours/day for time above 8.9 and below 3.9 mmol/l
    data_mmol = to_mmol_l(data)
    days = pd.to_datetime(data_mmol.t).dt.floor('D')
    daily = data_mmol.glucose.groupby(days)

    mbg = mean_glucose(data_mmol)
    rng = np.nanmean(daily.max() - daily.min())
    mdd = modd(data_mmol)
    tar = time_in_given_above_range(data, th=160.) * 24 / 100
    tbr = time_in_hypoglycemia(data) * 24 / 100

    return 8 + (mbg - 7.8) / 1.7 + (rng - 7.5) / 2.9 + (tbr - 0.6) / 1.2 + (tar - 6.2) / 5.7 + (mdd - 1.8) / 0.9


def gdi(data):
    """
       Computes the Glucose deviation index
       Parameters
       ----------
       data: pd.DataFrame
           Pandas dataframe with a column `glucose` containing the glucose data
           to analyze (in mg/dl)

       Returns
       -------
       gdi

       References
       ----------
        Zou et al. "Glycemic deviation index: a novel method of integrating glycemic numerical value and variability."
        BMC Endocr Disord. 2021;21(1):52. Published 2021 Mar 19. doi:10.1186/s12902-021-00691-z
   """

    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0:
        return np.nan

    c = time_in_hypoglycemia(data) / 100

    data = to_mmol_l(data)

    import math
    # hypoglycemia
    mg1 = np.nanmean(data.loc[data.glucose < 3.9, 'glucose'])  # mmol/L
    y11 = (math.log(mg1)) ** -0.801 - 0.672
    y21 = 6.337 * (np.abs(y11) + 0.664) ** 2 - 2.796
    mgi1 = y21 ** 2
    mgi1 = 0 if np.isnan(mgi1) else mgi1

    # hyperglycemia
    mg2 = np.nanmean(data.loc[data.glucose > 3.9, 'glucose'])  # mmol/L
    y12 = (math.log(mg2)) ** -0.801 - 0.672
    y22 = 6.337 * (np.abs(y12) + 0.664) ** 2 - 2.796
    mgi2 = y22 ** 2

    mgip = c * mgi1 + (1 - c) * mgi2

    sdg = std_glucose(data)
    sdgi = 1.375 ** sdg - 0.562

    g = mgip / (mgip + sdgi)
    return g * mgip + (1 - g) * sdgi


def int_hypo(data, th=70.):
    """
      Computes the intensity of hypoglycemia (IntHypo), the length of the vector of the per-day AUC below `th`
      (in mg/dl*min) and the per-day time below `th` (in min).
      Parameters
      ----------
      data: pd.DataFrame
          Pandas dataframe with a column `glucose` containing the glucose data
          to analyze (in mg/dl)
      th: float, optional, default: 70
          Hypoglycemia threshold (in mg/dl).

      Returns
      -------
      IntHypo

      References
      ----------
      Thomas et al. "The Development of New Composite Metrics for the Comprehensive Analytic and Visual
      Assessment of Hypoglycemia Using the Hypo-Triad." J Diabetes Sci Technol.
      2018;12(1):69-75. doi:10.1177/1932296817721242
    """

    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0:
        return np.nan

    # The sampling time needs at least two samples
    if data.t.values.size < 2:
        return np.nan

    # Per-day AUC below th [mg/dl*min] and time below th [min], over the observed time
    sample_time_minutes = data.t.diff().dt.total_seconds().dropna().iloc[0] / 60
    observed_days = values.size * sample_time_minutes / 1440
    auc = positive_auc_glucose(data.assign(glucose=-data.glucose), -float(th)) / observed_days
    min_hypo = time_in_given_below_range(data, th) / 100 * 1440

    return math.sqrt(auc ** 2 + min_hypo ** 2)


def int_hyper(data, th=180.):
    """
    Computes the intensity of hyperglycemia (IntHyper), the length of the vector of the per-day AUC above `th`
    (in mg/dl*min) and the per-day time above `th` (in min).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)
    th: float, optional, default: 180
        Hyperglycemia threshold (in mg/dl).

    Returns
    -------
    int_hyper: float
        IntHyper. It is nan without glucose values or with fewer than two samples.

    References
    ----------
    Vigersky et al. "The Comprehensive Glucose Pentagon: A Glucose-Centric Composite Metric for Assessing Glycemic
    Control in Persons With Diabetes." J Diabetes Sci Technol. 2018;12(1):114-123. doi:10.1177/1932296817718561
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0 or data.t.values.size < 2:
        return np.nan

    # Per-day AUC above th [mg/dl*min] and time above th [min], over the observed time
    sample_time_minutes = data.t.diff().dt.total_seconds().dropna().iloc[0] / 60
    observed_days = values.size * sample_time_minutes / 1440
    auc = positive_auc_glucose(data, float(th)) / observed_days
    min_hyper = time_in_given_above_range(data, th) / 100 * 1440

    return math.sqrt(auc ** 2 + min_hyper ** 2)


# Axes of the comprehensive glucose pentagon, clockwise from the top as in Vigersky et al. (2018), Figure 1
CGP_AXES = ['time_out_of_range', 'cv_glucose', 'int_hypo', 'int_hyper', 'mean_glucose']

# Healthy reference of Vigersky et al. (2018). CV 17% instead of 16.7% gives exactly the 466 mm^2 the paper reports
CGP_HEALTHY = {'time_out_of_range': 0., 'cv_glucose': 17., 'int_hypo': 0., 'int_hyper': 0., 'mean_glucose': 90.}


def comprehensive_glucose_pentagon(data, th_l=70., th_h=180.):
    """
    Computes the five axes of the comprehensive glucose pentagon (CGP).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)
    th_l: float, optional, default: 70
        Lower bound of the target range and hypoglycemia threshold (in mg/dl).
    th_h: float, optional, default: 180
        Upper bound of the target range and hyperglycemia threshold (in mg/dl).

    Returns
    -------
    cgp: dict
        Keys of `CGP_AXES`: time out of range (1440 min minus the per-day time in [th_l, th_h], in min),
        CV (in %), IntHypo, IntHyper and mean glucose (in mg/dl). Values are nan without glucose values or with
        fewer than two samples.

    References
    ----------
    Vigersky et al. "The Comprehensive Glucose Pentagon: A Glucose-Centric Composite Metric for Assessing Glycemic
    Control in Persons With Diabetes." J Diabetes Sci Technol. 2018;12(1):114-123. doi:10.1177/1932296817718561
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0 or data.t.values.size < 2:
        return {k: np.nan for k in CGP_AXES}

    tir = time_in_given_range(data, th_l, th_h, include_th_l=True, include_th_h=True)
    return {
        'time_out_of_range': (100 - tir) / 100 * 1440,
        'cv_glucose': cv_glucose(data),
        'int_hypo': int_hypo(data, th_l),
        'int_hyper': int_hyper(data, th_h),
        'mean_glucose': mean_glucose(data),
    }


def cgp_axis_lengths(cgp):
    """
    Converts the CGP axes to their lengths (in mm, out of 76) with the scaling formulae of Vigersky et al. (2018).
    Mean glucose below 90 mg/dl keeps the 14 mm of the healthy reference.

    Parameters
    ----------
    cgp: dict
        CGP axes, as returned by `comprehensive_glucose_pentagon`.

    Returns
    -------
    lengths: np.ndarray
        Axis lengths in the order of `CGP_AXES`.
    """
    return np.array([
        (cgp['time_out_of_range'] * 0.00614) ** 1.581 + 14,
        (cgp['cv_glucose'] - 17) * 0.92 + 14,
        np.exp(cgp['int_hypo'] * 0.00057) + 13,
        (cgp['int_hyper'] * 0.000115) ** 1.51 + 14,
        (max(cgp['mean_glucose'] - 90, 0) * 0.0217) ** 2.63 + 14,
    ])


def _pentagon_area(lengths):
    # Sum of the five 72-degree triangles between adjacent axes
    return 0.5 * np.sin(np.radians(72)) * np.sum(lengths * np.roll(lengths, -1))


def pgr(data, th_l=70., th_h=180.):
    """
    Computes the prognostic glycemic risk (PGR), the area of the comprehensive glucose pentagon divided by the area
    of the healthy reference (466 mm^2). Vigersky et al. (2018) read it as very low risk up to 2, low up to 3,
    moderate up to 4, high up to 4.5 and extremely high above.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)
    th_l: float, optional, default: 70
        Lower bound of the target range and hypoglycemia threshold (in mg/dl).
    th_h: float, optional, default: 180
        Upper bound of the target range and hyperglycemia threshold (in mg/dl).

    Returns
    -------
    pgr: float
        The PGR. It is nan without glucose values or with fewer than two samples.

    References
    ----------
    Vigersky et al. "The Comprehensive Glucose Pentagon: A Glucose-Centric Composite Metric for Assessing Glycemic
    Control in Persons With Diabetes." J Diabetes Sci Technol. 2018;12(1):114-123. doi:10.1177/1932296817718561
    """
    cgp = comprehensive_glucose_pentagon(data, th_l, th_h)
    if np.isnan(cgp['mean_glucose']):
        return np.nan
    return _pentagon_area(cgp_axis_lengths(cgp)) / _pentagon_area(cgp_axis_lengths(CGP_HEALTHY))

def hypo_rv(data):
    """
      Computes the hypo_rv value from the data
      Parameters
      ----------
      data: pd.DataFrame
          Pandas dataframe with a column `glucose` containing the glucose data
          to analyze (in mg/dl)

      Returns
      -------
      hypo_rv

      References
      ----------
      Thomas et al. "The Development of New Composite Metrics for the Comprehensive Analytic and Visual
      Assessment of Hypoglycemia Using the Hypo-Triad." J Diabetes Sci Technol.
      2018;12(1):69-75. doi:10.1177/1932296817721242
    """

    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0:
        return np.nan

    inthypo = int_hypo(data)
    episodes_per_day = find_hypoglycemic_events(data)['events_per_week'] / 7

    return inthypo * episodes_per_day

