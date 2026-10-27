import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.signal import find_peaks
from datetime import timedelta

from py_agata.inspection import find_hypoglycemic_events
from py_agata.time_in_ranges import time_in_target, time_in_hypoglycemia, time_in_hyperglycemia, \
    time_in_given_above_range
from py_agata.input_validator import *
from py_agata.utils import to_mmol_l
from py_agata.variability import mean_glucose, modd, std_glucose, positive_auc_glucose, ea1c


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


def glucose_pentagon(data, a1c=None):
    """
       Computes the components of the glucose pentagon
       Parameters
       ----------
       data: pd.DataFrame
           Pandas dataframe with a column `glucose` containing the glucose data
           to analyze (in mg/dl)
       a1c: float
           HbA1c of the patient

       Returns
       -------
       m, sd, tar, auc160, a1c: various parameters of the glucose pentagon

       References
       ----------
       Thomas et al. "The “Glucose Pentagon”: Assessing Glycemic Control of Patients with Diabetes
       Mellitus by a Model Integrating Different Parameters from Glucose Profiles" (2009) Diabetes Technology & Therapeutics,
       11(6), 399–409. doi:10.1089/dia.2008.0119
       """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if values.size == 0:
        return np.nan

    m = mean_glucose(data)
    sd = std_glucose(data)
    tar = time_in_given_above_range(data, th=160.) * 24 / 100  # hours
    sample_time_minutes = data.t.diff().dt.total_seconds().dropna().iloc[0] / 60
    n_days = data.shape[0] * sample_time_minutes / 1440
    auc160 = positive_auc_glucose(data, 160.) / 1440 / n_days  # mg/dl*24h per day
    if a1c == None:
        warnings.warn("The A1C value was not provided, proceeding by estimating it from the data.")
        a1c = ea1c(data)

    return m, sd, tar, auc160, a1c


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


def int_hypo(data):
    """
      Computes the IntHypo value from the data
      Parameters
      ----------
      data: pd.DataFrame
          Pandas dataframe with a column `glucose` containing the glucose data
          to analyze (in mg/dl)

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

    # Per-day AUC below 70 mg/dl [mg/dl*min] and time below 70 mg/dl [min], over the observed time
    sample_time_minutes = data.t.diff().dt.total_seconds().dropna().iloc[0] / 60
    observed_days = values.size * sample_time_minutes / 1440
    auc = positive_auc_glucose(data.assign(glucose=-data.glucose), -70) / observed_days
    min_hypo = time_in_hypoglycemia(data) / 100 * 1440

    return math.sqrt(auc ** 2 + min_hypo ** 2)

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

