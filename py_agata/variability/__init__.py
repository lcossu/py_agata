from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.signal import find_peaks
from datetime import timedelta

from py_agata.input_validator import *
from py_agata.time_in_ranges import time_in_target, time_in_hypoglycemia


def mean_glucose(data):
    """
    Computes the mean glucose level (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mean_glucose: float
        The mean glucose level.

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
    Wikipedia on mean: https://en.wikipedia.org/wiki/Mean (Accessed: 2020-12-10).
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

    # Return the result
    return np.mean(values)


def median_glucose(data):
    """
    Computes the median glucose level (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mean_glucose: float
        The median glucose level.

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
    Wikipedia on median: https://en.wikipedia.org/wiki/Median (Accessed: 2020-12-10).
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

    # Return the result
    return np.median(values)


def std_glucose(data):
    """
    Computes the std glucose level (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    std_glucose: float
        The std glucose level.

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
    Wikipedia on standard deviation: https://en.wikipedia.org/wiki/Standard_deviation (Accessed: 2020-12-10).
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

    # Return the result
    return np.std(values, ddof=1)


def cv_glucose(data):
    """
    Computes the coefficient of variation of glucose (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    cv_glucose: float
        The cv of glucose.

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
    Wikipedia on coefficient of variation: https://en.wikipedia.org/wiki/Coefficient_of_variation (Accessed: 2020-12-10).
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Return the result
    return 100 * std_glucose(data) / mean_glucose(data)


def range_glucose(data):
    """
    Computes the spanned glucose range (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    range: float
        The range of glucose.

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
    Wikipedia on range: https://en.wikipedia.org/wiki/Range_(statistics) (Accessed: 2020-12-10).
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Return the result
    if (data.glucose.values.size == 0):
        return np.nan
    return np.nanmax(data.glucose.values) - np.nanmin(data.glucose.values)


def iqr_glucose(data):
    """
    Computes the interquartile range of glucose values (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    iqr_glucose: float
        The interquartile range of glucose.

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
    Wikipedia on IQR: https://en.wikipedia.org/wiki/Interquartile_range (Accessed: 2020-12-10).
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get rid of nans
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return the result
    return iqr(values)


def auc_glucose_over_basal(data, basal):
    """
    Computes the area under the glucose curve using a given basal offset (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    auc_glucose_over_basal: float
        The area under the glucose curve.

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
    None
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_float_parameter(basal)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Shift the trace
    values = values - basal

    # Get ts
    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    ts = (t1 - t0).total_seconds() / 60

    # Return the result
    return np.sum(values * ts)


def positive_auc_glucose(data, basal):
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get non-nan values
    values = data.glucose.values[~np.isnan(data.glucose.values)]

    # Return nan if all values are nan
    if values.size == 0:
        return np.nan

    # Shift the trace
    values = values - basal

    values = np.array([x if x > 0 else 0 for x in values])

    # Get ts
    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    ts = (t1 - t0).total_seconds() / 60

    # Return the result
    return np.sum(values * ts)


def auc_glucose(data):
    """
    Computes the area under the glucose curve (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    auc_glucose: float
        The area under the glucose curve.

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
    None
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Return results
    return auc_glucose_over_basal(data, 0.)


def gmi(data):
    """
    Computes the glucose management indicator of the given data (ignoring nan values).
    It should be computed only if more than 12 days are available.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    gmi: float
        The glucose management indicator of the given data.

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
    Bergenstal et al., "Glucose Management Indicator (GMI): A new term
    for estimating A1C from continuous glucose monitoring", Diabetes Care,
    2018, vol. 41, pp. 2275-2280. DOI: 10.2337/dc18-1581.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Return results
    return 3.31 + 0.02392 * mean_glucose(data)


def cogi(data):
    """
    Computes the Continuous Glucose Monitoring Index (COGI) of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    cogi: float
        The Continuous Glucose Monitoring Index (COGI) of the given data.

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
    Leelaranthna et al., "Evaluating glucose control with a novel composite
    Continuous Glucose Monitoring Index", Journal of Diabetes Science and Technology,
    2019, vol. 14, pp. 277-283. DOI: 10.1177/1932296819838525.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Compute TIR component
    tir = time_in_target(data) * 0.5

    # Compute TBR component
    tbr = np.min([15, time_in_hypoglycemia(data)])
    tbr = (100 - 100 / 15 * tbr) * 0.35

    # Compute GV component
    gv = np.min([np.max([std_glucose(data) / 18.018, 1]), 6])
    gv = (120 - 20 * gv) * 0.15

    # Return results
    return tir + tbr + gv


def conga(data):
    """
    Computes the Continuous Overall Net Glycemic Action (CONGA) of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    conga: float
        The Continuous Overall Net Glycemic Action (CONGA) of the given data.

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
    McDonnell et al., "A novel approach to continuous glucose analysis
    utilizing glycemic variation", Diabetes Technol Ther, 2005, vol. 7,
    pp. 253–263. DOI: 10.1089/dia.2005.7.253.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Set the CONGAOrd hyperparameter to 4 (number of hours in the past it
    # refers to)
    conga_ord = 4

    # Build vectors
    n = data.glucose.values.size
    dc = np.empty(shape=(0,))

    for i in range(1, n):

        # Find the index referring to conga_ord hours ago
        j = np.where(data.t <= (pd.to_datetime(data.t.values[i]).to_pydatetime() - timedelta(hours=conga_ord)))[0]

        if not j.size == 0:
            j = j[-1]
            dc = np.append(dc, data.glucose.values[i] - data.glucose[j])

    # Return results
    if dc.size == 0:
        return np.nan
    else:
        return np.nanstd(dc, ddof=1)


def j_index(data):
    """
    Computes the J-Index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    j_index: float
        The J-Index of the given data.

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
    - Wójcicki, "J-index. A new proposition of the assessment of current
    glucose control in diabetic patients", Hormone and Metabolic Reseach,
    1995, vol. 27, pp. 41-42. DOI: 10.1055/s-2007-979906.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    return 1e-3 * (mean_glucose(data) + std_glucose(data)) ** 2


def mage_plus_index(data):
    """
    Computes the mean amplitude of positive glycemic excursion (MAGE+) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mage_plus_index: float
        The mean amplitude of positive glycemic excursion (MAGE+) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Get the first and last day limits
    if data.t.values.size == 0:
        return np.nan

    first_day = pd.to_datetime(data.t.values[0]).to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = pd.to_datetime(data.t.values[-1]).to_pydatetime()
    last_day = last_day.__add__(timedelta(days=1)).replace(hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    mage_day_plus = np.empty(shape=(n_days,))

    for d in range(0, n_days):

        # Step 0: parameters

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get glucose values (might be nan)
        std_within = np.nanstd(day_data, ddof=1)
        n = day_data.size

        if n > 3:
            # Step 1: turning points are only local extrema
            i_max = find_peaks(day_data)[0]
            i_min = find_peaks(-day_data)[0]
            i_turning = np.union1d([0, n - 1], np.union1d(i_max, i_min)).astype(int)

            turning = day_data[i_turning]
            n_turning = i_turning.size

            # Step 2: Turning points of no interest are removed
            # A turning point is removed if it's not significantly different from
            # BOTH its left and right-hand side RETAINED neighbours.

            to_be_kept = [True] * n_turning

            for i in range(1, n_turning - 1):  # First and last samples are retained

                condition_1 = abs(turning[i] - turning[i - 1]) < std_within
                condition_2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not (condition_1 and condition_2)

            i_turning = i_turning[to_be_kept]

            # Step 3: Turning points are removed again or moved appropriately
            i = 1
            while i < len(i_turning) - 1:
                prev = i_turning[i - 1]
                curr = i_turning[i]
                next = i_turning[i + 1]
                prev_slope = day_data[curr] - day_data[prev]
                next_slope = day_data[next] - day_data[curr]

                if prev_slope < 0 and next_slope > 0:  # Minimum
                    # The actual current turning point is the min in the interval
                    temp = np.nanargmin(day_data[prev:(next + 1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the max to the left of the current turning point.
                    temp = np.nanargmax(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the max to the right of the current turning point.
                    temp = np.nanargmax(day_data[(curr + 1):(next + 1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                elif prev_slope > 0 and next_slope < 0:  # Maximum
                    # The actual current turning point is the max in the interval
                    temp = np.nanargmax(day_data[prev:(next + 1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the min to the left of the current turning point.
                    temp = np.nanargmin(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the min to the right of the current turning point.
                    temp = np.nanargmin(day_data[(curr + 1):(next + 1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                else:  # Middle point
                    i_turning = np.delete(i_turning, i)

            # Step 4: Remove residual spurious turning points.
            # Turning points not significantly different from EITHER neighbour are
            # removed. Some extra processing is needed for the first and last sample.
            sample1 = day_data[i_turning[0]]
            sample2 = day_data[i_turning[1]]

            if abs(sample2 - sample1) < std_within:
                i_turning = np.delete(i_turning, 0)

            if len(i_turning) > 1:
                # Last sample processing
                sample1 = day_data[i_turning[-2]]
                sample2 = day_data[i_turning[-1]]
                if abs(sample2 - sample1) < std_within:
                    i_turning = np.delete(i_turning, len(i_turning) - 1)

            turning = day_data[i_turning]
            n_turning = len(i_turning)

            # Internal points
            to_be_kept = np.ones(n_turning, dtype=bool)
            for i in range(1, n_turning - 1):
                condition1 = abs(turning[i] - turning[i - 1]) < std_within
                condition2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not (condition1 or condition2)

            i_turning = i_turning[to_be_kept]
            turning = day_data[i_turning]

            # Step 5: Compute daily MAGE+
            excursions = np.diff(turning)
            mage_day_plus[d] = np.nanmean(excursions[excursions > 0])
        else:
            mage_day_plus[d] = np.nan

    # Compute index
    mage_day_plus[np.isnan(mage_day_plus)] = 0  # Correct for 'mean' behavior
    mpi = np.mean(mage_day_plus)

    # Manage all nan data
    if np.all(np.isnan(data['glucose'])):
        mpi = np.nan

    return mpi


def mage_minus_index(data):
    """
    Computes the mean amplitude of negative glycemic excursion (MAGE-) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mage_minus_index: float
        The mean amplitude of negative glycemic excursion (MAGE+) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size == 0:
        return np.nan
    # Get the first and last day limits
    first_day = pd.to_datetime(data.t.values[0]).to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = pd.to_datetime(data.t.values[-1]).to_pydatetime()
    last_day = last_day.__add__(timedelta(days=1)).replace(hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    mage_day_minus = np.empty(shape=(n_days,))

    for d in range(0, n_days):

        # Step 0: parameters

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get glucose values (might be nan)
        std_within = np.nanstd(day_data, ddof=1)
        n = day_data.size

        if n > 3:
            # Step 1: turning points are only local extrema
            i_max = find_peaks(day_data)[0]
            i_min = find_peaks(-day_data)[0]
            i_turning = np.union1d([0, n - 1], np.union1d(i_max, i_min)).astype(int)

            turning = day_data[i_turning]
            n_turning = i_turning.size

            # Step 2: Turning points of no interest are removed
            # A turning point is removed if it's not significantly different from
            # BOTH its left and right-hand side RETAINED neighbours.

            to_be_kept = [True] * n_turning

            for i in range(1, n_turning - 1):  # First and last samples are retained

                condition_1 = abs(turning[i] - turning[i - 1]) < std_within
                condition_2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not (condition_1 and condition_2)

            i_turning = i_turning[to_be_kept]

            # Step 3: Turning points are removed again or moved appropriately
            i = 1
            while i < len(i_turning) - 1:
                prev = i_turning[i - 1]
                curr = i_turning[i]
                next = i_turning[i + 1]
                prev_slope = day_data[curr] - day_data[prev]
                next_slope = day_data[next] - day_data[curr]

                if prev_slope < 0 and next_slope > 0:  # Minimum
                    # The actual current turning point is the min in the interval
                    temp = np.nanargmin(day_data[prev:(next + 1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the max to the left of the current turning point.
                    temp = np.nanargmax(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the max to the right of the current turning point.
                    temp = np.nanargmax(day_data[(curr + 1):(next + 1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                elif prev_slope > 0 and next_slope < 0:  # Maximum
                    # The actual current turning point is the max in the interval
                    temp = np.nanargmax(day_data[prev:(next + 1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the min to the left of the current turning point.
                    temp = np.nanargmin(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the min to the right of the current turning point.
                    temp = np.nanargmin(day_data[(curr + 1):(next + 1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                else:  # Middle point
                    i_turning = np.delete(i_turning, i)

            # Step 4: Remove residual spurious turning points.
            # Turning points not significantly different from EITHER neighbour are
            # removed. Some extra processing is needed for the first and last sample.
            sample1 = day_data[i_turning[0]]
            sample2 = day_data[i_turning[1]]

            if abs(sample2 - sample1) < std_within:
                i_turning = np.delete(i_turning, 0)

            if len(i_turning) > 1:
                # Last sample processing
                sample1 = day_data[i_turning[-2]]
                sample2 = day_data[i_turning[-1]]
                if abs(sample2 - sample1) < std_within:
                    i_turning = np.delete(i_turning, len(i_turning) - 1)

            turning = day_data[i_turning]
            n_turning = len(i_turning)

            # Internal points
            to_be_kept = np.ones(n_turning, dtype=bool)
            for i in range(1, n_turning - 1):
                condition1 = abs(turning[i] - turning[i - 1]) < std_within
                condition2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not (condition1 or condition2)

            i_turning = i_turning[to_be_kept]
            turning = day_data[i_turning]

            # Step 5: Compute daily MAGE+
            excursions = np.diff(turning)
            mage_day_minus[d] = np.nanmean(excursions[excursions < 0])
        else:
            mage_day_minus[d] = np.nan

    # Compute index
    mage_day_minus[np.isnan(mage_day_minus)] = 0  # Correct for 'mean' behavior
    mmi = -np.mean(mage_day_minus)

    # Manage all nan data
    if np.all(np.isnan(data['glucose'])):
        mmi = np.nan

    return mmi


def mage_index(data):
    """
    Computes the mean amplitude of glycemic excursion (MAGE) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    mage_index: float
        The mean amplitude of glycemic excursion (MAGE) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if np.isnan(mage_plus_index(data)) and np.isnan(mage_minus_index(data)):
        return np.nan
    return np.nanmean([mage_plus_index(data), mage_minus_index(data)])


def ef_index(data):
    """
    Computes the excursion frequency (EF) index of the given data, i.e., the number of excursion > 75 (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    ef_index: float
        The excursion frequency (EF) index of the given data.

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
    - Service et al., "Mean amplitude of glycemic excursions, a measure of
    diabetic instability", Diabetes, 1970, vol. 19, pp. 644-655. DOI:
    10.2337/diab.19.9.644.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size == 0:
        return np.nan

    # Set the fixed parameter
    ef_th = 75

    # Get the first and last day limits
    first_day = pd.to_datetime(data.t.values[0]).to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = pd.to_datetime(data.t.values[-1]).to_pydatetime()
    last_day = last_day.__add__(timedelta(days=1)).replace(hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    ef_day = np.empty(shape=(n_days,))

    for d in range(0, n_days):

        # Step 0: parameters

        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get glucose values (might be nan)
        std_within = np.nanstd(day_data, ddof=1)
        n = day_data.size

        if n > 3:
            # Step 1: turning points are only local extrema
            i_max = find_peaks(day_data)[0]
            i_min = find_peaks(-day_data)[0]
            i_turning = np.union1d([0, n - 1], np.union1d(i_max, i_min)).astype(int)

            turning = day_data[i_turning]
            n_turning = i_turning.size

            # Step 2: Turning points of no interest are removed
            # A turning point is removed if it's not significantly different from
            # BOTH its left and right-hand side RETAINED neighbours.

            to_be_kept = [True] * n_turning

            for i in range(1, n_turning - 1):  # First and last samples are retained

                condition_1 = abs(turning[i] - turning[i - 1]) < std_within
                condition_2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not (condition_1 and condition_2)

            i_turning = i_turning[to_be_kept]

            # Step 3: Turning points are removed again or moved appropriately
            i = 1
            while i < len(i_turning) - 1:
                prev = i_turning[i - 1]
                curr = i_turning[i]
                next = i_turning[i + 1]
                prev_slope = day_data[curr] - day_data[prev]
                next_slope = day_data[next] - day_data[curr]

                if prev_slope < 0 and next_slope > 0:  # Minimum
                    # The actual current turning point is the min in the interval
                    temp = np.nanargmin(day_data[prev:(next + 1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the max to the left of the current turning point.
                    temp = np.nanargmax(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the max to the right of the current turning point.
                    temp = np.nanargmax(day_data[(curr + 1):(next + 1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                elif prev_slope > 0 and next_slope < 0:  # Maximum
                    # The actual current turning point is the max in the interval
                    temp = np.nanargmax(day_data[prev:(next + 1)]) + prev
                    curr = temp
                    i_turning[i] = curr
                    # The actual previous turning point is the min to the left of the current turning point.
                    temp = np.nanargmin(day_data[prev:curr]) + prev
                    i_turning[i - 1] = temp
                    # The actual following turning point is the min to the right of the current turning point.
                    temp = np.nanargmin(day_data[(curr + 1):(next + 1)]) + curr + 1
                    i_turning[i + 1] = temp

                    i += 1
                else:  # Middle point
                    i_turning = np.delete(i_turning, i)

            # Step 4: Remove residual spurious turning points.
            # Turning points not significantly different from EITHER neighbour are
            # removed. Some extra processing is needed for the first and last sample.
            sample1 = day_data[i_turning[0]]
            sample2 = day_data[i_turning[1]]

            if abs(sample2 - sample1) < std_within:
                i_turning = np.delete(i_turning, 0)

            if len(i_turning) > 1:
                # Last sample processing
                sample1 = day_data[i_turning[-2]]
                sample2 = day_data[i_turning[-1]]
                if abs(sample2 - sample1) < std_within:
                    i_turning = np.delete(i_turning, len(i_turning) - 1)

            turning = day_data[i_turning]
            n_turning = len(i_turning)

            # Internal points
            to_be_kept = np.ones(n_turning, dtype=bool)
            for i in range(1, n_turning - 1):
                condition1 = abs(turning[i] - turning[i - 1]) < std_within
                condition2 = abs(turning[i + 1] - turning[i]) < std_within
                to_be_kept[i] = not (condition1 or condition2)

            i_turning = i_turning[to_be_kept]
            turning = day_data[i_turning]

            # Step 5: Compute daily MAGE+
            excursions = np.diff(turning)
            ef_day[d] = np.where(abs(excursions) > ef_th)[0].size
        else:
            ef_day[d] = np.nan

    # Compute index
    ei = np.nansum(ef_day) / n_days

    # Manage all nan data
    if np.all(np.isnan(data['glucose'])):
        ei = np.nan

    return ei


def modd(data):
    """
    Computes the mean of daily differences (MODD) of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    modd: float
        The mean of daily differences (MODD) of the given data.

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
    - Molnar et al., "Day-to-day variation of continuously monitored
    glycaemia: a further measure of diabetic instability", Diabetologia,
    1972, vol. 8, pp. 342–348. DOI: 10.1007/BF01218495.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    # Build vectors
    yesterday = timedelta(minutes=1440)

    n = len(data)

    Dm = []

    for i in range(1, n):

        # Find the index referring to the same time yesterday
        j = np.where(data.t <= (pd.to_datetime(data.t.values[i]).to_pydatetime() - yesterday))[0]

        if j.size > 0:  # if there is a meaningful sample in data[j]
            j = j[-1]
            Dm.append(abs(data.glucose.values[i] - data.glucose.values[j]))

    if Dm:
        modd = np.nanmean(Dm)
    else:
        modd = np.nan
    return modd


def sddm_index(data):
    """
    Computes the standard deviation of within-day means (SDDM) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    sddm_index: float
        The standard deviation of within-day means (SDDM) index of the given data.

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
    - Rodbard et al., "New and improved methods to characterize glycemic
    variability using continuous glucose monitoring", Diabetes Technology &
    Therapeutics, 2009, vol. 11, pp. 551-565. DOI: 10.1089/dia.2009.0015.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size == 0:
        return np.nan
    # Get the first and last day limits
    first_day = pd.to_datetime(data.t.values[0]).to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = pd.to_datetime(data.t.values[-1]).to_pydatetime()
    last_day = last_day.__add__(timedelta(days=1)).replace(hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    mean_within = np.zeros(shape=(n_days,))

    for d in range(0, n_days):
        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get daily mean and std
        mean_within[d] = np.nanmean(day_data)

    # Compute index
    return np.nanstd(mean_within, ddof=1)


def sdw_index(data):
    """
    Computes the mean of within-day standard deviation (SDW) index of the given data (ignoring nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    sdw_index: float
        The mean of within-day standard deviation (SDW) index of the given data.

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
    - Rodbard et al., "New and improved methods to characterize glycemic
    variability using continuous glucose monitoring", Diabetes Technology &
    Therapeutics, 2009, vol. 11, pp. 551-565. DOI: 10.1089/dia.2009.0015.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size == 0:
        return np.nan
    # Get the first and last day limits
    first_day = pd.to_datetime(data.t.values[0]).to_pydatetime()
    first_day = first_day.replace(hour=0, minute=0, second=0)
    last_day = pd.to_datetime(data.t.values[-1]).to_pydatetime()
    last_day = last_day.__add__(timedelta(days=1)).replace(hour=0, minute=0, second=0)

    # Calculate the number of days and preallocate
    n_days = (last_day - first_day).days
    std_within = np.zeros(shape=(n_days,))

    for d in range(0, n_days):
        # Get the day of data
        low_limit = data.t >= (first_day + timedelta(days=d))
        high_limit = data.t < (first_day + timedelta(days=d + 1))
        flags = np.logical_and(low_limit, high_limit)
        day_data = data.glucose.values[flags]

        # Get daily mean and std
        std_within[d] = np.nanstd(day_data, ddof=1)

    # Compute index
    return np.nanmean(std_within)


def glucose_roc(data):
    """
    Computes the glucose rate-of-change (ROC) trace.
    As defined in the given reference, ROC at time t is defined as the difference
    between the glucose at time t and t-15 minutes divided by 15. By
    definition, the first two samples are always nan (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    glucose_roc: pd.Dataframe
        Dataframe with column `t` and `glucose_roc` containing the glucose ROC (in mg/dl/min).

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
    - Clarke et al., "Statistical Tools to Analyze Continuous Glucose
    Monitor Data", Diabetes Technol Ther, 2009,
    vol. 11, pp. S45-S54. DOI: 10.1089=dia.2008.0138.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    g_roc = np.empty(shape=(data.glucose.values.size,))
    g_roc.fill(np.nan)

    if g_roc.size > 4:

        for t in range(3, g_roc.size):
            g_roc[t] = (data.glucose.values[t] - data.glucose.values[t - 3]) / 15

    return pd.DataFrame(data={'t': data.t.values, 'glucose_roc': g_roc})


def std_glucose_roc(data):
    """
    Computes the standard of glucose rate-of-change of given data (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with a column `glucose` containing the glucose data
        to analyze (in mg/dl)

    Returns
    -------
    std_glucose_roc: float
        The standard of glucose rate-of-change of given data.

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
    - Clarke et al., "Statistical Tools to Analyze Continuous Glucose
    Monitor Data", Diabetes Technol Ther, 2009,
    vol. 11, pp. S45-S54. DOI: 10.1089=dia.2008.0138.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    roc = glucose_roc(data)

    return np.nanstd(roc.glucose_roc.values, ddof=1)


def cvga(glucose_profiles: pd.DataFrame | list[pd.DataFrame]):
    """
    Control Variability Grid Analysis (CVGA) as in Magni et al. (2008).

    Parameters
    ----------
    glucose_profiles : list of pd.DataFrame or pd.DataFrame
        Each DataFrame must have columns 'Time' and 'glucose' (in mg/dL).
        If a single DataFrame is provided, it is treated as a single profile.

    Returns
    -------
    profile_assessment : np.ndarray
        Euclidean distance squared from the CVGA origin for each profile.
    profile_coordinates : np.ndarray
        Array of shape (n_profiles, 2) with CVGA (x, y) coordinates for each profile.
    best_control_index : int
        Index of the best-controlled profile (minimum distance).
    best_controlled_coordinates : np.ndarray
        (x, y) coordinates of the best-controlled profile.
    """
    def cvga_coordinates(max_bg, min_bg):
        x = np.clip(110 - min_bg, 0, 60)
        p = np.polyfit([110, 180, 300, 400], [0, 20, 40, 60], 3)
        y = np.polyval(p, max_bg)
        return x, y

    # Accept single DataFrame or list
    if isinstance(glucose_profiles, pd.DataFrame):
        glucose_profiles = [glucose_profiles]

    n = len(glucose_profiles)
    min_bg = np.zeros(n)
    max_bg = np.zeros(n)
    profile_coordinates = np.zeros((n, 2))
    profile_assessment = np.zeros(n)

    for i, profile in enumerate(glucose_profiles):
        check_dataframe(profile)
        if 'glucose' not in profile.columns:
            raise ValueError('Each profile must have a glucose column.')
        glucose_values = profile['glucose'].values
        if glucose_values.size == 0 or np.all(np.isnan(glucose_values)):
            min_bg[i] = np.nan
            max_bg[i] = np.nan
        else:
            min_bg[i] = np.nanmin(glucose_values)
            max_bg[i] = np.nanmax(glucose_values)
        profile_coordinates[i, 0], profile_coordinates[i, 1] = cvga_coordinates(max_bg[i], min_bg[i])
        profile_assessment[i] = profile_coordinates[i, 0] ** 2 + profile_coordinates[i, 1] ** 2

    has_valid_assessment = n > 0 and not np.all(np.isnan(profile_assessment))
    best_control_index = int(np.nanargmin(profile_assessment)) if has_valid_assessment else None
    best_controlled_coordinates = (
        profile_coordinates[best_control_index] if best_control_index is not None else (np.nan, np.nan)
    )

    return profile_assessment, profile_coordinates, best_control_index, best_controlled_coordinates


def ea1c(data):
    """
        Computes the estimaned A1C value from CGM data

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe with a column `glucose` containing the glucose data
            to analyze (in mg/dl)

        Returns
        -------
        eA1C: float
            The standard of glucose rate-of-change of given data.

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
        - Nathan et al. "Translating the A1C assay into estimated average glucose values"
            Diabetes Care. 2008;31(8):1473-1478. doi:10.2337/dc08-0545
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size == 0:
        return np.nan
    return (46.7 + mean_glucose(data)) / 28.7


def gvp(data):
    """
        Computes the glycemic variability percentage from CGM data

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe with a column `glucose` containing the glucose data
            to analyze (in mg/dl)

        Returns
        -------
        GVP: float
            The GVP calculated from the data

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
        - Peyser et al. "Glycemic Variability Percentage: A Novel Method for Assessing Glycemic Variability from Continuous Glucose Monitor Data."
         Diabetes Technol Ther. 2018;20(1):6-16. doi:10.1089/dia.2017.0187
    """

    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size < 2:
        return np.nan

    dx = (data.t.iloc[1] - data.t.iloc[0]).total_seconds() / 60
    L0 = dx * len(data.t[~np.isnan(data.glucose.values)])
    dy = np.diff(data.glucose.values[~np.isnan(data.glucose.values)])
    L = np.sum(np.sqrt(dx ** 2 + dy ** 2))

    return (L / L0 - 1) * 100


def mag(data, n=60):
    """
            Computes the mean absolute glucose (MAG) as the sum of all absolute glucose changes during admission and dividing this by the total time

            Parameters
            ----------
            data: pd.DataFrame
                Pandas dataframe with a column `glucose` containing the glucose data
                to analyze (in mg/dl)
            n = int
                number of minutes of interval to consider for the computation; effectively a down-sampling of the data.
            Returns
            -------
            MAG: float
                The MAG calculated from the data

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
            - Hermanides et al. "Glucose variability is associated with intensive care unit mortality"
             Critical Care Medicine 38(3):p 838-842, March 2010. | DOI: 10.1097/CCM.0b013e3181cc4be9

        """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if data.t.values.size < 2:
        return np.nan
    dx = (data.t.iloc[1] - data.t.iloc[0]).total_seconds() / 60
    if n < dx:
        n = dx
        warnings.warn(
            f"The parameter n is required to be at least equal to the sampling time of the data. Proceeding to calculate with n={dx}")

    idx = np.arange(0, data.shape[0], step=round(n / dx))
    idx_gl = data.iloc[idx].glucose.to_numpy().flatten()

    return np.nansum(np.abs(np.diff(idx_gl))) / (np.count_nonzero(~np.isnan(idx_gl)) * n / 60)


def mad(data):
    """
                Computes the mean absolute distance (MAD) as the median of distances of glucose values from their median

                Parameters
                ----------
                data: pd.DataFrame
                    Pandas dataframe with a column `glucose` containing the glucose data
                    to analyze (in mg/dl)

                Returns
                -------
                MAD: float
                    The MAD calculated from the data

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
                - Millard et al. "GLU: a software package for analysing continuously measured glucose levels in epidemiology."
                 Int J Epidemiol. 2020;49(3):744-757. doi:10.1093/ije/dyaa004

            """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    median = median_glucose(data)
    return np.median(np.abs(data.glucose[~np.isnan(data.glucose)] - median))


def distance_traveled(data):
    """
                   Computes the distance traveled as sum of absolute difference between CGM samples

                   Parameters
                   ----------
                   data: pd.DataFrame
                       Pandas dataframe with a column `glucose` containing the glucose data
                       to analyze (in mg/dl)

                   Returns
                   -------
                   distance_travelled: float
                       The distance traveled calculated from the data

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
                   - Marling et al. "Characterizing blood glucose variability using new metrics with continuous glucose monitoring data."
                    J Diabetes Sci Technol. 2011;5(4):871-878. Published 2011 Jul 1. doi:10.1177/193229681100500408

               """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if np.all(np.isnan(data.glucose.values)):
        return np.nan

    dy = np.abs(np.diff(data.glucose.values))
    return np.nansum(dy)


def sample_entropy(time_series, m=2, r=0.15):
    """
    Computes the sample entropy of a time series.
    Parameters
    ----------
    time_series: np.ndarray
        1D array of data
    m: int
        Embedding dimension
    r: float
        Tolerance (as a proportion of the std of the data)
    Returns
    -------
    se: float
        Sample entropy
    """
    time_series = np.asarray(time_series)
    N = len(time_series)
    if N <= m + 1 or np.all(np.isnan(time_series)):
        return np.nan
    r *= np.nanstd(time_series)
    def _count_matches(m):
        # Pairs i < j of the windows starting at 0..N-m, skipping windows that contain nan
        windows = np.lib.stride_tricks.sliding_window_view(time_series, m)[:N - m + 1]
        valid = ~np.isnan(windows).any(axis=1)
        windows = windows[valid]
        count = 0
        for i in range(len(windows) - 1):
            count += np.count_nonzero(np.max(np.abs(windows[i + 1:] - windows[i]), axis=1) <= r)
        return count
    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0:
        return np.nan
    if A == 0:
        return -np.log(1.0 / (B + 1))  # add-one smoothing
    return -np.log(A / B)

def multiscale_entropy(data, m=2, r=0.15, max_scale=20):
    """
    Computes the Multi Scale Entropy (MSE) of the data
    Parameters
    ----------
    data: pd.DataFrame
          Pandas dataframe with a column `glucose` containing the glucose data
                       to analyze (in mg/dl)
    m: int
        Embedding dimension for sample entropy
    r: float
        Tolerance (as a proportion of the std of the data)
    max_scale: int
        Maximum scale factor
    Returns
    -------
    mse: float
        The calculated MSE of the data
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    values = data.glucose.values[~np.isnan(data.glucose.values)]
    if len(values) < m + 2:
        return np.nan
    mse_values = []
    for scale in range(1, max_scale + 1):
        if len(values) // scale < m + 2:
            break
        # Coarse-graining
        cg = np.mean(values[:len(values) // scale * scale].reshape(-1, scale), axis=1)
        se = sample_entropy(cg, m=m, r=r)
        mse_values.append(se)
    if len(mse_values) == 0:
        return np.nan
    return np.nanmean(mse_values)


