import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from copy import copy

from py_agata.input_validator import *

def rmse(data, data_hat):
    """
    Computes the root mean squared error (RMSE) between two glucose traces (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data

    Returns
    -------
    rmse: float
        The computed root mean squared error (mg/dl).

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    Wikipedia on RMSE: https://en.wikipedia.org/wiki/Root-mean-square_deviation. (Accessed: 2020-12-10)
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)

    if data.glucose.values.size == 0:
        return np.nan
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return np.nan
    return np.sqrt(np.mean((data.glucose.values[idxs] - data_hat.glucose.values[idxs]) ** 2))


def mard(data, data_hat):
    """
    Computes the mean absolute relative difference (MARD) between two glucose traces (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data

    Returns
    -------
    mard: float
        The computed MARD (%).

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    Gini, "Measurement of Inequality and Incomes", The Economic Journal,
    vol. 31, 1921, pp. 124–126. DOI:10.2307/2223319.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)

    if data.glucose.values.size == 0:
        return np.nan
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return np.nan
    return 100 * np.mean(np.abs(np.divide(data.glucose.values[idxs] - data_hat.glucose.values[idxs],data.glucose.values[idxs])))


def cod(data, data_hat):
    """
    Computes the coefficient of determination (COD) between two glucose traces (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data

    Returns
    -------
    cod: float
        The computed COD (%).

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    Wright, "Correlation and causation", Journal of Agricultural Research, vol. 20, 1921, pp. 557–585.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)

    if data.glucose.values.size == 0:
        return np.nan
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return np.nan
    residuals = data.glucose.values[idxs] - data_hat.glucose.values[idxs]
    return 100 * (1 - np.linalg.norm(residuals,ord=2)**2 / np.linalg.norm(data.glucose.values[idxs] - np.mean(data.glucose.values[idxs]),ord=2)**2)

def clarke(data, data_hat):
    """
    Computes the Clarke Error Grid Analysis (CEGA) (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data

    Returns
    -------
    clarke: dict
        A dictionary containing the results of the CEGA with fields:
        - a: float
            The percentage of time spent in Zone A
        - b: float
            The percentage of time spent in Zone B
        - c: float
            The percentage of time spent in Zone C
        - d: float
            The percentage of time spent in Zone D
        - e: float
            The percentage of time spent in Zone E

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    Clarke et al., "Evaluating clinical accuracy of systems for self-monitoring
    of blood glucose", Diabetes Care, 1987, vol. 10, pp. 622–628. DOI: 10.2337/diacare.10.5.622.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)

    results = dict()
    results["a"] = np.nan
    results["b"] = np.nan
    results["c"] = np.nan
    results["d"] = np.nan
    results["e"] = np.nan
    if data.glucose.values.size == 0:
        return results
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return results

    y = data.glucose.values[idxs]
    yp = data_hat.glucose.values[idxs]

    total = np.zeros(shape=(5,))
    n = y.size

    for i in range(n):

        if (yp[i] <= 70 and y[i] <= 70) or (yp[i] <= 1.2*y[i] and yp[i] >= 0.8*y[i]):
            total[0] += 1
        elif (y[i] >= 180 and yp[i] <= 70) or (y[i] <= 70 and yp[i] >= 180):
            total[4] += 1
        elif ((y[i] >= 70 and y[i] <= 290) and (yp[i] >= y[i] + 110)) or ( (y[i] >= 130 and y[i] <= 180) and (yp[i] <= 7/5*y[i] - 182)):
            total[2] += 1
        elif (y[i] >= 240 and (yp[i] >= 70 and yp[i] <= 180)) or (y[i] <= 175/3 and yp[i] <= 180 and yp[i] >= 70) or ((y[i] >= 175/3 and y[i] <= 70) and yp[i] >= 6/5*y[i]):
            total[3] += 1
        else:
            total[1] += 1

    total = total/n*100
    results["a"] = total[0]
    results["b"] = total[1]
    results["c"] = total[2]
    results["d"] = total[3]
    results["e"] = total[4]

    return results


# DTS Error Grid borders (Klonoff et al. 2024, Supplemental Table A1), as (reference, monitor) in mg/dL.
# Lower border: vertical at reference = x_lower up to monitor 50, then a line to (600, y_600).
# Upper border: horizontal at monitor = y_upper up to reference 50, then a line to (x_600, 600).
# Rows are the outer borders of zones A, B, C, D, in order.
_DTS_BORDERS = (
    # x_lower, y_600, y_upper, x_600
    (62.5, 480, 60, 500),
    (97.5, 307, 86.5, 347),
    (153, 197, 124, 241),
    (238, 126, 179, 167),
)


def dts_error_grid(data, data_hat):
    """
    Computes the Diabetes Technology Society (DTS) Error Grid analysis (ignores nan values).

    Points lying exactly on a border are assigned to the lower-risk zone.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data (reference)
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data (monitor)

    Returns
    -------
    dts: dict
        A dictionary containing the results of the DTS Error Grid analysis with fields:
        - a: float
            The percentage of time spent in Zone A (no risk)
        - b: float
            The percentage of time spent in Zone B (mild risk)
        - c: float
            The percentage of time spent in Zone C (moderate risk)
        - d: float
            The percentage of time spent in Zone D (high risk)
        - e: float
            The percentage of time spent in Zone E (extreme risk)

    Raises
    ------
    None

    See Also
    --------
    clarke

    Examples
    --------
    None

    References
    ----------
    Klonoff et al., "The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for Glucose
    Monitors", Journal of Diabetes Science and Technology, 2024. DOI: 10.1177/19322968241275701.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)

    results = {zone: np.nan for zone in ("a", "b", "c", "d", "e")}
    if data.glucose.values.size == 0:
        return results
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return results

    y = data.glucose.values[idxs].astype(float)
    yp = data_hat.glucose.values[idxs].astype(float)

    # Zone index = outermost border the point lies strictly outside of
    zone = np.zeros(y.size, dtype=int)
    for k, (x_lower, y_600, y_upper, x_600) in enumerate(_DTS_BORDERS):
        lower = 50 + (y - x_lower) * (y_600 - 50) / (600 - x_lower)
        upper = np.maximum(y_upper, y_upper + (y - 50) * (600 - y_upper) / (x_600 - 50))
        outside = ((y > x_lower) & (yp < lower)) | (yp > upper)
        zone[outside] = k + 1

    counts = np.bincount(zone, minlength=5) / y.size * 100
    for k, z in enumerate(("a", "b", "c", "d", "e")):
        results[z] = counts[k]

    return results


def c2_sigmoid(x_vector, a, d, type):

    if a.size == 1:
        a = a*np.ones(shape=(x_vector.size,))

    # Auxiliary function
    y = np.zeros(shape=(x_vector.size,))

    for i in range(x_vector.size):

        x = x_vector[i]

        if type == '>=':
            x_i = 2 / d * (x - a[i] - d/2)
        elif type == '<=':
            x_i = 2 / d * (x - a[i] + d / 2)

        if x_i <= -1:
            y[i] = 0
        elif x_i >= 1:
            y[i] = 1
        elif x_i <= 0:
            y[i] = 0.5 * (-(x_i**4) - 2*(x_i**3) + 2*x_i + 1)
        else:
            y[i] = 0.5 * (x_i ** 4 - 2 * (x_i ** 3) + 2 * x_i + 1)

    return y

def g_rmse(data, data_hat):
    """
    Computes the glucose root mean squared erro (gRMSE) (ignores nan values).

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data

    Returns
    -------
    g_rmse: float
        The computed glucose root mean squared error (mg/dl).

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    Del Favero et al., "A glucose-specific metric to assess predictors and
    identify models", IEEE Transactions on Biomedical Engineering, 2012,
    vol. 59, pp. 1281-1290. DOI: 10.1109/TBME.2012.2185234.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)

    if data.glucose.values.size == 0:
        return np.nan
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return np.nan

    y = data.glucose.values[idxs]
    yp = data_hat.glucose.values[idxs]

    # Parameters
    alpha_l = 1.5
    alpha_h = 1
    d_l1 = 10
    d_l2 = 30
    d_h1 = 20
    d_h2 = 100

    # Compute the cost
    term_l = alpha_l * c2_sigmoid(y, yp, d_l1,'<=') * c2_sigmoid(y, np.array([80]),d_l2,'<=')
    term_h = alpha_h * c2_sigmoid(y, yp, d_h1, '>=') * c2_sigmoid(y, np.array([250]), d_h2, '>=')
    error_grid_inspired_cost = 1 + term_l + term_h

    # Compute the quadratic cost function
    quadratic_cost = (y - yp)*(y - yp)

    # Compute the gMSE
    g_mse = np.nanmean(quadratic_cost * error_grid_inspired_cost)

    # Compute the gRMSE
    return np.sqrt(g_mse)


def time_delay(data, data_hat, ph):
    """
    Computes the delay of a predicted glucose trace.
    The time delay is computed as the time shift necessary to maximize the
    correlation between the two traces.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe, numpy array or float containing the glucose data
    data_hat: pd.DataFrame
        Pandas dataframe, numpy array or float containing the inferred glucose data
    ph: int

    Returns
    -------
    time_delay: int
        The computed delay (min)

    Raises
    ------
    None

    See Also
    --------
    None
Ï
    Examples
    --------
    None

    References
    ----------
    Del Favero et al., "A glucose-specific metric to assess predictors and
    identify models", IEEE Transactions on Biomedical Engineering, 2012,
    vol. 59, pp. 1281-1290. DOI: 10.1109/TBME.2012.2185234.
    """
    # Check input
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)
    check_dataframe(data_hat)
    check_data_columns(data_hat)
    check_homogeneous_timegrid(data_hat)
    check_comparable_data(data, data_hat)
    check_int_parameter(ph)

    if data.glucose.values.size == 0:
        return np.nan
    idxs = np.where(np.logical_and(~np.isnan(data.glucose.values), ~np.isnan(data_hat.glucose.values)))[0]
    if idxs.size == 0:
        return np.nan

    y = data.glucose.values[idxs]
    yp = data_hat.glucose.values[idxs]

    t0 = pd.to_datetime(data.t.values[0]).to_pydatetime()
    t1 = pd.to_datetime(data.t.values[1]).to_pydatetime()
    sample_time = int((t1 - t0).total_seconds() / 60)

    errors = np.zeros(shape=(int(ph/sample_time)+1,))

    for steps in range(errors.size):
            errors[steps] = np.sqrt(np.mean((y - yp) ** 2))
            yp = yp[1:]
            y = y[0:-1]

    idx_min_error = np.where(np.min(errors)==errors)[0][0]
    return ph - sample_time*idx_min_error
