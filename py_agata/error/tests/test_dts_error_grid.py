import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_agata.error import dts_error_grid


def test_dts_error_grid():
    """
    Unit test of dts_error_grid function.

    Parameters
    ----------
    None

    Returns
    -------
    None

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
    # Set test data: (reference, monitor) pairs and their expected zone
    # (40, 10) A: both values low | (100, 120) A: on the +20% border | (100, 121) B | (100, 79) B
    # (300, 100) C | (300, 80) D | (60, 200) D | (300, 60) E | (30, 190) E | (120, nan) ignored
    reference = [40, 100, 100, 100, 300, 300, 60, 300, 30, 120]
    monitor = [10, 120, 121, 79, 100, 80, 200, 60, 190, np.nan]
    t = np.arange(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 0, 50, 0), timedelta(minutes=5)).astype(
        datetime)
    data = pd.DataFrame(data={'t': t, 'glucose': np.array(reference, dtype=float)})
    data_hat = pd.DataFrame(data={'t': t, 'glucose': np.array(monitor, dtype=float)})

    # Tests
    results = dts_error_grid(data, data_hat)
    assert type(results) is dict
    assert np.isclose(results["a"], 2 / 9 * 100)
    assert np.isclose(results["b"], 2 / 9 * 100)
    assert np.isclose(results["c"], 1 / 9 * 100)
    assert np.isclose(results["d"], 2 / 9 * 100)
    assert np.isclose(results["e"], 2 / 9 * 100)

    d = {'t': [], 'glucose': []}
    data_hat = pd.DataFrame(data=d)
    d = {'t': [], 'glucose': []}
    data = pd.DataFrame(data=d)

    # Tests
    for zone in ("a", "b", "c", "d", "e"):
        assert np.isnan(dts_error_grid(data, data_hat)[zone])

    d = {'t': [t[0]], 'glucose': [np.nan]}
    data_hat = pd.DataFrame(data=d)
    d = {'t': [t[0]], 'glucose': [120]}
    data = pd.DataFrame(data=d)

    # Tests
    for zone in ("a", "b", "c", "d", "e"):
        assert np.isnan(dts_error_grid(data, data_hat)[zone])
