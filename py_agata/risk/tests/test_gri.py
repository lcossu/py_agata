import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

from py_agata.risk import gri


def test_gri():
    """
    Unit test of gri function.

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
    # Set test data
    t = np.arange(datetime(2000, 1, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = 40
    glucose[1:3] = [80, 60]
    glucose[3] = 80
    glucose[4:6] = [120, 120]
    glucose[6:8] = [120, 200]
    glucose[8:10] = [260, 260]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    #Tests
    assert np.isnan(gri(data)) == False
    assert np.round(gri(data)*1000)/1000 == 94

    # Set empty data
    t = np.arange(datetime(2000, 1, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 55, 0), timedelta(minutes=5)).astype(
        datetime)
    glucose = np.zeros(shape=(t.shape[0],))
    glucose[0] = np.nan
    glucose[1:3] = [np.nan, np.nan]
    glucose[3] = np.nan
    glucose[4:6] = [np.nan, np.nan]
    glucose[6:8] = [np.nan, np.nan]
    glucose[8:10] = [np.nan, np.nan]
    glucose[10] = np.nan
    d = {'t': t, 'glucose': glucose}
    data = pd.DataFrame(data=d)

    # Tests
    assert np.isnan(gri(data))


def test_gri_components_rebuild_gri():
    import numpy as np
    import pandas as pd
    from py_agata.risk import gri, gri_components

    t = pd.date_range('2024-01-01', periods=100, freq='5min')
    glucose = np.r_[[50] * 5, [65] * 10, [120] * 60, [200] * 15, [300] * 10]
    data = pd.DataFrame({'t': t, 'glucose': glucose})

    hypo, hyper = gri_components(data)
    assert np.isclose(hypo, 5 + 0.8 * 10)
    assert np.isclose(hyper, 10 + 0.5 * 15)
    assert np.isclose(gri(data), 3.0 * hypo + 1.6 * hyper)
