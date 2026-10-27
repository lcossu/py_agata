import pandas as pd
import pytest

from py_agata.visualization import export_agp_pdf


def _build_profile(start='2024-01-01', periods=288, offset=0):
    t = pd.date_range(start=start, periods=periods, freq='5min')
    glucose = 120 + offset + 25 * ((pd.Series(range(periods)) % 48) / 48)
    return pd.DataFrame({'t': t, 'glucose': glucose.astype(float)})


def test_export_agp_pdf_single_dataframe(tmp_path):
    data = _build_profile()
    output = tmp_path / 'single_agp.pdf'

    result = export_agp_pdf(data, file_path=str(output))

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_export_agp_pdf_list_of_dataframes(tmp_path):
    data_1 = _build_profile(start='2024-01-01', offset=0)
    data_2 = _build_profile(start='2024-01-02', offset=15)
    output = tmp_path / 'multiple_agp.pdf'

    result = export_agp_pdf([data_1, data_2], file_path=str(output))

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_export_agp_pdf_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError, match='at least one dataframe'):
        export_agp_pdf([], file_path=str(tmp_path / 'empty.pdf'))

