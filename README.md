# PYAGATA

<img src="https://i.postimg.cc/G3YXvWWh/agata-logo.png" width="250" height="250">

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/gcappon/py_agata/COPYING)
[![GitHub commit](https://img.shields.io/github/last-commit/gcappon/py_agata)](https://github.com/gcappon/py_agata/commits/master)
[![CircleCI](https://circleci.com/gh/gcappon/py_agata.svg?style=shield)](https://circleci.com/gh/gcappon/py_agata)
[![codecov](https://codecov.io/gh/gcappon/py_agata/graph/badge.svg?token=RR39WKLDIK)](https://codecov.io/gh/gcappon/py_agata)

Official Python porting of AGATA (Automated Glucose dATa Analysis): a toolbox to analyse glucose data.

## Installation

```
pip install py_agata
```

- Python >= 3.9
- Tested with pandas 2.2 and 3.x

## Input data

Every function takes a `pd.DataFrame` with two columns:

| Column | Content |
|---|---|
| `t` | Timestamps (`datetime64`) on a uniform grid |
| `glucose` | Glucose in mg/dL, `nan` for missing samples |

Readers and helpers in `py_agata.utils` and `py_agata.processing` build this frame:

- `read_dexcom_data`, `read_eversense_data`, `read_freestyle_libre_data`
- `glucose_vector_to_dataframe`, `glucose_time_vectors_to_dataframe`
- `retime_glucose` puts irregular data on a uniform grid
- `impute_glucose` fills gaps up to a given length

## Quick start

```python
import pandas as pd
from py_agata.py_agata import Agata
from py_agata.processing import retime_glucose

data = pd.read_csv("example/data/data.csv").rename(columns={"Time": "t"})
data["t"] = pd.to_datetime(data["t"])
data = retime_glucose(data, timestep=5)

agata = Agata(glycemic_target="diabetes")

# One profile
results = agata.analyze_glucose_profile(data)
results["time_in_ranges"]["time_in_target"]

# One arm (list of profiles)
arm = agata.analyze_one_arm([data, data])
arm["variability"]["mean_glucose"]["median"]

# Two arms, with statistical tests
arms, stats = agata.compare_two_arms([data, data], [data, data], is_paired=True, alpha=0.05)
stats["risk"]["gri"]["p"]
```

`glycemic_target` is `"diabetes"` or `"pregnancy"`. It sets the time-in-range and event thresholds.

## Agata output

| Entrypoint | Shape |
|---|---|
| `analyze_glucose_profile` | `results[family][metric]` is a scalar; events are dicts |
| `analyze_one_arm` | `results[family][metric]` has `values`, `mean`, `std`, `median`, `prc_5`, `prc_25`, `prc_75`, `prc_95` |
| `compare_two_arms` | `(arms, stats)`; `arms` has `arm_1` and `arm_2`; `stats[family][metric]` has `p` and `h` |

| Family | Metrics |
|---|---|
| `variability` | `mean_glucose`, `median_glucose`, `std_glucose`, `cv_glucose`, `range_glucose`, `iqr_glucose`, `auc_glucose`, `gmi`, `cogi`, `conga`, `j_index`, `mage_index`, `mage_plus_index`, `mage_minus_index`, `ef_index`, `modd`, `sddm_index`, `sdw_index`, `std_glucose_roc`, `cvga`, `gvp`, `mag`, `mad`, `ea1c`, `distance_traveled` |
| `time_in_ranges` | `time_in_target`, `time_in_tight_target`, `time_in_hypoglycemia`, `time_in_l1_hypoglycemia`, `time_in_l2_hypoglycemia`, `time_in_hyperglycemia`, `time_in_l1_hyperglycemia`, `time_in_l2_hyperglycemia` |
| `risk` | `adrr`, `lbgi`, `hbgi`, `bgri`, `gri` |
| `glycemic_transformation` | `grade_score`, `grade_hypo_score`, `grade_hyper_score`, `grade_eu_score`, `igc`, `hypo_index`, `hyper_index`, `mr_index`, `q_score`, `gdi`, `int_hypo`, `hypo_rv` |
| `events` | `hypoglycemic_events`, `hyperglycemic_events`, `extended_hypoglycemic_events` |
| `data_quality` | `number_days_of_observation`, `missing_glucose_percentage`, `number_days_at_least_data` |

Every metric is also a plain function in the module of the same name, e.g. `from py_agata.variability import mage_index`.

## Functions outside Agata

| Module | Functions |
|---|---|
| `variability` | `glucose_roc`, `auc_glucose_over_basal`, `positive_auc_glucose`, `sample_entropy`, `multiscale_entropy` |
| `time_in_ranges` | `time_in_given_range`, `time_in_given_above_range`, `time_in_given_below_range` |
| `risk` | `dynamic_risk`, `glycemic_risk_parameter` |
| `glycemic_transformation` | `glucose_pentagon` |
| `inspection` | `find_nan_islands`, `find_hypoglycemic_events`, `find_hyperglycemic_events` |
| `events` | `after_meal_auc`, `meal_response_metrics` (need a `cho` column), `tdi` (needs `bolus` and `basal` columns) |
| `error` | `rmse`, `mard`, `cod`, `clarke`, `dts_error_grid`, `g_rmse`, `time_delay` (compare two traces) |
| `processing` | `detrend_glucose`, `merge_additional_data`, `select_time_window` |
| `utils` | `to_mmol_l`, `to_mg_dl` |

## Visualization

`py_agata.visualization` returns Plotly figures:

- `plot_glucose`, `plot_glucose_events`, `plot_glucose_3d`
- `plot_poincare`, `plot_roc_histogram`, `plot_lbgi_hbgi_hourly`, `plot_glucose_pentagon`
- `plot_cvga`, `plot_gri` (several profiles)
- `plot_clarke_error_grid`, `plot_dts_error_grid` (return `(fig, percentages)`)

`export_agp_pdf` writes a one-page AGP report to PDF (IDC AGP v5 layout, type 1 diabetes targets).

Every plot takes `width` and `height` in pixels. Defaults suit each plot (900×450 for time series, 600×600 for grids). Pass `None` to let the figure fill its container, e.g. in a web page.

```python
from py_agata.visualization import plot_glucose

fig = plot_glucose(data, width=1200, height=400)
fig.show()                    # interactive
fig.write_image("trace.pdf")  # static export, needs `pip install kaleido`
```

## Development

```
pip install -e ".[dev]"
pytest
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Citing

If you are using AGATA in your research, please cite it:
```
Cappon G, Sparacino G, Facchinetti A. AGATA: A toolbox for automated glucose data analysis.
J Diabetes Sci Technol. 2023. DOI: 10.1177/19322968221147570
```

Link to the paper --> [here](https://doi.org/10.1177/19322968221147570)
