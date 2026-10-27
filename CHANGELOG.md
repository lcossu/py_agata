# Changelog

## 0.1.0

### Added

- `Agata` reports new metrics in all three entrypoints:
  - `variability`: `gvp`, `mag`, `mad`, `ea1c`, `distance_traveled`
  - `glycemic_transformation`: `q_score`, `gdi`, `int_hypo`, `hypo_rv`
  - `data_quality`: `number_days_at_least_data`
- `variability`: `positive_auc_glucose`, `sample_entropy`, `multiscale_entropy`
- `glycemic_transformation`: `glucose_pentagon`
- `risk`: `glycemic_risk_parameter`
- `error`: `dts_error_grid` (DTS Error Grid, Klonoff 2024)
- `events` module: `after_meal_auc`, `meal_response_metrics`, `tdi`
- `inspection`: `number_days_at_least_data`
- `processing`: `merge_additional_data`, `select_time_window`
- `visualization` module: Plotly figures for glucose traces, events, CVGA, Poincaré, ROC histogram, LBGI/HBGI per hour, GRI, glucose pentagon, 3D glucose, Clarke and DTS error grids
  - One shared style, with `width` and `height` parameters on every plot
- `visualization.export_agp_pdf`: one-page AGP report (IDC AGP v5 layout, type 1 diabetes targets)
- Support for pandas 3

### Changed (breaking)

- `cvga` takes one profile or a list of profiles and returns `(profile_assessment, profile_coordinates, best_control_index, best_controlled_coordinates)`. `Agata` still reports the scalar per profile.
- `find_extended_hypoglycemic_events` default threshold moves from 54 to 70 mg/dL, following the Battelino 2023 consensus (>120 min below 70 mg/dL).
- Dependencies use lower bounds instead of exact pins:
  - numpy >= 2.0
  - pandas >= 2.2.2
  - scipy >= 1.13
- `pytest` and `pytest-cov` move to the `dev` extra.

### Fixed

- `detrend_glucose`, `impute_glucose` and `retime_glucose` returned unmodified data under pandas 3 (chained assignment).
- `compare_two_arms` now selects tests as MATLAB AGATA `compareTwoArms` does. P-values can change from 0.0.8.
  - Gaussian samples with `is_paired=True` use the paired t-test. 0.0.8 always ran the unpaired one.
  - Paired non-Gaussian samples skipped the Wilcoxon test and reported p = 1 when any single pair was equal. Now only fully identical samples give p = 1.
  - Mann-Whitney U ignores nan values. 0.0.8 returned nan.
  - `is_paired=True` with arms of different length raises `ValueError`.

All 0.0.8 metrics in `Agata` give identical values on the bundled example data.
