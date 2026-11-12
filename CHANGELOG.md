# Changelog

## 0.1.0

### Added

- `Agata` reports new metrics in all three entrypoints:
  - `variability`: `gvp`, `mag`, `mad`, `ea1c`, `distance_traveled`, `li`
  - `glycemic_transformation`: `q_score`, `gdi`, `int_hypo`, `hypo_rv`
  - `data_quality`: `number_days_at_least_data`
- `variability`: `positive_auc_glucose`, `sample_entropy`, `multiscale_entropy`
- `variability`: `glucodensity` (Matabuena 2021) and `glucodensity_distance` (2-Wasserstein distance between glucose distributions)
- `variability`: `li` (Lability Index, Ryan 2004, CGM readings paired 60 min apart)
- `visualization.plot_glucodensity`: glucodensity curves over the target range, analyzed profile in green and reference in black
- `visualization.plot_lasagna`: mean glucose by time of day, one row per day or per profile, colored by AGP range
- `glycemic_transformation`: `glucose_pentagon`
- `risk`: `glycemic_risk_parameter`
- `error`: `dts_error_grid` (DTS Error Grid, Klonoff 2024)
- `events` module: `after_meal_auc`, `meal_response_metrics` (peak searched in the 120 min after the meal), `tdi`
- `inspection`: `number_days_at_least_data`
- `processing`: `merge_additional_data`, `select_time_window`
- `visualization` module: Plotly figures for glucose traces, events, CVGA, Poincaré, ROC histogram, LBGI/HBGI per hour, GRI, glucose pentagon, 3D glucose, Clarke and DTS error grids, dynamic risk
  - One shared style, with `width` and `height` parameters on every plot
- `visualization.export_agp_pdf`: one-page AGP report (IDC AGP v5 layout, type 1 diabetes targets)
- `visualization.agp_percentiles`: AGP percentile curves over the time of day, pooled across profiles
- `visualization.lbgi_hbgi_hourly`: LBGI and HBGI per 1-hour block, or averaged by hour of day
- `visualization.roc_histogram`: rate-of-change histogram counts
- `risk.gri_components`: hypoglycemia and hyperglycemia components of GRI
- `visualization.plot_meal_response`: glucose around one meal with the baseline and peak of `meal_response_metrics`
- Support for pandas 3

### Changed (breaking)

- `cvga` takes one profile or a list of profiles and returns `(profile_assessment, profile_coordinates, best_control_index, best_controlled_coordinates)`. `Agata` still reports the scalar per profile.
- `find_extended_hypoglycemic_events` default threshold moves from 54 to 70 mg/dL, following the Battelino 2023 consensus (>120 min below 70 mg/dL).
- Dependencies use lower bounds instead of exact pins:
  - numpy >= 1.23.5
  - pandas >= 2.2.2
  - scipy >= 1.11
  - statsmodels >= 0.14.5 on Python < 3.11, >= 0.15.0 on Python >= 3.11 (older releases fail to import with numpy 2.1+, scipy 1.16+ or pandas 3)
- `pytest` and `pytest-cov` move to the `dev` extra.

### Fixed

- Profiles with a single sample no longer crash `Agata`. Metrics that need the sampling time return nan, event finders return no events, `impute_glucose` returns the data unchanged, and `plot_glucose_pentagon` raises `ValueError`
- `li` returns nan with a warning when 60 min is not a multiple of the sampling time, so `Agata` no longer fails on such grids
- `roc_histogram`, `plot_roc_histogram` and `plot_poincare` check for a homogeneous time grid, which their pairing of consecutive samples assumes
- `risk.dynamic_risk` no longer prints a debug value on every call
- `sample_entropy` and `multiscale_entropy` compare windows with numpy instead of a pairwise Python loop. On 13 days of 5-minute data, `sample_entropy` drops from 73 s to 0.2 s, with identical values.
- `detrend_glucose`, `impute_glucose` and `retime_glucose` returned unmodified data under pandas 3 (chained assignment).
- `compare_two_arms` now selects tests as MATLAB AGATA `compareTwoArms` does. P-values can change from 0.0.8.
  - Gaussian samples with `is_paired=True` use the paired t-test. 0.0.8 always ran the unpaired one.
  - Paired non-Gaussian samples skipped the Wilcoxon test and reported p = 1 when any single pair was equal. Now only fully identical samples give p = 1.
  - Mann-Whitney U ignores nan values. 0.0.8 returned nan.
  - `is_paired=True` with arms of different length raises `ValueError`.

All 0.0.8 metrics in `Agata` give identical values on the bundled example data.
