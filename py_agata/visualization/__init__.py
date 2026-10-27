import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from plotly.graph_objs import Scatter, Scatter3d, Scatterpolar, Bar, Histogram, Figure
from plotly.subplots import make_subplots

from py_agata.error import clarke, dts_error_grid, _DTS_BORDERS
from py_agata.glycemic_transformation import glucose_pentagon
from py_agata.input_validator import check_dataframe, check_data_columns, check_homogeneous_timegrid
from py_agata.risk import lbgi, hbgi, gri
from py_agata.inspection import missing_glucose_percentage
from py_agata.time_in_ranges import time_in_target, time_in_hypoglycemia, time_in_hyperglycemia, \
    time_in_l1_hypoglycemia, time_in_l2_hypoglycemia, time_in_l1_hyperglycemia, time_in_l2_hyperglycemia, \
    time_in_tight_target
from py_agata.variability import mean_glucose, median_glucose, std_glucose, cv_glucose, gmi

# Trace colors: glucose data, reference or baseline, and the hypo/hyper thresholds
CGM_COLOR = 'steelblue'
REFERENCE_COLOR = 'black'
HYPO_COLOR = 'firebrick'
HYPER_COLOR = 'goldenrod'

# Risk-zone fills from lowest to highest risk, shared by CVGA and error grids
ZONE_COLORS = ['#b8e186', '#e6f598', '#fee08b', '#fdae61', '#f46d43']

GRID_COLOR = 'rgb(232,232,232)'


def _style(fig, title, width, height, font_size=14, square=False):
    """
    Applies the shared layout to a figure. Width and height are in pixels; None lets the figure fill its container.
    Square figures keep one data unit equal on both axes.
    """
    fig.update_layout(
        template='simple_white',
        title=dict(text=title, x=0.01, xanchor='left') if title else None,
        width=width,
        height=height,
        font=dict(family='Arial', size=font_size, color='black'),
        margin=dict(l=70, r=30, t=60 if title else 30, b=60),
        legend=dict(bgcolor='rgba(255,255,255,0.8)'),
    )

    # simple_white draws shapes at 30% opacity; colors here are set at full strength
    fig.layout.template.layout.shapedefaults.opacity = 1
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, mirror=True, ticks='outside')
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, mirror=True, ticks='outside')
    if square:
        fig.update_xaxes(constrain='domain')
        fig.update_yaxes(scaleanchor='x', scaleratio=1, constrain='domain')
    return fig


def plot_glucose(data, above=180, below=70, width=900, height=450) -> Figure:
    """
    Plots the glucose trace with the target range. Columns `cho`, `bolus` and `basal`, if present, are drawn in
    panels below the glucose.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with columns `t` and `glucose` (in mg/dl), and optionally `cho`, `bolus` and `basal`.
    above: float, optional, default: 180
        Upper threshold of the target range (in mg/dl).
    below: float, optional, default: 70
        Lower threshold of the target range (in mg/dl).
    width: int or None, optional, default: 900
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 450
        Total figure height in pixels, shared by all panels. None fills the container.

    Returns
    -------
    fig: plotly.graph_objs.Figure
    """
    check_dataframe(data)
    check_data_columns(data)

    ytitles = ['Glucose [mg/dL]']
    traces = [Scatter(x=data['t'], y=data['glucose'], mode='lines', name='Glucose',
                      line=dict(color=CGM_COLOR, width=1.5), showlegend=False,
                      hovertemplate='%{x|%d %b %H:%M}<br>%{y:.0f} mg/dL<extra></extra>')]

    # Discrete inputs are drawn only where they occur
    for col, ytitle, symbol in [('cho', 'CHO [g]', 'star'), ('bolus', 'Bolus [U]', 'triangle-up')]:
        if col in data.columns:
            ytitles.append(ytitle)
            traces.append(Scatter(x=data['t'], y=data[col].where(data[col] > 0), mode='markers',
                                  marker=dict(symbol=symbol, size=9, color='dimgray'), name=ytitle,
                                  showlegend=False))
    if 'basal' in data.columns:
        ytitles.append('Basal [U/min]')
        traces.append(Scatter(x=data['t'], y=data['basal'], mode='lines', line=dict(color='dimgray', shape='hv'),
                              name='Basal [U/min]', showlegend=False))

    # The glucose panel is three times as tall as each input panel
    weights = [3] + [1] * (len(traces) - 1)
    fig = make_subplots(rows=len(traces), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[w / sum(weights) for w in weights])
    for i, (trace, ytitle) in enumerate(zip(traces, ytitles)):
        fig.add_trace(trace, row=i + 1, col=1)
        fig.update_yaxes(title_text=ytitle, row=i + 1, col=1)

    fig.add_hrect(y0=below, y1=above, fillcolor='rgba(184,225,134,0.25)', line_width=0, layer='below', row=1, col=1)
    fig.add_hline(y=below, line=dict(color=HYPO_COLOR, width=1.5, dash='dash'), row=1, col=1)
    fig.add_hline(y=above, line=dict(color=HYPER_COLOR, width=1.5, dash='dash'), row=1, col=1)
    fig.update_yaxes(range=[40, 400], row=1, col=1)

    _style(fig, None, width, height)
    fig.update_layout(selectdirection='h', dragmode='select')
    return fig


def plot_glucose_pentagon(data, a1c=None, width=600, height=550) -> Figure:
    """
    Plots the glucose pentagon of the given data against the healthy reference.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with columns `t` and `glucose` (in mg/dl).
    a1c: float, optional, default: None
        HbA1c of the patient (in %). If None, it is estimated from the data.
    width: int or None, optional, default: 600
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 550
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig: plotly.graph_objs.Figure

    References
    ----------
    Thomas et al. "The “Glucose Pentagon”: Assessing Glycemic Control of Patients with Diabetes Mellitus by a Model
    Integrating Different Parameters from Glucose Profiles." (2009). Diabetes Technology & Therapeutics, 11(6),
    399–409. doi:10.1089/dia.2008.0119
    """
    values = np.asarray(glucose_pentagon(data, a1c), dtype=float)

    # Axis name, unit, value range and healthy reference
    names = ['Mean', 'SD', 'Time >160', 'AUC >160', 'HbA1c']
    units = ['mg/dL', 'mg/dL', 'h/day', 'mg/dL·day', '%']
    min_vals = np.array([40, 0, 0, 0, 5.5])
    max_vals = np.array([310, 100, 24, 150, 12])
    healthy = np.array([90, 10, 0, 0, 5.5])

    # Offset by 0.1 so the healthy minimum is still visible
    def scale(v):
        r = (v - min_vals) / (max_vals - min_vals) + .1
        return np.append(r, r[0])

    theta = [f'{n}<br><b>{v:.1f}</b> {u}' for n, v, u in zip(names, values, units)]
    theta.append(theta[0])
    hover = [f'{n}: {v:.1f} {u} (range {lo}-{hi})' for n, v, u, lo, hi in zip(names, values, units, min_vals, max_vals)]
    hover.append(hover[0])

    fig = Figure()
    fig.add_trace(Scatterpolar(r=scale(healthy), theta=theta, mode='lines', name='Healthy reference',
                               line=dict(color=REFERENCE_COLOR, width=1.5, dash='dash'), hoverinfo='skip'))
    fig.add_trace(Scatterpolar(r=scale(values), theta=theta, mode='lines+markers', name='Data', fill='toself',
                               fillcolor='rgba(70,130,180,0.25)', line=dict(color=CGM_COLOR, width=2),
                               marker=dict(size=7), text=hover, hoverinfo='text'))

    _style(fig, None, width, height)
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1.1], showticklabels=False, ticks='', showline=False,
                                   gridcolor=GRID_COLOR),
                   angularaxis=dict(direction='clockwise', rotation=90, gridcolor=GRID_COLOR, ticks='')),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.08),
        margin=dict(l=90, r=90, t=50, b=50),
    )
    return fig


def plot_glucose_3d(data, width=800, height=600):
    """
    Plots one glucose line per day in 3D, with time of day on one axis and the day on the other.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe with columns `t` and `glucose` (in mg/dl). At least 3 days are required.
    width: int or None, optional, default: 800
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 600
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig: plotly.graph_objs.Figure or None
        None if the data cover fewer than 3 days.
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    data = data.assign(day=data['t'].dt.normalize(), hour=data['t'].dt.hour + data['t'].dt.minute / 60)
    if data['day'].nunique() < 3:
        warnings.warn("At least 3 distinct days required.")
        return

    fig = Figure()
    for day, d in data.groupby('day'):
        fig.add_trace(Scatter3d(x=d['hour'], y=d['day'], z=d['glucose'], mode='lines',
                                line=dict(color=CGM_COLOR, width=3), name=day.strftime('%d %b'), showlegend=False,
                                hovertemplate='%{y|%d %b} %{x:.2f} h<br>%{z:.0f} mg/dL<extra></extra>'))

    _style(fig, None, width, height)
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis=dict(title='Time of day [h]', range=[0, 24], tickvals=[0, 6, 12, 18, 24]),
            yaxis=dict(title='Day', tickformat='%d %b'),
            zaxis=dict(title='Glucose [mg/dL]', range=[40, 400], tickvals=[100, 200, 300, 400]),
            aspectratio=dict(x=1.6, y=1.3, z=0.8),
            camera=dict(eye=dict(x=-0.4, y=-2.1, z=0.9), center=dict(x=0, y=0, z=-0.15)),
        ),
    )
    return fig


# Clarke error grid borders as (x, y) polylines on a 0-400 mg/dL grid
_CLARKE_LINES = [
    ([0, 175 / 3], [70, 70]),
    ([175 / 3, 400 / 1.2], [70, 400]),
    ([70, 70], [84, 400]),
    ([0, 70], [180, 180]),
    ([70, 290], [180, 400]),
    ([70, 70], [0, 56]),
    ([70, 400], [56, 320]),
    ([180, 180], [0, 70]),
    ([180, 400], [70, 70]),
    ([240, 240], [70, 180]),
    ([240, 400], [180, 180]),
    ([130, 180], [0, 70]),
]
_CLARKE_LABELS = [('A', 30, 15), ('B', 370, 260), ('B', 280, 370), ('C', 160, 370), ('C', 160, 15),
                  ('D', 30, 140), ('D', 370, 120), ('E', 30, 370), ('E', 370, 15)]


def plot_clarke_error_grid(data, reference_col='reference_glucose', predicted_col='glucose',
                           title='Clarke Error Grid', width=600, height=600) -> (Figure, dict):
    """
    Plots the Clarke Error Grid comparing measured glucose to a reference.

    - Zone A: clinically accurate, within 20% of reference
    - Zone B: benign errors, would lead to appropriate or no treatment
    - Zone C: overcorrection errors, would lead to unnecessary treatment
    - Zone D: dangerous failure to detect or treat
    - Zone E: erroneous treatment, opposite to what is needed

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing glucose measurements
    reference_col : str, default='reference_glucose'
        Column name for reference glucose values
    predicted_col : str, default='glucose'
        Column name for predicted/measured glucose values
    title : str, default='Clarke Error Grid'
        Title of the plot
    width: int or None, optional, default: 600
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 600
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig: plotly.graph_objs.Figure
    percentages : dict
        Percentage of points in each zone (keys `a`-`e`), as returned by py_agata.error.clarke

    References
    ----------
    Clarke WL, et al. "Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose"
    Diabetes Care 1987;10(5):622-628.
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if reference_col not in data.columns:
        raise ValueError(f"Column '{reference_col}' not found in the dataframe.")
    if predicted_col not in data.columns:
        raise ValueError(f"Column '{predicted_col}' not found in the dataframe.")

    reference = data[reference_col].values.astype(float)
    predicted = data[predicted_col].values.astype(float)
    percentages = clarke(pd.DataFrame({'t': data.t, 'glucose': reference}),
                         pd.DataFrame({'t': data.t, 'glucose': predicted}))

    fig = Figure()
    fig.add_trace(Scatter(x=[0, 400], y=[0, 400], mode='lines', line=dict(color='gray', width=1, dash='dot'),
                          hoverinfo='skip', showlegend=False))
    for x, y in _CLARKE_LINES:
        fig.add_trace(Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=1.2), hoverinfo='skip',
                              showlegend=False))
    fig.add_trace(Scatter(x=reference, y=predicted, mode='markers', name='Glucose pairs', showlegend=False,
                          marker=dict(color=CGM_COLOR, size=5, opacity=0.5),
                          hovertemplate='Reference %{x:.0f}<br>Measured %{y:.0f}<extra></extra>'))
    for zone, x, y in _CLARKE_LABELS:
        fig.add_annotation(x=x, y=y, text=f'<b>{zone}</b>', showarrow=False, font=dict(size=18))

    _style(fig, title, width, height, square=True)
    fig.update_xaxes(title_text='Reference glucose [mg/dL]', range=[0, 400], dtick=50, showgrid=False)
    fig.update_yaxes(title_text='Measured glucose [mg/dL]', range=[0, 400], dtick=50, showgrid=False)
    return fig, percentages


def plot_dts_error_grid(data, reference_col='reference_glucose', predicted_col='glucose',
                        title='DTS Error Grid', max_val=None, width=750, height=600) -> (Figure, dict):
    """
    Plots the Diabetes Technology Society (DTS) Error Grid for glucose measurements comparison.

    The grid is divided into five risk zones with straight-line borders:

    - Zone A: no risk
    - Zone B: mild risk
    - Zone C: moderate risk
    - Zone D: high risk
    - Zone E: extreme risk

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing glucose measurements
    reference_col : str, default='reference_glucose'
        Column name for reference glucose values
    predicted_col : str, default='glucose'
        Column name for predicted/measured glucose values
    title : str, default='DTS Error Grid'
        Title of the plot
    max_val : float, default=None
        Upper limit of both axes in mg/dL. If None, it is set to max(400, 1.1 * largest value), capped at 600
    width: int or None, optional, default: 750
        Figure width in pixels, legend included. None fills the container.
    height: int or None, optional, default: 600
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig: plotly.graph_objs.Figure
    percentages : dict
        Dictionary containing the percentage of points in each zone, as returned by py_agata.error.dts_error_grid

    References
    ----------
    Klonoff et al., "The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for Glucose
    Monitors", Journal of Diabetes Science and Technology, 2024. DOI: 10.1177/19322968241275701.
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    if reference_col not in data.columns:
        raise ValueError(f"Column '{reference_col}' not found in the dataframe.")
    if predicted_col not in data.columns:
        raise ValueError(f"Column '{predicted_col}' not found in the dataframe.")

    reference = data[reference_col].values.astype(float)
    predicted = data[predicted_col].values.astype(float)

    percentages = dts_error_grid(pd.DataFrame({'t': data.t, 'glucose': reference}),
                                 pd.DataFrame({'t': data.t, 'glucose': predicted}))

    if max_val is None:
        max_val = min(600, max(400, np.nanmax(np.concatenate([reference, predicted])) * 1.1))

    fig = Figure()

    # Zones are painted from the outside in. E fills the whole grid, then each lower-risk zone covers it.
    zone_names = ['E (extreme risk)', 'D (high risk)', 'C (moderate risk)', 'B (mild risk)', 'A (no risk)']
    zone_colors = ZONE_COLORS[::-1]
    fig.add_trace(Scatter(x=[0, 600, 600, 0, 0], y=[0, 0, 600, 600, 0], mode='lines', fill='toself',
                          fillcolor=zone_colors[0], line=dict(width=0), name=zone_names[0], hoverinfo='skip'))
    for (x_lower, y_600, y_upper, x_600), name, color in zip(reversed(_DTS_BORDERS), zone_names[1:],
                                                             zone_colors[1:]):
        x = [0, x_lower, x_lower, 600, 600, x_600, 50, 0, 0]
        y = [0, 0, 50, y_600, 600, 600, y_upper, y_upper, 0]
        fig.add_trace(Scatter(x=x, y=y, mode='lines', fill='toself', fillcolor=color,
                              line=dict(color='black', width=1), name=name, hoverinfo='skip'))

    fig.add_trace(Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='gray', width=1, dash='dot'),
                          hoverinfo='skip', showlegend=False))
    fig.add_trace(Scatter(x=reference, y=predicted, mode='markers', name='Glucose pairs',
                          marker=dict(color='black', size=4, opacity=0.6),
                          hovertemplate='Reference %{x:.0f}<br>Measured %{y:.0f}<extra></extra>'))

    _style(fig, title, width, height, square=True)
    fig.update_xaxes(title_text='Reference glucose [mg/dL]', range=[0, max_val], showgrid=False)
    fig.update_yaxes(title_text='Measured glucose [mg/dL]', range=[0, max_val], showgrid=False)
    fig.update_layout(legend=dict(traceorder='reversed'))
    return fig, percentages


def plot_roc_histogram(data, width=700, height=450):
    """
    Plots the histogram of the glucose rate of change between consecutive samples.

    Parameters
    ----------
    data : pandas.DataFrame
        Pandas dataframe with columns `t` and `glucose` (in mg/dl).
    width: int or None, optional, default: 700
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 450
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig: plotly.graph_objs.Figure
    """
    check_dataframe(data)
    check_data_columns(data)

    ts = (pd.to_datetime(data.t.values[1]) - pd.to_datetime(data.t.values[0])).total_seconds() / 60
    roc = np.diff(data.glucose.values) / ts
    roc = roc[~np.isnan(roc)]

    fig = Figure(Histogram(x=roc, xbins=dict(size=0.25), marker=dict(color=CGM_COLOR, line=dict(color='white', width=0.5)),
                           hovertemplate='%{x} mg/dL/min<br>%{y} samples<extra></extra>', showlegend=False))
    fig.add_vline(x=0, line=dict(color='black', width=1, dash='dot'))

    _style(fig, 'Glucose rate of change', width, height)
    fig.update_xaxes(title_text='Rate of change [mg/dL/min]')
    fig.update_yaxes(title_text='Samples')
    fig.update_layout(bargap=0)
    return fig


def plot_cvga(glucose_profiles, plot_zone_names=True, highlight_best_control=True, font_size=14, labels=None,
              width=600, height=600):
    """
    Plots the Control Variability Grid Analysis (CVGA) as in Magni et al. (2008)

    Parameters
    ----------
    glucose_profiles : list of pd.DataFrame or pd.DataFrame
        Each DataFrame must have columns 't' and 'glucose' (in mg/dL).
        If a single DataFrame is provided, it is treated as a single profile.
    plot_zone_names : bool, optional
        Whether to plot the zone names in the CVGA plot (default: True).
    highlight_best_control : bool, optional
        Whether to highlight the best controlled profile (default: True).
    font_size : int, optional
        Font size for labels (default: 14).
    labels : list of str, optional
        Profile names shown on hover. If None, profiles are numbered.
    width: int or None, optional, default: 600
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 600
        Figure height in pixels. None fills the container.

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly figure object containing the CVGA plot.
    """
    from py_agata.variability import cvga as cvga_metric

    _, profile_coordinates, _, best_controlled_coordinates = cvga_metric(glucose_profiles)
    if labels is None:
        labels = [f'Profile {i + 1}' for i in range(len(profile_coordinates))]

    fig = Figure()

    # Zone D fills the grid, then the quarter circles of C, B and A cover it
    fig.add_shape(type='rect', x0=0, y0=0, x1=60, y1=60, fillcolor=ZONE_COLORS[3], line=dict(width=0),
                  layer='below')
    theta = np.linspace(0, np.pi / 2, 200)
    for radius, color in [(60, ZONE_COLORS[2]), (40, ZONE_COLORS[1]), (20, ZONE_COLORS[0])]:
        fig.add_trace(Scatter(x=np.r_[radius * np.cos(theta), 0], y=np.r_[radius * np.sin(theta), 0], fill='toself',
                              mode='lines', line=dict(color='black', width=1), fillcolor=color, showlegend=False,
                              hoverinfo='skip'))

    if plot_zone_names:
        for text, x, y in [('<b>A</b>', 8, 8), ('Upper B', 10, 30), ('Upper C', 10, 50), ('Lower B', 30, 10),
                           ('C', 35, 35), ('Lower C', 50, 10), ('D', 50, 50)]:
            fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=font_size - 2))

    if len(profile_coordinates) > 0:
        fig.add_trace(Scatter(x=profile_coordinates[:, 0], y=profile_coordinates[:, 1], mode='markers',
                              marker=dict(color=REFERENCE_COLOR, size=10, line=dict(color='white', width=1)),
                              text=labels, hovertemplate='%{text}<extra></extra>', cliponaxis=False,
                              showlegend=False))
        if highlight_best_control:
            fig.add_trace(Scatter(x=[best_controlled_coordinates[0]], y=[best_controlled_coordinates[1]],
                                  mode='markers', marker=dict(size=20, symbol='circle-open', color=REFERENCE_COLOR,
                                                              line=dict(color=REFERENCE_COLOR, width=2)),
                                  cliponaxis=False, hoverinfo='skip', showlegend=False))

    _style(fig, None, width, height, font_size=font_size, square=True)
    fig.update_xaxes(title_text='Minimum glucose [mg/dL]', tickvals=[0, 20, 40, 60],
                     ticktext=['>110', '90', '70', '<50'], range=[0, 60], showgrid=False, mirror=False)
    fig.update_yaxes(title_text='Maximum glucose [mg/dL]', tickvals=[0, 20, 40, 60],
                     ticktext=['<110', '180', '300', '>400'], range=[0, 60], showgrid=False, mirror=False)
    return fig


def plot_poincare(data, font_size=14, width=600, height=600):
    """
    Plots a Poincaré plot for glucose data (glucose[n] vs glucose[n+1]) with its SD1/SD2 ellipse.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 't' and 'glucose'.
    font_size : int, optional
        Font size for labels (default: 14).
    width: int or None, optional, default: 600
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 600
        Figure height in pixels. None fills the container.

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly figure object containing the Poincaré plot.
    """
    glucose = data['glucose'].values
    x = glucose[:-1]
    y = glucose[1:]

    # SD1 is the spread across the identity line, SD2 the spread along it
    sd1 = np.nanstd((y - x) / np.sqrt(2))
    sd2 = np.nanstd((y + x) / np.sqrt(2))
    theta = np.linspace(0, 2 * np.pi, 200)
    ellipse_x = np.nanmean(x) + (sd2 * np.cos(theta) - sd1 * np.sin(theta)) / np.sqrt(2)
    ellipse_y = np.nanmean(y) + (sd2 * np.cos(theta) + sd1 * np.sin(theta)) / np.sqrt(2)

    lo = np.floor(np.nanmin(glucose) / 50) * 50
    hi = np.ceil(np.nanmax(glucose) / 50) * 50

    fig = Figure()
    fig.add_trace(Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line=dict(color='gray', width=1, dash='dot'),
                          hoverinfo='skip', showlegend=False))
    fig.add_trace(Scatter(x=x, y=y, mode='markers', marker=dict(color=CGM_COLOR, size=5, opacity=0.5),
                          name='Pairs', showlegend=False,
                          hovertemplate='%{x:.0f} → %{y:.0f} mg/dL<extra></extra>'))
    fig.add_trace(Scatter(x=ellipse_x, y=ellipse_y, mode='lines', line=dict(color=REFERENCE_COLOR, width=2),
                          name=f'SD1 {sd1:.1f}, SD2 {sd2:.1f} mg/dL', hoverinfo='name', showlegend=False))
    fig.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper', xanchor='left', yanchor='top', showarrow=False,
                       align='left', text=f'SD1 = {sd1:.1f} mg/dL<br>SD2 = {sd2:.1f} mg/dL',
                       bgcolor='rgba(255,255,255,0.8)')

    _style(fig, None, width, height, font_size=font_size, square=True)
    fig.update_xaxes(title_text='Glucose[n] [mg/dL]', range=[lo, hi], dtick=50)
    fig.update_yaxes(title_text='Glucose[n+1] [mg/dL]', range=[lo, hi], dtick=50)
    return fig


def plot_lbgi_hbgi_hourly(data, font_size=14, width=900, height=400):
    """
    Computes LBGI and HBGI in 1-hour blocks and plots them as a single bar chart.
    LBGI is shown as negative values, HBGI as positive values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 't' (datetime) and 'glucose' (mg/dL)
    font_size : int
        Font size for plot labels
    width: int or None, optional, default: 900
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 400
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    hourly = data.set_index(pd.to_datetime(data['t']))['glucose'].resample('1h')

    hours, lbgi_vals, hbgi_vals = [], [], []
    for t, block in hourly:
        block_df = pd.DataFrame({'glucose': block.dropna(), 't': t})
        hours.append(t)
        lbgi_vals.append(np.nan if block_df.empty else -lbgi(block_df))
        hbgi_vals.append(np.nan if block_df.empty else hbgi(block_df))

    # Symmetric y range so zero sits in the middle
    top = np.nanmax(np.abs(np.r_[lbgi_vals, hbgi_vals, 1])) * 1.1

    fig = Figure()
    fig.add_trace(Bar(x=hours, y=lbgi_vals, name='LBGI', marker=dict(color=HYPO_COLOR, line_width=0),
                      customdata=-np.array(lbgi_vals), hovertemplate='%{x|%d %b %H:00}<br>LBGI %{customdata:.1f}<extra></extra>'))
    fig.add_trace(Bar(x=hours, y=hbgi_vals, name='HBGI', marker=dict(color=HYPER_COLOR, line_width=0),
                      hovertemplate='%{x|%d %b %H:00}<br>HBGI %{y:.1f}<extra></extra>'))

    _style(fig, None, width, height, font_size=font_size)
    fig.update_layout(barmode='relative', bargap=0.1,
                      legend=dict(orientation='h', x=1, xanchor='right', y=1.02, yanchor='bottom'))
    fig.update_yaxes(title_text='LBGI (−) / HBGI (+)', range=[-top, top], zeroline=True, zerolinecolor='black')
    fig.update_xaxes(title_text='Time')
    return fig


def plot_gri(datasets, labels=None, font_size=14, width=750, height=600):
    """
    Plots the Glycemia Risk Index (GRI) grid with one point per profile:
    - x axis: hypoglycemia component (%)
    - y axis: hyperglycemia component (%)

    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of DataFrames to compute GRI for.
    labels : list of str, optional
        List of labels for each dataset. If None, uses generic labels.
    font_size : int
        Font size for plot labels
    width: int or None, optional, default: 750
        Figure width in pixels, legend included. None fills the container.
    height: int or None, optional, default: 600
        Figure height in pixels. None fills the container.

    Returns
    -------
    fig : plotly.graph_objs.Figure

    References
    ----------
    Klonoff et al., "A Glycemia Risk Index (GRI) of Hypoglycemia and Hyperglycemia for Continuous Glucose
    Monitoring Validated by Clinician Ratings", Journal of Diabetes Science and Technology, 2023.
    DOI: 10.1177/19322968221085273.
    """
    if labels is None:
        labels = [f"Profile {i + 1}" for i in range(len(datasets))]

    x_hypo, y_hyper, gris = [], [], []
    for data in datasets:
        check_dataframe(data)
        check_data_columns(data)
        check_homogeneous_timegrid(data)
        if data.shape[0] < 2:
            raise ValueError("Data must span more than one timepoint.")
        vlow, low = time_in_l2_hypoglycemia(data), time_in_l1_hypoglycemia(data)
        vhigh, high = time_in_l2_hyperglycemia(data), time_in_l1_hyperglycemia(data)
        x_hypo.append(vlow + 0.8 * low)
        y_hyper.append(vhigh + 0.5 * high)
        gris.append(min(100, 3.0 * vlow + 2.4 * low + 1.6 * vhigh + 0.8 * high))

    fig = Figure()

    # Zones are bands of GRI = 3 * hypo + 1.6 * hyper. E fills the grid, then D to A cover it.
    fig.add_trace(Scatter(x=[0, 30, 30, 0, 0], y=[0, 0, 60, 60, 0], mode='lines', fill='toself',
                          fillcolor=ZONE_COLORS[4], line=dict(width=0), hoverinfo='skip', showlegend=False))
    for bound, color in zip([80, 60, 40, 20], ZONE_COLORS[3::-1]):
        fig.add_trace(Scatter(x=[0, bound / 3, 0, 0], y=[0, 0, bound / 1.6, 0], mode='lines', fill='toself',
                              fillcolor=color, line=dict(width=0), hoverinfo='skip', showlegend=False))

    # Zone letters sit on the x axis at the GRI midpoint of each band
    for zone, mid in zip('ABCDE', [10, 30, 50, 70, 88]):
        fig.add_annotation(x=mid / 3, y=0, yshift=12, text=f'<b>{zone}</b>', showarrow=False,
                           font=dict(size=font_size))

    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x', 'star', 'hexagon',
               'pentagon']
    for i, (x, y, g, label) in enumerate(zip(x_hypo, y_hyper, gris, labels)):
        fig.add_trace(Scatter(x=[x], y=[y], mode='markers', name=f'{label} (GRI {g:.1f})', cliponaxis=False,
                              marker=dict(size=12, color=REFERENCE_COLOR, symbol=symbols[i % len(symbols)],
                                          line=dict(color='white', width=1)),
                              hovertemplate=f'{label}<br>GRI {g:.1f}<br>Hypo %{{x:.1f}}%<br>Hyper %{{y:.1f}}%<extra></extra>'))

    _style(fig, None, width, height, font_size=font_size)
    fig.update_xaxes(title_text='Hypoglycemia component [%]', range=[0, 30], dtick=5, showgrid=False)
    fig.update_yaxes(title_text='Hyperglycemia component [%]', range=[0, 60], dtick=10, showgrid=False)
    return fig


def plot_glucose_events(data, events: dict, above=180, below=70, width=900, height=450) -> Figure:
    """
    Plots glucose data with event intervals (e.g., hypo/hyper events) as shaded areas.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns 't' and 'glucose'.
    events : dict
        Event type to the output of `find_hypoglycemic_events` or `find_hyperglycemic_events`. Types containing
        'hypo' are shaded red, types containing 'hyper' amber.
    width: int or None, optional, default: 900
        Figure width in pixels. None fills the container.
    height: int or None, optional, default: 450
        Total figure height in pixels. None fills the container.

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly figure object.
    """
    check_homogeneous_timegrid(data)
    fig = plot_glucose(data, above=above, below=below, width=width, height=height)

    for event_type, intervals in events.items():
        if 'hypo' in event_type:
            color = 'rgba(178,34,34,0.3)'
        elif 'hyper' in event_type:
            color = 'rgba(218,165,32,0.35)'
        else:
            color = 'rgba(0,0,0,0.1)'
        if intervals['events_per_week'] > 0:
            for start, end in zip(intervals['time_start'], intervals['time_end']):
                fig.add_vrect(x0=start, x1=end, fillcolor=color, layer='below', line_width=0, row=1, col=1)

    return fig


def _prepare_agp_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    check_dataframe(data)
    check_data_columns(data)
    check_homogeneous_timegrid(data)

    df = data.copy()
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').dropna(subset=['t'])
    if df.empty:
        raise ValueError('`data` must contain at least one valid timestamp.')

    df['minute_of_day'] = df['t'].dt.hour * 60 + df['t'].dt.minute + df['t'].dt.second / 60
    df['day'] = df['t'].dt.normalize()
    return df




# Colors and thresholds of the IDC AGP Report v5

_AGP_COLORS = {
    'very_high': '#E8711C',
    'high': '#FCB813',
    'target': '#3BA652',
    'low': '#ED1C24',
    'very_low': '#A0101C',
    'section': '#E6E6E6',
    'text': '#333333',
}

# Type 1 diabetes consensus ranges

_AGP_THRESHOLDS = {'very_low': 54., 'low': 70., 'high': 180., 'very_high': 250.}


def _agp_ticks():
    ticks = np.arange(0, 1441, 180)
    labels = ['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm', '12am']
    return ticks, labels


def _agp_percentiles(data: pd.DataFrame, bin_minutes: int = 15, smooth_bins: int = 3) -> pd.DataFrame:

    # Pool samples in time-of-day bins, then smooth circularly across midnight
    bins = (data['minute_of_day'] // bin_minutes) * bin_minutes
    agp = data.groupby(bins)['glucose'].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack()
    agp.columns = ['p05', 'p25', 'p50', 'p75', 'p95']
    agp = agp.reindex(np.arange(0, 1440, bin_minutes)).interpolate(limit_direction='both')
    padded = pd.concat([agp.iloc[-smooth_bins:], agp, agp.iloc[:smooth_bins]])
    agp = padded.rolling(2 * smooth_bins + 1, center=True, min_periods=1).mean().iloc[smooth_bins:-smooth_bins]
    agp.index = agp.index + bin_minutes / 2

    # Close the curve at both ends of the day
    edges = agp.iloc[[0, -1]].mean()
    agp.loc[0] = edges
    agp.loc[1440] = edges
    return agp.sort_index()


def _build_agp_summary(data: pd.DataFrame) -> dict:
    t = data['t'].dropna()
    start, end = t.min().normalize(), t.max().normalize()
    return {
        'start': start,
        'end': end,
        'days': (end - start).days + 1,
        'active': 100 - missing_glucose_percentage(data),
        'mean_glucose': mean_glucose(data),
        'gmi': gmi(data),
        'cv_glucose': cv_glucose(data),
        'titr': time_in_tight_target(data),
        'gri': gri(data),
        'very_high': time_in_l2_hyperglycemia(data),
        'high': time_in_l1_hyperglycemia(data),
        'target': time_in_target(data),
        'low': time_in_l1_hypoglycemia(data),
        'very_low': time_in_l2_hypoglycemia(data),
    }


def _agp_time_per_day(percentage: float) -> str:
    minutes = int(round(percentage / 100 * 1440))
    hours, minutes = divmod(minutes, 60)
    if hours == 0:
        return f'{minutes}min'
    return f'{hours}h' if minutes == 0 else f'{hours}h {minutes}min'


def _agp_section(fig, y: float, title: str, subtitle: str = None):
    fig.add_artist(Rectangle((0.04, y), 0.92, 0.022, transform=fig.transFigure,
                             facecolor=_AGP_COLORS['section'], edgecolor='none', zorder=0))
    fig.text(0.05, y + 0.011, title, fontsize=9.5, fontweight='bold', va='center', color=_AGP_COLORS['text'])
    if subtitle is not None:
        fig.text(0.05, y - 0.012, subtitle, fontsize=7, va='center', color='dimgray', style='italic')


def _agp_draw_statistics(fig, summary: dict, th: dict):
    x_l, x_r = 0.05, 0.53
    c = _AGP_COLORS['text']

    def row(y, left, right, bold_right=True, size=8.5):
        fig.text(x_l, y, left, fontsize=size, va='center', color=c)
        fig.text(x_r, y, right, fontsize=size, va='center', ha='right', color=c,
                 fontweight='bold' if bold_right else 'normal')

    row(0.885, f"{summary['start']:%d %b %Y} - {summary['end']:%d %b %Y}", f"{summary['days']} days")
    row(0.865, '% Time CGM is Active', f"{summary['active']:.1f}%")

    fig.text(x_l, 0.842, 'Glucose Ranges and Targets for Type 1 Diabetes', fontsize=8,
             fontweight='bold', va='center', color=c)
    fig.text(x_l, 0.828, 'Glucose Ranges', fontsize=7, va='center', color='dimgray')
    fig.text(x_r, 0.828, 'Targets % of Readings (Time/Day)', fontsize=7, ha='right', va='center', color='dimgray')

    low, high, very_low, very_high = th['low'], th['high'], th['very_low'], th['very_high']
    targets = [
        (f'Target Range {low:.0f}-{high:.0f} mg/dL', 'Greater than', 70),
        (f'Below {low:.0f} mg/dL', 'Less than', 4),
        (f'Below {very_low:.0f} mg/dL', 'Less than', 1),
        (f'Above {high:.0f} mg/dL', 'Less than', 25),
        (f'Above {very_high:.0f} mg/dL', 'Less than', 5),
    ]
    y = 0.814
    for label, comparison, value in targets:
        row(y, label, f'{comparison} {value}% ({_agp_time_per_day(value)})', bold_right=False, size=7.5)
        y -= 0.015
    fig.text(x_l, y, f'Each 5% increase in time in range ({low:.0f}-{high:.0f} mg/dL) is clinically beneficial.',
             fontsize=6.5, va='center', color='dimgray', style='italic')

    fig.add_artist(plt.Line2D([x_l, x_r], [0.725, 0.725], transform=fig.transFigure, color='lightgray', lw=0.8))
    row(0.711, 'Average Glucose', f"{summary['mean_glucose']:.0f} mg/dL")
    row(0.694, 'Glucose Management Indicator (GMI)', f"{summary['gmi']:.1f}%")
    row(0.677, 'Glucose Variability', f"{summary['cv_glucose']:.1f}%")
    fig.text(x_l, 0.664, 'Defined as percent coefficient of variation (%CV); target ≤36%',
             fontsize=6.5, va='center', color='dimgray', style='italic')
    row(0.647, 'Time in Tight Range (70-140 mg/dL)', f"{summary['titr']:.1f}%")

    # GRI zones A-E span 20 points each
    zone = 'ABCDE'[min(int(summary['gri'] // 20), 4)]
    row(0.630, 'Glycemia Risk Index (GRI)', f"{summary['gri']:.0f} (Zone {zone})")


def _agp_draw_time_in_ranges(fig, summary: dict, th: dict):
    keys = ['very_low', 'low', 'target', 'high', 'very_high']
    names = {
        'very_high': ('Very High', f">{th['very_high']:.0f} mg/dL"),
        'high': ('High', f"{th['high']:.0f}-{th['very_high']:.0f} mg/dL"),
        'target': ('Target Range', f"{th['low']:.0f}-{th['high']:.0f} mg/dL"),
        'low': ('Low', f"{th['very_low']:.0f}-{th['low']:.0f} mg/dL"),
        'very_low': ('Very Low', f"<{th['very_low']:.0f} mg/dL"),
    }
    bar_x, bar_w, bar_y0, bar_h = 0.60, 0.05, 0.640, 0.245

    # Every range keeps a visible slice so its label always has a segment to point at
    shown = np.array([max(summary[k], 1.5) for k in keys])
    shown = shown / shown.sum() * bar_h

    label_ys = np.linspace(bar_y0 + bar_h - 0.012, bar_y0 + 0.012, 5)
    y = bar_y0
    for k, h, label_y in zip(keys, shown, label_ys[::-1]):
        fig.add_artist(Rectangle((bar_x, y), bar_w, h, transform=fig.transFigure,
                                 facecolor=_AGP_COLORS[k], edgecolor='white', linewidth=1.2))
        fig.add_artist(plt.Line2D([bar_x + bar_w + 0.005, bar_x + bar_w + 0.02], [y + h / 2, label_y],
                                  transform=fig.transFigure, color='gray', lw=0.6))
        name, rng = names[k]
        fig.text(bar_x + bar_w + 0.025, label_y, name, fontsize=8, fontweight='bold', va='center',
                 color=_AGP_COLORS['text'])
        fig.text(bar_x + bar_w + 0.025, label_y - 0.012, rng, fontsize=6.5, va='center', color='dimgray')
        fig.text(0.95, label_y, f'{summary[k]:.0f}%', fontsize=10, fontweight='bold', ha='right', va='center',
                 color=_AGP_COLORS['text'])
        fig.text(0.95, label_y - 0.012, f'({_agp_time_per_day(summary[k])})', fontsize=6.5, ha='right',
                 va='center', color='dimgray')
        y += h


def _agp_draw_profile(fig, agp: pd.DataFrame, th: dict):
    ax = fig.add_axes([0.09, 0.355, 0.80, 0.200])
    x = agp.index.values
    ticks, tick_labels = _agp_ticks()
    clip = lambda v: np.clip(v, 0, 350)

    ax.fill_between(x, clip(agp['p05']), clip(agp['p95']), color='steelblue', alpha=0.25, linewidth=0)
    ax.fill_between(x, clip(agp['p25']), clip(agp['p75']), color='steelblue', alpha=0.60, linewidth=0)
    ax.plot(x, clip(agp['p05']), color='steelblue', linestyle='--', linewidth=0.9)
    ax.plot(x, clip(agp['p95']), color='steelblue', linestyle='--', linewidth=0.9)
    ax.plot(x, clip(agp['p50']), color='#1F3A5F', linewidth=2.2)
    for v in (th['low'], th['high']):
        ax.axhline(v, color=_AGP_COLORS['target'], linewidth=1.4)
    for v in (th['very_low'], th['very_high']):
        ax.axhline(v, color='lightgray', linewidth=0.8, zorder=0)

    # Percentile labels at the right edge, pushed apart when the curves converge
    label_ys = clip(agp[['p05', 'p25', 'p50', 'p75', 'p95']].iloc[-1].values)
    for i in range(1, 5):
        label_ys[i] = max(label_ys[i], label_ys[i - 1] + 14)
    for col, label, y in zip(['p05', 'p25', 'p50', 'p75', 'p95'], ['5%', '25%', '50%', '75%', '95%'], label_ys):
        ax.text(1452, y, label, fontsize=7, va='center', color=_AGP_COLORS['text'],
                fontweight='bold' if col == 'p50' else 'normal', clip_on=False)

    ax.set_xlim(0, 1440)
    ax.set_ylim(0, 350)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_yticks([th['very_low'], th['low'], th['high'], th['very_high'], 350])
    ax.tick_params(axis='y', labelsize=7.5, length=0)
    ax.tick_params(axis='x', length=0)
    ax.text(-0.065, 1.03, 'mg/dL', transform=ax.transAxes, fontsize=7, va='bottom', color='dimgray')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('lightgray')


def _agp_draw_daily_profiles(fig, data: pd.DataFrame, th: dict, end: pd.Timestamp):
    days = pd.date_range(end=end, periods=14, freq='D')
    x0, y_top, width, height, gap = 0.05, 0.270, 0.90 / 7, 0.105, 0.025
    low, high = th['low'], th['high']

    for i, day in enumerate(days):
        r, c = divmod(i, 7)
        ax = fig.add_axes([x0 + c * width, y_top - r * (height + gap) - height, width, height])
        day_df = data[data['day'] == day]
        if not day_df.empty:
            x = day_df['minute_of_day'].values
            g = np.clip(day_df['glucose'].values, 0, 350)
            ax.fill_between(x, high, g, where=g > high, color=_AGP_COLORS['high'], interpolate=True, linewidth=0)
            ax.fill_between(x, low, g, where=g < low, color=_AGP_COLORS['low'], interpolate=True, linewidth=0)
            ax.plot(x, g, color='steelblue', linewidth=0.7)
        ax.axhline(low, color=_AGP_COLORS['target'], linewidth=0.8)
        ax.axhline(high, color=_AGP_COLORS['target'], linewidth=0.8)
        ax.set_xlim(0, 1440)
        ax.set_ylim(0, 350)
        ax.set_xticks([720] if r == 1 else [])
        ax.set_xticklabels(['12pm'] if r == 1 else [], fontsize=6)
        ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_color('lightgray')
            spine.set_linewidth(0.6)
        ax.text(0.04, 0.92, f'{day:%d}', transform=ax.transAxes, fontsize=6.5, va='top', color='dimgray')
        if r == 0:
            ax.set_title(f'{day:%A}', fontsize=7, color=_AGP_COLORS['text'], pad=2)
        if c == 0:
            for v in (low, high):
                ax.text(-0.04, v, f'{v:.0f}', transform=ax.get_yaxis_transform(), fontsize=5.5,
                        ha='right', va='center', color='dimgray')


def export_agp_pdf(glucose_profiles, file_path: str = 'agp_report.pdf',
                   title: str = 'AGP Report: Continuous Glucose Monitoring') -> Path:
    """
    Exports a one-page A4 Ambulatory Glucose Profile (AGP) report following the
    IDC AGP Report v5 layout and the type 1 diabetes consensus targets: glucose statistics and targets, time in ranges,
    AGP percentile curves, and the daily glucose profiles of the last 14 days.

    Parameters
    ----------
    glucose_profiles : pd.DataFrame or list of pd.DataFrame
        One glucose profile or a list of glucose profiles. Each dataframe must contain
        columns `t` and `glucose` and have a homogeneous time grid.
    file_path : str, optional, default: 'agp_report.pdf'
        Destination PDF path.
    title : str, optional, default: 'AGP Report: Continuous Glucose Monitoring'
        Title shown in the PDF header.

    Returns
    -------
    pathlib.Path
        Path to the generated PDF file.

    References
    ----------
    Battelino et al., "Clinical targets for continuous glucose monitoring data
    interpretation: recommendations from the international consensus on time in
    range", Diabetes Care, 2019, vol. 42, pp. 1593-1603. DOI: 10.2337/dci19-0028.
    """
    if isinstance(glucose_profiles, pd.DataFrame):
        glucose_profiles = [glucose_profiles]
    elif isinstance(glucose_profiles, list):
        if len(glucose_profiles) == 0:
            raise ValueError('`glucose_profiles` must contain at least one dataframe.')
    else:
        raise TypeError('`glucose_profiles` must be a pd.DataFrame or a list of pd.DataFrame.')

    th = _AGP_THRESHOLDS

    prepared = [_prepare_agp_dataframe(data) for data in glucose_profiles]
    aggregated = pd.concat(prepared, ignore_index=True)
    if aggregated['glucose'].dropna().empty:
        raise ValueError('`glucose_profiles` must contain at least one non-NaN glucose value.')

    agp = _agp_percentiles(aggregated.dropna(subset=['glucose']))
    summary = _build_agp_summary(aggregated[['t', 'glucose']])

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.04, 0.955, title, fontsize=15, fontweight='bold', va='center', color=_AGP_COLORS['text'])
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.938, 0.938], transform=fig.transFigure, color='black', lw=1.2))

    _agp_section(fig, 0.905, 'GLUCOSE STATISTICS AND TARGETS')
    fig.text(0.58, 0.916, 'TIME IN RANGES', fontsize=9.5, fontweight='bold', va='center',
             color=_AGP_COLORS['text'])
    _agp_draw_statistics(fig, summary, th)
    _agp_draw_time_in_ranges(fig, summary, th)

    _agp_section(fig, 0.600, 'AMBULATORY GLUCOSE PROFILE (AGP)',
                 'AGP is a summary of glucose values from the report period, with median (50%) and other '
                 'percentiles shown as if they occurred in a single day.')
    _agp_draw_profile(fig, agp, th)

    _agp_section(fig, 0.300, 'DAILY GLUCOSE PROFILES',
                 'Each daily profile represents a midnight-to-midnight period.')
    _agp_draw_daily_profiles(fig, aggregated, th, summary['end'])

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return output_path
