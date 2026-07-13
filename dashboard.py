import json
import os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import yaml
from PIL import Image

from models.convlstm import ConvLSTMModel


LOCATION_DATA_PATH = "data/metadata/india_locations.json"
INDIA_REGION_BOUNDS = {
    "lat_min": 5.0,
    "lat_max": 35.0,
    "lon_min": 65.0,
    "lon_max": 100.0,
}
SHORT_TERM_FUTURE_DAYS = 14
MAX_DASHBOARD_FORECAST_DAYS = 10
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


st.set_page_config(
    page_title="ConvLSTM Climate Forecasting Evaluation",
    page_icon="Globe",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_styling():
    """Apply dashboard styling."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        body {
            color: #e0e0e0;
            background-color: #0c0f1e;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        h2 {
            font-size: 1.6rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-bottom: 1rem;
        }
        h3 {
            font-size: 1.2rem;
            font-weight: 600;
            color: #d1d1d1;
        }
        .metric-card {
            background-color: #1E293B;
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 100%;
        }
        .metric-card .stMetric {
            background-color: transparent;
            border: none;
            padding: 0;
        }
        .metric-card .stMetric > div:nth-child(1) {
            font-size: 1rem;
            color: #94A3B8;
        }
        .metric-card .stMetric > div:nth-child(2) {
            font-size: 2.25rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .metric-card.good {
            border-left: 5px solid #4ADE80;
        }
        hr {
            margin: 2.5rem 0;
            border-color: #334155;
        }
        .stDataFrame {
            background-color: #1E293B;
            border-radius: 0.5rem;
            border: 1px solid #334155;
        }
        .stDataFrame table {
            width: 100%;
        }
        .stDataFrame th {
            background-color: #334155;
            color: #FFFFFF;
            font-size: 1rem;
            font-weight: 600;
            text-align: left;
            padding: 0.75rem;
        }
        .stDataFrame td {
            padding: 0.75rem;
            font-size: 0.95rem;
            height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_evaluation_results(split="test"):
    """Load evaluation results from JSON."""
    results_path = f"outputs/evaluation/{split}_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_tensor_data(split="test"):
    """Load preprocessed tensors for prediction."""
    x_path = f"data/processed/tensors/{split}_X.npy"
    y_path = f"data/processed/tensors/{split}_y.npy"
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
    return np.load(x_path), np.load(y_path)


@st.cache_resource
def load_prediction_artifacts(model_path="experiments/latest/model.pth"):
    """Load the trained model and normalization metadata."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    mean = 288.15
    std = 10.0
    stats_path = "data/processed/mean_std.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        mean = float(stats.get("mean", mean))
        std = float(stats.get("std", std))

    if not os.path.exists(model_path):
        return None, config, mean, std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, config, mean, std


def denormalize_to_celsius(data, mean, std):
    """Convert normalized values to Celsius."""
    return data * std + mean - 273.15


@st.cache_data
def load_india_locations():
    """Load India state and city metadata for dashboard selectors."""
    if not os.path.exists(LOCATION_DATA_PATH):
        return None

    with open(LOCATION_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def lat_lon_to_grid_index(latitude, longitude, grid_shape):
    """Map a latitude/longitude pair to the nearest model grid cell."""
    rows, cols = grid_shape

    lat_ratio = (latitude - INDIA_REGION_BOUNDS["lat_min"]) / (
        INDIA_REGION_BOUNDS["lat_max"] - INDIA_REGION_BOUNDS["lat_min"]
    )
    lon_ratio = (longitude - INDIA_REGION_BOUNDS["lon_min"]) / (
        INDIA_REGION_BOUNDS["lon_max"] - INDIA_REGION_BOUNDS["lon_min"]
    )

    lat_idx = int(round(lat_ratio * (rows - 1)))
    lon_idx = int(round(lon_ratio * (cols - 1)))

    lat_idx = int(np.clip(lat_idx, 0, rows - 1))
    lon_idx = int(np.clip(lon_idx, 0, cols - 1))
    return lat_idx, lon_idx


def get_dashboard_today():
    """Get today's date in India time for forecast selection."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


@st.cache_data(ttl=1800)
def fetch_live_city_forecast(latitude, longitude, start_date, end_date):
    """Fetch a city-level weather forecast from Open-Meteo for the requested date window."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "Asia/Kolkata",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "weather_code",
                    "wind_speed_10m",
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                ]
            ),
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    forecast_df = pd.DataFrame(
        {
            "date": daily.get("time", []),
            "weather_code": daily.get("weather_code", []),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
            "precip_prob_max": daily.get("precipitation_probability_max", []),
            "wind_speed_max": daily.get("wind_speed_10m_max", []),
        }
    )
    if forecast_df.empty:
        raise ValueError("No daily forecast data returned from the live weather service.")

    forecast_df["summary"] = forecast_df["weather_code"].map(
        lambda code: WMO_WEATHER_CODES.get(int(code), "Unknown")
    )
    return {
        "current": payload.get("current", {}),
        "daily": forecast_df,
    }


def build_heatmap(data, title, color_scale="Turbo", cmid=None, point=None):
    """Create a consistent heatmap figure."""
    fig = px.imshow(
        data,
        color_continuous_scale=color_scale,
        aspect="auto",
        origin="lower",
    )
    fig.update_layout(
        title=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar_title="deg C",
    )
    if cmid is not None:
        fig.update_coloraxes(cmid=cmid)
        
    # Add specific city marker if provided
    if point is not None:
        lat_idx, lon_idx, city_name = point
        fig.add_trace(go.Scatter(
            x=[lon_idx],
            y=[lat_idx],
            mode='markers+text',
            marker=dict(color='white', size=12, symbol='star', line=dict(color='black', width=1)),
            text=[f"  {city_name}"],
            textposition="middle right",
            textfont=dict(color="white", size=14, weight="bold"),
            name=city_name,
            showlegend=False
        ))
    return fig


def run_single_prediction(split, sample_idx):
    """Run one-step prediction for the selected sample."""
    model, config, mean, std = load_prediction_artifacts()
    X_data, y_data = load_tensor_data(split)

    if model is None or X_data is None or y_data is None:
        return None

    device = next(model.parameters()).device
    sample_x = X_data[sample_idx:sample_idx + 1]
    sample_y = y_data[sample_idx:sample_idx + 1]

    with torch.no_grad():
        pred = model(torch.tensor(sample_x, dtype=torch.float32, device=device))

    pred_np = pred.cpu().numpy()[0, 0]
    target_np = sample_y[0, 0]
    input_last_day_np = sample_x[0, -1, 0]

    pred_c = denormalize_to_celsius(pred_np, mean, std)
    target_c = denormalize_to_celsius(target_np, mean, std)
    input_last_day_c = denormalize_to_celsius(input_last_day_np, mean, std)
    error_c = pred_c - target_c

    return {
        "prediction": pred_c,
        "target": target_c,
        "last_input_day": input_last_day_c,
        "error": error_c,
        "metrics": {
            "pred_mean": float(np.mean(pred_c)),
            "target_mean": float(np.mean(target_c)),
            "rmse": float(np.sqrt(np.mean((pred_c - target_c) ** 2))),
            "mae": float(np.mean(np.abs(pred_c - target_c))),
            "error_min": float(np.min(error_c)),
            "error_max": float(np.max(error_c)),
            "error_mean": float(np.mean(error_c)),
        },
        "sequence_length": int(config.get("sequence_length", 7)),
    }


def display_header(split, results_loaded):
    """Display the page header and status message."""
    st.title("ConvLSTM Climate Forecasting Model Evaluation")
    st.subheader(f"ERA5 {split.capitalize()} Set Results")
    if results_loaded:
        st.success("Evaluation results loaded successfully. The dashboard is up to date.")
    else:
        st.error(f"No evaluation results found for '{split}'. Run `python evaluate_model.py` first.")
    st.markdown("---")


def display_summary_metrics(metrics):
    """Display top-line metrics."""
    st.subheader("Top-Line Performance")
    cols = st.columns(4)

    with cols[0]:
        st.markdown("<div class='metric-card good'>", unsafe_allow_html=True)
        st.metric(label="RMSE (deg C)", value=f"{metrics.get('rmse_celsius', 0):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="MAE (deg C)", value=f"{metrics.get('mae_celsius', 0):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Correlation", value=f"{metrics.get('correlation', 0):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[3]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Normalized RMSE", value=f"{metrics.get('rmse_normalized', 0):.4f}")
        st.markdown("</div>", unsafe_allow_html=True)


def display_overview_tab(results):
    """Display overview content."""
    display_summary_metrics(results["metrics"])
    st.markdown("---")
    st.subheader("Visual Error Analysis")

    eval_path = f"outputs/visualizations/{results['split']}_evaluation.png"
    if os.path.exists(eval_path):
        st.image(
            Image.open(eval_path),
            caption="Error distribution, prediction accuracy, and spatial RMSE",
            use_container_width=True,
        )
    else:
        st.warning("Evaluation visualization not found. Run `evaluate_model.py` to generate it.")


def display_spatial_analysis_tab(results):
    """Display spatial analysis content."""
    st.subheader("Sample Prediction vs Actual Map")
    map_path = f"outputs/visualizations/{results['split']}_prediction_map.png"
    if os.path.exists(map_path):
        st.image(
            Image.open(map_path),
            caption="Comparison of a sample prediction against ground truth",
            use_container_width=True,
        )
    else:
        st.warning("Prediction map not found.")

    st.markdown("---")
    st.subheader("Spatial Performance Metrics")

    spatial = results.get("spatial_metrics", {})
    df = pd.DataFrame(
        {
            "Metric": [
                "Min RMSE (Best Grid)",
                "Mean RMSE",
                "Max RMSE (Worst Grid)",
                "Std Dev of RMSE",
            ],
            "Value (K)": [
                spatial.get("spatial_rmse_min", 0),
                spatial.get("spatial_rmse_mean", 0),
                spatial.get("spatial_rmse_max", 0),
                spatial.get("spatial_rmse_std", 0),
            ],
        }
    )

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.dataframe(df.style.format({"Value (K)": "{:.4f}"}), use_container_width=True, hide_index=True)

    with col2:
        fig = px.bar(
            df,
            x="Metric",
            y="Value (K)",
            title="RMSE Distribution Across Grid Points",
            color="Value (K)",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)


def display_detailed_metrics_tab(results):
    """Display detailed metrics content."""
    col1, col2 = st.columns(2)

    with col2:
        st.subheader("Full Metrics Breakdown")
        df = pd.DataFrame(
            {
                "Metric": [
                    "RMSE (normalized)",
                    "MAE (normalized)",
                    "RMSE (Kelvin)",
                    "MAE (Kelvin)",
                    "RMSE (Celsius)",
                    "MAE (Celsius)",
                    "Correlation",
                ],
                "Value": [
                    results["metrics"]["rmse_normalized"],
                    results["metrics"]["mae_normalized"],
                    results["metrics"]["rmse_kelvin"],
                    results["metrics"]["mae_kelvin"],
                    results["metrics"]["rmse_celsius"],
                    results["metrics"]["mae_celsius"],
                    results["metrics"]["correlation"],
                ],
            }
        )
        st.dataframe(
            df.style.format({"Value": "{:.4f}"}).set_properties(**{"text-align": "right"}, subset=["Value"]),
            use_container_width=True,
            hide_index=True,
        )

    with col1:
        st.subheader("Mean Temperature Comparison")
        ranges = results.get("value_ranges", {})
        pred_mean = ranges.get("pred_mean", 0)
        target_mean = ranges.get("target_mean", 0)

        fig = go.Figure(
            data=[
                go.Bar(name="Actual", x=["Mean Temperature"], y=[target_mean], marker_color="#1f77b4"),
                go.Bar(name="Predicted", x=["Mean Temperature"], y=[pred_mean], marker_color="#ff7f0e"),
            ]
        )
        fig.update_layout(
            title="Mean Temperature Comparison",
            yaxis_title="Temperature (deg C)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            height=400,
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Mean Prediction Error", f"{abs(pred_mean - target_mean):.2f} deg C")

        range_df = pd.DataFrame(
            {
                "Type": ["Actual Min", "Actual Max", "Predicted Min", "Predicted Max"],
                "Temperature (deg C)": [
                    ranges.get("target_min", 0),
                    ranges.get("target_max", 0),
                    ranges.get("pred_min", 0),
                    ranges.get("pred_max", 0),
                ],
            }
        )
        st.dataframe(
            range_df.style.format({"Temperature (deg C)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )


def display_report_tab(results):
    """Display report tab content."""
    st.subheader("Evaluation Summary Report")

    report = f"""
    ### Model Performance Summary

    - **Dataset**: {results['split'].upper()} Set
    - **Number of Samples**: {results['num_samples']}
    - **Spatial Resolution**: {results['spatial_shape'][0]} x {results['spatial_shape'][1]} grid points

    ---

    ### Accuracy Metrics

    - **Root Mean Square Error (RMSE)**: **{results['metrics']['rmse_celsius']:.4f} deg C** (Kelvin: {results['metrics']['rmse_kelvin']:.4f} K)
    - **Mean Absolute Error (MAE)**: **{results['metrics']['mae_celsius']:.4f} deg C** (Kelvin: {results['metrics']['mae_kelvin']:.4f} K)
    - **Correlation Coefficient**: **{results['metrics']['correlation']:.4f}**

    ---

    ### Spatial Analysis

    - **Best Performance (Min RMSE)**: {results['spatial_metrics']['spatial_rmse_min']:.4f} K
    - **Average Performance (Mean RMSE)**: {results['spatial_metrics']['spatial_rmse_mean']:.4f} K
    - **Worst Performance (Max RMSE)**: {results['spatial_metrics']['spatial_rmse_max']:.4f} K

    ---

    ### Temperature Ranges

    - **Actual Data Range**: {results['value_ranges']['target_min']:.2f} deg C to {results['value_ranges']['target_max']:.2f} deg C (Mean: {results['value_ranges']['target_mean']:.2f} deg C)
    - **Model Prediction Range**: {results['value_ranges']['pred_min']:.2f} deg C to {results['value_ranges']['pred_max']:.2f} deg C (Mean: {results['value_ranges']['pred_mean']:.2f} deg C)

    ---

    ### Interpretation
    """

    rmse_c = results["metrics"]["rmse_celsius"]
    if rmse_c < 1.0:
        report += f"Excellent performance with an RMSE of **{rmse_c:.4f} deg C**."
    elif rmse_c < 2.0:
        report += f"Good performance with an RMSE of **{rmse_c:.4f} deg C**."
    else:
        report += f"There is room for improvement with an RMSE of **{rmse_c:.4f} deg C**."

    st.markdown(report, unsafe_allow_html=True)





def get_date_ranges_for_split(split):
    """Get date ranges for each split."""
    date_ranges = {
        "train": (datetime(2019, 1, 1), datetime(2023, 12, 31)),
        "val": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "test": (datetime(2025, 1, 1), datetime(2025, 12, 31)),
    }
    return date_ranges.get(split, date_ranges["test"])


def get_target_dates_for_split(split, num_samples, sequence_length=7):
    """Get the target date represented by each sample in a split."""
    start_date, end_date = get_date_ranges_for_split(split)
    first_target_date = start_date + timedelta(days=sequence_length)
    return [first_target_date + timedelta(days=idx) for idx in range(num_samples)]


def circular_day_distance(day_a, day_b, year_length=366):
    """Smallest circular distance between two day-of-year values."""
    distance = abs(day_a - day_b)
    return min(distance, year_length - distance)


@st.cache_data
def get_climatology_for_day(target_day_of_year, sequence_length=7, window_days=10):
    """Average normalized target grids near a target day-of-year across all splits."""
    matched_grids = []

    for split in ["train", "val", "test"]:
        _, y_data = load_tensor_data(split)
        if y_data is None:
            continue

        sample_dates = get_target_dates_for_split(split, len(y_data), sequence_length=sequence_length)
        matched_indices = [
            idx
            for idx, sample_date in enumerate(sample_dates)
            if circular_day_distance(sample_date.timetuple().tm_yday, target_day_of_year) <= window_days
        ]

        if matched_indices:
            matched_grids.append(y_data[matched_indices, 0])

    if not matched_grids:
        raise ValueError("No climatology samples found for the requested day.")

    return np.concatenate(matched_grids, axis=0).mean(axis=0)


@st.cache_data
def get_recent_anomaly_grid(split="test", sequence_length=7, climatology_window_days=10):
    """Estimate the recent anomaly relative to climatology in normalized space."""
    _, y_data = load_tensor_data(split)
    if y_data is None:
        raise ValueError(f"Tensor data for split '{split}' could not be loaded.")

    _, end_date = get_date_ranges_for_split(split)
    baseline_grid = get_climatology_for_day(
        end_date.timetuple().tm_yday,
        sequence_length=sequence_length,
        window_days=climatology_window_days,
    )
    recent_mean_grid = y_data[-min(sequence_length, len(y_data)) :, 0].mean(axis=0)
    return recent_mean_grid - baseline_grid


def get_sample_idx_from_date(target_date, split):
    """Approximate sample index from a target date."""
    X_data, _ = load_tensor_data(split)
    if X_data is None:
        return None
    
    # Convert date to datetime if needed
    if isinstance(target_date, date) and not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())
    
    start_date, end_date = get_date_ranges_for_split(split)
    total_samples = len(X_data)
    
    # Calculate percentage through the date range
    total_days = (end_date - start_date).days
    days_from_start = (target_date - start_date).days
    
    # Clamp to valid range
    if days_from_start < 0:
        days_from_start = 0
    elif days_from_start > total_days:
        days_from_start = total_days
    
    # Approximate sample index
    approximate_idx = int((days_from_start / total_days) * total_samples)
    approximate_idx = min(approximate_idx, total_samples - 1)
    
    return approximate_idx


def predict_future_single_date(target_date, split="test"):
    """
    Predict for a specific future date using auto-regressive forecasting.
    
    Args:
        target_date: datetime object for the future date
        split: which split to use for initial sequence
    
    Returns:
        dict with prediction data for that date
    """
    from future_predict import load_model_and_stats, denormalize

    _, _, mean, std = load_prediction_artifacts()
    
    date_ranges = {
        "train": (datetime(2019, 1, 1), datetime(2023, 12, 31)),
        "val": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "test": (datetime(2025, 1, 1), datetime(2025, 12, 31)),
    }
    _, end_date = date_ranges.get(split, date_ranges["test"])
    
    # Convert date to datetime if needed
    if isinstance(target_date, date) and not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())
    
    days_ahead = (target_date - end_date).days
    
    if days_ahead < 1:
        raise ValueError(f"Target date {target_date} is not in the future")
    
    if days_ahead > 365:
        raise ValueError(f"Cannot predict more than 365 days ahead")

    sequence_length = 7
    target_day_of_year = target_date.timetuple().tm_yday
    climatology_norm = get_climatology_for_day(
        target_day_of_year,
        sequence_length=sequence_length,
        window_days=10,
    )
    recent_anomaly_norm = get_recent_anomaly_grid(
        split=split,
        sequence_length=sequence_length,
        climatology_window_days=10,
    )

    # For long horizons, use a climatology-based estimate with a decaying recent anomaly.
    if days_ahead > SHORT_TERM_FUTURE_DAYS:
        anomaly_scale = float(np.exp(-(days_ahead - SHORT_TERM_FUTURE_DAYS) / 30.0))
        pred_norm_grid = climatology_norm + recent_anomaly_norm * anomaly_scale
        pred_celsius = denormalize(pred_norm_grid, mean=mean, std=std)
        return {
            "prediction": pred_celsius,
            "mean_temp": float(np.mean(pred_celsius)),
            "min_temp": float(np.min(pred_celsius)),
            "max_temp": float(np.max(pred_celsius)),
            "std_temp": float(np.std(pred_celsius)),
            "is_future": True,
            "method": "seasonal_climatology",
            "days_ahead": days_ahead,
            "confidence_note": "Seasonal baseline with recent anomaly decay for long-range outlooks.",
        }

    model, _, mean, std, device = load_model_and_stats()

    y_path = f"data/processed/tensors/{split}_y.npy"
    y_data = np.load(y_path)
    sequence = y_data[-sequence_length:].copy()
    current_date = end_date

    for day_idx in range(days_ahead):
        current_date = current_date + timedelta(days=1)

        X_input = torch.from_numpy(sequence).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_norm = model(X_input)

        pred_norm_np = pred_norm.squeeze(0).cpu().numpy()
        target_climatology_norm = get_climatology_for_day(
            current_date.timetuple().tm_yday,
            sequence_length=sequence_length,
            window_days=10,
        )
        blend_weight = min(0.12 + 0.04 * day_idx, 0.55)
        pred_norm_np = pred_norm_np.squeeze(0)
        pred_norm_np = (1.0 - blend_weight) * pred_norm_np + blend_weight * target_climatology_norm
        pred_norm_np = pred_norm_np + recent_anomaly_norm * np.exp(-day_idx / 10.0) * 0.15
        pred_norm_np = np.clip(pred_norm_np, -3.0, 3.0)

        if current_date == target_date:
            pred_celsius = denormalize(pred_norm_np, mean, std)

            return {
                "prediction": pred_celsius,
                "mean_temp": float(np.mean(pred_celsius)),
                "min_temp": float(np.min(pred_celsius)),
                "max_temp": float(np.max(pred_celsius)),
                "std_temp": float(np.std(pred_celsius)),
                "is_future": True,
                "method": "hybrid_short_term",
                "days_ahead": days_ahead,
                "confidence_note": "Short-range ConvLSTM forecast blended toward seasonal climatology for stability.",
            }

        sequence = np.vstack([sequence[1:], pred_norm_np[np.newaxis, np.newaxis, ...]])

    raise ValueError(f"Failed to predict for {target_date}")



def display_daywise_prediction_tab_legacy(split):
    """Display day-wise prediction with date picker (historical and future)."""
    st.subheader("📅 Day-Wise Climate Prediction")
    st.caption("Choose today or any of the next 10 days to predict weather for a city in India.")

    prediction_split = "test"
    X_data, y_data = load_tensor_data(prediction_split)
    model, config, mean, std = load_prediction_artifacts()
    india_locations = load_india_locations()
    
    if X_data is None or y_data is None:
        st.warning("Tensor files not found. Run preprocessing first.")
        return
    
    if model is None:
        st.warning("Model not found. Train the model first.")
        return

    if india_locations is None:
        st.warning("India location file not found. Please make sure data/metadata/india_locations.json exists.")
        return

    grid_shape = (y_data.shape[-2], y_data.shape[-1])
    state_names = sorted(india_locations["states"].keys())
    default_state = "Maharashtra" if "Maharashtra" in state_names else state_names[0]

    st.markdown("### Location Selection")
    location_col1, location_col2, location_col3 = st.columns(3)

    with location_col1:
        selected_country = st.selectbox(
            "Country",
            [india_locations["country"]],
            index=0,
            key=f"country_select_{split}",
        )

    with location_col2:
        selected_state = st.selectbox(
            "State / Union Territory",
            state_names,
            index=state_names.index(default_state),
            key=f"state_select_{split}",
        )

    state_data = india_locations["states"][selected_state]
    city_records = state_data.get("cities", [])
    city_names = [city["name"] for city in city_records]

    if not city_records:
        st.warning(f"No cities found for {selected_state}.")
        return

    with location_col3:
        selected_city = st.selectbox(
            "City",
            city_names,
            key=f"city_select_{split}",
        )

    selected_city_record = next(city for city in city_records if city["name"] == selected_city)
    lat = float(selected_city_record["latitude"])
    lon = float(selected_city_record["longitude"])
    lat_idx, lon_idx = lat_lon_to_grid_index(lat, lon, grid_shape)
    point_data = (lat_idx, lon_idx, selected_city)

    st.caption(
        f"{selected_country}: {len(state_names)} states/UTs loaded, "
        f"{sum(len(state['cities']) for state in india_locations['states'].values()):,} cities available."
    )
    
    _, end_date = get_date_ranges_for_split(prediction_split)
    today_date = get_dashboard_today()
    prediction_window_start = max(today_date, date(2026, 1, 1))
    latest_supported_date = (end_date + timedelta(days=365)).date()

    if prediction_window_start > latest_supported_date:
        st.warning(
            f"Predictions are currently available only until {latest_supported_date.strftime('%B %d, %Y')}."
        )
        return

    prediction_window_end = min(
        prediction_window_start + timedelta(days=MAX_DASHBOARD_FORECAST_DAYS),
        latest_supported_date,
    )
    available_future_days = (prediction_window_end - prediction_window_start).days
    
    # Date picker with better layout
    st.markdown("### 📍 Select a Date")
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    
    with col1:
        selected_date = st.date_input(
            "Pick a date to forecast",
            value=prediction_window_start,
            min_value=prediction_window_start,
            max_value=prediction_window_end,
            key="prediction_date"
        )
    
    with col2:
        sequence_length = int(config.get("sequence_length", 7))
        st.info(
            f"📊 Today + next {available_future_days} days "
            f"(through {prediction_window_end.strftime('%b %d, %Y')})"
        )
    
    with col3:
        if isinstance(selected_date, date) and not isinstance(selected_date, datetime):
            selected_datetime = datetime.combine(selected_date, datetime.min.time())
        else:
            selected_datetime = selected_date

        days_from_today = (selected_datetime.date() - prediction_window_start).days
        if days_from_today == 0:
            st.success("✅ Today")
        else:
            st.warning(f"🔮 Next {days_from_today} day(s)")
    
    if st.button("🎯 Generate Forecast", type="primary", key="daywise_predict", use_container_width=True):
        # Convert date to datetime
        if isinstance(selected_date, date) and not isinstance(selected_date, datetime):
            selected_datetime = datetime.combine(selected_date, datetime.min.time())
        else:
            selected_datetime = selected_date
        days_from_today = (selected_datetime.date() - prediction_window_start).days
        
        try:
            if selected_datetime > end_date:
                with st.spinner(f"🔮 Forecasting weather for {selected_date.strftime('%B %d, %Y')}..."):
                    future_pred = predict_future_single_date(selected_datetime, prediction_split)
                
                # Better header
                col_head1, col_head2 = st.columns([3, 1])
                with col_head1:
                    if days_from_today == 0:
                        st.markdown(f"### 🔮 Today Forecast - {selected_date.strftime('%B %d, %Y')}")
                        st.markdown("*Predicting weather for today*")
                    else:
                        st.markdown(f"### 🔮 Forecast - {selected_date.strftime('%B %d, %Y')}")
                        st.markdown(f"*Predicting weather for **{days_from_today} day(s)** from today*")
                with col_head2:
                    st.markdown("### Today" if days_from_today == 0 else f"### +{days_from_today}d")

                method_label = future_pred.get("method", "future_forecast").replace("_", " ").title()
                st.info(
                    f"Forecast method: **{method_label}**. {future_pred.get('confidence_note', '')}"
                )
                
                st.markdown("---")
                
                # City-specific display card
                st.subheader(f"🏙️ Forecast specifically for {selected_city}, {selected_state}")
                
                city_temp = future_pred['prediction'][lat_idx, lon_idx]
                regional_delta = city_temp - future_pred['mean_temp']
                city_col1, city_col2, city_col3 = st.columns(3)
                with city_col1:
                    st.metric(f"{selected_city} Temperature", f"{city_temp:.1f}°C")
                with city_col2:
                    st.metric(
                        "Vs Regional Average",
                        f"{future_pred['mean_temp']:.1f}°C",
                        delta=f"{regional_delta:+.1f}°C at {selected_city}",
                    )
                with city_col3:
                    st.write(f"**Coordinates:** {lat:.2f}°N, {lon:.2f}°E")
                    st.write(f"**Grid Index:** [{lat_idx}, {lon_idx}]")
                    
                st.markdown("---")
                
                # Main temperature display card
                st.subheader("🌏 Regional Temperature Context")
                st.caption("These values describe the whole India forecast grid for the selected date, so they do not change when only the city changes.")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "Regional Average",
                        f"{future_pred['mean_temp']:.1f}°C",
                        help="Average across the entire region"
                    )
                
                with metric_col2:
                    st.metric(
                        "Regional Maximum",
                        f"{future_pred['max_temp']:.1f}°C",
                        delta=f"+{future_pred['max_temp'] - future_pred['mean_temp']:.1f}°C",
                        help="Highest temperature in the region"
                    )
                
                with metric_col3:
                    st.metric(
                        "Regional Minimum",
                        f"{future_pred['min_temp']:.1f}°C",
                        delta=f"{future_pred['min_temp'] - future_pred['mean_temp']:.1f}°C",
                        help="Lowest temperature in the region"
                    )
                
                with metric_col4:
                    temp_range = future_pred['max_temp'] - future_pred['min_temp']
                    st.metric(
                        "Temperature Variation",
                        f"{temp_range:.1f}°C",
                        help="Difference between hottest and coolest areas"
                    )
                
                st.markdown("---")
                
                # Temperature gauge visualization
                st.subheader("📈 City Temperature Summary")
                
                col_gauge1, col_gauge2 = st.columns([1.5, 1])
                
                with col_gauge1:
                    # Create a temperature scale visualization for the selected city.
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=city_temp,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{selected_city} Temperature (°C)"},
                        delta={'reference': future_pred['mean_temp'], 'position': 'top'},
                        gauge={
                            'axis': {'range': [-10, 50]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-10, 10], 'color': "#1f77b4"},  # Cold
                                {'range': [10, 20], 'color': "#2ca02c"},  # Cool
                                {'range': [20, 30], 'color': "#ff7f0e"},  # Warm
                                {'range': [30, 50], 'color': "#d62728"},  # Hot
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 30
                            }
                        }
                    ))
                    fig.update_layout(
                        height=350,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_gauge2:
                    # Interpretation guide
                    st.markdown("### 📋 What It Means")
                    
                    avg_temp = city_temp
                    if avg_temp < 10:
                        interpretation = "❄️ **Freezing Cold** - Expect frost and possibly snow"
                    elif avg_temp < 15:
                        interpretation = "🧊 **Cold** - Wear warm clothes"
                    elif avg_temp < 20:
                        interpretation = "🌤️ **Cool** - Comfortable weather"
                    elif avg_temp < 25:
                        interpretation = "☀️ **Warm** - Pleasant and sunny"
                    elif avg_temp < 30:
                        interpretation = "🔥 **Hot** - Stay hydrated"
                    else:
                        interpretation = "🌡️ **Very Hot** - Extreme heat conditions"
                    
                    st.markdown(interpretation)
                    st.caption(
                        f"{selected_city} is {regional_delta:+.1f}°C compared with the regional average on this date."
                    )
                
                st.markdown("---")
                
                # Spatial forecast map
                st.subheader("🗺️ Regional Temperature Map")
                st.caption("Hover over the map to see exact temperatures for each area")
                
                st.plotly_chart(
                    build_heatmap(
                        future_pred["prediction"],
                        f"Temperature Distribution - {selected_date.strftime('%B %d, %Y')}",
                        color_scale="RdYlBu_r",
                        point=point_data
                    ),
                    use_container_width=True
                )
                
                st.info(
                    "💡 **Note:** This is an AI forecast based on historical patterns. "
                    "Accuracy decreases the further into the future we predict."
                )
                
            else:
                # Use historical prediction for dates within data range
                sample_idx = get_sample_idx_from_date(selected_datetime, split)
                
                if sample_idx is None:
                    st.error("Could not determine sample index for selected date.")
                    return
                
                # Run prediction
                prediction = run_single_prediction(split, sample_idx)
                
                if prediction is None:
                    st.error("Prediction failed.")
                    return
                
                # Better header for historical
                st.markdown(f"### ✅ Historical Analysis - {selected_date.strftime('%B %d, %Y')}")
                st.markdown("*Comparing AI prediction vs actual recorded data*")
                st.markdown("---")
                
                # City-specific display card
                st.subheader(f"🏙️ Forecast specifically for {selected_city}, {selected_state}")
                
                city_temp = prediction['prediction'][lat_idx, lon_idx]
                actual_temp = prediction['target'][lat_idx, lon_idx]
                city_err = city_temp - actual_temp
                
                city_col1, city_col2, city_col3 = st.columns(3)
                with city_col1:
                    st.metric(f"Predicted Temperature", f"{city_temp:.1f}°C")
                with city_col2:
                    st.metric(
                        f"Actual Temperature", 
                        f"{actual_temp:.1f}°C",
                        delta=f"{city_err:+.1f}°C Error",
                        delta_color="inverse"
                    )
                with city_col3:
                    st.write(f"**Coordinates:** {lat}°N, {lon}°E")
                st.markdown("---")
                
                # Main metrics with better layout
                st.subheader("🎯 Regional Model Accuracy")
                
                acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
                
                # Color based on RMSE performance
                rmse = prediction['metrics']['rmse']
                if rmse < 1.0:
                    rmse_color = "🟢"
                    rmse_desc = "Excellent"
                elif rmse < 2.0:
                    rmse_color = "🟡"
                    rmse_desc = "Good"
                else:
                    rmse_color = "🔴"
                    rmse_desc = "Moderate"
                
                with acc_col1:
                    st.metric(
                        "Prediction Error (RMSE)",
                        f"{rmse:.2f}°C",
                        f"{rmse_color} {rmse_desc}",
                        help="How far off the predictions were on average"
                    )
                
                with acc_col2:
                    st.metric(
                        "Average Error (MAE)",
                        f"{prediction['metrics']['mae']:.2f}°C",
                        help="Typical prediction mistake"
                    )
                
                with acc_col3:
                    st.metric(
                        "AI Predicted Mean",
                        f"{prediction['metrics']['pred_mean']:.1f}°C",
                        help="What the AI thought"
                    )
                
                with acc_col4:
                    st.metric(
                        "Actual Mean",
                        f"{prediction['metrics']['target_mean']:.1f}°C",
                        help="What really happened"
                    )
                
                st.markdown("---")
                
                # Temperature comparison with better visuals
                st.subheader("🌡️ Temperature Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Better histogram
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=prediction['target'].flatten(),
                        name="Actual",
                        opacity=0.6,
                        marker_color="#1f77b4",
                        nbinsx=25
                    ))
                    fig.add_trace(go.Histogram(
                        x=prediction['prediction'].flatten(),
                        name="AI Prediction",
                        opacity=0.6,
                        marker_color="#ff7f0e",
                        nbinsx=25
                    ))
                    fig.update_layout(
                        title="Distribution of Temperatures Across Region",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Number of Locations",
                        barmode="overlay",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Statistics comparison
                    stats_data = {
                        "📊 Metric": ["🌡️ Average", "🥶 Coldest", "🔥 Hottest", "📈 Variation"],
                        "🤖 AI Predicted": [
                            f"{np.mean(prediction['prediction']):.1f}°C",
                            f"{np.min(prediction['prediction']):.1f}°C",
                            f"{np.max(prediction['prediction']):.1f}°C",
                            f"{np.std(prediction['prediction']):.1f}°C"
                        ],
                        "📡 Actual": [
                            f"{np.mean(prediction['target']):.1f}°C",
                            f"{np.min(prediction['target']):.1f}°C",
                            f"{np.max(prediction['target']):.1f}°C",
                            f"{np.std(prediction['target']):.1f}°C"
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True, column_config={
                        "📊 Metric": st.column_config.TextColumn(width="medium"),
                        "🤖 AI Predicted": st.column_config.TextColumn(width="medium"),
                        "📡 Actual": st.column_config.TextColumn(width="medium"),
                    })
                
                st.markdown("---")
                
                # Spatial maps with better explanations
                st.subheader("🗺️ Temperature Maps by Region")
                st.caption("Compare how temperature varied across different areas")
                
                map_cols = st.columns(3)
                with map_cols[0]:
                    st.plotly_chart(
                        build_heatmap(
                            prediction["last_input_day"],
                            "7 Days Before"
                        ),
                        use_container_width=True
                    )
                    st.caption("Reference: Temperature 7 days ago")
                
                with map_cols[1]:
                    st.plotly_chart(
                        build_heatmap(
                            prediction["prediction"],
                            f"AI Prediction",
                            point=point_data
                        ),
                        use_container_width=True
                    )
                    st.caption("What the AI model predicted")
                
                with map_cols[2]:
                    st.plotly_chart(
                        build_heatmap(
                            prediction["target"],
                            f"Actual Temperature",
                            point=point_data
                        ),
                        use_container_width=True
                    )
                    st.caption("What actually happened")
                
                st.markdown("---")
                
                # Error analysis with better visualization
                st.subheader("📊 Prediction Error Analysis")
                
                error_min = prediction["metrics"]["error_min"]
                error_max = prediction["metrics"]["error_max"]
                error_mean = prediction["metrics"]["error_mean"]
                
                err_col1, err_col2, err_col3 = st.columns(3)
                
                with err_col1:
                    st.metric(
                        "Best Prediction",
                        f"{abs(error_min):.2f}°C error",
                        "Closest to reality",
                    )
                
                with err_col2:
                    st.metric(
                        "Average Error",
                        f"{abs(error_mean):.2f}°C",
                        "Typical difference",
                    )
                
                with err_col3:
                    st.metric(
                        "Worst Prediction",
                        f"{abs(error_max):.2f}°C error",
                        "Furthest from reality",
                    )
                
                st.plotly_chart(
                    build_heatmap(
                        prediction["error"],
                        "Prediction Error Map (Red = Overestimated | Blue = Underestimated)",
                        color_scale="RdBu_r",
                        cmid=0
                    ),
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Forecast generation failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())



def predict_future_autoregressive(num_days, split="test"):
    """
    Auto-regressive prediction for multiple future days.
    Predicts forward from the last available data point.
    """
    from future_predict import load_model_and_stats, get_last_sequence, denormalize
    
    model, config, mean, std, device = load_model_and_stats()
    
    # Get the last 7-day sequence to start with
    y_path = f"data/processed/tensors/{split}_y.npy"
    y_data = np.load(y_path)
    
    # Use last 7 samples as starting sequence
    sequence = y_data[-7:].copy()  # Shape: (7, 121, 141)
    
    # Determine start date
    date_ranges = {
        "train": (datetime(2019, 1, 1), datetime(2023, 12, 31)),
        "val": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "test": (datetime(2025, 1, 1), datetime(2025, 12, 31)),
    }
    _, end_date = date_ranges.get(split, date_ranges["test"])
    
    predictions_by_date = {}
    current_date = end_date
    
    for day_idx in range(num_days):
        current_date = current_date + timedelta(days=1)
        
        # Prepare input for model: sequence is (7, 1, 121, 141)
        X_input = torch.from_numpy(sequence).unsqueeze(0).float().to(device)  # (1, 7, 1, 121, 141)
        
        # Make prediction
        with torch.no_grad():
            pred_norm = model(X_input)  # (1, 1, 121, 141)
        
        pred_norm_np = pred_norm.squeeze(0).cpu().numpy()  # (1, 121, 141)
        
        # Prevent exploding predictions over multiple days
        pred_norm_np = np.clip(pred_norm_np, -4.0, 4.0)
        pred_norm_np = pred_norm_np * 0.95
        
        pred_celsius = denormalize(pred_norm_np.squeeze(0), mean, std)  # (121, 141)
        
        # Store result
        predictions_by_date[current_date] = {
            "prediction": pred_celsius,
            "mean_temp": float(np.mean(pred_celsius)),
            "min_temp": float(np.min(pred_celsius)),
            "max_temp": float(np.max(pred_celsius)),
            "std_temp": float(np.std(pred_celsius)),
        }
        
        # Slide window: remove oldest day, add new prediction
        sequence = np.vstack([sequence[1:], pred_norm_np[np.newaxis, ...]])
    
    return predictions_by_date, end_date


def display_future_prediction_tab(split):
    """Display future predictions (beyond training data)."""
    st.subheader("🔮 Future Climate Prediction (Auto-Regressive)")
    st.caption("Predict temperatures for future dates using auto-regressive forecasting. "
               "Each prediction uses the previous 7 days to forecast the next day.")
    
    model, config, _, _ = load_prediction_artifacts()
    
    if model is None:
        st.warning("Model not found. Train the model first.")
        return
    
    # Input controls
    col1, col2, col3 = st.columns([1, 1, 1.2])
    
    with col1:
        num_days = st.number_input(
            "Days to predict",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            help="Maximum 120 days into the future"
        )
    
    with col2:
        st.metric("Forecast Method", "Auto-Regressive", help="7-day rolling window")
    
    with col3:
        st.metric("Input Window", "7 days", help="Uses previous 7 days")
    
    if st.button("🚀 Generate Future Forecast", type="primary", key="future_predict"):
        with st.spinner(f"Generating {num_days}-day forecast..."):
            try:
                predictions, start_date = predict_future_autoregressive(num_days, split)
                st.success(f"✅ Forecast generated! Predicting from {start_date.strftime('%Y-%m-%d')} + {num_days} days")
                
                # Convert to DataFrame for display
                forecast_data = []
                for date, pred in sorted(predictions.items()):
                    forecast_data.append({
                        "Date": date.strftime('%Y-%m-%d'),
                        "Mean (°C)": pred['mean_temp'],
                        "Min (°C)": pred['min_temp'],
                        "Max (°C)": pred['max_temp'],
                        "Std Dev (°C)": pred['std_temp'],
                    })
                
                forecast_df = pd.DataFrame(forecast_data)
                
                # Summary statistics
                st.markdown("---")
                st.subheader("📊 Forecast Summary")
                
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric(
                        "Period",
                        f"{forecast_df['Date'].iloc[0]} to {forecast_df['Date'].iloc[-1]}"
                    )
                with summary_cols[1]:
                    st.metric(
                        "Avg Forecast Temp",
                        f"{forecast_df['Mean (°C)'].mean():.2f}°C"
                    )
                with summary_cols[2]:
                    st.metric(
                        "Hottest Day",
                        f"{forecast_df['Max (°C)'].max():.2f}°C"
                    )
                with summary_cols[3]:
                    st.metric(
                        "Coldest Day",
                        f"{forecast_df['Min (°C)'].min():.2f}°C"
                    )
                
                # Forecast table
                st.markdown("---")
                st.subheader("📋 Day-by-Day Forecast")
                st.dataframe(
                    forecast_df.style.format({
                        "Mean (°C)": "{:.2f}",
                        "Min (°C)": "{:.2f}",
                        "Max (°C)": "{:.2f}",
                        "Std Dev (°C)": "{:.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Charts
                st.markdown("---")
                st.subheader("📈 Forecast Trends")
                
                chart_cols = st.columns(2)
                
                with chart_cols[0]:
                    # Mean temperature line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Mean (°C)'],
                        mode='lines+markers',
                        name='Mean Temp',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Max (°C)'],
                        mode='lines',
                        name='Max Temp',
                        line=dict(color='#ff7f0e', dash='dash'),
                        opacity=0.7
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Min (°C)'],
                        mode='lines',
                        name='Min Temp',
                        line=dict(color='#2ca02c', dash='dash'),
                        opacity=0.7
                    ))
                    fig.update_layout(
                        title="Temperature Forecast Trend",
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_cols[1]:
                    # Range (Max-Min) chart
                    forecast_df['Range'] = forecast_df['Max (°C)'] - forecast_df['Min (°C)']
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=forecast_df['Date'],
                        y=forecast_df['Range'],
                        name='Daily Range',
                        marker=dict(
                            color=forecast_df['Range'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Range (°C)")
                        )
                    ))
                    fig.update_layout(
                        title="Daily Temperature Range",
                        xaxis_title="Date",
                        yaxis_title="Temperature Range (°C)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#FAFAFA",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Spatial forecast for selected day
                st.markdown("---")
                st.subheader("🗺️ Spatial Forecast Maps")
                
                selected_forecast_idx = st.slider(
                    "Select day to view spatial map",
                    0,
                    len(forecast_data) - 1,
                    0
                )
                
                selected_date = forecast_data[selected_forecast_idx]["Date"]
                selected_pred = predictions[datetime.strptime(selected_date, '%Y-%m-%d')]
                
                st.plotly_chart(
                    build_heatmap(
                        selected_pred['prediction'],
                        f"Predicted Temperature Map - {selected_date}",
                        color_scale="RdYlBu_r"
                    ),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def display_daywise_prediction_tab(split):
    """Display city-level live weather forecasts for today and the next 10 days."""
    india_locations = load_india_locations()
    if india_locations is None:
        st.warning("India location file not found. Please make sure data/metadata/india_locations.json exists.")
        return

    st.subheader("📅 Day-Wise Climate Prediction")
    st.caption("Choose today or any of the next 10 days to view city-level weather for India.")

    state_names = sorted(india_locations["states"].keys())
    default_state = "Himachal Pradesh" if "Himachal Pradesh" in state_names else state_names[0]

    st.markdown("### Location Selection")
    location_col1, location_col2, location_col3 = st.columns(3)

    with location_col1:
        selected_country = st.selectbox(
            "Country",
            [india_locations["country"]],
            index=0,
            key=f"country_select_live_{split}",
        )

    with location_col2:
        selected_state = st.selectbox(
            "State / Union Territory",
            state_names,
            index=state_names.index(default_state),
            key=f"state_select_live_{split}",
        )

    state_data = india_locations["states"][selected_state]
    city_records = state_data.get("cities", [])
    if not city_records:
        st.warning(f"No cities found for {selected_state}.")
        return

    city_names = [city["name"] for city in city_records]
    default_city = "Dalhousie" if "Dalhousie" in city_names else city_names[0]

    with location_col3:
        selected_city = st.selectbox(
            "City",
            city_names,
            index=city_names.index(default_city),
            key=f"city_select_live_{split}",
        )

    selected_city_record = next(city for city in city_records if city["name"] == selected_city)
    lat = float(selected_city_record["latitude"])
    lon = float(selected_city_record["longitude"])

    st.caption(
        f"{selected_country}: {len(state_names)} states/UTs loaded, "
        f"{sum(len(state['cities']) for state in india_locations['states'].values()):,} cities available."
    )

    prediction_window_start = get_dashboard_today()
    prediction_window_end = prediction_window_start + timedelta(days=MAX_DASHBOARD_FORECAST_DAYS)
    available_future_days = (prediction_window_end - prediction_window_start).days

    try:
        live_forecast = fetch_live_city_forecast(
            lat,
            lon,
            prediction_window_start,
            prediction_window_end,
        )
    except Exception as exc:
        st.error(f"Unable to load live weather data for {selected_city}: {exc}")
        return

    live_forecast_df = live_forecast["daily"]

    st.markdown("### 📍 Select a Date")
    col1, col2, col3 = st.columns([2, 1.5, 1.5])

    with col1:
        selected_date = st.date_input(
            "Pick a date to forecast",
            value=prediction_window_start,
            min_value=prediction_window_start,
            max_value=prediction_window_end,
            key=f"prediction_date_live_{split}",
        )

    with col2:
        st.info(
            f"📊 Today + next {available_future_days} days "
            f"(through {prediction_window_end.strftime('%b %d, %Y')})"
        )

    with col3:
        days_from_today = (selected_date - prediction_window_start).days
        if days_from_today == 0:
            st.success("✅ Today")
        else:
            st.warning(f"🌤️ Next {days_from_today} day(s)")

    if st.button("🎯 Generate Forecast", type="primary", key=f"daywise_predict_live_{split}", use_container_width=True):
        try:
            selected_key = selected_date.strftime("%Y-%m-%d")
            selected_rows = live_forecast_df.loc[live_forecast_df["date"] == selected_key]
            if selected_rows.empty:
                st.error(f"No live forecast data is available for {selected_key}.")
                return

            selected_forecast = selected_rows.iloc[0]
            current_weather = live_forecast.get("current", {})

            col_head1, col_head2 = st.columns([3, 1])
            with col_head1:
                if days_from_today == 0:
                    st.markdown(f"### 🌤️ Today Forecast - {selected_date.strftime('%B %d, %Y')}")
                    st.markdown("*Live city weather forecast for today*")
                else:
                    st.markdown(f"### 🌤️ Forecast - {selected_date.strftime('%B %d, %Y')}")
                    st.markdown(f"*Live city weather forecast for **{days_from_today} day(s)** from today*")
            with col_head2:
                st.markdown("### Today" if days_from_today == 0 else f"### +{days_from_today}d")

            st.info(
                "Source: **Open-Meteo live forecast**. This avoids the coarse regional temperature grid "
                "that can overestimate temperatures for hill stations like Dalhousie."
            )

            st.markdown("---")
            st.subheader(f"🏙️ Weather for {selected_city}, {selected_state}")

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("High Temperature", f"{float(selected_forecast['temp_max']):.1f}°C")
            with metric_col2:
                st.metric("Low Temperature", f"{float(selected_forecast['temp_min']):.1f}°C")
            with metric_col3:
                precip_value = selected_forecast["precip_prob_max"]
                precip_display = "N/A" if pd.isna(precip_value) else f"{float(precip_value):.0f}%"
                st.metric("Rain Chance", precip_display)
            with metric_col4:
                wind_value = selected_forecast["wind_speed_max"]
                wind_display = "N/A" if pd.isna(wind_value) else f"{float(wind_value):.0f} km/h"
                st.metric("Max Wind", wind_display)

            detail_col1, detail_col2 = st.columns([1.5, 1])
            with detail_col1:
                st.markdown(f"### {selected_forecast['summary']}")
                st.write(f"**Coordinates:** {lat:.2f}°N, {lon:.2f}°E")
                if days_from_today == 0 and current_weather:
                    current_temp = current_weather.get("temperature_2m")
                    humidity = current_weather.get("relative_humidity_2m")
                    current_wind = current_weather.get("wind_speed_10m")
                    current_code = current_weather.get("weather_code")
                    if current_temp is not None and humidity is not None and current_wind is not None and current_code is not None:
                        current_summary = WMO_WEATHER_CODES.get(int(current_code), "Current conditions")
                        st.write(
                            f"**Current:** {float(current_temp):.1f}°C, {current_summary}, "
                            f"Humidity {float(humidity):.0f}%, Wind {float(current_wind):.0f} km/h"
                        )
            with detail_col2:
                avg_temp = (float(selected_forecast["temp_max"]) + float(selected_forecast["temp_min"])) / 2.0
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=avg_temp,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": f"{selected_city} Avg Temp (°C)"},
                        gauge={
                            "axis": {"range": [-10, 45]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [-10, 10], "color": "#1f77b4"},
                                {"range": [10, 20], "color": "#2ca02c"},
                                {"range": [20, 30], "color": "#ff7f0e"},
                                {"range": [30, 45], "color": "#d62728"},
                            ],
                        },
                    )
                )
                fig.update_layout(
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#FAFAFA",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📈 10-Day City Outlook")
            trend_fig = go.Figure()
            trend_fig.add_trace(
                go.Scatter(
                    x=live_forecast_df["date"],
                    y=live_forecast_df["temp_max"],
                    mode="lines+markers",
                    name="Daily High",
                    line=dict(color="#ff7f0e", width=3),
                )
            )
            trend_fig.add_trace(
                go.Scatter(
                    x=live_forecast_df["date"],
                    y=live_forecast_df["temp_min"],
                    mode="lines+markers",
                    name="Daily Low",
                    line=dict(color="#1f77b4", width=3),
                )
            )
            trend_fig.update_layout(
                title=f"{selected_city} High / Low Temperatures",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FAFAFA",
                margin=dict(t=50, l=10, r=10, b=10),
            )
            st.plotly_chart(trend_fig, use_container_width=True)

            outlook_df = live_forecast_df.rename(
                columns={
                    "date": "Date",
                    "summary": "Condition",
                    "temp_max": "High (°C)",
                    "temp_min": "Low (°C)",
                    "precip_prob_max": "Rain Chance (%)",
                    "wind_speed_max": "Wind (km/h)",
                }
            )[["Date", "Condition", "High (°C)", "Low (°C)", "Rain Chance (%)", "Wind (km/h)"]]
            st.dataframe(
                outlook_df.style.format(
                    {
                        "High (°C)": "{:.1f}",
                        "Low (°C)": "{:.1f}",
                        "Rain Chance (%)": "{:.0f}",
                        "Wind (km/h)": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"❌ Forecast generation failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


def main():
    apply_custom_styling()

    st.sidebar.title("Configuration")
    split = st.sidebar.selectbox("Select Dataset", ["test", "val", "train"], index=0)

    results = load_evaluation_results(split)

    display_header(split, results is not None)

    if results:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Spatial Analysis", "Detailed Metrics", "Report", "Predict Weather"]
        )

        with tab1:
            display_overview_tab(results)

        with tab2:
            display_spatial_analysis_tab(results)

        with tab3:
            display_detailed_metrics_tab(results)

        with tab4:
            display_report_tab(results)

        with tab5:
            display_daywise_prediction_tab(split)
        
        # with tab6:
        #     display_future_prediction_tab(split)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94A3B8;">
        <b>Model Information</b> | Architecture: ConvLSTM | Training: 100 epochs on ERA5 | Region: India (5-35N, 65-100E)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
