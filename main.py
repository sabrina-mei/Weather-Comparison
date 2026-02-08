from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_daily_usage(csv_path: str | Path) -> List[Dict[str, str | float]]:
    """
    Read an electric interval CSV and return a list of daily usage totals.

    Output format:
        [{"date": "YYYY-MM-DD", "usage": <float_kwh>}, ...]
    """
    csv_path = Path(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        # Skip metadata lines until we reach the data header
        for line in f:
            if line.strip().startswith("Start Interval Date/Time"):
                header = line
                break
        else:
            raise ValueError("Data header not found in CSV.")

        reader = csv.DictReader(f, fieldnames=next(csv.reader([header])))

        totals: defaultdict[str, float] = defaultdict(float)
        for row in reader:
            start_ts = row.get("Start Interval Date/Time")
            usage = row.get("Usage")
            if not start_ts or not usage:
                continue

            # Convert to date string (YYYY-MM-DD)
            date_obj = datetime.strptime(start_ts.strip(), "%m/%d/%Y %H:%M").date()
            date_key = date_obj.isoformat()

            totals[date_key] += float(usage)

    return [{"date": d, "usage": totals[d]} for d in sorted(totals.keys())]


def load_weather_mean(csv_path: str | Path) -> List[Dict[str, str | float]]:
    """
    Read the weather CSV and return a list of daily mean temperatures.

    Output format:
        [{"date": "YYYY-MM-DD", "mean": <float_f>}, ...]
    """
    csv_path = Path(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        # Skip any pre-header rows until we reach the header line
        for line in f:
            if line.strip().startswith("Date,"):
                header = line
                break
        else:
            raise ValueError("Weather header not found in CSV.")

        reader = csv.DictReader(f, fieldnames=next(csv.reader([header])))

        rows: List[Dict[str, str | float]] = []
        for row in reader:
            date_str = row.get("Date")
            mean = row.get("Mean")
            if not date_str or not mean:
                continue

            date_obj = datetime.strptime(date_str.strip(), "%m/%d/%Y").date()
            rows.append({"date": date_obj.isoformat(), "mean": float(mean)})

    return rows


def plot_usage_vs_temperature(
    usage_data: List[Dict[str, str | float]],
    weather_data: List[Dict[str, str | float]],
) -> None:
    """
    Plot electric usage (left axis, kWh) and mean temperature (right axis, °F)
    over time. Uses dates present in both datasets.
    """
    usage_map = {row["date"]: float(row["usage"]) for row in usage_data}
    weather_map = {row["date"]: float(row["mean"]) for row in weather_data}

    shared_dates = sorted(set(usage_map.keys()) & set(weather_map.keys()))
    if not shared_dates:
        raise ValueError("No overlapping dates between usage and weather data.")

    x_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in shared_dates]
    usage_vals = [usage_map[d] for d in shared_dates]
    temp_vals = [weather_map[d] for d in shared_dates]

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_left.plot(x_dates, usage_vals, color="tab:blue", label="Electric usage")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Electric use (kWh)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")

    ax_right = ax_left.twinx()
    ax_right.plot(x_dates, temp_vals, color="tab:red", label="Mean temperature")
    ax_right.set_ylabel("Temperature (°F)", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.show()


def plot_usage_by_temperature(
    usage_data: List[Dict[str, str | float]],
    weather_data: List[Dict[str, str | float]],
    exclude_start: str | None = "2025-12-22",
    exclude_end: str | None = "2026-01-19",
    fit_degree: int = 2,
) -> None:
    """
    Plot daily electric usage vs mean temperature.
    X-axis: temperature (°F)
    Y-axis: electric use (kWh)
    Optionally exclude a date range (inclusive) using YYYY-MM-DD strings.
    Fits a polynomial curve of degree `fit_degree`.
    """
    usage_map = {row["date"]: float(row["usage"]) for row in usage_data}
    weather_map = {row["date"]: float(row["mean"]) for row in weather_data}

    shared_dates = sorted(set(usage_map.keys()) & set(weather_map.keys()))
    if exclude_start or exclude_end:
        start_date = (
            datetime.strptime(exclude_start, "%Y-%m-%d").date()
            if exclude_start
            else None
        )
        end_date = (
            datetime.strptime(exclude_end, "%Y-%m-%d").date()
            if exclude_end
            else None
        )

        def _in_excluded_range(date_str: str) -> bool:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start_date and d < start_date:
                return False
            if end_date and d > end_date:
                return False
            return True

        shared_dates = [d for d in shared_dates if not _in_excluded_range(d)]
    if not shared_dates:
        raise ValueError("No overlapping dates between usage and weather data.")

    temps = [weather_map[d] for d in shared_dates]
    usage_vals = [usage_map[d] for d in shared_dates]

    plt.figure(figsize=(7, 5))
    plt.scatter(temps, usage_vals, color="tab:purple", alpha=0.7, label="Data")

    if fit_degree >= 1 and len(temps) > fit_degree:
        x = np.array(temps, dtype=float)
        y = np.array(usage_vals, dtype=float)
        coeffs = np.polyfit(x, y, deg=fit_degree)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = poly(x_fit)
        plt.plot(x_fit, y_fit, color="tab:orange", linewidth=2, label="Curve fit")
    plt.xlabel("Temperature (°F)")
    plt.ylabel("Electric use (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_daily_usage("local/electric.csv")
    weather = load_weather_mean("local/weather-data - Sheet1.csv")
    plot_usage_vs_temperature(data, weather)
    plot_usage_by_temperature(data, weather)
