#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbital_phase_tools.py

Utilities to:
1. identify pericentres and apocentres from R_host_kpc(t);
2. align each orbit by t - t_peri;
3. interpolate quantities onto a common orbital-phase grid;
4. plot smooth comparison curves.

Designed to be used with the `results` dictionary returned by
orbit_satellite_analysis.analyze_all(cfg).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter


def log_safe(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, np.log10(values), np.nan)


def get_time_values(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    Prefer time_gyr. Fall back to snapshot_number if needed.
    """
    if "time_gyr" in df.columns:
        t = np.asarray(df["time_gyr"], dtype=float)
        if np.count_nonzero(np.isfinite(t)) >= 3:
            return t, "Gyr"

    if "snapshot_number" not in df.columns:
        raise ValueError("DataFrame needs either time_gyr or snapshot_number.")

    return np.asarray(df["snapshot_number"], dtype=float), "snapshot"


def _savgol_window(n: int, requested: int = 7) -> Optional[int]:
    """
    Return a valid odd Savitzky-Golay window length or None.
    """
    if n < 5:
        return None
    w = min(requested, n if n % 2 == 1 else n - 1)
    if w < 5:
        return None
    if w % 2 == 0:
        w -= 1
    return w


def _prepare_orbit_arrays(
    df: pd.DataFrame,
    radius_col: str = "R_host_kpc",
) -> Tuple[np.ndarray, np.ndarray]:
    t, _ = get_time_values(df)

    if radius_col not in df.columns:
        raise ValueError(f"Column {radius_col} not found.")

    R = np.asarray(df[radius_col], dtype=float)

    mask = np.isfinite(t) & np.isfinite(R)
    t = t[mask]
    R = R[mask]

    if len(t) < 3:
        raise ValueError("Need at least 3 finite points to identify extrema.")

    order = np.argsort(t)
    t = t[order]
    R = R[order]

    # Remove repeated time values, keeping the first occurrence.
    _, unique_idx = np.unique(t, return_index=True)
    unique_idx = np.sort(unique_idx)

    return t[unique_idx], R[unique_idx]


def _refine_extremum_with_pchip(
    t: np.ndarray,
    y: np.ndarray,
    i: int,
    kind: str,
) -> Tuple[float, float]:
    """
    Refine a local extremum around index i using a PCHIP interpolation
    and a bounded scalar minimization/maximization inside [t[i-1], t[i+1]].
    """
    lo, hi = t[i - 1], t[i + 1]

    if hi <= lo:
        return float(t[i]), float(y[i])

    interp = PchipInterpolator(t, y, extrapolate=False)

    if kind == "pericentre":
        objective = lambda xx: float(interp(xx))
    elif kind == "apocentre":
        objective = lambda xx: -float(interp(xx))
    else:
        raise ValueError("kind must be 'pericentre' or 'apocentre'.")

    try:
        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        te = float(res.x)
        ye = float(interp(te))
    except Exception:
        te = float(t[i])
        ye = float(y[i])

    return te, ye


def _prune_events(
    events: pd.DataFrame,
    min_separation: float,
) -> pd.DataFrame:
    """
    Remove multiple nearby extrema caused by noise. If two pericentres are too
    close, keep the one with smaller R. If two apocentres are too close, keep
    the one with larger R.
    """
    if len(events) == 0 or min_separation <= 0:
        return events.copy()

    out_rows = []

    for kind, group in events.groupby("kind", sort=False):
        rows = group.sort_values("t_event").to_dict("records")

        kept = []
        for row in rows:
            if not kept:
                kept.append(row)
                continue

            dt = abs(row["t_event"] - kept[-1]["t_event"])

            if dt >= min_separation:
                kept.append(row)
            else:
                if kind == "pericentre":
                    if row["R_event_kpc"] < kept[-1]["R_event_kpc"]:
                        kept[-1] = row
                else:
                    if row["R_event_kpc"] > kept[-1]["R_event_kpc"]:
                        kept[-1] = row

        out_rows.extend(kept)

    out = pd.DataFrame(out_rows)
    if len(out) == 0:
        return out

    out = out.sort_values("t_event").reset_index(drop=True)

    # Number pericentres/apocentres separately.
    out["event_number"] = -1
    for kind in ["pericentre", "apocentre"]:
        m = out["kind"] == kind
        out.loc[m, "event_number"] = np.arange(np.count_nonzero(m)) + 1

    return out


def find_orbital_extrema(
    df: pd.DataFrame,
    radius_col: str = "R_host_kpc",
    smooth_window: int = 7,
    smooth_polyorder: int = 2,
    min_separation: float = 0.05,
    refine: bool = True,
) -> pd.DataFrame:
    """
    Identify pericentres and apocentres from R_host_kpc(t).

    Parameters
    ----------
    smooth_window:
        Odd Savitzky-Golay window length. If too few snapshots are available,
        no smoothing is applied.
    min_separation:
        Minimum separation between same-kind events in the same unit as time.
        If time_gyr is present, this is in Gyr.
    refine:
        If True, refine the event time using PCHIP interpolation around the
        bracketing snapshots.

    Returns
    -------
    pandas.DataFrame with:
        kind, event_number, t_event, R_event_kpc, index_nearest_snapshot
    """
    t, R = _prepare_orbit_arrays(df, radius_col=radius_col)

    w = _savgol_window(len(R), requested=smooth_window)
    if w is not None and w > smooth_polyorder:
        R_det = savgol_filter(R, window_length=w, polyorder=smooth_polyorder)
    else:
        R_det = R.copy()

    rows = []

    for i in range(1, len(R_det) - 1):
        is_min = (R_det[i] <= R_det[i - 1]) and (R_det[i] < R_det[i + 1])
        is_max = (R_det[i] >= R_det[i - 1]) and (R_det[i] > R_det[i + 1])

        if not (is_min or is_max):
            continue

        kind = "pericentre" if is_min else "apocentre"

        if refine:
            te, Re = _refine_extremum_with_pchip(t, R, i, kind=kind)
        else:
            te, Re = float(t[i]), float(R[i])

        idx_near = int(np.argmin(np.abs(t - te)))

        rows.append(
            {
                "kind": kind,
                "t_event": te,
                "R_event_kpc": Re,
                "index_nearest_snapshot": idx_near,
                "t_nearest_snapshot": float(t[idx_near]),
                "R_nearest_snapshot_kpc": float(R[idx_near]),
            }
        )

    events = pd.DataFrame(rows)

    if len(events) == 0:
        return pd.DataFrame(
            columns=[
                "kind",
                "event_number",
                "t_event",
                "R_event_kpc",
                "index_nearest_snapshot",
                "t_nearest_snapshot",
                "R_nearest_snapshot_kpc",
            ]
        )

    events = _prune_events(events, min_separation=min_separation)
    return events


def choose_pericentre(
    events: pd.DataFrame,
    which: str | int = "first",
) -> pd.Series:
    """
    Select the reference pericentre for phase alignment.

    which:
        "first"   -> first pericentre in time
        "deepest" -> pericentre with smallest R
        integer   -> pericentre event_number
    """
    peris = events[events["kind"] == "pericentre"].copy()

    if len(peris) == 0:
        raise ValueError("No pericentre found.")

    if which == "first":
        return peris.sort_values("t_event").iloc[0]

    if which == "deepest":
        return peris.sort_values("R_event_kpc").iloc[0]

    if isinstance(which, int):
        hit = peris[peris["event_number"] == which]
        if len(hit) == 0:
            raise ValueError(f"Pericentre number {which} not found.")
        return hit.iloc[0]

    raise ValueError("which must be 'first', 'deepest', or an integer.")


def align_results_by_pericentre(
    results: Dict[str, pd.DataFrame],
    which_pericentre: str | int = "first",
    radius_col: str = "R_host_kpc",
    smooth_window: int = 7,
    min_separation: float = 0.05,
    output_dir: Optional[str | Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Add t_minus_tperi to each result DataFrame.

    Returns
    -------
    aligned:
        dictionary of DataFrames with new column t_minus_tperi
    all_events:
        dictionary with pericentre/apocentre tables
    summary:
        table with the selected reference pericentre for each label
    """
    aligned = {}
    all_events = {}
    summary_rows = []

    for label, df in results.items():
        events = find_orbital_extrema(
            df,
            radius_col=radius_col,
            smooth_window=smooth_window,
            min_separation=min_separation,
            refine=True,
        )

        if len(events) == 0:
            print(f"[{label}] No extrema found.")
            continue

        peri = choose_pericentre(events, which=which_pericentre)
        tperi = float(peri["t_event"])

        t, unit = get_time_values(df)
        df2 = df.copy()
        df2["t_minus_tperi"] = t - tperi
        df2["t_peri_ref"] = tperi
        df2["time_unit_for_alignment"] = unit

        aligned[label] = df2
        all_events[label] = events

        summary_rows.append(
            {
                "label": label,
                "which_pericentre": which_pericentre,
                "t_peri": tperi,
                "R_peri_kpc": float(peri["R_event_kpc"]),
                "nearest_snapshot_index": int(peri["index_nearest_snapshot"]),
                "time_unit": unit,
            }
        )

    summary = pd.DataFrame(summary_rows)

    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        summary.to_csv(outdir / "selected_pericentres.csv", index=False)

        for label, events in all_events.items():
            safe_label = label.replace("/", "_")
            events.to_csv(outdir / f"orbital_extrema_{safe_label}.csv", index=False)

    return aligned, all_events, summary


def common_phase_grid(
    aligned: Dict[str, pd.DataFrame],
    ngrid: int = 400,
    xlim: Optional[Tuple[float, float]] = None,
    overlap_only: bool = True,
) -> np.ndarray:
    """
    Build a common t - t_peri grid.

    overlap_only=True uses only the interval covered by all simulations.
    overlap_only=False uses the union and interpolates only where each run exists.
    """
    ranges = []

    for df in aligned.values():
        if "t_minus_tperi" not in df.columns:
            continue
        x = np.asarray(df["t_minus_tperi"], dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        ranges.append((float(np.min(x)), float(np.max(x))))

    if not ranges:
        raise ValueError("No valid aligned ranges.")

    if xlim is not None:
        xmin, xmax = xlim
    elif overlap_only:
        xmin = max(r[0] for r in ranges)
        xmax = min(r[1] for r in ranges)
    else:
        xmin = min(r[0] for r in ranges)
        xmax = max(r[1] for r in ranges)

    if xmax <= xmin:
        raise ValueError("Invalid common grid. Try overlap_only=False or set xlim manually.")

    return np.linspace(xmin, xmax, ngrid)


def interpolate_aligned_quantity(
    df: pd.DataFrame,
    column: str,
    xgrid: np.ndarray,
    log: bool = False,
) -> np.ndarray:
    """
    PCHIP interpolation of a column as a function of t_minus_tperi.
    PCHIP is preferred over cubic splines here because it avoids strong overshoot.
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found.")

    x = np.asarray(df["t_minus_tperi"], dtype=float)
    y = np.asarray(df[column], dtype=float)

    if log:
        y = log_safe(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.full_like(xgrid, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    if len(x_unique) < 3:
        return np.full_like(xgrid, np.nan, dtype=float)

    interp = PchipInterpolator(x_unique, y_unique, extrapolate=False)
    return np.asarray(interp(xgrid), dtype=float)


def plot_aligned_quantity(
    aligned: Dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    filename: str | Path,
    title: Optional[str] = None,
    log: bool = False,
    ngrid: int = 400,
    xlim: Optional[Tuple[float, float]] = None,
    overlap_only: bool = True,
    show_points: bool = True,
    show_pericentre_line: bool = True,
):
    """
    Plot a smooth comparison curve versus t - t_peri.

    Original snapshots are optionally shown as faint points, and the smooth
    curve is a PCHIP interpolation on a common phase grid.
    """
    xgrid = common_phase_grid(
        aligned,
        ngrid=ngrid,
        xlim=xlim,
        overlap_only=overlap_only,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for label, df in aligned.items():
        if column not in df.columns:
            print(f"[{label}] column not found: {column}")
            continue

        ygrid = interpolate_aligned_quantity(df, column, xgrid, log=log)
        ax.plot(xgrid, ygrid, lw=2.0, label=label)

        if show_points:
            x = np.asarray(df["t_minus_tperi"], dtype=float)
            y = np.asarray(df[column], dtype=float)
            if log:
                y = log_safe(y)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[mask], y[mask], marker="o", lw=0, ms=3, alpha=0.35)

    if show_pericentre_line:
        ax.axvline(0.0, ls="--", lw=1.2, alpha=0.8)

    ax.set_xlabel(r"$t - t_{\rm peri}$ [Gyr]")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename, dpi=220, facecolor="white", transparent=False)
    plt.show()


def plot_orbital_extrema_check(
    results: Dict[str, pd.DataFrame],
    all_events: Dict[str, pd.DataFrame],
    filename: str | Path,
):
    """
    Diagnostic plot: R_host(t) with pericentres and apocentres.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in results.items():
        if label not in all_events:
            continue

        t, unit = get_time_values(df)
        R = np.asarray(df["R_host_kpc"], dtype=float)

        ax.plot(t, R, marker="o", lw=1.5, ms=3, label=label)

        events = all_events[label]

        peris = events[events["kind"] == "pericentre"]
        apos = events[events["kind"] == "apocentre"]

        ax.scatter(peris["t_event"], peris["R_event_kpc"], marker="v", s=80)
        ax.scatter(apos["t_event"], apos["R_event_kpc"], marker="^", s=80)

    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel(r"$R_{\rm host}$ [kpc]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename, dpi=220, facecolor="white", transparent=False)
    plt.show()
