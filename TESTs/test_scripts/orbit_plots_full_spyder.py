#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_plots_full_spyder.py

Plotting and post-processing companion for orbit_analysis_full_spyder.py.

This script does NOT reread HDF5 snapshots. It uses:
  - `results` and `cfg` already in memory from the analysis script, OR
  - saved CSV files in cfg.output_dir / output_dir.

Designed for Spyder:
  1. Run orbit_analysis_full_spyder.py first.
  2. Then run this file.
  3. Edit x_axis_mode to compare time, snapshot, or t - first pericentre.

Main functions:
  - run_all_comparison_plots(...)
  - run_standard_and_pericentre_normalized_plots(...)
  - load_results_from_output(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G_KPC_KMS2_MSUN = 4.30091e-6


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log10_safe(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, np.log10(values), np.nan)


def safe_divide(num, den) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where((den > 0) & np.isfinite(num) & np.isfinite(den), num / den, np.nan)


def first_available_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def available_columns(results: Dict[str, pd.DataFrame]) -> set:
    cols = set()
    for df in results.values():
        cols.update(df.columns)
    return cols


def load_results_from_output(output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    results = {}
    for path in sorted(output_dir.glob("*/orbit_full_timeseries.csv")):
        results[path.parent.name] = pd.read_csv(path)
    if not results:
        for path in sorted(output_dir.glob("*/center_tracking_timeseries.csv")):
            results[path.parent.name] = pd.read_csv(path)
    if not results:
        raise FileNotFoundError(f"No per-label timeseries CSVs found in {output_dir}")
    return results


# =============================================================================
# EXTREMA AND TIME AXES
# =============================================================================

def compute_orbital_extrema(df: pd.DataFrame) -> pd.DataFrame:
    if "R_host_kpc" not in df.columns or len(df) == 0:
        return pd.DataFrame()

    r = np.asarray(df["R_host_kpc"], dtype=float)
    rows = []

    for i in range(1, len(df) - 1):
        if not np.all(np.isfinite([r[i - 1], r[i], r[i + 1]])):
            continue

        event_type = None
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            event_type = "pericentre"
        elif r[i] > r[i - 1] and r[i] > r[i + 1]:
            event_type = "apocentre"

        if event_type is None:
            continue

        row = {
            "event_type": event_type,
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else i,
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else i,
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
        }

        for col in [
            "V_3d_kms", "V_rad_kms", "V_tan_kms",
            "r_tidal_kpc", "rhalf_star_member_kpc",
            "Mstar_member_msun", "Mgas_inside_rt_msun", "Mdm_inside_rt_msun",
            "SFR_gas_tracked_msun_yr", "SFR_gas_inside_rt_msun_yr",
            "sSFR_inside_rt_yr", "SFE_inside_rt_yr",
            "P_ram_dyne_cm2", "rho_cgm_msun_kpc3", "V_rel_cgm_kms",
            "tidal_field_kms2_kpc2", "tidal_accel_across_rhalf_kms2_kpc",
        ]:
            if col in df.columns:
                val = df.iloc[i][col]
                row[col] = float(val) if np.isfinite(val) else np.nan

        rows.append(row)

    if not any(row["event_type"] == "pericentre" for row in rows) and np.any(np.isfinite(r)):
        i = int(np.nanargmin(r))
        rows.append({
            "event_type": "pericentre_candidate_global_minimum",
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else i,
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else i,
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
        })

    return pd.DataFrame(rows)


def get_first_pericentre_time(df: pd.DataFrame) -> float:
    if "t_first_pericentre_gyr" in df.columns:
        vals = np.asarray(df["t_first_pericentre_gyr"], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            return float(vals[0])

    ev = compute_orbital_extrema(df)
    if len(ev) == 0 or "time_gyr" not in ev.columns:
        return np.nan

    peri = ev[ev["event_type"] == "pericentre"]
    if len(peri) == 0:
        peri = ev[ev["event_type"].astype(str).str.startswith("pericentre")]
    if len(peri) == 0:
        return np.nan

    return float(peri.iloc[0]["time_gyr"])


def add_time_since_first_pericentre(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time_since_first_pericentre_gyr" in out.columns:
        return out

    if "time_gyr" in out.columns:
        tperi = get_first_pericentre_time(out)
        out["t_first_pericentre_gyr"] = tperi
        out["time_since_first_pericentre_gyr"] = out["time_gyr"] - tperi if np.isfinite(tperi) else np.nan
    else:
        out["t_first_pericentre_gyr"] = np.nan
        out["time_since_first_pericentre_gyr"] = np.nan

    ev = compute_orbital_extrema(out)
    if len(ev) > 0:
        peri = ev[ev["event_type"] == "pericentre"]
        if len(peri) == 0:
            peri = ev[ev["event_type"].astype(str).str.startswith("pericentre")]
        if len(peri) > 0:
            idx_peri = int(peri.iloc[0]["snapshot_index"])
            if "snapshot_index" in out.columns:
                out["snapshot_since_first_pericentre"] = out["snapshot_index"] - idx_peri
            else:
                out["snapshot_since_first_pericentre"] = np.arange(len(out)) - idx_peri
        else:
            out["snapshot_since_first_pericentre"] = np.nan
    else:
        out["snapshot_since_first_pericentre"] = np.nan

    return out


def get_time_axis(df: pd.DataFrame, x_axis_mode: str = "time") -> Tuple[np.ndarray, str]:
    mode = x_axis_mode.lower()

    if mode in ["time_since_first_pericentre", "t_since_peri", "tminusperi", "t-tperi", "t_minus_tperi"]:
        tmp = add_time_since_first_pericentre(df)
        return tmp["time_since_first_pericentre_gyr"].values, r"$t - t_{\rm first\ peri}$ [Gyr]"

    if mode in ["snapshot_since_first_pericentre", "snap_since_peri"]:
        tmp = add_time_since_first_pericentre(df)
        return tmp["snapshot_since_first_pericentre"].values, r"Snapshot $-$ first pericentre snapshot"

    if mode in ["snapshot", "snapshot_number"]:
        if "snapshot_number" in df.columns:
            return df["snapshot_number"].values, "Snapshot"
        return np.arange(len(df)), "Snapshot index"

    if "time_gyr" in df.columns and np.any(np.isfinite(df["time_gyr"].values)):
        return df["time_gyr"].values, "Time [Gyr]"

    if "snapshot_number" in df.columns:
        return df["snapshot_number"].values, "Snapshot"

    return np.arange(len(df)), "Index"


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def add_derived_tidal_quantities(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    out = df.copy()

    if "R_host_kpc" not in out.columns:
        return out

    if "tidal_field_kms2_kpc2" not in out.columns and cfg is not None:
        R = np.asarray(out["R_host_kpc"], dtype=float)
        Mhost = np.full(len(out), np.nan, dtype=float)

        for i, r in enumerate(R):
            if np.isfinite(r) and r > 0:
                try:
                    Mhost[i] = float(cfg.host.mass_enclosed_msun(r))
                except Exception:
                    Mhost[i] = np.nan

        out["Mhost_enclosed_msun"] = Mhost
        out["tidal_field_kms2_kpc2"] = np.where(
            (R > 0) & np.isfinite(Mhost),
            G_KPC_KMS2_MSUN * Mhost / R**3,
            np.nan,
        )
        out["tidal_field_proxy_msun_kpc3"] = np.where(R > 0, Mhost / R**3, np.nan)

    rhalf_col = first_available_column(
        out,
        ["rhalf_star_member_kpc", "rhalf_star_kpc", "rhalf_star_all_kpc"],
    )

    if "tidal_accel_across_rhalf_kms2_kpc" not in out.columns:
        if rhalf_col is not None and "tidal_field_kms2_kpc2" in out.columns:
            out["tidal_accel_across_rhalf_kms2_kpc"] = out["tidal_field_kms2_kpc2"] * out[rhalf_col]

    if "rhalf_over_rt" not in out.columns:
        if "r_tidal_kpc" in out.columns and rhalf_col is not None:
            out["rhalf_over_rt"] = out[rhalf_col] / out["r_tidal_kpc"]

    return out


def add_derived_sfr_quantities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sfr_tracked_col = first_available_column(
        out,
        [
            "SFR_gas_tracked_msun_yr",
            "SFR_tracked_msun_yr",
            "SFR_total_msun_yr",
            "SFR_msun_yr",
            "SFR",
        ],
    )

    sfr_inside_rt_col = first_available_column(
        out,
        [
            "SFR_gas_inside_rt_msun_yr",
            "SFR_inside_rt_msun_yr",
            "SFR_inside_tidal_radius_msun_yr",
        ],
    )

    if sfr_tracked_col is not None:
        out["SFR_tracked_use_msun_yr"] = np.asarray(out[sfr_tracked_col], dtype=float)
    else:
        out["SFR_tracked_use_msun_yr"] = np.nan

    if sfr_inside_rt_col is not None:
        out["SFR_inside_rt_use_msun_yr"] = np.asarray(out[sfr_inside_rt_col], dtype=float)
    else:
        out["SFR_inside_rt_use_msun_yr"] = np.nan

    mstar_tracked_col = first_available_column(out, ["Mstar_tracked_msun", "Mstar_total_msun"])
    mstar_member_col = first_available_column(out, ["Mstar_member_msun", "Mstar_inside_rt_msun", "Mstar_tracked_msun"])
    mstar_inside_rt_col = first_available_column(out, ["Mstar_inside_rt_msun", "Mstar_member_msun", "Mstar_tracked_msun"])

    mgas_tracked_col = first_available_column(out, ["Mgas_tracked_msun", "Mgas_total_msun"])
    mgas_inside_rt_col = first_available_column(out, ["Mgas_inside_rt_msun", "Mgas_member_msun", "Mgas_tracked_msun"])

    if "sSFR_tracked_yr" not in out.columns:
        out["sSFR_tracked_yr"] = safe_divide(out["SFR_tracked_use_msun_yr"], out[mstar_tracked_col]) if mstar_tracked_col else np.nan

    if "sSFR_member_yr" not in out.columns:
        out["sSFR_member_yr"] = safe_divide(out["SFR_tracked_use_msun_yr"], out[mstar_member_col]) if mstar_member_col else np.nan

    if "sSFR_inside_rt_yr" not in out.columns:
        out["sSFR_inside_rt_yr"] = safe_divide(out["SFR_inside_rt_use_msun_yr"], out[mstar_inside_rt_col]) if mstar_inside_rt_col else np.nan

    if "SFE_tracked_yr" not in out.columns:
        out["SFE_tracked_yr"] = safe_divide(out["SFR_tracked_use_msun_yr"], out[mgas_tracked_col]) if mgas_tracked_col else np.nan

    if "SFE_inside_rt_yr" not in out.columns:
        out["SFE_inside_rt_yr"] = safe_divide(out["SFR_inside_rt_use_msun_yr"], out[mgas_inside_rt_col]) if mgas_inside_rt_col else np.nan

    if "tdep_tracked_gyr" not in out.columns:
        out["tdep_tracked_gyr"] = safe_divide(out[mgas_tracked_col], out["SFR_tracked_use_msun_yr"]) / 1.0e9 if mgas_tracked_col else np.nan

    if "tdep_inside_rt_gyr" not in out.columns:
        out["tdep_inside_rt_gyr"] = safe_divide(out[mgas_inside_rt_col], out["SFR_inside_rt_use_msun_yr"]) / 1.0e9 if mgas_inside_rt_col else np.nan

    return out


def prepare_results_for_comparison(results: Dict[str, pd.DataFrame], cfg=None) -> Dict[str, pd.DataFrame]:
    prepared = {}
    for label, df in results.items():
        tmp = df.copy()
        tmp = add_time_since_first_pericentre(tmp)
        tmp = add_derived_tidal_quantities(tmp, cfg=cfg)
        tmp = add_derived_sfr_quantities(tmp)
        tmp["label"] = label
        prepared[label] = tmp
    return prepared


# =============================================================================
# PRINTING
# =============================================================================

def print_and_save_extrema(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    all_events = []

    for label, df in results.items():
        ev = compute_orbital_extrema(df)
        if len(ev) == 0:
            print("\n" + "=" * 80)
            print(label)
            print("=" * 80)
            print("No pericentre/apocentre detected.")
            continue

        ev = ev.assign(label=label)
        tperi = get_first_pericentre_time(df)
        ev["t_first_pericentre_gyr"] = tperi
        ev["time_since_first_pericentre_gyr"] = ev["time_gyr"] - tperi if "time_gyr" in ev.columns else np.nan

        all_events.append(ev)

        label_dir = ensure_dir(output_dir / label)
        ev.to_csv(label_dir / "orbital_extrema.csv", index=False)

        cols = [
            "event_type", "snapshot_number", "time_gyr", "time_since_first_pericentre_gyr",
            "R_host_kpc", "V_3d_kms", "V_rad_kms", "V_tan_kms",
            "r_tidal_kpc", "rhalf_star_member_kpc",
        ]
        cols = [c for c in cols if c in ev.columns]

        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)
        print(ev[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if all_events:
        out = pd.concat(all_events, ignore_index=True)
        out.to_csv(output_dir / "comparison_orbital_extrema_all_labels.csv", index=False)
        return out

    return pd.DataFrame()


# =============================================================================
# PLOT HELPERS
# =============================================================================

def _add_pericentre_reference_line(ax, x_axis_mode: str) -> None:
    mode = x_axis_mode.lower()
    if mode in [
        "time_since_first_pericentre",
        "t_since_peri",
        "tminusperi",
        "t-tperi",
        "t_minus_tperi",
        "snapshot_since_first_pericentre",
        "snap_since_peri",
    ]:
        ax.axvline(0.0, lw=1.0, ls="--", alpha=0.6)


def _plot_extrema_markers(ax, df: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> None:
    ev = compute_orbital_extrema(df)
    if len(ev) == 0:
        return

    for _, row in ev.iterrows():
        idx = int(row["snapshot_index"])
        if idx < 0 or idx >= len(x):
            continue
        if row["event_type"].startswith("pericentre"):
            ax.scatter(x[idx], y[idx], marker="v", s=55, zorder=4)
        elif row["event_type"] == "apocentre":
            ax.scatter(x[idx], y[idx], marker="^", s=55, zorder=4)


def plot_compare_quantity(
    results: Dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    output_dir: Union[str, Path],
    filename: str,
    title: Optional[str] = None,
    log10_y: bool = False,
    annotate_extrema: bool = False,
    x_axis_mode: str = "time",
) -> Optional[Path]:
    output_dir = ensure_dir(output_dir)

    if not any(column in df.columns for df in results.values()):
        print(f"[skip] Column not found: {column}")
        return None

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    plotted = False
    xlabel = "Time [Gyr]"

    for label, df in results.items():
        if column not in df.columns:
            continue

        x, xlabel = get_time_axis(df, x_axis_mode=x_axis_mode)
        y = np.asarray(df[column], dtype=float)
        if log10_y:
            y = log10_safe(y)

        if not np.any(np.isfinite(y)) or not np.any(np.isfinite(x)):
            continue

        ax.plot(x, y, marker="o", lw=1.5, label=label)

        if annotate_extrema:
            _plot_extrema_markers(ax, df, x, y)

        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"[skip] No finite values for {column}")
        return None

    _add_pericentre_reference_line(ax, x_axis_mode)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_compare_orbits_xy(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    filename: str = "comparison_orbits_xy.png",
) -> Optional[Path]:
    output_dir = ensure_dir(output_dir)

    if not all(any(c in df.columns for df in results.values()) for c in ["x_sat_kpc", "y_sat_kpc"]):
        print("[skip] Missing x_sat_kpc/y_sat_kpc.")
        return None

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

    for label, df in results.items():
        if "x_sat_kpc" not in df.columns or "y_sat_kpc" not in df.columns:
            continue

        ax.plot(df["x_sat_kpc"], df["y_sat_kpc"], marker="o", lw=1.5, label=label)
        ax.scatter(df["x_sat_kpc"].iloc[0], df["y_sat_kpc"].iloc[0], marker="s", s=55)
        ax.scatter(df["x_sat_kpc"].iloc[-1], df["y_sat_kpc"].iloc[-1], marker="x", s=55)

        ev = compute_orbital_extrema(df)
        for _, row in ev.iterrows():
            i = int(row["snapshot_index"])
            if i < 0 or i >= len(df):
                continue
            if row["event_type"].startswith("pericentre"):
                ax.scatter(df["x_sat_kpc"].iloc[i], df["y_sat_kpc"].iloc[i], marker="v", s=70)
            elif row["event_type"] == "apocentre":
                ax.scatter(df["x_sat_kpc"].iloc[i], df["y_sat_kpc"].iloc[i], marker="^", s=70)

    ax.scatter([0], [0], marker="+", s=150, label="host")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title("Orbit comparison in the x-y plane")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grid(
    results: Dict[str, pd.DataFrame],
    panels: Sequence[Tuple[str, str, bool, str]],
    output_dir: Union[str, Path],
    filename: str,
    suptitle: str,
    annotate_extrema: bool = True,
    x_axis_mode: str = "time",
    ncols: int = 3,
) -> Optional[Path]:
    output_dir = ensure_dir(output_dir)

    useful_panels = []
    for col, ylabel, logy, title in panels:
        has_finite = False
        for df in results.values():
            if col in df.columns and np.any(np.isfinite(np.asarray(df[col], dtype=float))):
                has_finite = True
                break
        if has_finite:
            useful_panels.append((col, ylabel, logy, title))

    if not useful_panels:
        print(f"[skip] No useful panels for {filename}")
        return None

    nrows = int(np.ceil(len(useful_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.1 * nrows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, (column, ylabel, logy, title) in zip(axes, useful_panels):
        xlabel = "Time [Gyr]"
        for label, df in results.items():
            if column not in df.columns:
                continue

            x, xlabel = get_time_axis(df, x_axis_mode=x_axis_mode)
            y = np.asarray(df[column], dtype=float)
            if logy:
                y = log10_safe(y)

            if not np.any(np.isfinite(y)) or not np.any(np.isfinite(x)):
                continue

            ax.plot(x, y, marker="o", lw=1.5, label=label)

            if annotate_extrema:
                _plot_extrema_markers(ax, df, x, y)

        _add_pericentre_reference_line(ax, x_axis_mode)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    for ax in axes[len(useful_panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(3, len(labels)))

    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = output_dir / filename
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# HIGH-LEVEL PLOTS
# =============================================================================

def run_all_comparison_plots(
    results: Dict[str, pd.DataFrame],
    cfg=None,
    output_dir: Optional[Union[str, Path]] = None,
    output_subdir: str = "comparison_plots",
    annotate_extrema: bool = True,
    x_axis_mode: str = "time",
) -> Dict[str, Path]:
    if output_dir is None:
        if cfg is not None and hasattr(cfg, "output_dir"):
            output_dir = Path(cfg.output_dir) / output_subdir
        else:
            output_dir = Path(output_subdir)

    output_dir = ensure_dir(output_dir)

    prepared = prepare_results_for_comparison(results, cfg=cfg)

    combined = pd.concat([df.assign(label=label) for label, df in prepared.items()], ignore_index=True)
    combined_path = output_dir / "combined_with_derived_quantities.csv"
    combined.to_csv(combined_path, index=False)

    outputs: Dict[str, Path] = {"combined_table": combined_path}

    events = print_and_save_extrema(prepared, output_dir)
    if len(events):
        outputs["orbital_extrema_table"] = output_dir / "comparison_orbital_extrema_all_labels.csv"

    p = plot_compare_orbits_xy(prepared, output_dir)
    if p is not None:
        outputs["orbits_xy"] = p

    orbit_panels = [
        ("R_host_kpc", r"$R_{\rm host}$ [kpc]", False, "Orbital radius"),
        ("V_3d_kms", r"$|v|$ [km s$^{-1}$]", False, "3D velocity"),
        ("V_rad_kms", r"$v_{\rm rad}$ [km s$^{-1}$]", False, "Radial velocity"),
        ("V_tan_kms", r"$v_{\rm tan}$ [km s$^{-1}$]", False, "Tangential velocity"),
    ]

    size_panels = [
        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", True, "Tidal radius"),
        ("rhalf_star_member_kpc", r"$\log(r_{1/2,\star,\rm member}/{\rm kpc})$", True, "Member stellar half-mass radius"),
        ("rhalf_star_all_kpc", r"$\log(r_{1/2,\star,\rm all}/{\rm kpc})$", True, "All-stars half-mass radius"),
        ("r90_star_member_kpc", r"$\log(r_{90,\star,\rm member}/{\rm kpc})$", True, "Member stellar r90"),
        ("rhalf_over_rt", r"$r_{1/2,\star}/r_t$", False, "Stellar size / tidal radius"),
        ("offset_all_stars_com_from_center_kpc", r"$\log(\Delta_{\rm COM}/{\rm kpc})$", True, "All-stars COM offset"),
    ]

    mass_panels = [
        ("Mstar_member_msun", r"$\log(M_{\star,\rm member}/M_\odot)$", True, "Member stellar mass"),
        ("Mstar_inside_rt_msun", r"$\log(M_{\star}(<r_t)/M_\odot)$", True, "Stellar mass inside rt"),
        ("Mgas_tracked_msun", r"$\log(M_{\rm gas,tracked}/M_\odot)$", True, "Tracked gas mass"),
        ("Mgas_inside_rt_msun", r"$\log(M_{\rm gas}(<r_t)/M_\odot)$", True, "Gas inside rt"),
        ("Mdm_tracked_msun", r"$\log(M_{\rm DM,tracked}/M_\odot)$", True, "Tracked DM mass"),
        ("Mdm_inside_rt_msun", r"$\log(M_{\rm DM}(<r_t)/M_\odot)$", True, "DM inside rt"),
        ("fstar_member", r"$f_{\star,\rm member}$", False, "Member stellar fraction"),
        ("fstar_stripped_definitive", r"$f_{\star,\rm stripped}$", False, "Definitively stripped stellar fraction"),
    ]

    sfr_panels = [
        ("SFR_tracked_use_msun_yr", r"$\log({\rm SFR}_{\rm tracked}/M_\odot\,{\rm yr}^{-1})$", True, "Tracked/global SFR"),
        ("SFR_inside_rt_use_msun_yr", r"$\log({\rm SFR}(<r_t)/M_\odot\,{\rm yr}^{-1})$", True, "SFR inside rt"),
        ("SFR_gas_inside_rhalf_msun_yr", r"$\log({\rm SFR}(<r_{1/2})/M_\odot\,{\rm yr}^{-1})$", True, "SFR inside rhalf"),
        ("sSFR_tracked_yr", r"$\log({\rm sSFR}_{\rm tracked}/{\rm yr}^{-1})$", True, "Tracked sSFR"),
        ("sSFR_member_yr", r"$\log({\rm sSFR}_{\rm member}/{\rm yr}^{-1})$", True, "Member sSFR"),
        ("sSFR_inside_rt_yr", r"$\log({\rm sSFR}(<r_t)/{\rm yr}^{-1})$", True, "sSFR inside rt"),
        ("SFE_tracked_yr", r"$\log({\rm SFE}_{\rm tracked}/{\rm yr}^{-1})$", True, "Tracked gas SFE"),
        ("SFE_inside_rt_yr", r"$\log({\rm SFE}(<r_t)/{\rm yr}^{-1})$", True, "SFE inside rt"),
        ("tdep_tracked_gyr", r"$\log(t_{\rm dep,tracked}/{\rm Gyr})$", True, "Tracked gas depletion time"),
        ("tdep_inside_rt_gyr", r"$\log(t_{\rm dep}(<r_t)/{\rm Gyr})$", True, "Depletion time inside rt"),
    ]

    environment_panels = [
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", True, "Ram pressure"),
        ("rho_cgm_msun_kpc3", r"$\log(\rho_{\rm CGM}/M_\odot\,{\rm kpc}^{-3})$", True, "Local CGM density"),
        ("V_rel_cgm_kms", r"$v_{\rm rel,CGM}$ [km s$^{-1}$]", False, "Relative velocity with CGM"),
        ("tidal_field_kms2_kpc2", r"$\log(GM_{\rm host}(<R)/R^3)$", True, "Tidal-field proxy"),
        ("tidal_accel_across_rhalf_kms2_kpc", r"$\log[(GM(<R)/R^3)r_{1/2}]$", True, "Tidal acceleration across stars"),
        ("rhalf_over_rt", r"$r_{1/2,\star}/r_t$", False, "Size relative to tidal radius"),
    ]

    summary_panels = [
        ("R_host_kpc", r"$R_{\rm host}$ [kpc]", False, "Orbital radius"),
        ("V_rad_kms", r"$v_{\rm rad}$ [km s$^{-1}$]", False, "Radial velocity"),
        ("V_tan_kms", r"$v_{\rm tan}$ [km s$^{-1}$]", False, "Tangential velocity"),
        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", True, "Tidal radius"),
        ("rhalf_star_member_kpc", r"$\log(r_{1/2,\star}/{\rm kpc})$", True, "Member stellar size"),
        ("Mstar_member_msun", r"$\log(M_{\star,\rm member}/M_\odot)$", True, "Member stellar mass"),
        ("SFR_tracked_use_msun_yr", r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$", True, "SFR"),
        ("sSFR_member_yr", r"$\log({\rm sSFR}/{\rm yr}^{-1})$", True, "sSFR"),
        ("SFE_tracked_yr", r"$\log({\rm SFE}/{\rm yr}^{-1})$", True, "SFE"),
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", True, "Ram pressure"),
        ("tidal_field_kms2_kpc2", r"$\log(GM(<R)/R^3)$", True, "Tidal field"),
        ("fstar_stripped_definitive", r"$f_{\star,\rm stripped}$", False, "Stripped fraction"),
    ]

    grids = [
        ("summary_grid", summary_panels, "comparison_summary_grid.png", "Summary comparison", 3),
        ("orbit_grid", orbit_panels, "comparison_orbit_grid.png", "Orbit comparison", 2),
        ("size_grid", size_panels, "comparison_size_grid.png", "Size comparison", 2),
        ("mass_grid", mass_panels, "comparison_mass_grid.png", "Mass and stripping comparison", 2),
        ("sfr_sfe_ssfr_grid", sfr_panels, "comparison_sfr_sfe_ssfr_grid.png", "SFR, sSFR, SFE and depletion time", 2),
        ("environment_grid", environment_panels, "comparison_environment_grid.png", "Ram pressure and tidal forcing", 2),
    ]

    for key, panels, filename, title, ncols in grids:
        if x_axis_mode.lower() in ["time_since_first_pericentre", "t_since_peri", "tminusperi", "t-tperi", "t_minus_tperi"]:
            filename = filename.replace(".png", "_t_minus_tperi.png")

        p = plot_grid(
            prepared,
            panels=panels,
            output_dir=output_dir,
            filename=filename,
            suptitle=title,
            annotate_extrema=annotate_extrema,
            x_axis_mode=x_axis_mode,
            ncols=ncols,
        )
        if p is not None:
            outputs[key] = p

    # Individual high-value plots.
    individual = [
        ("R_host_kpc", r"$R_{\rm host}$ [kpc]", "comparison_Rhost.png", "Orbital radius", False),
        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", "comparison_log_rt.png", "Tidal radius", True),
        ("Mstar_member_msun", r"$\log(M_{\star,\rm member}/M_\odot)$", "comparison_log_Mstar_member.png", "Member stellar mass", True),
        ("SFR_tracked_use_msun_yr", r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$", "comparison_log_SFR.png", "SFR", True),
        ("sSFR_member_yr", r"$\log({\rm sSFR}/{\rm yr}^{-1})$", "comparison_log_sSFR.png", "sSFR", True),
        ("SFE_tracked_yr", r"$\log({\rm SFE}/{\rm yr}^{-1})$", "comparison_log_SFE.png", "SFE", True),
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", "comparison_log_Pram.png", "Ram pressure", True),
        ("tidal_field_kms2_kpc2", r"$\log(GM_{\rm host}(<R)/R^3)$", "comparison_log_tidal_field.png", "Tidal field", True),
    ]

    for column, ylabel, filename, title, logy in individual:
        if x_axis_mode.lower() in ["time_since_first_pericentre", "t_since_peri", "tminusperi", "t-tperi", "t_minus_tperi"]:
            filename = filename.replace(".png", "_t_minus_tperi.png")

        p = plot_compare_quantity(
            prepared,
            column=column,
            ylabel=ylabel,
            output_dir=output_dir,
            filename=filename,
            title=title,
            log10_y=logy,
            annotate_extrema=annotate_extrema,
            x_axis_mode=x_axis_mode,
        )
        if p is not None:
            outputs[column] = p

    print("\nSaved comparison outputs in:")
    print(output_dir)

    return outputs


def run_standard_and_pericentre_normalized_plots(
    results: Dict[str, pd.DataFrame],
    cfg=None,
    annotate_extrema: bool = True,
) -> Dict[str, Dict[str, Path]]:
    out_time = run_all_comparison_plots(
        results,
        cfg=cfg,
        output_subdir="comparison_plots_time",
        annotate_extrema=annotate_extrema,
        x_axis_mode="time",
    )

    out_tperi = run_all_comparison_plots(
        results,
        cfg=cfg,
        output_subdir="comparison_plots_t_minus_tperi",
        annotate_extrema=annotate_extrema,
        x_axis_mode="time_since_first_pericentre",
    )

    return {
        "time": out_time,
        "time_since_first_pericentre": out_tperi,
    }


# =============================================================================
# USER SETTINGS FOR SPYDER
# =============================================================================

# This file can be run after orbit_analysis_full_spyder.py in the same Spyder
# session. It will try to use the existing `results` and `cfg` variables.
# If they do not exist, it loads CSVs from OUTPUT_DIR_FOR_LOADING.

OUTPUT_DIR_FOR_LOADING = "orbit_full_analysis_outputs"

RUN_PLOTS = True
GENERATE_BOTH_TIME_AXES = True

# Use one of:
#   "time"
#   "snapshot"
#   "time_since_first_pericentre"
#   "snapshot_since_first_pericentre"
X_AXIS_MODE = "time_since_first_pericentre"

ANNOTATE_EXTREMA = True

if __name__ == "__main__":
    if RUN_PLOTS:
        # If this script is run in the same Spyder namespace after the analysis
        # script, `results` and `cfg` may already exist. Otherwise, load CSVs.
        try:
            results  # type: ignore[name-defined]
        except NameError:
            results = load_results_from_output(OUTPUT_DIR_FOR_LOADING)

        try:
            cfg  # type: ignore[name-defined]
        except NameError:
            cfg = None

        if GENERATE_BOTH_TIME_AXES:
            comparison_outputs = run_standard_and_pericentre_normalized_plots(
                results,
                cfg=cfg,
                annotate_extrema=ANNOTATE_EXTREMA,
            )
        else:
            comparison_outputs = run_all_comparison_plots(
                results,
                cfg=cfg,
                output_subdir=f"comparison_plots_{X_AXIS_MODE}",
                annotate_extrema=ANNOTATE_EXTREMA,
                x_axis_mode=X_AXIS_MODE,
            )

        print("\nObjects available in Spyder:")
        print("  results")
        print("  comparison_outputs")
