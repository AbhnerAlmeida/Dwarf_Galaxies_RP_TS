#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_comparison_tools_spyder.py

Post-processing and comparison plots for the satellite-orbit analysis scripts.

Designed for Spyder
-------------------
Run your main HDF5 analysis first, so that you have in memory:

    results
    cfg

where `results` is a dictionary:

    results[label] = pandas.DataFrame

Then run:

    from orbit_comparison_tools_spyder import run_all_comparison_plots

    comparison_outputs = run_all_comparison_plots(
        results,
        cfg,
        output_subdir="comparison_plots_tperi",
        x_axis_mode="time_since_first_pericentre",
        annotate_extrema=True,
    )

Main features
-------------
1. Comparison plots between all simulation labels.
2. Optional x-axis normalization by first pericentre:
       x = t - t_first_pericentre
3. Derived tidal-force proxies:
       tidal_field = G M_host(<R) / R^3
4. Efficient derived SFR/SFE/sSFR quantities, if SFR columns exist.
5. Pericentre/apocentre tables saved for each label and for all labels combined.

Notes
-----
This file does not reread HDF5 snapshots. It only works with the DataFrames
already produced by the main analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G_KPC_KMS2_MSUN = 4.30091e-6


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Orbital extrema and time normalization
# ---------------------------------------------------------------------

def compute_orbital_extrema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify local pericentres and apocentres from R_host_kpc.

    Pericentre = local minimum of R_host_kpc.
    Apocentre  = local maximum of R_host_kpc.

    If no bracketed pericentre is found, the global minimum is returned as a
    pericentre candidate.
    """
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
            "r_tidal_kpc",
            "rhalf_star_member_kpc", "rhalf_star_kpc", "rhalf_star_all_kpc",
            "Mstar_member_msun", "Mstar_inside_rt_msun", "Mstar_tracked_msun",
            "Mgas_tracked_msun", "Mgas_inside_rt_msun",
            "Mdm_tracked_msun", "Mdm_inside_rt_msun",
            "P_ram_dyne_cm2", "rho_cgm_msun_kpc3", "V_rel_cgm_kms",
            "SFR_gas_tracked_msun_yr", "SFR_gas_inside_rt_msun_yr",
            "SFR_tracked_msun_yr", "SFR_inside_rt_msun_yr",
            "SFR_msun_yr", "SFR",
            "tidal_field_kms2_kpc2",
            "tidal_accel_across_rhalf_kms2_kpc",
        ]:
            if col in df.columns:
                val = df.iloc[i][col]
                row[col] = float(val) if np.isfinite(val) else np.nan

        rows.append(row)

    if not any(row["event_type"] == "pericentre" for row in rows) and np.any(np.isfinite(r)):
        i = int(np.nanargmin(r))
        row = {
            "event_type": "pericentre_candidate_global_minimum",
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else i,
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else i,
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
        }
        for col in [
            "V_3d_kms", "V_rad_kms", "V_tan_kms",
            "r_tidal_kpc",
            "rhalf_star_member_kpc", "rhalf_star_kpc", "rhalf_star_all_kpc",
            "Mstar_member_msun", "Mstar_inside_rt_msun", "Mstar_tracked_msun",
            "Mgas_tracked_msun", "Mgas_inside_rt_msun",
            "Mdm_tracked_msun", "Mdm_inside_rt_msun",
            "P_ram_dyne_cm2", "rho_cgm_msun_kpc3", "V_rel_cgm_kms",
            "SFR_gas_tracked_msun_yr", "SFR_gas_inside_rt_msun_yr",
            "SFR_tracked_msun_yr", "SFR_inside_rt_msun_yr",
            "SFR_msun_yr", "SFR",
            "tidal_field_kms2_kpc2",
            "tidal_accel_across_rhalf_kms2_kpc",
        ]:
            if col in df.columns:
                val = df.iloc[i][col]
                row[col] = float(val) if np.isfinite(val) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def get_first_pericentre_time(df: pd.DataFrame) -> float:
    """
    Return time_gyr of the first detected pericentre.

    Priority:
    1. first local pericentre
    2. first pericentre candidate
    3. NaN
    """
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
    """
    Add:
        t_first_pericentre_gyr
        time_since_first_pericentre_gyr
        snapshot_since_first_pericentre

    If no pericentre is detected, the new columns are NaN.
    """
    out = df.copy()

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
    """
    x_axis_mode options:
        "time" or "time_gyr"
        "snapshot" or "snapshot_number"
        "time_since_first_pericentre", "t_since_peri", or "tminusperi"
        "snapshot_since_first_pericentre"
    """
    mode = x_axis_mode.lower()

    if mode in ["time_since_first_pericentre", "t_since_peri", "tminusperi", "t-tperi", "t_minus_tperi"]:
        if "time_since_first_pericentre_gyr" in df.columns:
            return df["time_since_first_pericentre_gyr"].values, r"$t - t_{\rm first\ peri}$ [Gyr]"
        tmp = add_time_since_first_pericentre(df)
        return tmp["time_since_first_pericentre_gyr"].values, r"$t - t_{\rm first\ peri}$ [Gyr]"

    if mode in ["snapshot_since_first_pericentre", "snap_since_peri"]:
        if "snapshot_since_first_pericentre" in df.columns:
            return df["snapshot_since_first_pericentre"].values, r"Snapshot $-$ first pericentre snapshot"
        tmp = add_time_since_first_pericentre(df)
        return tmp["snapshot_since_first_pericentre"].values, r"Snapshot $-$ first pericentre snapshot"

    if mode in ["snapshot", "snapshot_number"]:
        if "snapshot_number" in df.columns:
            return df["snapshot_number"].values, "Snapshot"
        return np.arange(len(df)), "Snapshot index"

    # Default: physical time.
    if "time_gyr" in df.columns and np.any(np.isfinite(df["time_gyr"].values)):
        return df["time_gyr"].values, "Time [Gyr]"

    if "snapshot_number" in df.columns:
        return df["snapshot_number"].values, "Snapshot"

    return np.arange(len(df)), "Index"


# ---------------------------------------------------------------------
# Derived quantities: tides, SFR, SFE, sSFR
# ---------------------------------------------------------------------

def add_derived_tidal_quantities(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Add host enclosed mass and simple tidal-field proxies.

    tidal_field_kms2_kpc2 = G M_host(<R) / R^3

    This is not the full NFW tidal tensor. It is a compact diagnostic of the
    external tidal field at the satellite position.
    """
    out = df.copy()

    if "R_host_kpc" not in out.columns:
        return out

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

    if rhalf_col is not None:
        out["tidal_accel_across_rhalf_kms2_kpc"] = out["tidal_field_kms2_kpc2"] * out[rhalf_col]

    if "r_tidal_kpc" in out.columns and rhalf_col is not None:
        out["rhalf_over_rt"] = out[rhalf_col] / out["r_tidal_kpc"]

    return out


def add_derived_sfr_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standardized SFR/SFE/sSFR columns when enough input data exists.

    Standardized columns created:
        SFR_tracked_use_msun_yr
        SFR_inside_rt_use_msun_yr
        sSFR_tracked_yr
        sSFR_member_yr
        sSFR_inside_rt_yr
        SFE_tracked_yr
        SFE_inside_rt_yr
        tdep_tracked_gyr
        tdep_inside_rt_gyr

    Definitions:
        sSFR = SFR / Mstar
        SFE  = SFR / Mgas
        tdep = Mgas / SFR

    Units:
        SFR: Msun/yr
        sSFR, SFE: yr^-1
        tdep: Gyr
    """
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
        # Fallback: if only one SFR exists, use it as a tracked/global SFR only.
        out["SFR_inside_rt_use_msun_yr"] = np.nan

    mstar_tracked_col = first_available_column(out, ["Mstar_tracked_msun", "Mstar_total_msun"])
    mstar_member_col = first_available_column(out, ["Mstar_member_msun", "Mstar_inside_rt_msun", "Mstar_tracked_msun"])
    mstar_inside_rt_col = first_available_column(out, ["Mstar_inside_rt_msun", "Mstar_member_msun", "Mstar_tracked_msun"])

    mgas_tracked_col = first_available_column(out, ["Mgas_tracked_msun", "Mgas_total_msun"])
    mgas_inside_rt_col = first_available_column(out, ["Mgas_inside_rt_msun", "Mgas_member_msun", "Mgas_tracked_msun"])

    if mstar_tracked_col is not None:
        out["sSFR_tracked_yr"] = safe_divide(out["SFR_tracked_use_msun_yr"], out[mstar_tracked_col])
    else:
        out["sSFR_tracked_yr"] = np.nan

    if mstar_member_col is not None:
        out["sSFR_member_yr"] = safe_divide(out["SFR_tracked_use_msun_yr"], out[mstar_member_col])
    else:
        out["sSFR_member_yr"] = np.nan

    if mstar_inside_rt_col is not None:
        out["sSFR_inside_rt_yr"] = safe_divide(out["SFR_inside_rt_use_msun_yr"], out[mstar_inside_rt_col])
    else:
        out["sSFR_inside_rt_yr"] = np.nan

    if mgas_tracked_col is not None:
        out["SFE_tracked_yr"] = safe_divide(out["SFR_tracked_use_msun_yr"], out[mgas_tracked_col])
        out["tdep_tracked_gyr"] = safe_divide(out[mgas_tracked_col], out["SFR_tracked_use_msun_yr"]) / 1.0e9
    else:
        out["SFE_tracked_yr"] = np.nan
        out["tdep_tracked_gyr"] = np.nan

    if mgas_inside_rt_col is not None:
        out["SFE_inside_rt_yr"] = safe_divide(out["SFR_inside_rt_use_msun_yr"], out[mgas_inside_rt_col])
        out["tdep_inside_rt_gyr"] = safe_divide(out[mgas_inside_rt_col], out["SFR_inside_rt_use_msun_yr"]) / 1.0e9
    else:
        out["SFE_inside_rt_yr"] = np.nan
        out["tdep_inside_rt_gyr"] = np.nan

    return out


def prepare_results_for_comparison(
    results: Dict[str, pd.DataFrame],
    cfg,
) -> Dict[str, pd.DataFrame]:
    """
    Return a copy of results with derived quantities added.
    """
    prepared = {}
    for label, df in results.items():
        tmp = df.copy()
        tmp = add_time_since_first_pericentre(tmp)
        tmp = add_derived_tidal_quantities(tmp, cfg)
        tmp = add_derived_sfr_quantities(tmp)
        tmp["label"] = label
        prepared[label] = tmp
    return prepared


# ---------------------------------------------------------------------
# Printing and tables
# ---------------------------------------------------------------------

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

        if "time_gyr" in ev.columns:
            tperi = get_first_pericentre_time(df)
            ev["t_first_pericentre_gyr"] = tperi
            ev["time_since_first_pericentre_gyr"] = ev["time_gyr"] - tperi

        all_events.append(ev)

        label_dir = ensure_dir(output_dir / label)
        ev.to_csv(label_dir / "orbital_extrema.csv", index=False)

        cols = [
            "event_type",
            "snapshot_number",
            "time_gyr",
            "time_since_first_pericentre_gyr",
            "R_host_kpc",
            "V_3d_kms",
            "V_rad_kms",
            "V_tan_kms",
            "r_tidal_kpc",
            "rhalf_star_member_kpc",
            "tidal_field_kms2_kpc2",
            "SFR_tracked_use_msun_yr",
            "sSFR_member_yr",
            "SFE_tracked_yr",
        ]
        cols = [c for c in cols if c in ev.columns]

        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)
        print(
            ev[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.3e}" if abs(x) > 1e4 or (abs(x) < 1e-2 and x != 0) else f"{x:.3f}",
            )
        )

    if all_events:
        out = pd.concat(all_events, ignore_index=True)
        out.to_csv(output_dir / "comparison_orbital_extrema_all_labels.csv", index=False)
        return out

    return pd.DataFrame()


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _plot_extrema_markers(
    ax,
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
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


def _add_pericentre_reference_line(ax, x_axis_mode: str) -> None:
    mode = x_axis_mode.lower()
    if mode in ["time_since_first_pericentre", "t_since_peri", "tminusperi", "t-tperi", "t_minus_tperi",
                "snapshot_since_first_pericentre", "snap_since_peri"]:
        ax.axvline(0.0, lw=1.0, ls="--", alpha=0.6)


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
    """
    Plot one quantity for all labels.
    """
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

    required = {"x_sat_kpc", "y_sat_kpc"}
    if not all(any(col in df.columns for df in results.values()) for col in required):
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


def plot_comparison_grid(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    filename: str = "comparison_summary_grid.png",
    annotate_extrema: bool = True,
    x_axis_mode: str = "time",
) -> Optional[Path]:
    output_dir = ensure_dir(output_dir)

    panels = [
        ("R_host_kpc", r"$R_{\rm host}$ [kpc]", False, "Orbital radius"),
        ("V_rad_kms", r"$v_{\rm rad}$ [km s$^{-1}$]", False, "Radial velocity"),
        ("V_tan_kms", r"$v_{\rm tan}$ [km s$^{-1}$]", False, "Tangential velocity"),

        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", True, "Tidal radius"),
        ("rhalf_star_member_kpc", r"$\log(r_{1/2,\star}/{\rm kpc})$", True, "Member stellar size"),
        ("rhalf_over_rt", r"$r_{1/2,\star}/r_t$", False, "Stellar size / tidal radius"),

        ("Mstar_member_msun", r"$\log(M_{\star,\rm member}/M_\odot)$", True, "Member stellar mass"),
        ("Mgas_tracked_msun", r"$\log(M_{\rm gas,tracked}/M_\odot)$", True, "Tracked gas mass"),
        ("Mdm_tracked_msun", r"$\log(M_{\rm DM,tracked}/M_\odot)$", True, "Tracked DM mass"),

        ("SFR_tracked_use_msun_yr", r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$", True, "SFR"),
        ("sSFR_member_yr", r"$\log({\rm sSFR}/{\rm yr}^{-1})$", True, "sSFR"),
        ("SFE_tracked_yr", r"$\log({\rm SFE}/{\rm yr}^{-1})$", True, "SFE"),

        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", True, "Ram pressure"),
        ("tidal_field_kms2_kpc2", r"$\log(GM(<R)/R^3)$", True, "Tidal-field proxy"),
        ("tidal_accel_across_rhalf_kms2_kpc", r"$\log[(GM(<R)/R^3)r_{1/2}]$", True, "Tidal acceleration"),
    ]

    cols_available = available_columns(results)
    panels = [p for p in panels if p[0] in cols_available]

    if not panels:
        print("[skip] No standard columns available for comparison grid.")
        return None

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, (column, ylabel, logy, title) in zip(axes, panels):
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

    for ax in axes[len(panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(3, len(labels)))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / filename
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sfr_sfe_ssfr_grid(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    filename: str = "comparison_sfr_sfe_ssfr_grid.png",
    annotate_extrema: bool = True,
    x_axis_mode: str = "time_since_first_pericentre",
) -> Optional[Path]:
    """
    Dedicated grid for SFR, SFE, sSFR and depletion time.

    It automatically skips panels that have no finite values.
    """
    output_dir = ensure_dir(output_dir)

    panels = [
        ("SFR_tracked_use_msun_yr", r"$\log({\rm SFR}_{\rm tracked}/M_\odot\,{\rm yr}^{-1})$", True, "Tracked/global SFR"),
        ("SFR_inside_rt_use_msun_yr", r"$\log({\rm SFR}(<r_t)/M_\odot\,{\rm yr}^{-1})$", True, "SFR inside tidal radius"),
        ("sSFR_member_yr", r"$\log({\rm sSFR}_{\rm member}/{\rm yr}^{-1})$", True, "Member sSFR"),
        ("sSFR_inside_rt_yr", r"$\log({\rm sSFR}(<r_t)/{\rm yr}^{-1})$", True, "sSFR inside tidal radius"),
        ("SFE_tracked_yr", r"$\log({\rm SFE}_{\rm tracked}/{\rm yr}^{-1})$", True, "Tracked gas SFE"),
        ("SFE_inside_rt_yr", r"$\log({\rm SFE}(<r_t)/{\rm yr}^{-1})$", True, "SFE inside tidal radius"),
        ("tdep_tracked_gyr", r"$\log(t_{\rm dep,tracked}/{\rm Gyr})$", True, "Tracked gas depletion time"),
        ("tdep_inside_rt_gyr", r"$\log(t_{\rm dep}(<r_t)/{\rm Gyr})$", True, "Depletion time inside tidal radius"),
    ]

    useful_panels = []
    for panel in panels:
        col = panel[0]
        has_finite = False
        for df in results.values():
            if col in df.columns and np.any(np.isfinite(np.asarray(df[col], dtype=float))):
                has_finite = True
                break
        if has_finite:
            useful_panels.append(panel)

    if not useful_panels:
        print("[skip] No finite SFR/SFE/sSFR quantities found.")
        return None

    ncols = 2
    nrows = int(np.ceil(len(useful_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), facecolor="white")
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

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / filename
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_environment_grid(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    filename: str = "comparison_environment_grid.png",
    annotate_extrema: bool = True,
    x_axis_mode: str = "time_since_first_pericentre",
) -> Optional[Path]:
    """
    Dedicated grid for ram pressure and tidal forcing.
    """
    output_dir = ensure_dir(output_dir)

    panels = [
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", True, "Ram pressure"),
        ("rho_cgm_msun_kpc3", r"$\log(\rho_{\rm CGM}/M_\odot\,{\rm kpc}^{-3})$", True, "Local CGM density"),
        ("V_rel_cgm_kms", r"$v_{\rm rel,CGM}$ [km s$^{-1}$]", False, "Relative velocity with CGM"),
        ("tidal_field_kms2_kpc2", r"$\log(GM_{\rm host}(<R)/R^3)$", True, "Tidal-field proxy"),
        ("tidal_accel_across_rhalf_kms2_kpc", r"$\log[(GM(<R)/R^3)r_{1/2}]$", True, "Tidal acceleration across stars"),
        ("rhalf_over_rt", r"$r_{1/2,\star}/r_t$", False, "Size relative to tidal radius"),
    ]

    useful_panels = []
    for panel in panels:
        col = panel[0]
        has_finite = False
        for df in results.values():
            if col in df.columns and np.any(np.isfinite(np.asarray(df[col], dtype=float))):
                has_finite = True
                break
        if has_finite:
            useful_panels.append(panel)

    if not useful_panels:
        print("[skip] No finite environmental quantities found.")
        return None

    ncols = 2
    nrows = int(np.ceil(len(useful_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), facecolor="white")
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

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / filename
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
# Main Spyder function
# ---------------------------------------------------------------------

def run_all_comparison_plots(
    results: Dict[str, pd.DataFrame],
    cfg,
    output_subdir: str = "comparison_plots",
    annotate_extrema: bool = True,
    x_axis_mode: str = "time",
) -> Dict[str, Path]:
    """
    Main function to call from Spyder.

    Parameters
    ----------
    results
        Dictionary of DataFrames produced by the main orbit-analysis script.
    cfg
        Configuration object from the main script. Must include cfg.output_dir
        and cfg.host.mass_enclosed_msun().
    output_subdir
        Name of subdirectory inside cfg.output_dir.
    annotate_extrema
        If True, mark pericentres and apocentres on the curves.
    x_axis_mode
        Options:
            "time"
            "snapshot"
            "time_since_first_pericentre"
            "snapshot_since_first_pericentre"

    Returns
    -------
    outputs
        Dictionary with saved file paths.
    """
    output_dir = ensure_dir(Path(cfg.output_dir) / output_subdir)

    prepared = prepare_results_for_comparison(results, cfg)

    # Save combined table with derived quantities.
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

    grid_name = "comparison_summary_grid.png"
    if x_axis_mode.lower() in ["time_since_first_pericentre", "t_since_peri", "tminusperi", "t-tperi", "t_minus_tperi"]:
        grid_name = "comparison_summary_grid_t_minus_tperi.png"

    p = plot_comparison_grid(
        prepared,
        output_dir,
        filename=grid_name,
        annotate_extrema=annotate_extrema,
        x_axis_mode=x_axis_mode,
    )
    if p is not None:
        outputs["summary_grid"] = p

    p = plot_sfr_sfe_ssfr_grid(
        prepared,
        output_dir,
        filename="comparison_sfr_sfe_ssfr_grid.png",
        annotate_extrema=annotate_extrema,
        x_axis_mode=x_axis_mode,
    )
    if p is not None:
        outputs["sfr_sfe_ssfr_grid"] = p

    p = plot_environment_grid(
        prepared,
        output_dir,
        filename="comparison_environment_grid.png",
        annotate_extrema=annotate_extrema,
        x_axis_mode=x_axis_mode,
    )
    if p is not None:
        outputs["environment_grid"] = p

    # Individual plots.
    standard_plots = [
        # Orbit.
        ("R_host_kpc", r"$R_{\rm host}$ [kpc]", "comparison_Rhost.png", "Orbital radius", False),
        ("V_3d_kms", r"$|v|$ [km s$^{-1}$]", "comparison_V3d.png", "3D velocity", False),
        ("V_rad_kms", r"$v_{\rm rad}$ [km s$^{-1}$]", "comparison_Vrad.png", "Radial velocity", False),
        ("V_tan_kms", r"$v_{\rm tan}$ [km s$^{-1}$]", "comparison_Vtan.png", "Tangential velocity", False),

        # Sizes.
        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", "comparison_log_rt.png", "Tidal radius", True),
        ("rhalf_star_member_kpc", r"$\log(r_{1/2,\star,\rm member}/{\rm kpc})$", "comparison_log_rhalf_member.png", "Member stellar half-mass radius", True),
        ("rhalf_star_all_kpc", r"$\log(r_{1/2,\star,\rm all}/{\rm kpc})$", "comparison_log_rhalf_all.png", "All-stars half-mass radius", True),
        ("rhalf_over_rt", r"$r_{1/2,\star}/r_t$", "comparison_rhalf_over_rt.png", "Stellar size relative to tidal radius", False),

        # Masses.
        ("Mstar_member_msun", r"$\log(M_{\star,\rm member}/M_\odot)$", "comparison_log_Mstar_member.png", "Member stellar mass", True),
        ("Mstar_inside_rt_msun", r"$\log(M_\star(<r_t)/M_\odot)$", "comparison_log_Mstar_inside_rt.png", "Stellar mass inside tidal radius", True),
        ("Mgas_tracked_msun", r"$\log(M_{\rm gas,tracked}/M_\odot)$", "comparison_log_Mgas_tracked.png", "Tracked gas mass", True),
        ("Mgas_inside_rt_msun", r"$\log(M_{\rm gas}(<r_t)/M_\odot)$", "comparison_log_Mgas_inside_rt.png", "Gas mass inside tidal radius", True),
        ("Mdm_tracked_msun", r"$\log(M_{\rm DM,tracked}/M_\odot)$", "comparison_log_Mdm_tracked.png", "Tracked DM mass", True),
        ("Mdm_inside_rt_msun", r"$\log(M_{\rm DM}(<r_t)/M_\odot)$", "comparison_log_Mdm_inside_rt.png", "DM mass inside tidal radius", True),

        # Stellar membership / stripping.
        ("fstar_member", r"$f_{\star,\rm member}$", "comparison_fstar_member.png", "Member stellar fraction", False),
        ("fstar_stripped_definitive", r"$f_{\star,\rm stripped}$", "comparison_fstar_stripped.png", "Definitively stripped stellar fraction", False),

        # Ram pressure and environment.
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", "comparison_log_Pram.png", "Ram pressure", True),
        ("rho_cgm_msun_kpc3", r"$\log(\rho_{\rm CGM}/M_\odot\,{\rm kpc}^{-3})$", "comparison_log_rho_cgm.png", "Local CGM density", True),
        ("V_rel_cgm_kms", r"$v_{\rm rel,CGM}$ [km s$^{-1}$]", "comparison_Vrel_CGM.png", "Relative velocity with CGM", False),

        # SFR, sSFR, SFE.
        ("SFR_tracked_use_msun_yr", r"$\log({\rm SFR}_{\rm tracked}/M_\odot\,{\rm yr}^{-1})$", "comparison_log_SFR_tracked.png", "Tracked/global SFR", True),
        ("SFR_inside_rt_use_msun_yr", r"$\log({\rm SFR}(<r_t)/M_\odot\,{\rm yr}^{-1})$", "comparison_log_SFR_inside_rt.png", "SFR inside tidal radius", True),
        ("sSFR_member_yr", r"$\log({\rm sSFR}_{\rm member}/{\rm yr}^{-1})$", "comparison_log_sSFR_member.png", "Member sSFR", True),
        ("sSFR_inside_rt_yr", r"$\log({\rm sSFR}(<r_t)/{\rm yr}^{-1})$", "comparison_log_sSFR_inside_rt.png", "sSFR inside tidal radius", True),
        ("SFE_tracked_yr", r"$\log({\rm SFE}_{\rm tracked}/{\rm yr}^{-1})$", "comparison_log_SFE_tracked.png", "Tracked gas SFE", True),
        ("SFE_inside_rt_yr", r"$\log({\rm SFE}(<r_t)/{\rm yr}^{-1})$", "comparison_log_SFE_inside_rt.png", "SFE inside tidal radius", True),
        ("tdep_tracked_gyr", r"$\log(t_{\rm dep,tracked}/{\rm Gyr})$", "comparison_log_tdep_tracked.png", "Tracked gas depletion time", True),
        ("tdep_inside_rt_gyr", r"$\log(t_{\rm dep}(<r_t)/{\rm Gyr})$", "comparison_log_tdep_inside_rt.png", "Depletion time inside tidal radius", True),

        # Tidal forcing.
        ("tidal_field_kms2_kpc2", r"$\log(GM_{\rm host}(<R)/R^3)$", "comparison_log_tidal_field.png", "Tidal-field proxy", True),
        ("tidal_accel_across_rhalf_kms2_kpc", r"$\log[(GM(<R)/R^3)r_{1/2}]$", "comparison_log_tidal_accel_rhalf.png", "Tidal acceleration across stellar body", True),
    ]

    for column, ylabel, filename, title, logy in standard_plots:
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


# ---------------------------------------------------------------------
# Convenience function to run both normal time and t - first pericentre
# ---------------------------------------------------------------------

def run_standard_and_pericentre_normalized_plots(
    results: Dict[str, pd.DataFrame],
    cfg,
    annotate_extrema: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Convenience wrapper that creates two folders:

        comparison_plots_time/
        comparison_plots_t_minus_tperi/

    Useful in Spyder when you want both views.
    """
    out_time = run_all_comparison_plots(
        results,
        cfg,
        output_subdir="comparison_plots_time",
        annotate_extrema=annotate_extrema,
        x_axis_mode="time",
    )

    out_tperi = run_all_comparison_plots(
        results,
        cfg,
        output_subdir="comparison_plots_t_minus_tperi",
        annotate_extrema=annotate_extrema,
        x_axis_mode="time_since_first_pericentre",
    )

    return {
        "time": out_time,
        "time_since_first_pericentre": out_tperi,
    }
