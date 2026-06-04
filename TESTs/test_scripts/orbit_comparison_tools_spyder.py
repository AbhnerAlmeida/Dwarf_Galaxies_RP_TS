#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_comparison_tools_spyder.py

Helper functions to compare several satellite-orbit simulations after running
orbit_center_tracking_spyder.py.

Usage in Spyder, after you have run the main analysis and have `results` and `cfg`
in memory:

    from orbit_comparison_tools_spyder import run_all_comparison_plots

    comparison_outputs = run_all_comparison_plots(
        results,
        cfg,
        output_subdir="comparison_plots",
        annotate_extrema=True,
    )

This file does not reread the HDF5 snapshots. It uses the DataFrames already
created by the centre-tracking script. It derives simple tidal-force proxies from
R_host_kpc and cfg.host.mass_enclosed_msun(R).

It will plot SFR and ram pressure only if the corresponding columns exist in the
DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G_KPC_KMS2_MSUN = 4.30091e-6


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log10_safe(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, np.log10(values), np.nan)


def get_time_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "time_gyr" in df.columns and np.any(np.isfinite(df["time_gyr"].values)):
        return df["time_gyr"].values, "Time [Gyr]"
    if "snapshot_number" in df.columns:
        return df["snapshot_number"].values, "Snapshot"
    return np.arange(len(df)), "Index"


def available_columns(results: Dict[str, pd.DataFrame]) -> set:
    cols = set()
    for df in results.values():
        cols.update(df.columns)
    return cols


def add_derived_tidal_quantities(
    df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """
    Add host enclosed mass and simple tidal-field proxies.

    tidal_field_kms2_kpc2 = G M_host(<R) / R^3

    This is not the full NFW tidal tensor. It is a compact diagnostic for how
    strong the external field is at the satellite location. For comparisons
    across runs with the same host, it is very useful.
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
    out["tidal_field_proxy_msun_kpc3"] = np.where(
        R > 0,
        Mhost / R**3,
        np.nan,
    )

    # Approximate tidal acceleration across the stellar body.
    rhalf_col = None
    for candidate in ["rhalf_star_member_kpc", "rhalf_star_kpc", "rhalf_star_all_kpc"]:
        if candidate in out.columns:
            rhalf_col = candidate
            break

    if rhalf_col is not None:
        out["tidal_accel_across_rhalf_kms2_kpc"] = (
            out["tidal_field_kms2_kpc2"] * out[rhalf_col]
        )

    # Useful ratios if available.
    if "r_tidal_kpc" in out.columns and rhalf_col is not None:
        out["rhalf_over_rt"] = out[rhalf_col] / out["r_tidal_kpc"]

    return out


def prepare_results_for_comparison(
    results: Dict[str, pd.DataFrame],
    cfg,
) -> Dict[str, pd.DataFrame]:
    """
    Return a copy of results with derived quantities added.
    """
    return {
        label: add_derived_tidal_quantities(df, cfg).assign(label=label)
        for label, df in results.items()
    }


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
            "V_3d_kms",
            "V_rad_kms",
            "V_tan_kms",
            "r_tidal_kpc",
            "rhalf_star_member_kpc",
            "rhalf_star_all_kpc",
            "Mstar_member_msun",
            "Mstar_inside_rt_msun",
            "Mgas_tracked_msun",
            "Mdm_tracked_msun",
            "P_ram_dyne_cm2",
            "SFR_gas_tracked_msun_yr",
            "SFR_gas_inside_rt_msun_yr",
            "tidal_field_kms2_kpc2",
            "tidal_accel_across_rhalf_kms2_kpc",
        ]:
            if col in df.columns:
                row[col] = float(df.iloc[i][col]) if np.isfinite(df.iloc[i][col]) else np.nan

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
            "V_3d_kms",
            "V_rad_kms",
            "V_tan_kms",
            "r_tidal_kpc",
            "rhalf_star_member_kpc",
            "rhalf_star_all_kpc",
            "Mstar_member_msun",
            "Mstar_inside_rt_msun",
            "Mgas_tracked_msun",
            "Mdm_tracked_msun",
            "P_ram_dyne_cm2",
            "SFR_gas_tracked_msun_yr",
            "SFR_gas_inside_rt_msun_yr",
            "tidal_field_kms2_kpc2",
            "tidal_accel_across_rhalf_kms2_kpc",
        ]:
            if col in df.columns:
                row[col] = float(df.iloc[i][col]) if np.isfinite(df.iloc[i][col]) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


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
        all_events.append(ev)

        label_dir = ensure_dir(output_dir / label if isinstance(output_dir, Path) else Path(output_dir) / label)
        ev.to_csv(label_dir / "orbital_extrema.csv", index=False)

        cols = [
            "event_type",
            "snapshot_number",
            "time_gyr",
            "R_host_kpc",
            "V_3d_kms",
            "V_rad_kms",
            "V_tan_kms",
            "r_tidal_kpc",
            "rhalf_star_member_kpc",
            "tidal_field_kms2_kpc2",
        ]
        cols = [c for c in cols if c in ev.columns]

        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)
        print(ev[cols].to_string(index=False, float_format=lambda x: f"{x:.3e}" if abs(x) > 1e4 or (abs(x) < 1e-2 and x != 0) else f"{x:.3f}"))

    if all_events:
        out = pd.concat(all_events, ignore_index=True)
        out.to_csv(output_dir / "comparison_orbital_extrema_all_labels.csv", index=False)
        return out

    return pd.DataFrame()


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
    for label, df in results.items():
        if column not in df.columns:
            continue

        x, xlabel = get_time_axis(df)
        y = np.asarray(df[column], dtype=float)
        if log10_y:
            y = log10_safe(y)

        if not np.any(np.isfinite(y)):
            continue

        ax.plot(x, y, marker="o", lw=1.5, label=label)
        if annotate_extrema:
            _plot_extrema_markers(ax, df, x, y)
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"[skip] No finite values for {column}")
        return None

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
) -> Optional[Path]:
    output_dir = ensure_dir(output_dir)

    panels = [
        ("R_host_kpc", r"$R_{\rm host}$ [kpc]", False, "Orbital radius"),
        ("V_rad_kms", r"$v_{\rm rad}$ [km s$^{-1}$]", False, "Radial velocity"),
        ("V_tan_kms", r"$v_{\rm tan}$ [km s$^{-1}$]", False, "Tangential velocity"),
        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", True, "Tidal radius"),
        ("rhalf_star_member_kpc", r"$\log(r_{1/2,\star}/{\rm kpc})$", True, "Member stellar size"),
        ("Mstar_member_msun", r"$\log(M_{\star,\rm member}/M_\odot)$", True, "Member stellar mass"),
        ("Mgas_tracked_msun", r"$\log(M_{\rm gas,tracked}/M_\odot)$", True, "Tracked gas mass"),
        ("Mdm_tracked_msun", r"$\log(M_{\rm DM,tracked}/M_\odot)$", True, "Tracked DM mass"),
        ("tidal_field_kms2_kpc2", r"$\log(GM(<R)/R^3)$", True, "Tidal-field proxy"),
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
        for label, df in results.items():
            if column not in df.columns:
                continue
            x, xlabel = get_time_axis(df)
            y = np.asarray(df[column], dtype=float)
            if logy:
                y = log10_safe(y)
            if not np.any(np.isfinite(y)):
                continue
            ax.plot(x, y, marker="o", lw=1.5, label=label)
            if annotate_extrema:
                _plot_extrema_markers(ax, df, x, y)

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


def run_all_comparison_plots(
    results: Dict[str, pd.DataFrame],
    cfg,
    output_subdir: str = "comparison_plots",
    annotate_extrema: bool = True,
) -> Dict[str, Path]:
    """
    Main function to call from Spyder.

    Returns a dictionary with the paths of the saved figures/tables.
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

    p = plot_comparison_grid(prepared, output_dir, annotate_extrema=annotate_extrema)
    if p is not None:
        outputs["summary_grid"] = p

    # Orbit.
    standard_plots = [
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
        ("Mdm_tracked_msun", r"$\log(M_{\rm DM,tracked}/M_\odot)$", "comparison_log_Mdm_tracked.png", "Tracked DM mass", True),

        # Stellar membership / stripping.
        ("fstar_member", r"$f_{\star,\rm member}$", "comparison_fstar_member.png", "Member stellar fraction", False),
        ("fstar_stripped_definitive", r"$f_{\star,\rm stripped}$", "comparison_fstar_stripped.png", "Definitively stripped stellar fraction", False),

        # Ram pressure. These are plotted only if the columns exist.
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", "comparison_log_Pram.png", "Ram pressure", True),
        ("rho_cgm_msun_kpc3", r"$\log(\rho_{\rm CGM}/M_\odot\,{\rm kpc}^{-3})$", "comparison_log_rho_cgm.png", "Local CGM density", True),
        ("V_rel_cgm_kms", r"$v_{\rm rel,CGM}$ [km s$^{-1}$]", "comparison_Vrel_CGM.png", "Relative velocity with CGM", False),

        # SFR. These are plotted only if the columns exist.
        ("SFR_gas_tracked_msun_yr", r"$\log({\rm SFR}_{\rm tracked}/M_\odot\,{\rm yr}^{-1})$", "comparison_log_SFR_tracked.png", "Tracked gas SFR", True),
        ("SFR_gas_inside_rt_msun_yr", r"$\log({\rm SFR}(<r_t)/M_\odot\,{\rm yr}^{-1})$", "comparison_log_SFR_inside_rt.png", "SFR inside tidal radius", True),

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
        )
        if p is not None:
            outputs[column] = p

    print("\nSaved comparison outputs in:")
    print(output_dir)

    return outputs

