#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_functions_gadget.py

Plotting utilities for the Gadget4 dwarf-galaxy orbit project.

This module is inspired by Abhner's TNG PlotFunctions.py workflow, but is
designed to be lightweight and independent of TNG-specific modules. It works
directly with the DataFrames produced by orbit_satellite_analysis.py /
orbit_satellite_analysis_extended.py, especially the dictionary returned by:

    results = analyze_all(cfg)

Main goals
----------
- Keep a consistent visual identity across Gadget dwarf simulations.
- Provide simple wrappers for comparison plots, phase-aligned plots, mass-budget
  panels, ram-pressure panels, star-formation panels, and retained-fraction plots.
- Avoid importing TNGFunctions, ExtractTNG, or project-specific TNG machinery.
- Use clear labels for columns produced by the Gadget analysis pipeline.

Typical use
-----------
from pathlib import Path
from plot_functions_gadget import (
    load_results_from_output,
    plot_standard_gadget_suite,
    plot_quantity,
    plot_mass_budget,
)

# If you already have results from analyze_all(cfg):
plot_standard_gadget_suite(results, outdir="orbit_analysis_outputs/gadget_plots")

# Or load previously saved satellite_evolution.csv files:
results = load_results_from_output(
    output_root="orbit_analysis_outputs",
    labels=["E_mid_L_radial", "E_mid_L_mid", "E_mid_L_high"],
)

plot_quantity(
    results,
    column="Mgas_inside_rt_msun",
    outdir="orbit_analysis_outputs/gadget_plots",
    filename="phase_log_Mgas_inside_rt",
    xmode="phase",
    log=True,
)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None


# =============================================================================
# Basic style registry
# =============================================================================

@dataclass(frozen=True)
class LineStyle:
    color: str = "black"
    linestyle: object = "solid"
    marker: str = "o"
    linewidth: float = 1.8
    markersize: float = 4.5
    alpha: float = 1.0
    label: Optional[str] = None


# Simulation/orbit styles.
# You can extend this dictionary with your actual run labels.
LABEL_STYLES: Dict[str, LineStyle] = {
    "E_mid_L_radial": LineStyle(
        color="tab:red", linestyle="solid", marker="o",
        label=r"$E_{\rm mid}$, radial"
    ),
    "E_mid_L_mid": LineStyle(
        color="tab:blue", linestyle="solid", marker="s",
        label=r"$E_{\rm mid}$, intermediate $L$"
    ),
    "E_mid_L_high": LineStyle(
        color="tab:green", linestyle="solid", marker="^",
        label=r"$E_{\rm mid}$, high $L$"
    ),
    "E_low_L_radial": LineStyle(
        color="tab:orange", linestyle="solid", marker="o",
        label=r"$E_{\rm low}$, radial"
    ),
    "E_high_L_radial": LineStyle(
        color="tab:purple", linestyle="solid", marker="o",
        label=r"$E_{\rm high}$, radial"
    ),
    "fiducial": LineStyle(
        color="black", linestyle="solid", marker="o", label="fiducial"
    ),
    "DMx4": LineStyle(
        color="tab:purple", linestyle="solid", marker="D", label="DMx4"
    ),
    "DMx5": LineStyle(
        color="tab:brown", linestyle="solid", marker="D", label="DMx5"
    ),
    "ZOOM": LineStyle(
        color="tab:cyan", linestyle="solid", marker="P", label="ZOOM"
    ),
}


# Component styles for mass-budget plots.
COMPONENT_STYLES: Dict[str, LineStyle] = {
    "stars": LineStyle(color="tab:orange", linestyle="solid", marker="o", label="stars"),
    "gas": LineStyle(color="tab:blue", linestyle=(0, (8, 5)), marker="s", label="gas"),
    "dm": LineStyle(color="tab:purple", linestyle=(0, (1, 2)), marker="^", label="DM"),
    "total": LineStyle(color="black", linestyle="solid", marker="D", label="total"),
    "sf_gas": LineStyle(color="tab:green", linestyle="solid", marker="s", label="star-forming gas"),
    "young_stars": LineStyle(color="tab:red", linestyle="solid", marker="*", label="young stars"),
    "ram": LineStyle(color="tab:brown", linestyle="solid", marker="o", label="ram pressure"),
}


@dataclass(frozen=True)
class QuantitySpec:
    column: str
    ylabel: str
    title: str
    log: bool = False
    default_filename: Optional[str] = None


QUANTITIES: Dict[str, QuantitySpec] = {
    # Orbit
    "R_host": QuantitySpec(
        "R_host_kpc", r"$\log(R_{\rm host}/{\rm kpc})$",
        "Orbital radius", log=True, default_filename="log_Rhost"
    ),
    "V_rad": QuantitySpec(
        "V_rad_kms", r"$v_{\rm rad}$ [km s$^{-1}$]",
        "Radial velocity", log=False, default_filename="Vrad"
    ),
    "V_tan": QuantitySpec(
        "V_tan_kms", r"$v_{\rm tan}$ [km s$^{-1}$]",
        "Tangential velocity", log=False, default_filename="Vtan"
    ),
    "V_3d": QuantitySpec(
        "V_3d_kms", r"$|v|$ [km s$^{-1}$]",
        "3D velocity", log=False, default_filename="V3d"
    ),

    # Sizes
    "r_tidal": QuantitySpec(
        "r_tidal_kpc", r"$\log(r_t/{\rm kpc})$",
        "Tidal radius", log=True, default_filename="log_rt"
    ),
    "rhalf_star": QuantitySpec(
        "rhalf_star_kpc", r"$\log(r_{1/2,\star}/{\rm kpc})$",
        "Stellar half-mass radius", log=True, default_filename="log_rhalf_star"
    ),
    "r90_star": QuantitySpec(
        "r90_star_kpc", r"$\log(r_{90,\star}/{\rm kpc})$",
        "Stellar 90 per cent radius", log=True, default_filename="log_r90_star"
    ),

    # Masses inside tidal radius
    "Mstar_rt": QuantitySpec(
        "Mstar_inside_rt_msun", r"$\log(M_\star(<r_t)/M_\odot)$",
        "Stellar mass inside tidal radius", log=True, default_filename="log_Mstar_rt"
    ),
    "Mgas_rt": QuantitySpec(
        "Mgas_inside_rt_msun", r"$\log(M_{\rm gas}(<r_t)/M_\odot)$",
        "Satellite gas mass inside tidal radius", log=True, default_filename="log_Mgas_rt"
    ),
    "Mdm_rt": QuantitySpec(
        "Mdm_inside_rt_msun", r"$\log(M_{\rm DM}(<r_t)/M_\odot)$",
        "Dark matter mass inside tidal radius", log=True, default_filename="log_Mdm_rt"
    ),
    "Msat_rt": QuantitySpec(
        "Msat_inside_rt_msun", r"$\log(M_{\rm sat}(<r_t)/M_\odot)$",
        "Total mass inside tidal radius", log=True, default_filename="log_Msat_rt"
    ),

    # Central masses
    "Mstar_rhalf": QuantitySpec(
        "Mstar_inside_rhalf_msun", r"$\log(M_\star(<r_{1/2,\star})/M_\odot)$",
        "Stellar mass inside stellar half-mass radius", log=True,
        default_filename="log_Mstar_rhalf"
    ),
    "Mgas_rhalf": QuantitySpec(
        "Mgas_inside_rhalf_msun", r"$\log(M_{\rm gas}(<r_{1/2,\star})/M_\odot)$",
        "Gas mass inside stellar half-mass radius", log=True,
        default_filename="log_Mgas_rhalf"
    ),
    "Mdm_rhalf": QuantitySpec(
        "Mdm_inside_rhalf_msun", r"$\log(M_{\rm DM}(<r_{1/2,\star})/M_\odot)$",
        "DM mass inside stellar half-mass radius", log=True,
        default_filename="log_Mdm_rhalf"
    ),

    # Fixed apertures produced by the extended analysis
    "Mstar_1kpc": QuantitySpec(
        "Mstar_inside_1kpc_msun", r"$\log(M_\star(<1\,{\rm kpc})/M_\odot)$",
        "Stellar mass inside 1 kpc", log=True, default_filename="log_Mstar_1kpc"
    ),
    "Mgas_1kpc": QuantitySpec(
        "Mgas_inside_1kpc_msun", r"$\log(M_{\rm gas}(<1\,{\rm kpc})/M_\odot)$",
        "Gas mass inside 1 kpc", log=True, default_filename="log_Mgas_1kpc"
    ),
    "Mdm_1kpc": QuantitySpec(
        "Mdm_inside_1kpc_msun", r"$\log(M_{\rm DM}(<1\,{\rm kpc})/M_\odot)$",
        "DM mass inside 1 kpc", log=True, default_filename="log_Mdm_1kpc"
    ),
    "Mstar_3kpc": QuantitySpec(
        "Mstar_inside_3kpc_msun", r"$\log(M_\star(<3\,{\rm kpc})/M_\odot)$",
        "Stellar mass inside 3 kpc", log=True, default_filename="log_Mstar_3kpc"
    ),
    "Mgas_3kpc": QuantitySpec(
        "Mgas_inside_3kpc_msun", r"$\log(M_{\rm gas}(<3\,{\rm kpc})/M_\odot)$",
        "Gas mass inside 3 kpc", log=True, default_filename="log_Mgas_3kpc"
    ),
    "Mdm_3kpc": QuantitySpec(
        "Mdm_inside_3kpc_msun", r"$\log(M_{\rm DM}(<3\,{\rm kpc})/M_\odot)$",
        "DM mass inside 3 kpc", log=True, default_filename="log_Mdm_3kpc"
    ),

    # Star formation / gas
    "SFR_tracked": QuantitySpec(
        "SFR_tracked_msun_per_yr", r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$",
        "Tracked satellite SFR", log=True, default_filename="log_SFR_tracked"
    ),
    "SFR_rt": QuantitySpec(
        "SFR_inside_rt_msun_per_yr", r"$\log({\rm SFR}(<r_t)/M_\odot\,{\rm yr}^{-1})$",
        "SFR inside tidal radius", log=True, default_filename="log_SFR_rt"
    ),
    "Mgas_SF_tracked": QuantitySpec(
        "Mgas_SF_tracked_msun", r"$\log(M_{\rm gas,SF}/M_\odot)$",
        "Star-forming gas mass", log=True, default_filename="log_Mgas_SF"
    ),
    "Mgas_SF_1kpc": QuantitySpec(
        "Mgas_SF_inside_1kpc_msun", r"$\log(M_{\rm gas,SF}(<1\,{\rm kpc})/M_\odot)$",
        "Star-forming gas inside 1 kpc", log=True, default_filename="log_Mgas_SF_1kpc"
    ),
    "Mstar_young_tracked": QuantitySpec(
        "Mstar_young_tracked_msun", r"$\log(M_{\star,{\rm young}}/M_\odot)$",
        "Young stellar mass", log=True, default_filename="log_Myoung"
    ),
    "Mstar_young_1kpc": QuantitySpec(
        "Mstar_young_inside_1kpc_msun", r"$\log(M_{\star,{\rm young}}(<1\,{\rm kpc})/M_\odot)$",
        "Young stellar mass inside 1 kpc", log=True, default_filename="log_Myoung_1kpc"
    ),

    # Ram pressure
    "P_ram": QuantitySpec(
        "P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\,cm^{-2}})$",
        "Ram pressure", log=True, default_filename="log_Pram"
    ),
    "rho_cgm": QuantitySpec(
        "rho_cgm_msun_kpc3", r"$\log(\rho_{\rm CGM}/M_\odot\,{\rm kpc}^{-3})$",
        "Local CGM density", log=True, default_filename="log_rho_cgm"
    ),
}


# =============================================================================
# General helpers
# =============================================================================

def configure_matplotlib(style_path: Optional[Union[str, Path]] = None) -> None:
    """
    Apply a matplotlib style file if provided and set conservative defaults.
    """
    if style_path is not None:
        plt.style.use(str(style_path))

    plt.rcParams.setdefault("axes.linewidth", 1.0)
    plt.rcParams.setdefault("xtick.direction", "in")
    plt.rcParams.setdefault("ytick.direction", "in")
    plt.rcParams.setdefault("xtick.top", True)
    plt.rcParams.setdefault("ytick.right", True)
    plt.rcParams.setdefault("legend.frameon", False)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name.strip("_")


def save_figure(
    fig,
    outdir: Union[str, Path],
    filename: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    transparent: bool = False,
    close: bool = False,
) -> List[Path]:
    """
    Save a figure in multiple formats.
    """
    plt.show()
    outdir = ensure_dir(outdir)
    filename = sanitize_filename(filename)
    paths = []

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt == "png":
            fmt_dir = ensure_dir(outdir / "PNG")
            path = fmt_dir / f"{filename}.png"
        else:
            path = outdir / f"{filename}.{fmt}"

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=transparent,
        )
        paths.append(path)

    if close:
        plt.close(fig)

    return paths


def log10_safe(values: Union[pd.Series, np.ndarray, Sequence[float]]) -> np.ndarray:
    """
    log10(values) with non-positive values mapped to NaN.
    """
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, np.log10(values), np.nan)


def ratio_safe(
    numerator: Union[pd.Series, np.ndarray],
    denominator: Union[pd.Series, np.ndarray],
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.where(denominator > 0, numerator / denominator, np.nan)


def finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def label_title(label: str) -> str:
    """
    Pretty label name. Uses LABEL_STYLES if available, otherwise replaces underscores.
    """
    if label in LABEL_STYLES and LABEL_STYLES[label].label is not None:
        return LABEL_STYLES[label].label
    return str(label).replace("_", " ")


def style_for_label(label: str, index: int = 0) -> LineStyle:
    """
    Resolve a line style for a simulation label.
    """
    if label in LABEL_STYLES:
        return LABEL_STYLES[label]

    # Deterministic fallback using matplotlib cycle.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
    color = cycle[index % len(cycle)]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    return LineStyle(color=color, marker=markers[index % len(markers)], label=label_title(label))


def component_style(component: str) -> LineStyle:
    return COMPONENT_STYLES.get(component, LineStyle(label=component))


def quantity_spec(key_or_column: str) -> QuantitySpec:
    """
    Return metadata for either a symbolic quantity key or a raw DataFrame column.
    """
    if key_or_column in QUANTITIES:
        return QUANTITIES[key_or_column]

    column = key_or_column
    title = column.replace("_", " ")
    log_guess = any(token in column.lower() for token in ["mass", "msun", "r_", "rho", "pressure", "sfr"])
    ylabel = column
    if log_guess:
        ylabel = rf"$\log({column})$"

    return QuantitySpec(column=column, ylabel=ylabel, title=title, log=log_guess)


# =============================================================================
# Loading analysis tables
# =============================================================================

def load_results_from_output(
    output_root: Union[str, Path],
    labels: Optional[Sequence[str]] = None,
    csv_name: str = "satellite_evolution.csv",
) -> Dict[str, pd.DataFrame]:
    """
    Load satellite_evolution.csv files produced by orbit_satellite_analysis.

    Expected layout:
        output_root / label / satellite_evolution.csv
    """
    output_root = Path(output_root)

    if labels is None:
        labels = [
            p.name for p in output_root.iterdir()
            if p.is_dir() and (p / csv_name).exists()
        ]

    results: Dict[str, pd.DataFrame] = {}

    for label in labels:
        path = output_root / label / csv_name
        if not path.exists():
            warnings.warn(f"Missing table for {label}: {path}")
            continue
        results[label] = pd.read_csv(path)

    return results


# =============================================================================
# X-axis and interpolation
# =============================================================================

def get_x_axis(
    df: pd.DataFrame,
    mode: str = "time",
    phase_column: str = "t_minus_tperi_gyr",
) -> Tuple[np.ndarray, str]:
    """
    Resolve x-axis.

    mode:
        "time"     -> time_gyr if available, otherwise snapshot_number
        "phase"    -> t_minus_tperi_gyr if available, otherwise time_gyr
        "snapshot" -> snapshot_number
    """
    mode = mode.lower()

    if mode in ("phase", "tperi", "t_minus_tperi"):
        if phase_column in df.columns:
            return np.asarray(df[phase_column], dtype=float), r"$t - t_{\rm peri}$ [Gyr]"
        if "t_minus_first_peri_gyr" in df.columns:
            return np.asarray(df["t_minus_first_peri_gyr"], dtype=float), r"$t - t_{\rm peri}$ [Gyr]"
        warnings.warn(f"{phase_column} not found; falling back to time.")

    if mode == "snapshot":
        if "snapshot_number" in df.columns:
            return np.asarray(df["snapshot_number"], dtype=float), "Snapshot"
        return np.arange(len(df), dtype=float), "Snapshot index"

    if "time_gyr" in df.columns:
        x = np.asarray(df["time_gyr"], dtype=float)
        if np.count_nonzero(np.isfinite(x)) >= 2:
            return x, "Time [Gyr]"

    if "snapshot_number" in df.columns:
        return np.asarray(df["snapshot_number"], dtype=float), "Snapshot"

    return np.arange(len(df), dtype=float), "Index"


def _unique_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y = finite_xy(x, y)
    if len(x) == 0:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # If repeated x values exist, keep the median y per x.
    unique_x = np.unique(x)
    if len(unique_x) == len(x):
        return x, y

    y_unique = np.array([np.nanmedian(y[x == ux]) for ux in unique_x])
    return unique_x, y_unique


def smooth_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    ngrid: int = 400,
    xgrid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smoothly interpolate a time series using PCHIP.

    PCHIP is shape-preserving and avoids strong overshoot, which is safer for
    masses, radii and retained fractions than cubic splines.
    """
    x, y = _unique_sorted_xy(x, y)

    if len(x) < 3:
        return x, y

    if xgrid is None:
        xgrid = np.linspace(np.nanmin(x), np.nanmax(x), ngrid)

    if PchipInterpolator is not None:
        f = PchipInterpolator(x, y, extrapolate=False)
        return xgrid, np.asarray(f(xgrid), dtype=float)

    # Fallback.
    return xgrid, np.interp(xgrid, x, y, left=np.nan, right=np.nan)


def common_xgrid(
    results: Mapping[str, pd.DataFrame],
    xmode: str = "phase",
    phase_column: str = "t_minus_tperi_gyr",
    ngrid: int = 400,
    overlap_only: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Common x grid for smooth comparison curves.
    """
    if xlim is not None:
        return np.linspace(xlim[0], xlim[1], ngrid)

    ranges = []
    for df in results.values():
        x, _ = get_x_axis(df, mode=xmode, phase_column=phase_column)
        x = x[np.isfinite(x)]
        if len(x) > 1:
            ranges.append((float(np.min(x)), float(np.max(x))))

    if not ranges:
        raise ValueError("Cannot build common grid; no valid x values.")

    if overlap_only:
        xmin = max(r[0] for r in ranges)
        xmax = min(r[1] for r in ranges)
    else:
        xmin = min(r[0] for r in ranges)
        xmax = max(r[1] for r in ranges)

    if xmax <= xmin:
        xmin = min(r[0] for r in ranges)
        xmax = max(r[1] for r in ranges)

    return np.linspace(xmin, xmax, ngrid)


# =============================================================================
# Event annotation
# =============================================================================

def add_pericentre_line(ax, xmode: str = "phase", alpha: float = 0.7) -> None:
    """
    Add a reference line at t - t_peri = 0 when using phase plots.
    """
    if xmode.lower() in ("phase", "tperi", "t_minus_tperi"):
        ax.axvline(0.0, ls="--", lw=1.1, color="0.35", alpha=alpha, zorder=0)


def add_event_lines_from_table(
    ax,
    event_table: Optional[pd.DataFrame],
    x_column: str = "t_event",
    kind_column: str = "kind",
) -> None:
    """
    Add pericentre/apocentre markers from an events table.

    Works with event tables from orbital_phase_tools.py.
    """
    if event_table is None or len(event_table) == 0:
        return

    for _, row in event_table.iterrows():
        if x_column not in row:
            continue
        x = row[x_column]
        if not np.isfinite(x):
            continue

        kind = str(row[kind_column]) if kind_column in row else ""
        if "peri" in kind:
            ax.axvline(x, ls="--", lw=0.9, color="0.25", alpha=0.55)
        elif "apo" in kind:
            ax.axvline(x, ls=":", lw=0.9, color="0.45", alpha=0.55)


# =============================================================================
# Core plotters
# =============================================================================

def plot_quantity(
    results: Mapping[str, pd.DataFrame],
    column: str,
    outdir: Union[str, Path],
    filename: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xmode: str = "time",
    phase_column: str = "t_minus_tperi_gyr",
    log: Optional[bool] = None,
    smooth: bool = False,
    show_points: bool = True,
    common_grid: bool = True,
    overlap_only: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend: bool = True,
    grid: bool = True,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
):
    """
    Compare one column across simulations.
    """
    spec = quantity_spec(column)
    col = spec.column
    log = spec.log if log is None else log
    ylabel = spec.ylabel if ylabel is None else ylabel
    title = spec.title if title is None else title
    filename = filename or spec.default_filename or col

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="white")

    xgrid = None
    if smooth and common_grid:
        try:
            xgrid = common_xgrid(
                results,
                xmode=xmode,
                phase_column=phase_column,
                overlap_only=overlap_only,
                xlim=xlim,
            )
        except Exception:
            xgrid = None

    for idx, (label, df) in enumerate(results.items()):
        if col not in df.columns:
            print(f"[plot_quantity] Column '{col}' not found for {label}.")
            continue

        st = style_for_label(label, idx)
        x, xlabel = get_x_axis(df, mode=xmode, phase_column=phase_column)
        y = np.asarray(df[col], dtype=float)
        if log:
            y = log10_safe(y)

        if smooth:
            xs, ys = smooth_interpolate(x, y, xgrid=xgrid)
            ax.plot(
                xs, ys,
                color=st.color,
                ls=st.linestyle,
                lw=st.linewidth,
                alpha=st.alpha,
                label=st.label or label_title(label),
            )
            if show_points:
                xp, yp = finite_xy(x, y)
                ax.plot(
                    xp, yp,
                    lw=0,
                    marker=st.marker,
                    ms=0.75 * st.markersize,
                    color=st.color,
                    alpha=0.35,
                )
        else:
            x, y = finite_xy(x, y)
            ax.plot(
                x, y,
                color=st.color,
                ls=st.linestyle,
                lw=st.linewidth,
                marker=st.marker if show_points else None,
                ms=st.markersize,
                alpha=st.alpha,
                label=st.label or label_title(label),
            )

    add_pericentre_line(ax, xmode=xmode)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if grid:
        ax.grid(alpha=0.3)
    if legend:
        ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, outdir, filename, formats=formats, dpi=dpi, close=True)
    return fig, ax


def plot_quantities_grid(
    results: Mapping[str, pd.DataFrame],
    quantities: Sequence[str],
    outdir: Union[str, Path],
    filename: str,
    ncols: int = 2,
    xmode: str = "time",
    phase_column: str = "t_minus_tperi_gyr",
    smooth: bool = False,
    show_points: bool = True,
    common_grid: bool = True,
    overlap_only: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
):
    """
    Multi-panel comparison plot.
    """
    n = len(quantities)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.7 * ncols, 4.2 * nrows),
        facecolor="white",
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, q in zip(axes_flat, quantities):
        spec = quantity_spec(q)
        col = spec.column

        xgrid = None
        if smooth and common_grid:
            try:
                xgrid = common_xgrid(
                    results,
                    xmode=xmode,
                    phase_column=phase_column,
                    overlap_only=overlap_only,
                    xlim=xlim,
                )
            except Exception:
                xgrid = None

        for idx, (label, df) in enumerate(results.items()):
            if col not in df.columns:
                continue

            st = style_for_label(label, idx)
            x, xlabel = get_x_axis(df, mode=xmode, phase_column=phase_column)
            y = np.asarray(df[col], dtype=float)
            if spec.log:
                y = log10_safe(y)

            if smooth:
                xs, ys = smooth_interpolate(x, y, xgrid=xgrid)
                ax.plot(xs, ys, color=st.color, ls=st.linestyle, lw=st.linewidth, label=st.label or label)
                if show_points:
                    xp, yp = finite_xy(x, y)
                    ax.plot(xp, yp, lw=0, marker=st.marker, ms=3.0, color=st.color, alpha=0.35)
            else:
                x, y = finite_xy(x, y)
                ax.plot(x, y, color=st.color, ls=st.linestyle, lw=st.linewidth,
                        marker=st.marker if show_points else None, ms=st.markersize,
                        label=st.label or label)

        add_pericentre_line(ax, xmode=xmode)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(spec.ylabel)
        ax.set_title(spec.title)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    # Put one legend in the first panel.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save_figure(fig, outdir, filename, formats=formats, dpi=dpi, close=True)
    return fig, axes


# =============================================================================
# Specialized science plots
# =============================================================================

def plot_orbit_panel(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "orbit_diagnostics",
    xmode: str = "time",
    smooth: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
):
    return plot_quantities_grid(
        results,
        quantities=["R_host", "V_rad", "V_tan", "r_tidal"],
        outdir=outdir,
        filename=filename,
        ncols=2,
        xmode=xmode,
        smooth=smooth,
        xlim=xlim,
    )


def _aperture_suffix(aperture: str) -> str:
    aperture = str(aperture).lower().strip()
    aliases = {
        "rt": "inside_rt",
        "tidal": "inside_rt",
        "rhalf": "inside_rhalf",
        "rhalf_star": "inside_rhalf",
        "1": "inside_1kpc",
        "1kpc": "inside_1kpc",
        "3": "inside_3kpc",
        "3kpc": "inside_3kpc",
        "tracked": "tracked",
        "total": "tracked",
    }
    return aliases.get(aperture, aperture)


def _mass_column(component: str, aperture: str) -> str:
    comp_map = {
        "stars": "Mstar",
        "star": "Mstar",
        "gas": "Mgas",
        "dm": "Mdm",
        "dark": "Mdm",
        "total": "Msat",
        "sat": "Msat",
    }
    comp = comp_map.get(component.lower(), component)
    suffix = _aperture_suffix(aperture)

    if suffix == "tracked":
        return f"{comp}_tracked_msun"
    return f"{comp}_{suffix}_msun"


def plot_mass_budget(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    aperture: str = "rt",
    filename: Optional[str] = None,
    xmode: str = "time",
    log: bool = True,
    smooth: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    components: Sequence[str] = ("stars", "gas", "dm", "total"),
):
    """
    Plot mass components for each simulation in separate panels.

    aperture:
        "rt", "rhalf", "1kpc", "3kpc", or "tracked"
    """
    n = len(results)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(7.5, max(4.0, 3.2 * n)),
        sharex=True,
        facecolor="white",
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (sim_idx, (label, df)) in zip(axes, enumerate(results.items())):
        x, xlabel = get_x_axis(df, mode=xmode)

        for comp in components:
            col = _mass_column(comp, aperture)
            if col not in df.columns:
                continue

            st = component_style(comp)
            y = np.asarray(df[col], dtype=float)
            if log:
                y = log10_safe(y)

            if smooth:
                xs, ys = smooth_interpolate(x, y)
                ax.plot(xs, ys, color=st.color, ls=st.linestyle, lw=st.linewidth, label=st.label or comp)
            else:
                xp, yp = finite_xy(x, y)
                ax.plot(xp, yp, color=st.color, ls=st.linestyle, marker=st.marker,
                        lw=st.linewidth, ms=st.markersize, label=st.label or comp)

        add_pericentre_line(ax, xmode=xmode)
        ax.set_title(label_title(label))
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncol=4, fontsize=9)

        if log:
            ax.set_ylabel(r"$\log(M/M_\odot)$")
        else:
            ax.set_ylabel(r"$M$ [$M_\odot$]")

    axes[-1].set_xlabel(xlabel)
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)

    fig.tight_layout()

    if filename is None:
        filename = f"mass_budget_{_aperture_suffix(aperture)}_{xmode}"

    save_figure(fig, outdir, filename, close=True)
    return fig, axes


def plot_retained_fraction(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "retained_fraction_inside_rt",
    xmode: str = "time",
    components: Sequence[str] = ("stars", "gas", "dm"),
    xlim: Optional[Tuple[float, float]] = None,
):
    """
    Plot M(<rt) / M_tracked for stars, gas, and DM.
    """
    fig, axes = plt.subplots(1, len(components), figsize=(4.8 * len(components), 4.2),
                             sharey=True, facecolor="white", squeeze=False)
    axes = axes.ravel()

    for ax, comp in zip(axes, components):
        inside_col = _mass_column(comp, "rt")
        total_col = _mass_column(comp, "tracked")
        st_comp = component_style(comp)

        for idx, (label, df) in enumerate(results.items()):
            if inside_col not in df.columns or total_col not in df.columns:
                continue

            st = style_for_label(label, idx)
            x, xlabel = get_x_axis(df, mode=xmode)
            frac = ratio_safe(df[inside_col], df[total_col])
            x, frac = finite_xy(x, frac)

            ax.plot(x, frac, color=st.color, ls=st.linestyle, marker=st.marker,
                    lw=st.linewidth, ms=st.markersize, label=st.label or label_title(label))

        add_pericentre_line(ax, xmode=xmode)
        ax.set_title(st_comp.label or comp)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"$M(<r_t)/M_{\rm tracked}$")
    axes[0].legend(frameon=False, fontsize=9)

    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)

    fig.tight_layout()
    save_figure(fig, outdir, filename, close=True)
    return fig, axes


def plot_star_formation_panel(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "star_formation_panel",
    xmode: str = "phase",
    smooth: bool = True,
    xlim: Optional[Tuple[float, float]] = (-0.4, 0.6),
):
    """
    Standard panel for SFR, star-forming gas, and young stars.

    Columns are only plotted if available.
    """
    quantities = [
        "SFR_tracked",
        "SFR_rt",
        "Mgas_SF_tracked",
        "Mgas_SF_1kpc",
        "Mstar_young_tracked",
        "Mstar_young_1kpc",
    ]
    return plot_quantities_grid(
        results,
        quantities=quantities,
        outdir=outdir,
        filename=filename,
        ncols=2,
        xmode=xmode,
        smooth=smooth,
        xlim=xlim,
    )


def plot_ram_pressure_panel(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "ram_pressure_panel",
    xmode: str = "phase",
    smooth: bool = True,
    xlim: Optional[Tuple[float, float]] = (-0.4, 0.6),
):
    quantities = ["P_ram", "rho_cgm", "R_host", "V_3d"]
    return plot_quantities_grid(
        results,
        quantities=quantities,
        outdir=outdir,
        filename=filename,
        ncols=2,
        xmode=xmode,
        smooth=smooth,
        xlim=xlim,
    )


def plot_size_mass_panel(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "size_mass_panel",
    xmode: str = "phase",
    smooth: bool = True,
    xlim: Optional[Tuple[float, float]] = (-0.4, 0.6),
):
    quantities = [
        "R_host",
        "r_tidal",
        "rhalf_star",
        "Msat_rt",
        "Mstar_rt",
        "Mgas_rt",
        "Mdm_rt",
        "Mstar_rhalf",
    ]
    return plot_quantities_grid(
        results,
        quantities=quantities,
        outdir=outdir,
        filename=filename,
        ncols=2,
        xmode=xmode,
        smooth=smooth,
        xlim=xlim,
    )


def plot_orbit_xy(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "orbit_xy",
    xcol: str = "x_sat_kpc",
    ycol: str = "y_sat_kpc",
    mark_start_end: bool = True,
):
    """
    Host-centered 2D orbit projection.
    """
    fig, ax = plt.subplots(figsize=(6.2, 6.0), facecolor="white")

    for idx, (label, df) in enumerate(results.items()):
        if xcol not in df.columns or ycol not in df.columns:
            continue

        st = style_for_label(label, idx)
        x = np.asarray(df[xcol], dtype=float)
        y = np.asarray(df[ycol], dtype=float)
        x, y = finite_xy(x, y)

        ax.plot(x, y, color=st.color, ls=st.linestyle, lw=st.linewidth,
                label=st.label or label_title(label))

        if mark_start_end and len(x) > 0:
            ax.scatter(x[0], y[0], color=st.color, marker="o", s=50, edgecolor="black", zorder=3)
            ax.scatter(x[-1], y[-1], color=st.color, marker="X", s=60, edgecolor="black", zorder=3)

    ax.scatter(0, 0, marker="+", s=120, color="black", label="host")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x_{\rm sat}$ [kpc]")
    ax.set_ylabel(r"$y_{\rm sat}$ [kpc]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, outdir, filename, close=True)
    return fig, ax


def plot_correlation(
    results: Mapping[str, pd.DataFrame],
    x_quantity: str,
    y_quantity: str,
    outdir: Union[str, Path],
    filename: Optional[str] = None,
    color_by: Optional[str] = "time_gyr",
    logx: Optional[bool] = None,
    logy: Optional[bool] = None,
    cmap: str = "viridis",
):
    """
    Co-evolution/correlation plot, analogous to the TNG co-evolution figures.
    """
    xspec = quantity_spec(x_quantity)
    yspec = quantity_spec(y_quantity)

    logx = xspec.log if logx is None else logx
    logy = yspec.log if logy is None else logy

    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="white")

    last_sc = None

    for idx, (label, df) in enumerate(results.items()):
        if xspec.column not in df.columns or yspec.column not in df.columns:
            continue

        st = style_for_label(label, idx)
        x = np.asarray(df[xspec.column], dtype=float)
        y = np.asarray(df[yspec.column], dtype=float)
        if logx:
            x = log10_safe(x)
        if logy:
            y = log10_safe(y)

        mask = np.isfinite(x) & np.isfinite(y)

        if color_by is not None and color_by in df.columns:
            c = np.asarray(df[color_by], dtype=float)
            mask = mask & np.isfinite(c)
            last_sc = ax.scatter(
                x[mask], y[mask],
                c=c[mask],
                cmap=cmap,
                marker=st.marker,
                s=45,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                label=st.label or label_title(label),
            )
        else:
            ax.plot(
                x[mask], y[mask],
                color=st.color,
                marker=st.marker,
                ls=st.linestyle,
                lw=st.linewidth,
                label=st.label or label_title(label),
            )

    ax.set_xlabel(xspec.ylabel if logx else xspec.column)
    ax.set_ylabel(yspec.ylabel if logy else yspec.column)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=ax)
        cbar.set_label(color_by)

    fig.tight_layout()

    if filename is None:
        filename = f"correlation_{xspec.column}_vs_{yspec.column}"

    save_figure(fig, outdir, filename, close=True)
    return fig, ax


# =============================================================================
# Standard suite
# =============================================================================

def plot_standard_gadget_suite(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    phase_xlim: Tuple[float, float] = (-0.4, 0.6),
    make_phase_plots: bool = True,
    make_time_plots: bool = True,
) -> Dict[str, List[Path]]:
    """
    Generate a standard set of diagnostic figures for the dwarf orbit project.
    """
    outdir = ensure_dir(outdir)

    paths_before = set(outdir.rglob("*"))

    if make_time_plots:
        plot_orbit_panel(results, outdir=outdir, filename="time_orbit_diagnostics", xmode="time")
        plot_size_mass_panel(results, outdir=outdir, filename="time_size_mass_panel", xmode="time", smooth=False, xlim=None)
        plot_mass_budget(results, outdir=outdir, aperture="rt", filename="time_mass_budget_rt", xmode="time")
        plot_retained_fraction(results, outdir=outdir, filename="time_retained_fraction_rt", xmode="time")

    if make_phase_plots:
        plot_orbit_panel(results, outdir=outdir, filename="phase_orbit_diagnostics", xmode="phase",
                         smooth=True, xlim=phase_xlim)
        plot_size_mass_panel(results, outdir=outdir, filename="phase_size_mass_panel", xmode="phase",
                             smooth=True, xlim=phase_xlim)
        plot_mass_budget(results, outdir=outdir, aperture="rt", filename="phase_mass_budget_rt",
                         xmode="phase", smooth=True, xlim=phase_xlim)
        plot_retained_fraction(results, outdir=outdir, filename="phase_retained_fraction_rt",
                               xmode="phase", xlim=phase_xlim)
        plot_ram_pressure_panel(results, outdir=outdir, xmode="phase", smooth=True, xlim=phase_xlim)
        plot_star_formation_panel(results, outdir=outdir, xmode="phase", smooth=True, xlim=phase_xlim)

    plot_orbit_xy(results, outdir=outdir, filename="orbit_xy")

    paths_after = set(outdir.rglob("*"))
    new_paths = sorted(paths_after - paths_before)

    return {"created": new_paths}


# =============================================================================
# Legend helpers
# =============================================================================

def legend_handles_for_labels(labels: Sequence[str]) -> Tuple[List[Line2D], List[str]]:
    handles = []
    texts = []
    for idx, label in enumerate(labels):
        st = style_for_label(label, idx)
        handles.append(
            Line2D(
                [0], [0],
                color=st.color,
                ls=st.linestyle,
                lw=st.linewidth,
                marker=st.marker,
                markersize=st.markersize,
            )
        )
        texts.append(st.label or label_title(label))
    return handles, texts


def legend_handles_for_components(components: Sequence[str]) -> Tuple[List[Line2D], List[str]]:
    handles = []
    texts = []
    for comp in components:
        st = component_style(comp)
        handles.append(
            Line2D(
                [0], [0],
                color=st.color,
                ls=st.linestyle,
                lw=st.linewidth,
                marker=st.marker,
                markersize=st.markersize,
            )
        )
        texts.append(st.label or comp)
    return handles, texts


# =============================================================================
# Convenience aliases close to your current notebook functions
# =============================================================================

def plot_compare_log_quantity(
    results: Mapping[str, pd.DataFrame],
    column: str,
    ylabel: Optional[str],
    filename: str,
    title: Optional[str] = None,
    outdir: Union[str, Path] = ".",
    xmode: str = "time",
    smooth: bool = False,
):
    """
    Backward-compatible replacement for the notebook helper you were using.
    """
    return plot_quantity(
        results=results,
        column=column,
        ylabel=ylabel,
        filename=filename,
        title=title,
        outdir=outdir,
        xmode=xmode,
        log=True,
        smooth=smooth,
    )


def plot_main_comparison_panel(
    results: Mapping[str, pd.DataFrame],
    outdir: Union[str, Path],
    filename: str = "main_comparison_panel",
    xmode: str = "time",
    smooth: bool = False,
):
    """
    Replacement for your manual 4x2 comparison panel.
    """
    quantities = [
        "R_host",
        "r_tidal",
        "rhalf_star",
        "Msat_rt",
        "Mstar_rt",
        "Mgas_rt",
        "Mdm_rt",
        "Mstar_rhalf",
    ]

    return plot_quantities_grid(
        results=results,
        quantities=quantities,
        outdir=outdir,
        filename=filename,
        ncols=2,
        xmode=xmode,
        smooth=smooth,
    )


if __name__ == "__main__":
    print(
        "plot_functions_gadget.py is a library module. "
        "Import it from your analysis notebook or script."
    )

