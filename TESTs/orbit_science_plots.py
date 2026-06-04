"""orbit_science_plots.py

Scientific plotting and post-processing tools for satellite-orbit analysis.

This file consolidates the useful plotting functionality from:
    - orbit_comparison_tools_spyder.py
    - orbit_comparison_tools_spyder_v2.py
    - orbit_plots_full_spyder.py

It does not reread HDF5 snapshots.  It works with:
    1. `results` already in memory after running `analyze_all(cfg)`, or
    2. saved CSV files written by `orbit_analysis_tools.py`.

Main capabilities
-----------------
1. Load saved time-series CSVs:
       load_results_from_output(output_dir)

2. Add derived plotting columns:
       time_since_first_pericentre_gyr
       snapshot_since_first_pericentre
       tidal_field_kms2_kpc2
       rhalf_over_rt
       standardized SFR/sSFR/SFE/tdep columns

3. Produce comparison plots across labels:
       orbit in xy
       orbital radius and velocity panels
       stellar/gas/DM mass panels
       size and stripping panels
       SFR/sSFR/SFE/depletion-time panels
       ram-pressure and tidal-forcing panels

4. Save combined derived tables and orbital-extrema tables.

Recommended use
---------------
    from orbit_science_plots import *

    comparison_outputs = run_standard_and_pericentre_normalized_plots(
        results,
        cfg=cfg,
        annotate_extrema=True,
    )

or, without rerunning HDF5 analysis:

    results = load_results_from_output("orbit_full_analysis_outputs")
    comparison_outputs = run_all_comparison_plots(
        results,
        cfg=None,
        output_dir="orbit_full_analysis_outputs/comparison_plots",
        x_axis_mode="time_since_first_pericentre",
    )"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G_KPC_KMS2_MSUN = 4.30091e-6

from dataclasses import dataclass, field


ArrayLike3 = Union[Sequence[float], np.ndarray]
FieldLike = Union[str, Tuple[str, str]]

KPC_IN_CM = 3.0856775814913673e21
MSUN_IN_G = 1.98847e33


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MapPlotConfig:
    """Configuration for yt map plots.

    The defaults are conservative and should work for quick inspection.  For
    publication figures you will probably adjust ``width_kpc``, ``axis``, and
    ``fields`` case by case.
    """

    output_dir: str = "orbit_map_outputs"
    fields: Tuple[FieldLike, ...] = ("gas_density", "gas_pressure", "gas_sfr")
    axis: str = "z"
    width_kpc: float = 80.0
    depth_kpc: Optional[float] = None
    center_on: str = "satellite"  # "satellite" or "host"

    # If True, save a yt-rendered plot.  The functions return the saved paths.
    save: bool = True

    # Optional yt loading kwargs.  Use this for non-standard Gadget frontends.
    yt_load_kwargs: Dict[str, object] = field(default_factory=dict)

    # Optional zlim by alias.  Values are yt/plot units after yt conversion.
    # Example: {"gas_density": (1e-30, 1e-24)}
    zlim: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Optional unit labels for plotted fields.
    # Example: {"gas_density": "Msun/kpc**2"}
    units: Dict[str, str] = field(default_factory=dict)

    # Annotations.
    annotate_center: bool = True
    annotate_host: bool = True
    annotate_orbit: bool = True
    annotate_tidal_radius: bool = True
    annotate_rhalf: bool = False

    # Orbit annotation details.
    orbit_use_full_track: bool = True
    orbit_n_segments_max: int = 300


def configure_matplotlib_for_paper(
    base_fontsize: int = 12,
    use_latex: bool = False,
) -> None:
    """
    Apply a clean, publication-oriented Matplotlib configuration.

    This does not force any specific colour palette, so the default Matplotlib
    colour cycle remains available.  Set `use_latex=True` only if your local
    Python/Matplotlib installation has a working LaTeX setup.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 250,
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.constrained_layout.use": False,
        "text.usetex": bool(use_latex),
    })


# =============================================================================
# Generic helpers
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_snapshot_number(path: Union[str, Path]) -> int:
    stem = Path(path).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    return int(digits[-1]) if digits else -1


def find_snapshots_for_label(cfg, label: str) -> List[Path]:
    """Return sorted snapshot files for ``root / label / output``."""
    root = Path(cfg.root)
    glob = getattr(cfg, "snapshot_glob", "snapshot_*.hdf5")
    out = root / label / "output"
    snaps = sorted(out.glob(glob), key=natural_snapshot_number)
    if not snaps:
        raise FileNotFoundError(f"No snapshots found for label={label!r} in {out}")
    return snaps


def _row_value(row: Union[pd.Series, Mapping[str, object]], key: str, default=np.nan):
    try:
        if key in row:
            return row[key]
    except Exception:
        pass
    return default


def get_satellite_center_from_row(row: Union[pd.Series, Mapping[str, object]]) -> np.ndarray:
    """Extract satellite centre from a time-series row."""
    for cols in [
        ("x_sat_kpc", "y_sat_kpc", "z_sat_kpc"),
        ("center_x_kpc", "center_y_kpc", "center_z_kpc"),
        ("x_kpc", "y_kpc", "z_kpc"),
    ]:
        if all(col in row for col in cols):
            return np.asarray([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
    raise KeyError(
        "Could not find satellite centre columns. Expected one of: "
        "(x_sat_kpc, y_sat_kpc, z_sat_kpc), "
        "(center_x_kpc, center_y_kpc, center_z_kpc), or (x_kpc, y_kpc, z_kpc)."
    )


def get_host_center_from_cfg(cfg) -> np.ndarray:
    return np.asarray(getattr(cfg.host, "host_center_kpc", (0.0, 0.0, 0.0)), dtype=float)


def get_snapshot_row(
    results: Dict[str, pd.DataFrame],
    label: str,
    snapshot_index: Optional[int] = None,
    snapshot_number: Optional[int] = None,
    time_gyr: Optional[float] = None,
) -> pd.Series:
    """Select one row from ``results[label]``.

    Priority:
    1. exact snapshot_number;
    2. exact snapshot_index;
    3. nearest time_gyr;
    4. first row.
    """
    if label not in results:
        raise KeyError(f"label={label!r} is not present in results")

    df = results[label]
    if len(df) == 0:
        raise ValueError(f"results[{label!r}] is empty")

    if snapshot_number is not None and "snapshot_number" in df.columns:
        sub = df[df["snapshot_number"].astype(int) == int(snapshot_number)]
        if len(sub):
            return sub.iloc[0]
        raise ValueError(f"snapshot_number={snapshot_number} not found for label={label!r}")

    if snapshot_index is not None and "snapshot_index" in df.columns:
        sub = df[df["snapshot_index"].astype(int) == int(snapshot_index)]
        if len(sub):
            return sub.iloc[0]
        raise ValueError(f"snapshot_index={snapshot_index} not found for label={label!r}")

    if time_gyr is not None and "time_gyr" in df.columns:
        t = np.asarray(df["time_gyr"], dtype=float)
        if np.any(np.isfinite(t)):
            idx = int(np.nanargmin(np.abs(t - float(time_gyr))))
            return df.iloc[idx]

    return df.iloc[0]


def get_snapshot_file_from_row_or_cfg(row: pd.Series, cfg, label: str) -> Path:
    """Find the HDF5 snapshot path associated with a selected row."""
    path = _row_value(row, "snapshot_file", None)
    if path is not None and isinstance(path, str) and len(path) > 0:
        p = Path(path)
        if p.exists():
            return p

    snaps = find_snapshots_for_label(cfg, label)

    if "snapshot_number" in row:
        wanted = int(row["snapshot_number"])
        for p in snaps:
            if natural_snapshot_number(p) == wanted:
                return p

    if "snapshot_index" in row:
        idx = int(row["snapshot_index"])
        if 0 <= idx < len(snaps):
            return snaps[idx]

    return snaps[0]


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
# yt loading and field resolution
# =============================================================================

def import_yt():
    """Import yt lazily, with a helpful error message if missing."""
    try:
        import yt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The map-plotting tools require yt. Install it in the environment "
            "used by Spyder, for example: conda install -c conda-forge yt"
        ) from exc
    return yt


def yt_unit_base_from_cfg(cfg) -> Dict[str, float]:
    """Build a Gadget-compatible yt unit_base dictionary from cfg."""
    return {
        "UnitLength_in_cm": float(getattr(cfg, "length_unit_to_kpc", 1.0)) * KPC_IN_CM,
        "UnitMass_in_g": float(getattr(cfg, "mass_unit_to_msun", 1.0)) * MSUN_IN_G,
        "UnitVelocity_in_cm_per_s": float(getattr(cfg, "velocity_unit_to_kms", 1.0)) * 1.0e5,
    }


def yt_load_snapshot(snapshot_file: Union[str, Path], cfg=None, yt_load_kwargs: Optional[Dict[str, object]] = None):
    """Load a Gadget/Gadget-4 HDF5 snapshot with yt.

    The function first tries with ``unit_base`` inferred from cfg.  If yt rejects
    the keyword for your frontend, it retries with the user-provided kwargs only.
    """
    yt = import_yt()
    snapshot_file = Path(snapshot_file)
    kwargs = dict(yt_load_kwargs or {})

    if cfg is not None and "unit_base" not in kwargs:
        kwargs["unit_base"] = yt_unit_base_from_cfg(cfg)

    try:
        return yt.load(str(snapshot_file), **kwargs)
    except TypeError:
        kwargs.pop("unit_base", None)
        return yt.load(str(snapshot_file), **kwargs)


FIELD_ALIASES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # Projected gas mass is best represented through gas density/column density.
    "gas_density": (
        ("gas", "density"),
        ("gas", "mass_density"),
        ("PartType0", "Density"),
        ("PartType0", "Densities"),
        ("PartType0", "density"),
        ("PartType0", "rho"),
        ("deposit", "PartType0_density"),
        ("deposit", "gas_density"),
    ),
    "gas_mass": (
        ("gas", "density"),
        ("PartType0", "Density"),
        ("deposit", "PartType0_density"),
    ),
    "gas_pressure": (
        ("gas", "pressure"),
        ("gas", "Pressure"),
        ("PartType0", "Pressure"),
        ("PartType0", "pressure"),
    ),
    "gas_temperature": (
        ("gas", "temperature"),
        ("PartType0", "Temperature"),
        ("PartType0", "temperature"),
    ),
    "gas_sfr": (
        ("gas", "star_formation_rate"),
        ("gas", "sfr"),
        ("PartType0", "StarFormationRate"),
        ("PartType0", "StarFormationRates"),
        ("PartType0", "SFR"),
        ("PartType0", "sfr"),
    ),
    "star_density": (
        ("deposit", "PartType4_density"),
        ("deposit", "stars_density"),
        ("stars", "density"),
        ("PartType4", "Masses"),
    ),
    "dm_density": (
        ("deposit", "PartType1_density"),
        ("deposit", "dark_matter_density"),
        ("PartType1", "Masses"),
    ),
}


def normalize_field_name(field: FieldLike) -> str:
    if isinstance(field, tuple):
        return "__".join(field)
    return str(field)


def resolve_yt_field(ds, field: FieldLike) -> Tuple[Tuple[str, str], str]:
    """Resolve a user field alias or explicit yt field tuple.

    Returns
    -------
    field_tuple, plot_kind
        plot_kind is either ``"projection"`` or ``"particle"``.
    """
    if isinstance(field, tuple):
        if field in ds.field_list or field in getattr(ds, "derived_field_list", []):
            kind = "particle" if field[0].startswith("PartType") else "projection"
            return field, kind
        raise KeyError(f"Field {field!r} not found in this yt dataset")

    key = str(field)
    candidates = FIELD_ALIASES.get(key, (("gas", key), ("PartType0", key)))

    all_fields = set(ds.field_list) | set(getattr(ds, "derived_field_list", []))
    for cand in candidates:
        if cand in all_fields:
            # Raw Gadget particle SFR is usually better shown as a ParticlePlot.
            if cand[0].startswith("PartType") and key in ["gas_sfr", "star_sfr"]:
                return cand, "particle"
            if cand[0].startswith("PartType") and cand[1].lower() in ["starformationrate", "starformationrates", "sfr"]:
                return cand, "particle"
            return cand, "projection"

    available = sorted([str(f) for f in all_fields])[:80]
    raise KeyError(
        f"Could not resolve field alias {field!r}. Use print_yt_fields_for_label(...) "
        f"to inspect available fields. First available fields include: {available}"
    )


def print_yt_fields(snapshot_file: Union[str, Path], cfg=None, max_lines: int = 250, yt_load_kwargs: Optional[Dict[str, object]] = None) -> None:
    """Print yt fields available in one snapshot."""
    ds = yt_load_snapshot(snapshot_file, cfg=cfg, yt_load_kwargs=yt_load_kwargs)
    print("\n=== yt field_list ===")
    for i, field in enumerate(ds.field_list[:max_lines]):
        print(f"{i:04d}: {field}")
    if len(ds.field_list) > max_lines:
        print(f"... truncated; total field_list length = {len(ds.field_list)}")

    derived = list(getattr(ds, "derived_field_list", []))
    print("\n=== yt derived_field_list ===")
    for i, field in enumerate(derived[:max_lines]):
        print(f"{i:04d}: {field}")
    if len(derived) > max_lines:
        print(f"... truncated; total derived_field_list length = {len(derived)}")


def print_yt_fields_for_label(
    cfg,
    label: str,
    snapshot_index: Optional[int] = 0,
    snapshot_number: Optional[int] = None,
    max_lines: int = 250,
    yt_load_kwargs: Optional[Dict[str, object]] = None,
) -> None:
    """Inspect yt fields for a snapshot selected by label/index/number."""
    snaps = find_snapshots_for_label(cfg, label)
    if snapshot_number is not None:
        selected = None
        for p in snaps:
            if natural_snapshot_number(p) == int(snapshot_number):
                selected = p
                break
        if selected is None:
            raise ValueError(f"snapshot_number={snapshot_number} not found for {label}")
    else:
        idx = 0 if snapshot_index is None else int(snapshot_index)
        selected = snaps[idx]

    print(f"Inspecting yt fields for: {selected}")
    print_yt_fields(selected, cfg=cfg, max_lines=max_lines, yt_load_kwargs=yt_load_kwargs)


# =============================================================================
# Geometry and annotations
# =============================================================================

def plane_coordinates_from_axis(axis: str) -> Tuple[int, int, int, str, str]:
    """Return x-index, y-index, line-of-sight-index and labels for a yt axis."""
    a = str(axis).lower()
    if a in ["z", "2"]:
        return 0, 1, 2, "x", "y"
    if a in ["y", "1"]:
        return 0, 2, 1, "x", "z"
    if a in ["x", "0"]:
        return 1, 2, 0, "y", "z"
    raise ValueError("axis must be one of 'x', 'y', 'z', 0, 1, or 2")


def get_orbit_points(results: Dict[str, pd.DataFrame], label: str) -> np.ndarray:
    """Return Nx3 satellite trajectory in kpc from a results DataFrame."""
    df = results[label]
    cols = ["x_sat_kpc", "y_sat_kpc", "z_sat_kpc"]
    if not all(c in df.columns for c in cols):
        return np.empty((0, 3), dtype=float)
    pts = df[cols].to_numpy(dtype=float)
    good = np.all(np.isfinite(pts), axis=1)
    return pts[good]


def _downsample_points(points: np.ndarray, nmax: int) -> np.ndarray:
    if len(points) <= nmax:
        return points
    idx = np.linspace(0, len(points) - 1, nmax).astype(int)
    return points[idx]


def annotate_orbit_on_yt_plot(plot, orbit_points_kpc: np.ndarray, axis: str, n_segments_max: int = 300) -> None:
    """Try to overlay the orbit path on a yt plot.

    This uses yt's ``annotate_line`` callback when available.  If the callback is
    unavailable for a frontend/plot type, the function silently skips the orbit
    overlay rather than failing the whole map.
    """
    if orbit_points_kpc is None or len(orbit_points_kpc) < 2:
        return

    pts = _downsample_points(np.asarray(orbit_points_kpc, dtype=float), n_segments_max + 1)
    try:
        for p0, p1 in zip(pts[:-1], pts[1:]):
            plot.annotate_line(
                p0,
                p1,
                coord_system="data",
                plot_args={"linewidth": 1.0, "alpha": 0.65},
            )
    except Exception:
        return


def annotate_basic_markers(
    plot,
    center_kpc: np.ndarray,
    host_center_kpc: Optional[np.ndarray] = None,
    r_tidal_kpc: Optional[float] = None,
    rhalf_kpc: Optional[float] = None,
    annotate_center: bool = True,
    annotate_host: bool = True,
    annotate_tidal_radius: bool = True,
    annotate_rhalf: bool = False,
) -> None:
    """Add common satellite/host/radius annotations to a yt plot."""
    if annotate_center:
        try:
            plot.annotate_marker(
                center_kpc,
                coord_system="data",
                plot_args={"marker": "x", "s": 80, "linewidths": 1.5},
            )
        except Exception:
            pass

    if annotate_host and host_center_kpc is not None:
        try:
            plot.annotate_marker(
                host_center_kpc,
                coord_system="data",
                plot_args={"marker": "+", "s": 90, "linewidths": 1.5},
            )
        except Exception:
            pass

    if annotate_tidal_radius and r_tidal_kpc is not None and np.isfinite(r_tidal_kpc) and r_tidal_kpc > 0:
        try:
            plot.annotate_sphere(center_kpc, radius=(float(r_tidal_kpc), "kpc"), circle_args={"linewidth": 1.0})
        except Exception:
            pass

    if annotate_rhalf and rhalf_kpc is not None and np.isfinite(rhalf_kpc) and rhalf_kpc > 0:
        try:
            plot.annotate_sphere(center_kpc, radius=(float(rhalf_kpc), "kpc"), circle_args={"linewidth": 1.0, "linestyle": "--"})
        except Exception:
            pass


# =============================================================================
# Main map plotting functions
# =============================================================================

def make_single_yt_map(
    snapshot_file: Union[str, Path],
    center_kpc: ArrayLike3,
    field: FieldLike = "gas_density",
    cfg=None,
    axis: str = "z",
    width_kpc: float = 80.0,
    depth_kpc: Optional[float] = None,
    output_dir: Union[str, Path] = "orbit_map_outputs",
    filename_prefix: str = "map",
    host_center_kpc: Optional[ArrayLike3] = None,
    r_tidal_kpc: Optional[float] = None,
    rhalf_kpc: Optional[float] = None,
    orbit_points_kpc: Optional[np.ndarray] = None,
    units: Optional[str] = None,
    zlim: Optional[Tuple[float, float]] = None,
    annotate_center: bool = True,
    annotate_host: bool = True,
    annotate_orbit: bool = True,
    annotate_tidal_radius: bool = True,
    annotate_rhalf: bool = False,
    yt_load_kwargs: Optional[Dict[str, object]] = None,
) -> Path:
    """Create one yt map for one snapshot and one field.

    The function uses ``ProjectionPlot`` for gas/grid-like fields and
    ``ParticlePlot`` for raw particle SFR fields.
    """
    yt = import_yt()
    outdir = ensure_dir(output_dir)
    snapshot_file = Path(snapshot_file)
    center = np.asarray(center_kpc, dtype=float)
    host_center = None if host_center_kpc is None else np.asarray(host_center_kpc, dtype=float)

    ds = yt_load_snapshot(snapshot_file, cfg=cfg, yt_load_kwargs=yt_load_kwargs)
    resolved_field, plot_kind = resolve_yt_field(ds, field)
    yt_center = ds.arr(center, "kpc")
    yt_width = (float(width_kpc), "kpc")

    if plot_kind == "particle":
        # ParticlePlot uses x/y particle positions.  Axis controls which plane is shown.
        ix, iy, _ilos, xname, yname = plane_coordinates_from_axis(axis)
        coord_fields = [
            ("all", f"particle_position_{xname}"),
            ("all", f"particle_position_{yname}"),
        ]
        # Some Gadget frontends keep particle coordinates under PartType0 only.
        # Try all-particle positions first, then PartType0 positions.
        try:
            plot = yt.ParticlePlot(ds, coord_fields[0], coord_fields[1], resolved_field, center=yt_center, width=yt_width)
        except Exception:
            coord_fields = [
                (resolved_field[0], f"particle_position_{xname}"),
                (resolved_field[0], f"particle_position_{yname}"),
            ]
            plot = yt.ParticlePlot(ds, coord_fields[0], coord_fields[1], resolved_field, center=yt_center, width=yt_width)
    else:
        data_source = None
        if depth_kpc is not None and np.isfinite(depth_kpc) and depth_kpc > 0:
            # Restrict projection to a box around the satellite.
            half = float(depth_kpc) / 2.0
            left = center.copy()
            right = center.copy()
            los = plane_coordinates_from_axis(axis)[2]
            left[:] = center - float(width_kpc) / 2.0
            right[:] = center + float(width_kpc) / 2.0
            left[los] = center[los] - half
            right[los] = center[los] + half
            data_source = ds.box(ds.arr(left, "kpc"), ds.arr(right, "kpc"))

        
         
        plot = yt.ProjectionPlot(
            ds,
            axis,
            resolved_field,
            center=yt_center,
            width=yt_width,
            data_source=data_source,
        )

    if units is not None:
        try:
            plot.set_unit(resolved_field, units)
        except Exception:
            pass

    if zlim is not None:
        try:
            plot.set_zlim(resolved_field, zlim[0], zlim[1])
        except Exception:
            pass

    # annotate_basic_markers(
    #     plot,
    #     center,
    #     host_center_kpc=host_center,
    #     r_tidal_kpc=r_tidal_kpc,
    #     rhalf_kpc=rhalf_kpc,
    #     annotate_center=annotate_center,
    #     annotate_host=annotate_host,
    #     annotate_tidal_radius=annotate_tidal_radius,
    #     annotate_rhalf=annotate_rhalf,
    # )

    # if annotate_orbit and orbit_points_kpc is not None:
    #     annotate_orbit_on_yt_plot(plot, orbit_points_kpc, axis=axis)

    snapnum = natural_snapshot_number(snapshot_file)
    field_name = normalize_field_name(field).replace(" ", "_").replace("/", "_")
    filename = outdir / f"{filename_prefix}_snap{snapnum:03d}_{field_name}_{axis}.png"

    plot.save(str(filename))
    # yt may append the field name to filename if it handles multiple fields.
    # Return the intended path if it exists; otherwise return the first matching file.
    if filename.exists():
        return filename
    matches = sorted(outdir.glob(f"{filename.stem}*.png"))
    if matches:
        return matches[0]
    return filename


def plot_map_suite(
    results: Dict[str, pd.DataFrame],
    cfg,
    label: str,
    snapshot_index: Optional[int] = None,
    snapshot_number: Optional[int] = None,
    time_gyr: Optional[float] = None,
    fields: Sequence[FieldLike] = ("gas_density", "gas_pressure", "gas_sfr"),
    width_kpc: float = 80.0,
    depth_kpc: Optional[float] = None,
    axis: str = "z",
    center_on: str = "satellite",
    output_dir: Optional[Union[str, Path]] = None,
    annotate_center: bool = True,
    annotate_host: bool = True,
    annotate_orbit: bool = True,
    annotate_tidal_radius: bool = True,
    annotate_rhalf: bool = False,
    yt_load_kwargs: Optional[Dict[str, object]] = None,
    zlim: Optional[Dict[str, Tuple[float, float]]] = None,
    units: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """Plot several yt maps for one simulation and one snapshot.

    Parameters
    ----------
    results, cfg
        Output from ``orbit_analysis_tools.analyze_all(cfg)`` and the same cfg.
    label
        Simulation label, e.g. ``"ORBIT01"``.
    snapshot_index, snapshot_number, time_gyr
        Select the snapshot.  ``snapshot_number`` has priority.
    fields
        Aliases or explicit yt field tuples.
    center_on
        ``"satellite"`` centers on the tracked satellite centre. ``"host"``
        centers on ``cfg.host.host_center_kpc``.
    """
    row = get_snapshot_row(
        results,
        label,
        snapshot_index=snapshot_index,
        snapshot_number=snapshot_number,
        time_gyr=time_gyr,
    )
    snapshot_file = get_snapshot_file_from_row_or_cfg(row, cfg, label)

    sat_center = get_satellite_center_from_row(row)
    host_center = get_host_center_from_cfg(cfg)
    map_center = host_center if str(center_on).lower() == "host" else sat_center

    if output_dir is None:
        output_dir = Path(getattr(cfg, "output_dir", ".")) / "map_plots" / label
    else:
        output_dir = Path(output_dir) / label
    ensure_dir(output_dir)

    r_tidal = float(row["r_tidal_kpc"]) if "r_tidal_kpc" in row and np.isfinite(row["r_tidal_kpc"]) else None
    rhalf = None
    for col in ["rhalf_star_member_kpc", "rhalf_star_all_kpc", "rhalf_star_kpc"]:
        if col in row and np.isfinite(row[col]):
            rhalf = float(row[col])
            break

    orbit_points = get_orbit_points(results, label) if annotate_orbit else None
    zlim = zlim or {}
    units = units or {}

    snapnum = int(row["snapshot_number"]) if "snapshot_number" in row else natural_snapshot_number(snapshot_file)
    prefix = f"{label}_map"

    outputs: Dict[str, Path] = {}
    for field in fields:
        field_key = normalize_field_name(field)
        try:
            outputs[field_key] = make_single_yt_map(
                snapshot_file=snapshot_file,
                center_kpc=map_center,
                field=field,
                cfg=cfg,
                axis=axis,
                width_kpc=width_kpc,
                depth_kpc=depth_kpc,
                output_dir=output_dir,
                filename_prefix=prefix,
                host_center_kpc=host_center,
                r_tidal_kpc=r_tidal if str(center_on).lower() == "satellite" else None,
                rhalf_kpc=rhalf if str(center_on).lower() == "satellite" else None,
                orbit_points_kpc=orbit_points,
                units=units.get(field_key) or units.get(str(field)),
                zlim=zlim.get(field_key) or zlim.get(str(field)),
                annotate_center=annotate_center,
                annotate_host=annotate_host,
                annotate_orbit=annotate_orbit,
                annotate_tidal_radius=annotate_tidal_radius,
                annotate_rhalf=annotate_rhalf,
                yt_load_kwargs=yt_load_kwargs,
            )
        except Exception as exc:
            print(f"[plot_map_suite] Skipping field={field!r} for {label} snapshot={snapnum}: {exc}")

    return outputs


# =============================================================================
# Event-based map suites
# =============================================================================

def compute_orbital_extrema_for_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Small local extrema finder, independent from orbit_science_plots.py."""
    if "R_host_kpc" not in df.columns or len(df) == 0:
        return pd.DataFrame()
    r = np.asarray(df["R_host_kpc"], dtype=float)
    rows = []
    for i in range(1, len(df) - 1):
        if not np.all(np.isfinite([r[i - 1], r[i], r[i + 1]])):
            continue
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            kind = "pericentre"
        elif r[i] > r[i - 1] and r[i] > r[i + 1]:
            kind = "apocentre"
        else:
            continue
        rows.append({
            "event_type": kind,
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else i,
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else i,
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(r[i]),
        })
    if not any(row["event_type"] == "pericentre" for row in rows) and np.any(np.isfinite(r)):
        i = int(np.nanargmin(r))
        rows.append({
            "event_type": "pericentre_candidate_global_minimum",
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else i,
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else i,
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(r[i]),
        })
    return pd.DataFrame(rows)


def get_first_pericentre_row(df: pd.DataFrame) -> pd.Series:
    events = compute_orbital_extrema_for_maps(df)
    if len(events) == 0:
        idx = int(np.nanargmin(np.asarray(df["R_host_kpc"], dtype=float))) if "R_host_kpc" in df.columns else 0
        return df.iloc[idx]
    peri = events[events["event_type"] == "pericentre"]
    if len(peri) == 0:
        peri = events[events["event_type"].astype(str).str.startswith("pericentre")]
    ev = peri.iloc[0] if len(peri) else events.iloc[0]
    if "snapshot_index" in df.columns:
        sub = df[df["snapshot_index"].astype(int) == int(ev["snapshot_index"])]
        if len(sub):
            return sub.iloc[0]
    if "snapshot_number" in df.columns:
        sub = df[df["snapshot_number"].astype(int) == int(ev["snapshot_number"])]
        if len(sub):
            return sub.iloc[0]
    return df.iloc[int(ev["snapshot_index"])]


def plot_first_pericentre_map_suite(
    results: Dict[str, pd.DataFrame],
    cfg,
    labels: Optional[Sequence[str]] = None,
    fields: Sequence[FieldLike] = ("gas_density", "gas_pressure", "gas_sfr"),
    width_kpc: float = 80.0,
    depth_kpc: Optional[float] = None,
    axis: str = "z",
    center_on: str = "satellite",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Dict[str, Path]]:
    """Create map suites at the first pericentre of each selected label."""
    selected = list(labels) if labels is not None else list(results.keys())
    all_outputs: Dict[str, Dict[str, Path]] = {}

    for label in selected:
        df = results[label]
        row = get_first_pericentre_row(df)
        snap_index = int(row["snapshot_index"]) if "snapshot_index" in row else None
        snap_number = int(row["snapshot_number"]) if "snapshot_number" in row else None
        print(f"[plot_first_pericentre_map_suite] {label}: snapshot_number={snap_number}, snapshot_index={snap_index}")
        all_outputs[label] = plot_map_suite(
            results,
            cfg,
            label=label,
            snapshot_index=snap_index,
            snapshot_number=snap_number,
            fields=fields,
            width_kpc=width_kpc,
            depth_kpc=depth_kpc,
            axis=axis,
            center_on=center_on,
            output_dir=output_dir,
            **kwargs,
        )

    return all_outputs


def plot_map_sequence(
    results: Dict[str, pd.DataFrame],
    cfg,
    label: str,
    snapshot_numbers: Optional[Sequence[int]] = None,
    snapshot_indices: Optional[Sequence[int]] = None,
    fields: Sequence[FieldLike] = ("gas_density",),
    width_kpc: float = 80.0,
    depth_kpc: Optional[float] = None,
    axis: str = "z",
    center_on: str = "satellite",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Dict[str, Path]]:
    """Create maps for a sequence of snapshots from one label."""
    if snapshot_numbers is None and snapshot_indices is None:
        raise ValueError("Provide snapshot_numbers or snapshot_indices")

    outputs: Dict[str, Dict[str, Path]] = {}

    if snapshot_numbers is not None:
        for snap in snapshot_numbers:
            outputs[f"snapshot_number_{snap}"] = plot_map_suite(
                results,
                cfg,
                label=label,
                snapshot_number=int(snap),
                fields=fields,
                width_kpc=width_kpc,
                depth_kpc=depth_kpc,
                axis=axis,
                center_on=center_on,
                output_dir=output_dir,
                **kwargs,
            )

    if snapshot_indices is not None:
        for idx in snapshot_indices:
            outputs[f"snapshot_index_{idx}"] = plot_map_suite(
                results,
                cfg,
                label=label,
                snapshot_index=int(idx),
                fields=fields,
                width_kpc=width_kpc,
                depth_kpc=depth_kpc,
                axis=axis,
                center_on=center_on,
                output_dir=output_dir,
                **kwargs,
            )

    return outputs

