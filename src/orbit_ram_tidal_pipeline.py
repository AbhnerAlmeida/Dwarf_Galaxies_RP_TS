#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_ram_tidal_pipeline.py

Notebook-oriented helper tools for controlled Gadget4 dwarf-galaxy simulations.

Expected project structure
--------------------------
Example:

    ./SIMULATIONS/ORBIT/HigherRes/E_mid_L_mid/output/snapshot_XXX.hdf5

This module does not replace dwarf_ram_analysis_ids.py.  It wraps it in a
friendlier workflow:

1. Discover one or more simulation outputs.
2. Run ram-pressure + tidal analysis tables.
3. Load the tables as RunData objects.
4. Make quick comparison plots in the notebook.
5. Make simple HDF5 particle maps for visualization.
6. Make an orbit GIF from the measured dwarf trajectory.

The scientific measurements should still come from dwarf_ram_analysis_ids.py
or dwarf_ram_analysis.py.  The visualization functions here are intentionally
lightweight and robust for interactive inspection.
"""

from __future__ import annotations

import os
import re
import glob
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import h5py
except Exception:
    h5py = None

KPC_CGS = 3.0856775814913673e21
SNAP_RE = re.compile(r"snapshot_(\d{3})\.hdf5$")


# =============================================================================
# Discovery and paths
# =============================================================================

def snapshot_number(path: str | Path) -> Optional[int]:
    m = SNAP_RE.match(Path(path).name)
    return int(m.group(1)) if m else None


def find_snapshots(output_dir: str | Path) -> List[Path]:
    output_dir = Path(output_dir)
    files = list(output_dir.glob("snapshot_*.hdf5"))

    def keyfunc(p: Path) -> int:
        n = snapshot_number(p)
        return n if n is not None else 10**9

    return sorted(files, key=keyfunc)


def discover_simulations(
    sim_root: str | Path = "./SIMULATIONS/ORBIT",
    suite: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    output_name: str = "output",
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Discover simulation directories.

    Examples
    --------
    discover_simulations("./SIMULATIONS/ORBIT", suite="HigherRes")
    discover_simulations("./SIMULATIONS/ORBIT/HigherRes", labels=["E_mid_L_mid"])

    Returns
    -------
    pandas.DataFrame with columns:
        label, suite, sim_dir, output_dir, n_snapshots, first_snapshot, last_snapshot
    """
    root = Path(sim_root)
    if suite is not None:
        root = root / suite

    pattern = f"**/{output_name}" if recursive else output_name
    output_dirs = sorted([p for p in root.glob(pattern) if p.is_dir()])

    rows: List[Dict[str, Any]] = []
    labels_set = None if labels is None else set(labels)

    for outdir in output_dirs:
        sim_dir = outdir.parent
        label = sim_dir.name
        if labels_set is not None and label not in labels_set:
            continue

        snaps = find_snapshots(outdir)
        if len(snaps) == 0:
            continue

        # The suite is the directory immediately above the label when possible.
        suite_name = sim_dir.parent.name if sim_dir.parent != root.parent else ""
        first = snaps[0].name
        last = snaps[-1].name
        rows.append({
            "label": label,
            "suite": suite_name,
            "sim_dir": str(sim_dir),
            "output_dir": str(outdir),
            "n_snapshots": len(snaps),
            "first_snapshot": first,
            "last_snapshot": last,
        })

    return pd.DataFrame(rows)


def analysis_dir_for_run(
    analysis_root: str | Path,
    label: str,
    suite: Optional[str] = None,
    keep_suite: bool = True,
) -> Path:
    root = Path(analysis_root)
    if keep_suite and suite:
        return root / str(suite) / str(label)
    return root / str(label)


# =============================================================================
# Running dwarf_ram_analysis_ids.py
# =============================================================================

def build_tracking_config(
    dra: Any,
    label: str,
    dwarf_init_center: Sequence[float],
    host_center: Optional[Sequence[float]],
    per_label_dwarf_centers: Optional[Dict[str, Sequence[float]]] = None,
    per_label_host_centers: Optional[Dict[str, Sequence[float]]] = None,
    **kwargs: Any,
) -> Any:
    """
    Build dra.TrackingConfig while allowing per-label centers.

    This is useful for cases like:
        ORBIT02 -> dwarf_init_center=[250,150,0]
        E_mid_L_mid -> custom center
    """
    dcenter = dwarf_init_center
    hcenter = host_center
    if per_label_dwarf_centers and label in per_label_dwarf_centers:
        dcenter = per_label_dwarf_centers[label]
    if per_label_host_centers and label in per_label_host_centers:
        hcenter = per_label_host_centers[label]

    return dra.TrackingConfig(
        dwarf_init_center=dcenter,
        host_center=hcenter,
        **kwargs,
    )


def run_analysis_table_for_simulations(
    dra: Any,
    sims: pd.DataFrame,
    analysis_root: str | Path = "./ANALYSIS/ORBIT",
    dwarf_init_center: Sequence[float] = (0.0, 0.0, 0.0),
    host_center: Optional[Sequence[float]] = (0.0, 0.0, 0.0),
    per_label_dwarf_centers: Optional[Dict[str, Sequence[float]]] = None,
    per_label_host_centers: Optional[Dict[str, Sequence[float]]] = None,
    table_name: str = "dwarf_ram_pressure_evolution_final.txt",
    force_reanalyze: bool = False,
    keep_suite: bool = True,
    config_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run dra.analyze_snapshot_directory for each discovered simulation.

    Returns a DataFrame with table paths and status.
    """
    config_kwargs = dict(config_kwargs or {})
    rows: List[Dict[str, Any]] = []

    for _, row in sims.iterrows():
        label = str(row["label"])
        suite = str(row.get("suite", ""))
        output_dir = Path(row["output_dir"])
        outdir = analysis_dir_for_run(analysis_root, label, suite=suite, keep_suite=keep_suite)
        outdir.mkdir(parents=True, exist_ok=True)
        table_path = outdir / table_name

        status = "existing"
        if force_reanalyze or not table_path.exists():
            status = "computed"
            if verbose:
                print(f"[RUN ] {label}: {output_dir}")
                print(f"      outdir: {outdir}")

            cfg = build_tracking_config(
                dra,
                label=label,
                dwarf_init_center=dwarf_init_center,
                host_center=host_center,
                per_label_dwarf_centers=per_label_dwarf_centers,
                per_label_host_centers=per_label_host_centers,
                **config_kwargs,
            )

            records, written = dra.analyze_snapshot_directory(
                str(output_dir),
                cfg,
                outdir=str(outdir),
                table_name=table_name,
                write_table=True,
                verbose=verbose,
            )
            table_path = Path(written) if written is not None else table_path
            n_records = len(records)
        else:
            if verbose:
                print(f"[SKIP] {label}: {table_path}")
            n_records = np.nan

        rows.append({
            "label": label,
            "suite": suite,
            "output_dir": str(output_dir),
            "analysis_dir": str(outdir),
            "table_path": str(table_path),
            "status": status,
            "n_records_computed": n_records,
        })

    return pd.DataFrame(rows)


def prepare_runs_from_analysis_table(
    dra: Any,
    analysis_table: pd.DataFrame,
    smooth: int = 5,
    pre_window: Tuple[float, float] = (-0.5, -0.1),
    force_window: Tuple[float, float] = (-0.3, 0.3),
    response_window: Tuple[float, float] = (-0.1, 1.0),
    entry_radius_kpc: float = 250.0,
    gas_loss_fraction: float = 0.05,
    gas_loss_mode: str = "total",
    min_gas_floor_abs: float = 1e5,
) -> List[Any]:
    table_paths = list(analysis_table["table_path"])
    labels = list(analysis_table["label"])
    snapshot_dirs = list(analysis_table["output_dir"])

    return dra.prepare_runs(
        table_paths,
        labels=labels,
        snapshot_dirs=snapshot_dirs,
        smooth=smooth,
        pre_window=pre_window,
        force_window=force_window,
        response_window=response_window,
        entry_radius_kpc=entry_radius_kpc,
        gas_loss_fraction=gas_loss_fraction,
        gas_loss_mode=gas_loss_mode,
        min_gas_floor_abs=min_gas_floor_abs,
    )


def metrics_dataframe(dra: Any, runs: Sequence[Any], outpath: Optional[str | Path] = None) -> pd.DataFrame:
    if hasattr(dra, "metrics_rows"):
        df = pd.DataFrame(dra.metrics_rows(runs))
    else:
        rows = []
        for run in runs:
            item = {"label": run.label}
            item.update(getattr(run, "epochs", {}))
            item.update(getattr(run, "metrics", {}))
            rows.append(item)
        df = pd.DataFrame(rows)

    if outpath is not None:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outpath, index=False)

    return df


# =============================================================================
# Plot helpers for RunData objects
# =============================================================================

def _col(run: Any, name: str, default: Optional[np.ndarray] = None) -> np.ndarray:
    if hasattr(run, "cols") and name in run.cols:
        return np.asarray(run.cols[name], dtype=float)
    if default is not None:
        return default
    return np.full_like(np.asarray(run.t, dtype=float), np.nan, dtype=float)


def _finite_xy(x: Sequence[float], y: Sequence[float], positive_y: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if positive_y:
        good &= y > 0
    return good


def running_nanmedian(y: Sequence[float], window: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    out = np.full_like(y, np.nan, dtype=float)
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        vals = y[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.nanmedian(vals)
    return out


def baseline_in_window(tau: Sequence[float], y: Sequence[float], tmin: float, tmax: float) -> float:
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = np.isfinite(tau) & np.isfinite(y) & (tau >= tmin) & (tau <= tmax)
    if not np.any(sel):
        return np.nan
    return float(np.nanmedian(y[sel]))


def normalize_to_window(tau: Sequence[float], y: Sequence[float], tmin: float, tmax: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    base = baseline_in_window(tau, y, tmin, tmax)
    if not np.isfinite(base) or base == 0:
        return np.full_like(y, np.nan, dtype=float)
    return y / base


def plot_runs_quantity(
    runs: Sequence[Any],
    col: str,
    ylabel: Optional[str] = None,
    logy: bool = False,
    normalized: bool = False,
    pre_window: Tuple[float, float] = (-0.5, -0.1),
    smooth: int = 1,
    xlim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    marker: Optional[str] = None,
    lw: float = 2.0,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    for run in runs:
        if not hasattr(run, "cols") or col not in run.cols:
            continue

        y = np.asarray(run.cols[col], dtype=float)
        if normalized:
            y = normalize_to_window(run.tau, y, pre_window[0], pre_window[1])
        if smooth and smooth > 1:
            y = running_nanmedian(y, smooth)

        good = _finite_xy(run.tau, y, positive_y=logy)
        if np.any(good):
            ax.plot(run.tau[good], y[good], lw=lw, marker=marker, ms=3, label=run.label)

    ax.axvline(0.0, ls="--", lw=1.1, color="0.35")
    ax.set_xlabel(r"$\tau = t - t_{\rm peri}$ [Gyr]")
    ax.set_ylabel(ylabel or col)
    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_forcing_response_panel(
    runs: Sequence[Any],
    xlim: Optional[Tuple[float, float]] = (-1.0, 1.5),
    smooth: int = 5,
    pre_window: Tuple[float, float] = (-0.5, -0.1),
) -> plt.Figure:
    pram_col = "pram_upstream_dyn_cm2"
    if len(runs) and hasattr(runs[0], "cols") and pram_col not in runs[0].cols:
        pram_col = "pram_dyn_cm2"

    specs = [
        ("dist_host_kpc", r"$d_{\rm host}$ [kpc]", False, False),
        (pram_col, r"$P_{\rm ram}$ [dyn cm$^{-2}$]", True, False),
        ("tidal_to_self_star", r"$a_{\rm tide}/a_{\rm self,\star}$", True, False),
        ("mgas_msun", r"$M_{\rm gas}/M_{\rm gas,pre}$", True, True),
        ("mgas_sf_msun", r"$M_{\rm gas,SF}/M_{\rm gas,SF,pre}$", True, True),
        ("rhalf_star_kpc", r"$R_{1/2,\star}/R_{\rm pre}$", False, True),
        ("fstar_dwarf_origin_stripped", r"$f_{\star,\rm stripped}$", False, False),
        ("fgas_dwarf_origin_stripped", r"$f_{\rm gas,dwarf,stripped}$", False, False),
    ]

    n = len(specs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows), sharex=True)
    axes = np.ravel(axes)

    for ax, (col, ylabel, logy, norm) in zip(axes, specs):
        plot_runs_quantity(
            runs, col, ylabel=ylabel, logy=logy, normalized=norm,
            pre_window=pre_window, smooth=smooth, xlim=xlim, ax=ax,
        )

    for ax in axes[len(specs):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


# =============================================================================
# Lightweight HDF5 snapshot reading for maps
# =============================================================================

def _require_h5py() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for lightweight HDF5 map generation.")


def read_header_units_kpc(f: Any) -> float:
    """
    Return conversion factor from code length to kpc.

    If the snapshot does not provide UnitLength_in_cm, assume coordinates are
    already in kpc-like units.
    """
    try:
        attrs = f["Header"].attrs
        unit_length_cm = attrs.get("UnitLength_in_cm", None)
        if unit_length_cm is not None and float(unit_length_cm) > 0:
            return float(unit_length_cm) / KPC_CGS
    except Exception:
        pass
    return 1.0


def read_parttype(
    snapshot_path: str | Path,
    ptype: str,
    fields: Sequence[str] = ("Coordinates", "Masses"),
    convert_coords_to_kpc: bool = True,
) -> Dict[str, Any]:
    """
    Read a Gadget HDF5 particle group using h5py.

    Returns an empty dict if the particle type or fields are absent.
    """
    _require_h5py()
    snapshot_path = Path(snapshot_path)
    out: Dict[str, Any] = {}
    with h5py.File(snapshot_path, "r") as f:
        if ptype not in f:
            return out
        g = f[ptype]
        conv = read_header_units_kpc(f) if convert_coords_to_kpc else 1.0
        for field in fields:
            if field in g:
                arr = np.asarray(g[field])
                if field == "Coordinates" and convert_coords_to_kpc:
                    arr = arr.astype(float) * conv
                out[field] = arr
    return out


def load_snapshot_particles_for_map(
    snapshot_path: str | Path,
    include_dm: bool = False,
    convert_coords_to_kpc: bool = True,
) -> Dict[str, Dict[str, Any]]:
    fields_gas = ["Coordinates", "Masses", "Density", "StarFormationRate", "ParticleIDs"]
    fields_star = ["Coordinates", "Masses", "ParticleIDs", "StellarFormationTime"]
    fields_dm = ["Coordinates", "Masses", "ParticleIDs"]

    data = {
        "gas": read_parttype(snapshot_path, "PartType0", fields_gas, convert_coords_to_kpc),
        "stars": read_parttype(snapshot_path, "PartType4", fields_star, convert_coords_to_kpc),
    }
    if include_dm:
        data["dm"] = read_parttype(snapshot_path, "PartType1", fields_dm, convert_coords_to_kpc)
    return data


def center_from_particles(
    pdata: Dict[str, Dict[str, Any]],
    mode: str = "stars",
    fallback: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if mode == "stars" and "Coordinates" in pdata.get("stars", {}):
        pos = np.asarray(pdata["stars"]["Coordinates"], dtype=float)
        if len(pos):
            return np.nanmedian(pos, axis=0)
    if mode == "gas" and "Coordinates" in pdata.get("gas", {}):
        pos = np.asarray(pdata["gas"]["Coordinates"], dtype=float)
        if len(pos):
            return np.nanmedian(pos, axis=0)
    if fallback is not None:
        return np.asarray(fallback, dtype=float)
    return np.zeros(3, dtype=float)


def project_positions(pos: np.ndarray, axis: str = "z") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=float)
    axis = axis.lower()
    if axis == "z":
        return pos[:, 0], pos[:, 1], pos[:, 2]
    if axis == "y":
        return pos[:, 0], pos[:, 2], pos[:, 1]
    if axis == "x":
        return pos[:, 1], pos[:, 2], pos[:, 0]
    raise ValueError("axis must be 'x', 'y', or 'z'")


def make_particle_map(
    snapshot_path: str | Path,
    outpath: str | Path,
    center: Optional[Sequence[float]] = None,
    center_mode: str = "stars",
    axis: str = "z",
    width_kpc: float = 80.0,
    bins: int = 256,
    include_stars: bool = True,
    include_gas: bool = True,
    sfr_only: bool = False,
    gas_weight: str = "Masses",
    title: Optional[str] = None,
    convert_coords_to_kpc: bool = True,
    cmap: str = "viridis",
) -> Optional[Path]:
    """
    Make a quick 2D histogram map from raw HDF5 particles.

    This is a visualization helper, not the preferred scientific measurement.
    """
    pdata = load_snapshot_particles_for_map(snapshot_path, convert_coords_to_kpc=convert_coords_to_kpc)
    if center is None:
        cen = center_from_particles(pdata, mode=center_mode)
    else:
        cen = np.asarray(center, dtype=float)

    half = 0.5 * float(width_kpc)
    extent = [-half, half, -half, half]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    has_any = False

    if include_gas and "Coordinates" in pdata.get("gas", {}):
        g = pdata["gas"]
        pos = np.asarray(g["Coordinates"], dtype=float) - cen[None, :]
        x, y, _ = project_positions(pos, axis=axis)
        sel = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= half) & (np.abs(y) <= half)

        if sfr_only and "StarFormationRate" in g:
            sfr = np.asarray(g["StarFormationRate"], dtype=float)
            sel &= sfr > 0

        if np.any(sel):
            if gas_weight in g:
                weights = np.asarray(g[gas_weight], dtype=float)
            elif gas_weight == "StarFormationRate" and "StarFormationRate" in g:
                weights = np.asarray(g["StarFormationRate"], dtype=float)
            else:
                weights = None

            H, xe, ye = np.histogram2d(
                x[sel], y[sel], bins=bins, range=[[-half, half], [-half, half]],
                weights=None if weights is None else weights[sel],
            )
            H = H.T
            H = np.where(H > 0, H, np.nan)
            im = ax.imshow(
                np.log10(H),
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
            )
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label(f"log weighted gas map: {gas_weight}")
            has_any = True

    if include_stars and "Coordinates" in pdata.get("stars", {}):
        s = pdata["stars"]
        spos = np.asarray(s["Coordinates"], dtype=float) - cen[None, :]
        sx, sy, _ = project_positions(spos, axis=axis)
        sel_s = np.isfinite(sx) & np.isfinite(sy) & (np.abs(sx) <= half) & (np.abs(sy) <= half)
        if np.any(sel_s):
            ax.scatter(sx[sel_s], sy[sel_s], s=2, alpha=0.65, c="white", linewidths=0)
            has_any = True

    ax.axhline(0, lw=0.5, color="0.8")
    ax.axvline(0, lw=0.5, color="0.8")
    ax.set_xlabel(f"{axis}-projection coordinate 1 [kpc]")
    ax.set_ylabel(f"{axis}-projection coordinate 2 [kpc]")
    ax.set_title(title or Path(snapshot_path).name)

    if not has_any:
        plt.close(fig)
        return None

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def make_map_series(
    output_dir: str | Path,
    outdir: str | Path,
    table_path: Optional[str | Path] = None,
    center_mode: str = "table",
    axis: str = "z",
    width_kpc: float = 80.0,
    bins: int = 256,
    max_snaps: Optional[int] = None,
    every: int = 1,
    sfr_only: bool = False,
    gas_weight: str = "Masses",
    convert_coords_to_kpc: bool = True,
) -> List[Path]:
    """
    Make a series of quick HDF5 particle maps.

    center_mode:
        "table" -> uses x_kpc,y_kpc,z_kpc from the analysis table
        "stars" -> recenters on median stellar position in each snapshot
        "origin" -> center=[0,0,0]
    """
    output_dir = Path(output_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snaps = find_snapshots(output_dir)
    snaps = snaps[::max(1, int(every))]
    if max_snaps is not None:
        snaps = snaps[:int(max_snaps)]

    table = None
    if table_path is not None and Path(table_path).exists():
        try:
            table = np.genfromtxt(table_path, names=True, dtype=None, encoding=None)
            if getattr(table, "shape", None) == ():
                table = np.array([table], dtype=table.dtype)
        except Exception:
            table = None

    centers_by_snapshot: Dict[str, np.ndarray] = {}
    if table is not None and table.dtype.names is not None:
        names = table.dtype.names
        if {"snapshot", "x_kpc", "y_kpc", "z_kpc"}.issubset(set(names)):
            for row in table:
                centers_by_snapshot[str(row["snapshot"])] = np.array(
                    [float(row["x_kpc"]), float(row["y_kpc"]), float(row["z_kpc"])],
                    dtype=float,
                )

    outputs: List[Path] = []
    for snap in snaps:
        center = None
        if center_mode == "table":
            center = centers_by_snapshot.get(snap.name, None)
        elif center_mode == "origin":
            center = np.zeros(3)

        out = outdir / f"map_{snap.stem}_{axis}.png"
        title = f"{snap.name} | {output_dir.parent.name}"
        made = make_particle_map(
            snap,
            out,
            center=center,
            center_mode="stars",
            axis=axis,
            width_kpc=width_kpc,
            bins=bins,
            sfr_only=sfr_only,
            gas_weight=gas_weight,
            title=title,
            convert_coords_to_kpc=convert_coords_to_kpc,
        )
        if made is not None:
            outputs.append(made)

    return outputs


# =============================================================================
# Orbit GIFs
# =============================================================================

def read_evolution_table(table_path: str | Path) -> pd.DataFrame:
    arr = np.genfromtxt(table_path, names=True, dtype=None, encoding=None)
    if getattr(arr, "shape", None) == ():
        arr = np.array([arr], dtype=arr.dtype)
    df = pd.DataFrame({name: arr[name] for name in arr.dtype.names})
    return df


def relative_orbit_xy(df: pd.DataFrame, xcol: str = "host_dx_kpc", ycol: str = "host_dy_kpc") -> Tuple[np.ndarray, np.ndarray]:
    if xcol in df.columns and ycol in df.columns and np.any(np.isfinite(df[xcol])) and np.any(np.isfinite(df[ycol])):
        return np.asarray(df[xcol], dtype=float), np.asarray(df[ycol], dtype=float)
    return np.asarray(df["x_kpc"], dtype=float), np.asarray(df["y_kpc"], dtype=float)


def make_orbit_frames(
    table_path: str | Path,
    outdir: str | Path,
    label: Optional[str] = None,
    xcol: str = "host_dx_kpc",
    ycol: str = "host_dy_kpc",
    trail: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 140,
) -> List[Path]:
    df = read_evolution_table(table_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label = label or Path(table_path).parent.name

    x, y = relative_orbit_xy(df, xcol=xcol, ycol=ycol)
    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        return []

    xg, yg = x[good], y[good]
    pad_x = 0.08 * max(np.nanmax(xg) - np.nanmin(xg), 1.0)
    pad_y = 0.08 * max(np.nanmax(yg) - np.nanmin(yg), 1.0)
    if xlim is None:
        xlim = (float(np.nanmin(xg) - pad_x), float(np.nanmax(xg) + pad_x))
    if ylim is None:
        ylim = (float(np.nanmin(yg) - pad_y), float(np.nanmax(yg) + pad_y))

    time = np.asarray(df["time_gyr"], dtype=float) if "time_gyr" in df else np.full(len(df), np.nan)
    dist = np.asarray(df["dist_host_kpc"], dtype=float) if "dist_host_kpc" in df else np.sqrt(x**2 + y**2)

    frames: List[Path] = []
    for i in range(len(df)):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue

        fig, ax = plt.subplots(figsize=(6.2, 6.0))
        ax.scatter([0], [0], s=160, marker="*", label="host")
        if trail:
            ax.plot(x[:i+1], y[:i+1], lw=1.8, alpha=0.8, label="orbit trail")
        else:
            ax.plot(x, y, lw=1.0, alpha=0.35)
        ax.scatter([x[i]], [y[i]], s=80, marker="o", label="dwarf")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\Delta x_{\rm dwarf-host}$ [kpc]")
        ax.set_ylabel(r"$\Delta y_{\rm dwarf-host}$ [kpc]")
        ax.set_title(f"{label} | frame {i:03d} | t={time[i]:.3f} Gyr | d={dist[i]:.1f} kpc")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        out = outdir / f"orbit_frame_{i:04d}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frames.append(out)

    return frames


def make_gif_from_frames(
    frames: Sequence[str | Path],
    outgif: str | Path,
    duration: float = 0.12,
    loop: int = 0,
) -> Optional[Path]:
    frames = [Path(f) for f in frames if Path(f).exists()]
    if len(frames) == 0:
        return None

    outgif = Path(outgif)
    outgif.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(str(f)) for f in frames]
        imageio.mimsave(str(outgif), imgs, duration=duration, loop=loop)
        return outgif
    except Exception:
        pass

    try:
        from PIL import Image
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
        imgs[0].save(
            outgif,
            save_all=True,
            append_images=imgs[1:],
            duration=int(duration * 1000),
            loop=loop,
        )
        return outgif
    except Exception as exc:
        warnings.warn(f"Could not create GIF: {exc}")
        return None


def make_orbit_gif(
    table_path: str | Path,
    outdir: str | Path,
    label: Optional[str] = None,
    duration: float = 0.12,
    dpi: int = 140,
) -> Optional[Path]:
    outdir = Path(outdir)
    frame_dir = outdir / "orbit_frames"
    frames = make_orbit_frames(table_path, frame_dir, label=label, dpi=dpi)
    if len(frames) == 0:
        return None
    outgif = outdir / f"orbit_{label or Path(table_path).parent.name}.gif"
    return make_gif_from_frames(frames, outgif, duration=duration)


# =============================================================================
# Convenience report
# =============================================================================

def write_run_summary_markdown(
    metrics_df: pd.DataFrame,
    outpath: str | Path,
    title: str = "Ram-pressure and tidal stripping summary",
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {title}", ""]
    if metrics_df.empty:
        lines.append("No metrics available.")
    else:
        lines.append("## Runs")
        lines.append("")
        for _, row in metrics_df.iterrows():
            label = row.get("label", "run")
            lines.append(f"### {label}")
            for key in [
                "d_peri_kpc", "t_peri_gyr", "pram_peak", "tau_pram_peak",
                "tidal_peak", "tau_tidal_peak", "star_norm_min",
                "young_norm_min", "gas_sf_norm_min",
                "fstar_dwarf_origin_stripped", "fgas_dwarf_origin_stripped",
            ]:
                if key in metrics_df.columns:
                    val = row.get(key, np.nan)
                    if isinstance(val, (int, float, np.floating)) and np.isfinite(val):
                        lines.append(f"- `{key}`: {val:.4e}")
            lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")
    return outpath
