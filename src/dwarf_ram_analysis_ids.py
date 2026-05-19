#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dwarf_ram_analysis_ids.py

ID-aware organized analysis module for controlled Gadget4 dwarf-galaxy runs.

This file consolidates the analysis logic that was spread across:
  - analyze_dwarf_ram_pressure_final.py
  - track_dwarf_evolution.py
  - compare_dwarf_ram_pressure_runs.py
  - compare_dwarf_runs_compaction_channels.py
  - compare_dwarf_runs_section5.py
  - pos_process.py

Design
------
1. No plotting code here. This module only reads snapshots/tables, measures
   physical quantities, computes epochs/metrics, and writes tables/CSV files.
2. Notebook-friendly: the main functions return Python dictionaries,
   dataclasses, and numpy arrays that can be inspected interactively.
3. CLI-friendly: use subcommands for the most common workflows.

Typical notebook usage
----------------------
import dwarf_ram_analysis_ids as dra

cfg = dra.TrackingConfig(
    dwarf_init_center=[x0, y0, z0],
    host_center=[xh, yh, zh],
    search_radius=15.0,
)

records, table_path = dra.analyze_snapshot_directory(
    "/path/to/output",
    cfg,
    outdir="/path/to/analysis",
)

runs = dra.prepare_runs(
    ["runA/dwarf_ram_pressure_evolution_final.txt",
     "runB/dwarf_ram_pressure_evolution_final.txt"],
    labels=["A", "B"],
)

channel_rows = [dra.mass_budget_for_run(r) for r in runs]

Command-line examples
---------------------
python dwarf_ram_analysis.py track /path/to/output \
    --dwarf-init-center X Y Z --host-center Xh Yh Zh

python dwarf_ram_analysis.py metrics runA/table.txt runB/table.txt --labels A B \
    --outdir compare_metrics

python dwarf_ram_analysis.py budgets runA/table.txt runB/table.txt \
    --labels A B --snapshot-dirs runA/output runB/output
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SNAP_RE = re.compile(r"snapshot_(\d{3})\.hdf5$")

G_CGS = 6.67430e-8
MSUN_CGS = 1.98847e33
KPC_CGS = 3.0856775814913673e21


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class TrackingConfig:
    """Configuration for snapshot-level dwarf tracking and environmental analysis."""

    dwarf_init_center: Sequence[float]
    host_center: Optional[Sequence[float]] = None
    track_host: bool = False

    search_radius: float = 15.0
    host_search_radius: float = 30.0
    aperture_factor: float = 2.0
    young_age_myr: float = 100.0

    cgm_rin: float = 3.0
    cgm_rout: float = 10.0
    cgm_temp_min: float = 1.0e5
    cgm_density_max: Optional[float] = None
    cgm_estimator: str = "cone"  # "cone" or "shell"
    cone_half_angle_deg: float = 35.0
    cone_min_cells: int = 8
    require_inflow_in_cone: bool = False

    tidal_delta_kpc: float = 1.0
    max_snaps: Optional[int] = None

    # -------------------------------------------------------------------------
    # Optional particle-ID based origin tracking.
    # These defaults keep the original behaviour unchanged unless
    # use_particle_ids=True.
    # -------------------------------------------------------------------------
    use_particle_ids: bool = False
    origin_ids_path: Optional[str] = None
    auto_create_origin_ids: bool = True

    # In your controlled runs only the dwarf has stars. If True, every
    # PartType4 particle in the first snapshot is treated as initially dwarf-born.
    origin_all_stars_are_dwarf: bool = True

    # Initial dwarf-origin gas/DM are selected around the stellar dwarf centre.
    # If origin_dwarf_radius_kpc is None, the radius is
    # max(origin_min_radius_kpc, origin_dwarf_aperture_factor * Rhalf_star_init).
    origin_dwarf_radius_kpc: Optional[float] = None
    origin_dwarf_aperture_factor: float = 5.0
    origin_min_radius_kpc: float = 5.0

    # If True, ram-pressure/CGM estimates use only gas whose IDs were not
    # initially assigned to the dwarf. This avoids counting stripped dwarf gas
    # as host CGM.
    restrict_cgm_to_host_gas_ids: bool = True

    # ID-based stripping diagnostics. IDs define origin; radius/tidal criteria
    # define whether the material is currently stripped.
    measure_id_stripping: bool = True
    stripped_radius_factor: float = 2.0
    stripped_use_tidal_radius: bool = True


@dataclass
class RunData:
    """Container for one already-produced evolution table."""

    label: str
    table_path: str
    snapshot_dir: Optional[str]
    tab: np.ndarray
    cols: Dict[str, np.ndarray]
    t: np.ndarray
    tau: np.ndarray
    t_peri: float
    epochs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OriginIDCatalog:
    """Particle-ID origin catalog built from the initial snapshot.

    IDs define where a particle came from. They should be combined with
    geometric or dynamical criteria to decide whether the particle is still
    part of the dwarf, in a tail, or in the host/CGM.
    """

    dwarf_star_ids: Optional[np.ndarray] = None
    dwarf_gas_ids: Optional[np.ndarray] = None
    host_gas_ids: Optional[np.ndarray] = None
    dwarf_dm_ids: Optional[np.ndarray] = None
    host_dm_ids: Optional[np.ndarray] = None
    origin_radius_kpc: float = np.nan
    source_snapshot: str = ""



# =============================================================================
# Output schema
# =============================================================================

OUTPUT_COLUMNS = [
    "snapshot", "time_gyr", "redshift",
    "x_kpc", "y_kpc", "z_kpc",
    "dist_host_kpc",
    "rhalf_star_kpc", "rhalf_old_kpc", "rhalf_young_kpc",
    "young_to_total_size_ratio", "old_to_total_size_ratio",
    "young_to_old_size_ratio", "gas_sf_to_star_size_ratio",
    "nstar_dwarf_origin", "mstar_dwarf_origin_msun",
    "mstar_dwarf_origin_core_msun", "mstar_dwarf_origin_stripped_msun",
    "fstar_dwarf_origin_stripped", "rhalf_star_dwarf_origin_kpc",
    "ngas_dwarf_origin", "mgas_dwarf_origin_msun",
    "mgas_dwarf_origin_core_msun", "mgas_dwarf_origin_stripped_msun",
    "fgas_dwarf_origin_stripped",
    "ngas_host_origin", "mgas_host_origin_msun",
    "ndm_dwarf_origin", "mdm_dwarf_origin_msun",
    "mdm_dwarf_origin_core_msun", "mdm_dwarf_origin_stripped_msun",
    "fdm_dwarf_origin_stripped",
    "id_stripped_radius_kpc", "id_origin_radius_kpc",
    "mstar_msun", "mstar_old_msun", "mstar_young_msun",
    "mdm_ap_msun", "mbar_ap_msun", "mdwarf_proxy_msun",
    "mgas_msun", "mgas_sf_msun",
    "sfr_msun_per_yr", "sfr_central_msun_per_yr", "f_sfr_central",
    "rhalf_gas_sf_kpc",
    "rho_cgm_g_cm3", "vrel_km_s", "pram_dyn_cm2", "n_env", "cgm_host_id_restricted",
    "rho_cgm_shell_g_cm3", "vrel_shell_km_s", "pram_shell_dyn_cm2", "n_shell",
    "rho_cgm_upstream_g_cm3", "vrel_upstream_km_s", "pram_upstream_dyn_cm2",
    "n_upstream", "upstream_x", "upstream_y", "upstream_z",
    "mgas_leading_frac", "mgas_sf_leading_frac", "sfr_leading_frac",
    "rho_leading_over_trailing", "gas_com_along_upstream_kpc",
    "host_dx_kpc", "host_dy_kpc", "host_dz_kpc",
    "host_radial_x", "host_radial_y", "host_radial_z",
    "cos_wind_host_radial",
    "mhost_inner_msun", "mhost_enclosed_msun", "mhost_outer_msun",
    "host_mean_density_msun_kpc3", "tidal_strength_msun_kpc3",
    "tidal_radius_kpc",
    "rhalf_star_over_rtidal", "rhalf_gas_sf_over_rtidal",
    "rhalf_young_over_rtidal", "rhalf_old_over_rtidal",
    "delta_g_tidal_cms2", "tidal_gradient_s2",
    "a_tide_rhalf_star_cms2", "a_tide_rhalf_gas_sf_cms2",
    "a_self_star_proxy_cms2", "a_self_gas_proxy_cms2",
    "tidal_to_self_star", "tidal_to_self_gas", "tidal_delta_kpc",
]


# =============================================================================
# Generic utilities
# =============================================================================

def try_import_yt():
    """Import yt lazily so that table-only workflows work without yt installed."""
    try:
        import yt  # type: ignore
        return yt
    except Exception as exc:
        raise RuntimeError("yt is required for snapshot-level analysis but could not be imported.") from exc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_snapshots(path: str) -> List[str]:
    files = glob.glob(os.path.join(path, "snapshot_*.hdf5"))

    def keyfunc(fname: str) -> int:
        base = os.path.basename(fname)
        m = SNAP_RE.match(base)
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=keyfunc)


def snapshot_number(path: str) -> Optional[int]:
    m = SNAP_RE.match(os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def to_nd(arr: Any) -> np.ndarray:
    try:
        return arr.to_ndarray()
    except Exception:
        return np.asarray(arr)


def choose_existing_field(ds: Any, candidates: Sequence[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    all_fields = set(ds.field_list) | set(ds.derived_field_list)
    for field in candidates:
        if field in all_fields:
            return field
    return None


def get_particle_id_field(ds: Any, ptype: str) -> Optional[Tuple[str, str]]:
    """Return the most likely particle-ID field for a Gadget/yt particle type."""
    return choose_existing_field(ds, [
        (ptype, "ParticleIDs"),
        (ptype, "particle_identity"),
        (ptype, "particle_index"),
        (ptype, "ParticleID"),
        (ptype, "ID"),
    ])


def _ids_to_int64(ids: Any) -> np.ndarray:
    arr = to_nd(ids)
    return np.asarray(arr, dtype=np.int64)


def ids_mask(ids: Optional[np.ndarray], allowed_ids: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Boolean mask selecting IDs contained in allowed_ids.

    Returns None when either side is unavailable, so callers can fall back to
    the original non-ID behaviour.
    """
    if ids is None or allowed_ids is None:
        return None
    allowed = np.asarray(allowed_ids, dtype=np.int64)
    if allowed.size == 0:
        return np.zeros(len(ids), dtype=bool)
    return np.isin(np.asarray(ids, dtype=np.int64), allowed)


def load_origin_id_catalog(path: str) -> OriginIDCatalog:
    data = np.load(path, allow_pickle=False)

    def get(name: str) -> Optional[np.ndarray]:
        if name not in data.files:
            return None
        arr = np.asarray(data[name], dtype=np.int64)
        return arr if arr.size else np.array([], dtype=np.int64)

    source = ""
    if "source_snapshot" in data.files:
        try:
            source = str(np.asarray(data["source_snapshot"]).item())
        except Exception:
            source = ""

    origin_radius = np.nan
    if "origin_radius_kpc" in data.files:
        try:
            origin_radius = float(np.asarray(data["origin_radius_kpc"]).item())
        except Exception:
            origin_radius = np.nan

    return OriginIDCatalog(
        dwarf_star_ids=get("dwarf_star_ids"),
        dwarf_gas_ids=get("dwarf_gas_ids"),
        host_gas_ids=get("host_gas_ids"),
        dwarf_dm_ids=get("dwarf_dm_ids"),
        host_dm_ids=get("host_dm_ids"),
        origin_radius_kpc=origin_radius,
        source_snapshot=source,
    )


def save_origin_id_catalog(catalog: OriginIDCatalog, path: str) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))

    def arr(x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            return np.array([], dtype=np.int64)
        return np.asarray(x, dtype=np.int64)

    np.savez(
        path,
        dwarf_star_ids=arr(catalog.dwarf_star_ids),
        dwarf_gas_ids=arr(catalog.dwarf_gas_ids),
        host_gas_ids=arr(catalog.host_gas_ids),
        dwarf_dm_ids=arr(catalog.dwarf_dm_ids),
        host_dm_ids=arr(catalog.host_dm_ids),
        origin_radius_kpc=np.array(catalog.origin_radius_kpc),
        source_snapshot=np.array(catalog.source_snapshot),
    )
    return path


def get_time_gyr(ds: Any) -> float:
    try:
        return float(ds.current_time.to("Gyr"))
    except Exception:
        try:
            return float(ds.current_time.in_units("Gyr"))
        except Exception:
            return np.nan


def get_redshift(ds: Any) -> float:
    try:
        return float(ds.current_redshift)
    except Exception:
        return np.nan


def safe_div(num: Any, den: Any) -> float:
    try:
        num_f = float(num)
        den_f = float(den)
        if np.isfinite(num_f) and np.isfinite(den_f) and den_f != 0.0:
            return num_f / den_f
    except Exception:
        pass
    return np.nan


def safe_frac_delta(final: Any, initial: Any) -> float:
    try:
        f = float(final)
        i = float(initial)
        if np.isfinite(f) and np.isfinite(i) and i != 0.0:
            return (f - i) / i
    except Exception:
        pass
    return np.nan


def normalize_vector(v: Sequence[float]) -> Optional[np.ndarray]:
    vec = np.asarray(v, dtype=float)
    n = float(np.sqrt(np.sum(vec**2)))
    if not np.isfinite(n) or n <= 0:
        return None
    return vec / n


def finite_mask(*arrs: np.ndarray) -> np.ndarray:
    if len(arrs) == 0:
        return np.array([], dtype=bool)
    m = np.ones(len(arrs[0]), dtype=bool)
    for arr in arrs:
        m &= np.isfinite(np.asarray(arr, dtype=float))
    return m


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


def baseline_in_window(tau: Sequence[float], y: Sequence[float], tmin: float = -0.6, tmax: float = -0.1) -> float:
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


def argmax_in_window(tau: Sequence[float], y: Sequence[float], tmin: float, tmax: float) -> Tuple[float, float]:
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(tau) & np.isfinite(y) & (tau >= tmin) & (tau <= tmax)
    if not np.any(m):
        return np.nan, np.nan
    i = int(np.nanargmax(y[m]))
    return float(y[m][i]), float(tau[m][i])


def argmin_in_window(tau: Sequence[float], y: Sequence[float], tmin: float, tmax: float) -> Tuple[float, float]:
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(tau) & np.isfinite(y) & (tau >= tmin) & (tau <= tmax)
    if not np.any(m):
        return np.nan, np.nan
    i = int(np.nanargmin(y[m]))
    return float(y[m][i]), float(tau[m][i])


def log10_positive(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x) & (x > 0)
    out[good] = np.log10(x[good])
    return out


# =============================================================================
# Table I/O
# =============================================================================

def read_table(filename: str) -> np.ndarray:
    tab = np.genfromtxt(filename, names=True, dtype=None, encoding=None)
    if np.size(tab) == 0:
        raise RuntimeError("Empty table: {0}".format(filename))
    if getattr(tab, "shape", None) == ():
        tab = np.array([tab], dtype=tab.dtype)
    if tab.dtype.names is None:
        raise RuntimeError("Could not read named columns from: {0}".format(filename))
    if "time_gyr" in tab.dtype.names:
        order = np.argsort(np.asarray(tab["time_gyr"], dtype=float))
        tab = tab[order]
    return tab


def table_to_cols(tab: np.ndarray) -> Dict[str, np.ndarray]:
    cols: Dict[str, np.ndarray] = {}
    if tab.dtype.names is None:
        return cols
    for name in tab.dtype.names:
        arr = np.asarray(tab[name])
        if arr.dtype.kind in "iufb":
            cols[name] = arr.astype(float)
        else:
            cols[name] = arr
    return cols


def write_records_table(
    records: Sequence[Dict[str, Any]],
    output_path: str,
    columns: Optional[Sequence[str]] = None,
) -> str:
    columns = list(columns) if columns is not None else list(OUTPUT_COLUMNS)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# " + " ".join(columns) + "\n")
        for rec in records:
            row = []
            for key in columns:
                val = rec.get(key, np.nan)
                if key == "snapshot":
                    row.append(str(val))
                else:
                    try:
                        row.append("{:.8e}".format(float(val)))
                    except Exception:
                        row.append("nan")
            f.write(" ".join(row) + "\n")
    return output_path


def write_csv(rows: Sequence[Dict[str, Any]], output_path: str) -> Optional[str]:
    if not rows:
        return None
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def structured_from_records(records: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> np.ndarray:
    """Convert numeric record fields to a structured array; useful for quick plotting."""
    columns = list(columns) if columns is not None else list(OUTPUT_COLUMNS)
    dtype = []
    for c in columns:
        dtype.append((c, "U128" if c == "snapshot" else "f8"))
    arr = np.empty(len(records), dtype=dtype)
    for i, rec in enumerate(records):
        for c in columns:
            if c == "snapshot":
                arr[c][i] = str(rec.get(c, ""))
            else:
                try:
                    arr[c][i] = float(rec.get(c, np.nan))
                except Exception:
                    arr[c][i] = np.nan
    return arr


# =============================================================================
# Field discovery and data access
# =============================================================================

def get_star_fields(ds: Any) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    pos = choose_existing_field(ds, [("PartType4", "Coordinates"), ("stars", "particle_position")])
    vel = choose_existing_field(ds, [("PartType4", "Velocities"), ("stars", "particle_velocity")])
    mass = choose_existing_field(ds, [("PartType4", "Masses"), ("PartType4", "particle_mass"), ("stars", "particle_mass")])
    birth = choose_existing_field(ds, [("PartType4", "StellarFormationTime"), ("PartType4", "creation_time"), ("stars", "creation_time")])
    return pos, vel, mass, birth


def get_gas_fields(ds: Any) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    pos = choose_existing_field(ds, [("PartType0", "Coordinates"), ("gas", "particle_position"), ("gas", "x")])
    vel = choose_existing_field(ds, [("PartType0", "Velocities"), ("gas", "particle_velocity"), ("gas", "velocity")])
    mass = choose_existing_field(ds, [("PartType0", "Masses"), ("PartType0", "particle_mass"), ("gas", "cell_mass"), ("gas", "mass")])
    sfr = choose_existing_field(ds, [("PartType0", "StarFormationRate"), ("gas", "star_formation_rate"), ("gas", "sfr")])
    dens = choose_existing_field(ds, [("PartType0", "Density"), ("gas", "density")])
    temp = choose_existing_field(ds, [("gas", "temperature"), ("PartType0", "Temperature")])
    return pos, vel, mass, sfr, dens, temp


def get_dm_fields(ds: Any) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    pos = choose_existing_field(ds, [("PartType1", "Coordinates"), ("darkmatter", "particle_position"), ("dm", "particle_position")])
    vel = choose_existing_field(ds, [("PartType1", "Velocities"), ("darkmatter", "particle_velocity"), ("dm", "particle_velocity")])
    mass = choose_existing_field(ds, [("PartType1", "Masses"), ("PartType1", "particle_mass"), ("darkmatter", "particle_mass"), ("dm", "particle_mass")])
    return pos, vel, mass


def get_component_catalog(ds: Any) -> List[Dict[str, Any]]:
    ad = ds.all_data()
    specs = [
        ("gas", [("PartType0", "Coordinates"), ("gas", "particle_position"), ("gas", "x")],
         [("PartType0", "Masses"), ("PartType0", "particle_mass"), ("gas", "cell_mass"), ("gas", "mass")]),
        ("dm", [("PartType1", "Coordinates"), ("darkmatter", "particle_position"), ("dm", "particle_position")],
         [("PartType1", "Masses"), ("PartType1", "particle_mass"), ("darkmatter", "particle_mass"), ("dm", "particle_mass")]),
        ("ptype2", [("PartType2", "Coordinates")], [("PartType2", "Masses"), ("PartType2", "particle_mass")]),
        ("ptype3", [("PartType3", "Coordinates")], [("PartType3", "Masses"), ("PartType3", "particle_mass")]),
        ("stars", [("PartType4", "Coordinates"), ("stars", "particle_position")],
         [("PartType4", "Masses"), ("PartType4", "particle_mass"), ("stars", "particle_mass")]),
        ("ptype5", [("PartType5", "Coordinates")], [("PartType5", "Masses"), ("PartType5", "particle_mass")]),
    ]

    comps: List[Dict[str, Any]] = []
    for name, pos_candidates, mass_candidates in specs:
        pos_f = choose_existing_field(ds, pos_candidates)
        mass_f = choose_existing_field(ds, mass_candidates)
        if pos_f is None or mass_f is None:
            continue
        try:
            pos_arr = ad[pos_f]
            mass_arr = ad[mass_f]
        except Exception:
            continue
        comps.append({
            "name": name,
            "pos": pos_arr,
            "mass": mass_arr,
            "pos_units": pos_arr.units,
            "mass_units": mass_arr.units,
        })
    return comps


def get_star_data(ds: Any) -> Optional[Dict[str, Any]]:
    ad = ds.all_data()
    pos_f, vel_f, mass_f, birth_f = get_star_fields(ds)
    id_f = get_particle_id_field(ds, "PartType4")
    if pos_f is None or mass_f is None:
        return None
    return {
        "pos": ad[pos_f],
        "vel": ad[vel_f] if vel_f is not None else None,
        "mass": ad[mass_f],
        "birth": ad[birth_f] if birth_f is not None else None,
        "id": ad[id_f] if id_f is not None else None,
        "pos_units": ad[pos_f].units,
        "vel_units": ad[vel_f].units if vel_f is not None else None,
        "mass_units": ad[mass_f].units,
    }


def get_gas_data(ds: Any) -> Optional[Dict[str, Any]]:
    ad = ds.all_data()
    pos_f, vel_f, mass_f, sfr_f, dens_f, temp_f = get_gas_fields(ds)
    id_f = get_particle_id_field(ds, "PartType0")
    if pos_f is None or mass_f is None:
        return None
    return {
        "pos": ad[pos_f],
        "vel": ad[vel_f] if vel_f is not None else None,
        "mass": ad[mass_f],
        "sfr": ad[sfr_f] if sfr_f is not None else None,
        "dens": ad[dens_f] if dens_f is not None else None,
        "temp": ad[temp_f] if temp_f is not None else None,
        "id": ad[id_f] if id_f is not None else None,
        "pos_units": ad[pos_f].units,
        "vel_units": ad[vel_f].units if vel_f is not None else None,
        "mass_units": ad[mass_f].units,
    }


def get_dm_data(ds: Any) -> Optional[Dict[str, Any]]:
    ad = ds.all_data()
    pos_f, vel_f, mass_f = get_dm_fields(ds)
    id_f = get_particle_id_field(ds, "PartType1")
    if pos_f is None or mass_f is None:
        return None
    return {
        "pos": ad[pos_f],
        "vel": ad[vel_f] if vel_f is not None else None,
        "mass": ad[mass_f],
        "id": ad[id_f] if id_f is not None else None,
        "pos_units": ad[pos_f].units,
        "vel_units": ad[vel_f].units if vel_f is not None else None,
        "mass_units": ad[mass_f].units,
    }


# =============================================================================
# Basic particle measurements
# =============================================================================

def weighted_center(pos: np.ndarray, mass: np.ndarray) -> Optional[np.ndarray]:
    pos = np.asarray(pos)
    mass = np.asarray(mass)
    if len(pos) == 0:
        return None
    mtot = np.sum(mass)
    if not np.isfinite(mtot) or mtot <= 0:
        return None
    return np.sum(pos * mass[:, None], axis=0) / mtot


def robust_center(pos: np.ndarray, mass: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    pos = np.asarray(pos)
    if len(pos) == 0:
        return None
    med = np.median(pos, axis=0)
    if mass is None or len(mass) != len(pos):
        return med
    dist = radius_3d(pos, med)
    good = np.isfinite(dist)
    if not np.any(good):
        return med
    rcut = np.percentile(dist[good], 50.0)
    sel = dist <= rcut
    if np.sum(sel) < 5:
        return med
    cen = weighted_center(pos[sel], np.asarray(mass)[sel])
    return med if cen is None else cen


def weighted_velocity(vel: np.ndarray, mass: np.ndarray) -> Optional[np.ndarray]:
    vel = np.asarray(vel)
    mass = np.asarray(mass)
    if len(vel) == 0:
        return None
    mtot = np.sum(mass)
    if not np.isfinite(mtot) or mtot <= 0:
        return None
    return np.sum(vel * mass[:, None], axis=0) / mtot


def radius_3d(pos: np.ndarray, center: Sequence[float]) -> np.ndarray:
    return np.sqrt(np.sum((np.asarray(pos) - np.asarray(center)[None, :])**2, axis=1))


def half_mass_radius(pos: np.ndarray, mass: np.ndarray, center: Sequence[float]) -> float:
    pos = np.asarray(pos)
    mass = np.asarray(mass)
    if len(pos) == 0 or len(mass) == 0:
        return np.nan
    r = radius_3d(pos, center)
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = mass[order]
    mcum = np.cumsum(m_sorted)
    if len(mcum) == 0 or not np.isfinite(mcum[-1]) or mcum[-1] <= 0:
        return np.nan
    i = int(np.searchsorted(mcum, 0.5 * mcum[-1]))
    i = min(i, len(r_sorted) - 1)
    return float(r_sorted[i])



def _mass_to_msun(ds: Any, mass: np.ndarray, mass_units: Any) -> np.ndarray:
    try:
        return ds.arr(mass, mass_units).to("Msun").to_ndarray()
    except Exception:
        return np.asarray(mass, dtype=float)


def _radius_to_kpc(ds: Any, radius_native: np.ndarray, pos_units: Any) -> np.ndarray:
    try:
        return ds.arr(radius_native, pos_units).to("kpc").to_ndarray()
    except Exception:
        return np.asarray(radius_native, dtype=float)


def create_origin_id_catalog_from_loaded_snapshot(
    ds: Any,
    snapshot_name: str,
    dwarf_center_guess_code: Sequence[float],
    cfg: TrackingConfig,
) -> OriginIDCatalog:
    """Build an ID-origin catalog from the initial snapshot.

    In the intended controlled setup, only the dwarf has stars. Therefore, by
    default all PartType4 IDs in the initial snapshot are assigned to the dwarf.
    Dwarf-origin gas and DM are selected inside a radius around the initial
    stellar centre; host gas/DM are all remaining gas/DM IDs.
    """
    dwarf = track_object_from_stars(ds, dwarf_center_guess_code, cfg.search_radius)
    if dwarf is None:
        raise RuntimeError("Could not locate dwarf in initial snapshot to build origin ID catalog.")

    center = dwarf["center"]
    pos_units = dwarf["pos_units"]
    rhalf_native = dwarf["rhalf_star_native"]
    try:
        rhalf_kpc = float(ds.quan(rhalf_native, pos_units).to("kpc"))
    except Exception:
        rhalf_kpc = float(rhalf_native)

    if cfg.origin_dwarf_radius_kpc is not None:
        origin_radius_kpc = float(cfg.origin_dwarf_radius_kpc)
    else:
        origin_radius_kpc = max(float(cfg.origin_min_radius_kpc), float(cfg.origin_dwarf_aperture_factor) * rhalf_kpc)

    def select_component_ids(component: Optional[Dict[str, Any]], all_are_dwarf: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if component is None or component.get("id") is None:
            return None, None
        ids = _ids_to_int64(component["id"])
        if all_are_dwarf:
            dwarf_ids = ids
        else:
            center_q = ds.arr(center, pos_units).to(component["pos_units"])
            pos = to_nd(component["pos"].to(component["pos_units"]))
            rr_native = radius_3d(pos, to_nd(center_q))
            rr_kpc = _radius_to_kpc(ds, rr_native, component["pos_units"])
            dwarf_ids = ids[np.isfinite(rr_kpc) & (rr_kpc <= origin_radius_kpc)]
        dwarf_ids = np.asarray(dwarf_ids, dtype=np.int64)
        host_ids = np.setdiff1d(ids, dwarf_ids, assume_unique=False)
        return dwarf_ids, host_ids

    star = get_star_data(ds)
    gas = get_gas_data(ds)
    dm = get_dm_data(ds)

    dwarf_star_ids, _ = select_component_ids(star, all_are_dwarf=cfg.origin_all_stars_are_dwarf)
    dwarf_gas_ids, host_gas_ids = select_component_ids(gas, all_are_dwarf=False)
    dwarf_dm_ids, host_dm_ids = select_component_ids(dm, all_are_dwarf=False)

    return OriginIDCatalog(
        dwarf_star_ids=dwarf_star_ids,
        dwarf_gas_ids=dwarf_gas_ids,
        host_gas_ids=host_gas_ids,
        dwarf_dm_ids=dwarf_dm_ids,
        host_dm_ids=host_dm_ids,
        origin_radius_kpc=origin_radius_kpc,
        source_snapshot=snapshot_name,
    )


def create_origin_id_catalog_from_snapshot(snapshot_path: str, cfg: TrackingConfig) -> OriginIDCatalog:
    yt = try_import_yt()
    ds = yt.load(snapshot_path)
    return create_origin_id_catalog_from_loaded_snapshot(
        ds,
        os.path.basename(snapshot_path),
        cfg.dwarf_init_center,
        cfg,
    )


def resolve_origin_id_catalog(snapfiles: Sequence[str], cfg: TrackingConfig, outdir: str, verbose: bool = True) -> Optional[OriginIDCatalog]:
    """Load or create the optional ID-origin catalog for a run."""
    if not cfg.use_particle_ids:
        return None
    if len(snapfiles) == 0:
        return None

    path = cfg.origin_ids_path or os.path.join(outdir, "origin_particle_ids.npz")
    if os.path.exists(path):
        if verbose:
            print("[INFO] Loading origin ID catalog: {0}".format(path))
        return load_origin_id_catalog(path)

    if not cfg.auto_create_origin_ids:
        warnings.warn("use_particle_ids=True but origin catalog does not exist: {0}".format(path))
        return None

    if verbose:
        print("[INFO] Creating origin ID catalog from: {0}".format(os.path.basename(snapfiles[0])))
    catalog = create_origin_id_catalog_from_snapshot(snapfiles[0], cfg)
    save_origin_id_catalog(catalog, path)
    if verbose:
        print("[INFO] Wrote origin ID catalog: {0}".format(path))
        print("       N dwarf stars: {0}".format(0 if catalog.dwarf_star_ids is None else len(catalog.dwarf_star_ids)))
        print("       N dwarf gas:   {0}".format(0 if catalog.dwarf_gas_ids is None else len(catalog.dwarf_gas_ids)))
        print("       N host gas:    {0}".format(0 if catalog.host_gas_ids is None else len(catalog.host_gas_ids)))
    return catalog


def _component_origin_budget(
    ds: Any,
    component: Optional[Dict[str, Any]],
    origin_ids: Optional[np.ndarray],
    center_native: Sequence[float],
    center_units: Any,
    core_radius_kpc: float,
    stripped_radius_kpc: float,
    prefix: str,
    compute_rhalf: bool = False,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if component is None or component.get("id") is None or origin_ids is None:
        out["n{0}_dwarf_origin".format(prefix)] = np.nan
        out["m{0}_dwarf_origin_msun".format(prefix)] = np.nan
        out["m{0}_dwarf_origin_core_msun".format(prefix)] = np.nan
        out["m{0}_dwarf_origin_stripped_msun".format(prefix)] = np.nan
        out["f{0}_dwarf_origin_stripped".format(prefix)] = np.nan
        if compute_rhalf:
            out["rhalf_{0}_dwarf_origin_kpc".format(prefix)] = np.nan
        return out

    ids = _ids_to_int64(component["id"])
    m_origin = ids_mask(ids, origin_ids)
    if m_origin is None or not np.any(m_origin):
        out["n{0}_dwarf_origin".format(prefix)] = 0.0
        out["m{0}_dwarf_origin_msun".format(prefix)] = 0.0
        out["m{0}_dwarf_origin_core_msun".format(prefix)] = 0.0
        out["m{0}_dwarf_origin_stripped_msun".format(prefix)] = 0.0
        out["f{0}_dwarf_origin_stripped".format(prefix)] = np.nan
        if compute_rhalf:
            out["rhalf_{0}_dwarf_origin_kpc".format(prefix)] = np.nan
        return out

    center_q = ds.arr(center_native, center_units).to(component["pos_units"])
    pos = to_nd(component["pos"].to(component["pos_units"]))
    mass = _mass_to_msun(ds, to_nd(component["mass"]), component["mass_units"])
    rr_native = radius_3d(pos, to_nd(center_q))
    rr_kpc = _radius_to_kpc(ds, rr_native, component["pos_units"])

    sel_core = m_origin & np.isfinite(rr_kpc) & (rr_kpc <= core_radius_kpc)
    sel_stripped = m_origin & np.isfinite(rr_kpc) & (rr_kpc >= stripped_radius_kpc)

    mtot = float(np.sum(mass[m_origin]))
    mcore = float(np.sum(mass[sel_core]))
    mstrip = float(np.sum(mass[sel_stripped]))

    out["n{0}_dwarf_origin".format(prefix)] = float(np.sum(m_origin))
    out["m{0}_dwarf_origin_msun".format(prefix)] = mtot
    out["m{0}_dwarf_origin_core_msun".format(prefix)] = mcore
    out["m{0}_dwarf_origin_stripped_msun".format(prefix)] = mstrip
    out["f{0}_dwarf_origin_stripped".format(prefix)] = safe_div(mstrip, mtot)

    if compute_rhalf:
        try:
            rhalf = half_mass_radius(pos[m_origin], mass[m_origin], to_nd(center_q))
            out["rhalf_{0}_dwarf_origin_kpc".format(prefix)] = float(ds.quan(rhalf, component["pos_units"]).to("kpc"))
        except Exception:
            out["rhalf_{0}_dwarf_origin_kpc".format(prefix)] = np.nan
    return out


def measure_origin_id_diagnostics(
    ds: Any,
    center_native: Sequence[float],
    center_units: Any,
    rhalf_star_native: float,
    aperture_factor: float,
    origin_ids: Optional[OriginIDCatalog],
    stripped_radius_factor: float = 2.0,
    stripped_use_tidal_radius: bool = True,
    tidal_radius_kpc: float = np.nan,
) -> Dict[str, float]:
    """Measure ID-based retained/stripped budgets.

    IDs define initial origin. The current state is still defined spatially: core
    material is inside aperture_factor*Rhalf_star, while stripped material is
    outside max(stripped_radius_factor*Rhalf_star, r_tidal) when r_tidal is
    available and stripped_use_tidal_radius=True.
    """
    out: Dict[str, float] = {
        "nstar_dwarf_origin": np.nan,
        "mstar_dwarf_origin_msun": np.nan,
        "mstar_dwarf_origin_core_msun": np.nan,
        "mstar_dwarf_origin_stripped_msun": np.nan,
        "fstar_dwarf_origin_stripped": np.nan,
        "rhalf_star_dwarf_origin_kpc": np.nan,
        "ngas_dwarf_origin": np.nan,
        "mgas_dwarf_origin_msun": np.nan,
        "mgas_dwarf_origin_core_msun": np.nan,
        "mgas_dwarf_origin_stripped_msun": np.nan,
        "fgas_dwarf_origin_stripped": np.nan,
        "ngas_host_origin": np.nan,
        "mgas_host_origin_msun": np.nan,
        "ndm_dwarf_origin": np.nan,
        "mdm_dwarf_origin_msun": np.nan,
        "mdm_dwarf_origin_core_msun": np.nan,
        "mdm_dwarf_origin_stripped_msun": np.nan,
        "fdm_dwarf_origin_stripped": np.nan,
        "id_stripped_radius_kpc": np.nan,
        "id_origin_radius_kpc": np.nan,
    }
    if origin_ids is None:
        return out

    try:
        rhalf_star_kpc = float(ds.quan(rhalf_star_native, center_units).to("kpc"))
    except Exception:
        rhalf_star_kpc = float(rhalf_star_native)

    core_radius_kpc = float(aperture_factor) * rhalf_star_kpc
    stripped_radius_kpc = float(stripped_radius_factor) * rhalf_star_kpc
    if stripped_use_tidal_radius and np.isfinite(tidal_radius_kpc) and tidal_radius_kpc > 0:
        stripped_radius_kpc = max(stripped_radius_kpc, float(tidal_radius_kpc))

    out["id_stripped_radius_kpc"] = stripped_radius_kpc
    out["id_origin_radius_kpc"] = origin_ids.origin_radius_kpc

    star = get_star_data(ds)
    gas = get_gas_data(ds)
    dm = get_dm_data(ds)

    out.update(_component_origin_budget(
        ds, star, origin_ids.dwarf_star_ids,
        center_native, center_units,
        core_radius_kpc, stripped_radius_kpc,
        prefix="star", compute_rhalf=True,
    ))
    out.update(_component_origin_budget(
        ds, gas, origin_ids.dwarf_gas_ids,
        center_native, center_units,
        core_radius_kpc, stripped_radius_kpc,
        prefix="gas", compute_rhalf=False,
    ))
    out.update(_component_origin_budget(
        ds, dm, origin_ids.dwarf_dm_ids,
        center_native, center_units,
        core_radius_kpc, stripped_radius_kpc,
        prefix="dm", compute_rhalf=False,
    ))

    # Host-origin gas budget, useful for sanity checks. This is not a CGM shell;
    # it is the total host-origin gas present in the snapshot/domain.
    if gas is not None and gas.get("id") is not None and origin_ids.host_gas_ids is not None:
        gids = _ids_to_int64(gas["id"])
        hmask = ids_mask(gids, origin_ids.host_gas_ids)
        if hmask is not None:
            mass = _mass_to_msun(ds, to_nd(gas["mass"]), gas["mass_units"])
            out["ngas_host_origin"] = float(np.sum(hmask))
            out["mgas_host_origin_msun"] = float(np.sum(mass[hmask]))

    return out


def get_star_current_ages_myr(ds: Any, star_birth_arr: Any) -> Optional[np.ndarray]:
    if star_birth_arr is None:
        return None
    try:
        age = ds.current_time.to("Myr") - star_birth_arr.to("Myr")
        return to_nd(age)
    except Exception:
        return None


def track_object_from_stars(ds: Any, center_guess_code: Sequence[float], search_radius_kpc: float, min_stars: int = 20) -> Optional[Dict[str, Any]]:
    """Track an object using the robust center of stars near a previous center guess."""
    star = get_star_data(ds)
    if star is None:
        return None

    pos_units = star["pos_units"]
    center_guess = ds.arr(center_guess_code, "code_length").to(pos_units)
    rsearch = ds.quan(search_radius_kpc, "kpc").to(pos_units)

    pos = to_nd(star["pos"].to(pos_units))
    mass = to_nd(star["mass"])
    sel = radius_3d(pos, to_nd(center_guess)) <= float(rsearch)

    if np.sum(sel) < min_stars:
        return None

    center = robust_center(pos[sel], mass[sel])
    if center is None:
        return None

    rhalf = half_mass_radius(pos[sel], mass[sel], center)
    vbulk = None
    if star["vel"] is not None:
        try:
            vel = to_nd(star["vel"][sel])
            vbulk = weighted_velocity(vel, mass[sel])
        except Exception:
            vbulk = None

    return {
        "center": center,
        "rhalf_star_native": rhalf,
        "nstar": int(np.sum(sel)),
        "pos_units": str(pos_units),
        "vbulk_star_native": vbulk,
    }


def measure_dwarf_properties(ds: Any, center_native: Sequence[float], rhalf_star_native: float, aperture_factor: float, young_age_myr: float = 100.0) -> Optional[Dict[str, float]]:
    star = get_star_data(ds)
    gas = get_gas_data(ds)
    dm = get_dm_data(ds)

    if star is None:
        return None

    center_star = ds.arr(center_native, star["pos_units"])
    rap_star = ds.quan(aperture_factor * rhalf_star_native, star["pos_units"])

    star_pos = to_nd(star["pos"].to(star["pos_units"]))
    star_mass = to_nd(star["mass"])
    rstar = radius_3d(star_pos, to_nd(center_star))
    sel_star = rstar <= float(rap_star)

    mstar_native = np.sum(star_mass[sel_star])

    rhalf_young = np.nan
    rhalf_old = np.nan
    mstar_young = np.nan
    mstar_old = np.nan

    if star["birth"] is not None:
        ages_myr = get_star_current_ages_myr(ds, star["birth"])
        if ages_myr is not None:
            sel_young = sel_star & np.isfinite(ages_myr) & (ages_myr >= 0.0) & (ages_myr <= young_age_myr)
            sel_old = sel_star & np.isfinite(ages_myr) & (ages_myr > young_age_myr)

            if np.sum(sel_young) > 5:
                mstar_young_native = np.sum(star_mass[sel_young])
                rhalf_tmp = half_mass_radius(star_pos[sel_young], star_mass[sel_young], to_nd(center_star))
                try:
                    rhalf_young = float(ds.quan(rhalf_tmp, star["pos_units"]).to("kpc"))
                except Exception:
                    rhalf_young = float(rhalf_tmp)
                try:
                    mstar_young = float(ds.quan(mstar_young_native, star["mass_units"]).to("Msun"))
                except Exception:
                    mstar_young = float(mstar_young_native)

            if np.sum(sel_old) > 5:
                mstar_old_native = np.sum(star_mass[sel_old])
                rhalf_tmp = half_mass_radius(star_pos[sel_old], star_mass[sel_old], to_nd(center_star))
                try:
                    rhalf_old = float(ds.quan(rhalf_tmp, star["pos_units"]).to("kpc"))
                except Exception:
                    rhalf_old = float(rhalf_tmp)
                try:
                    mstar_old = float(ds.quan(mstar_old_native, star["mass_units"]).to("Msun"))
                except Exception:
                    mstar_old = float(mstar_old_native)

    mgas = np.nan
    mgas_sf = np.nan
    sfr_total = np.nan
    sfr_central = np.nan
    f_sfr_central = np.nan
    rhalf_gas_sf = np.nan

    if gas is not None:
        center_gas = ds.arr(center_native, gas["pos_units"])
        rap_gas = ds.quan(aperture_factor * rhalf_star_native, gas["pos_units"])
        gas_pos = to_nd(gas["pos"].to(gas["pos_units"]))
        gas_mass = to_nd(gas["mass"])
        rgas = radius_3d(gas_pos, to_nd(center_gas))
        sel_gas = rgas <= float(rap_gas)
        mgas_native = np.sum(gas_mass[sel_gas])

        try:
            mgas = float(ds.quan(mgas_native, gas["mass_units"]).to("Msun"))
        except Exception:
            mgas = float(mgas_native)

        if gas["sfr"] is not None:
            try:
                gas_sfr = to_nd(gas["sfr"].to("Msun/yr"))
            except Exception:
                gas_sfr = to_nd(gas["sfr"])

            sfr_total = float(np.sum(gas_sfr[sel_gas]))
            sel_sf = sel_gas & (gas_sfr > 0.0)
            mgas_sf_native = np.sum(gas_mass[sel_sf])

            try:
                mgas_sf = float(ds.quan(mgas_sf_native, gas["mass_units"]).to("Msun"))
            except Exception:
                mgas_sf = float(mgas_sf_native)

            rcen = ds.quan(rhalf_star_native, gas["pos_units"])
            sfr_central = float(np.sum(gas_sfr[rgas <= float(rcen)]))
            f_sfr_central = safe_div(sfr_central, sfr_total)

            if np.sum(sel_sf) > 5:
                rhalf_tmp = half_mass_radius(gas_pos[sel_sf], gas_mass[sel_sf], to_nd(center_gas))
                try:
                    rhalf_gas_sf = float(ds.quan(rhalf_tmp, gas["pos_units"]).to("kpc"))
                except Exception:
                    rhalf_gas_sf = float(rhalf_tmp)

    mdm_ap = np.nan
    if dm is not None:
        center_dm = ds.arr(center_native, dm["pos_units"])
        rap_dm = ds.quan(aperture_factor * rhalf_star_native, dm["pos_units"])
        dm_pos = to_nd(dm["pos"].to(dm["pos_units"]))
        dm_mass = to_nd(dm["mass"])
        rdm = radius_3d(dm_pos, to_nd(center_dm))
        mdm_native = np.sum(dm_mass[rdm <= float(rap_dm)])
        try:
            mdm_ap = float(ds.quan(mdm_native, dm["mass_units"]).to("Msun"))
        except Exception:
            mdm_ap = float(mdm_native)

    try:
        rhalf_star_kpc = float(ds.quan(rhalf_star_native, star["pos_units"]).to("kpc"))
    except Exception:
        rhalf_star_kpc = float(rhalf_star_native)

    try:
        mstar = float(ds.quan(mstar_native, star["mass_units"]).to("Msun"))
    except Exception:
        mstar = float(mstar_native)

    mbar_ap = 0.0
    has_bar = False
    for val in [mstar, mgas]:
        if np.isfinite(val):
            mbar_ap += float(val)
            has_bar = True
    mbar_ap = mbar_ap if has_bar else np.nan

    mdwarf_proxy = 0.0
    has_proxy = False
    for val in [mbar_ap, mdm_ap]:
        if np.isfinite(val):
            mdwarf_proxy += float(val)
            has_proxy = True
    mdwarf_proxy = mdwarf_proxy if has_proxy else np.nan

    return {
        "rhalf_star_kpc": rhalf_star_kpc,
        "rhalf_old_kpc": rhalf_old,
        "rhalf_young_kpc": rhalf_young,
        "mstar_msun": mstar,
        "mstar_old_msun": mstar_old,
        "mstar_young_msun": mstar_young,
        "mdm_ap_msun": mdm_ap,
        "mbar_ap_msun": mbar_ap,
        "mdwarf_proxy_msun": mdwarf_proxy,
        "mgas_msun": mgas,
        "mgas_sf_msun": mgas_sf,
        "sfr_msun_per_yr": float(sfr_total) if np.isfinite(sfr_total) else np.nan,
        "sfr_central_msun_per_yr": float(sfr_central) if np.isfinite(sfr_central) else np.nan,
        "f_sfr_central": float(f_sfr_central) if np.isfinite(f_sfr_central) else np.nan,
        "rhalf_gas_sf_kpc": rhalf_gas_sf,
        "young_to_total_size_ratio": safe_div(rhalf_young, rhalf_star_kpc),
        "old_to_total_size_ratio": safe_div(rhalf_old, rhalf_star_kpc),
        "young_to_old_size_ratio": safe_div(rhalf_young, rhalf_old),
        "gas_sf_to_star_size_ratio": safe_div(rhalf_gas_sf, rhalf_star_kpc),
    }


# =============================================================================
# Ram pressure, directional, and tidal diagnostics
# =============================================================================

def build_cgm_shell_mask(
    ds: Any,
    gas: Dict[str, Any],
    dwarf_center_native: Sequence[float],
    rin_kpc: float,
    rout_kpc: float,
    cgm_temp_min: float,
    cgm_density_max: Optional[float],
    host_gas_ids: Optional[np.ndarray] = None,
    restrict_to_host_ids: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_q = ds.arr(dwarf_center_native, gas["pos_units"])
    gas_pos = to_nd(gas["pos"].to(gas["pos_units"]))

    rin = ds.quan(rin_kpc, "kpc").to(gas["pos_units"])
    rout = ds.quan(rout_kpc, "kpc").to(gas["pos_units"])
    rr = radius_3d(gas_pos, to_nd(center_q))
    sel_env = (rr >= float(rin)) & (rr <= float(rout))

    # Optional origin restriction: the local CGM/ram-pressure medium should be
    # host-origin gas, not stripped gas that originally belonged to the dwarf.
    if restrict_to_host_ids and host_gas_ids is not None:
        if gas.get("id") is None:
            warnings.warn("restrict_to_host_ids=True but no gas ParticleIDs field was found; using geometric CGM selection.")
        else:
            gids = _ids_to_int64(gas["id"])
            mid = ids_mask(gids, host_gas_ids)
            if mid is not None:
                sel_env &= mid

    if gas["temp"] is not None:
        try:
            temp_vals = to_nd(gas["temp"].to("K"))
            sel_env &= np.isfinite(temp_vals) & (temp_vals >= cgm_temp_min)
        except Exception:
            pass

    if cgm_density_max is not None and gas["dens"] is not None:
        try:
            dens_vals = to_nd(gas["dens"].to("g/cm**3"))
            sel_env &= np.isfinite(dens_vals) & (dens_vals <= cgm_density_max)
        except Exception:
            pass

    return sel_env, rr, gas_pos

def mass_weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    return float(np.sum(values[good] * weights[good]) / np.sum(weights[good]))


def summarize_environment(ds: Any, gas: Dict[str, Any], sel: np.ndarray, dwarf_vbulk_native: Optional[Sequence[float]]) -> Optional[Dict[str, Any]]:
    if np.sum(sel) < 1:
        return None

    gas_vel = to_nd(gas["vel"])
    gas_mass = to_nd(gas["mass"])
    gas_dens = gas["dens"]

    try:
        rho_val = float(gas_dens[sel].mean().to("g/cm**3"))
    except Exception:
        try:
            rho_val = float(np.mean(to_nd(gas_dens[sel])))
        except Exception:
            rho_val = np.nan

    v_cgm = weighted_velocity(gas_vel[sel], gas_mass[sel])
    out: Dict[str, Any] = {"rho_g_cm3": rho_val, "vbulk_native": v_cgm, "n_cells": int(np.sum(sel))}

    if dwarf_vbulk_native is None or v_cgm is None:
        out.update({"vrel_km_s": np.nan, "pram_dyn_cm2": np.nan, "wind_hat": None})
        return out

    vrel_vec = np.asarray(dwarf_vbulk_native, dtype=float) - np.asarray(v_cgm, dtype=float)
    try:
        vrel_native = float(np.sqrt(np.sum(vrel_vec**2)))
        vrel_kms = float(ds.arr([vrel_native], gas["vel_units"]).to("km/s")[0])
        vrel_cms = float(ds.arr([vrel_native], gas["vel_units"]).to("cm/s")[0])
        pram = rho_val * vrel_cms**2
    except Exception:
        vrel_kms = float(np.sqrt(np.sum(vrel_vec**2)))
        pram = np.nan

    out["vrel_km_s"] = vrel_kms
    out["pram_dyn_cm2"] = pram
    out["wind_hat"] = normalize_vector(vrel_vec)
    return out


def measure_local_cgm_and_pram(
    ds: Any,
    dwarf_center_native: Sequence[float],
    dwarf_vbulk_native: Optional[Sequence[float]],
    rin_kpc: float = 3.0,
    rout_kpc: float = 10.0,
    cgm_temp_min: float = 1.0e5,
    cgm_density_max: Optional[float] = None,
    estimator: str = "cone",
    cone_half_angle_deg: float = 35.0,
    cone_min_cells: int = 8,
    require_inflow_in_cone: bool = False,
    host_gas_ids: Optional[np.ndarray] = None,
    restrict_to_host_ids: bool = False,
) -> Optional[Dict[str, Any]]:
    gas = get_gas_data(ds)
    if gas is None or gas["vel"] is None or gas["dens"] is None:
        return None

    sel_shell, rr, gas_pos = build_cgm_shell_mask(
        ds, gas, dwarf_center_native, rin_kpc, rout_kpc, cgm_temp_min, cgm_density_max,
        host_gas_ids=host_gas_ids,
        restrict_to_host_ids=restrict_to_host_ids,
    )
    if np.sum(sel_shell) < 10:
        return None

    shell = summarize_environment(ds, gas, sel_shell, dwarf_vbulk_native)
    if shell is None:
        return None

    cone: Dict[str, Any] = {
        "rho_g_cm3": np.nan,
        "vrel_km_s": np.nan,
        "pram_dyn_cm2": np.nan,
        "n_cells": 0,
        "vbulk_native": None,
        "wind_hat": shell.get("wind_hat"),
    }

    if shell.get("wind_hat") is not None:
        center_q = ds.arr(dwarf_center_native, gas["pos_units"])
        rel_pos = gas_pos - to_nd(center_q)
        rr_safe = np.where(rr > 0, rr, np.nan)
        rel_hat = rel_pos / rr_safe[:, None]
        cos_half = np.cos(np.deg2rad(cone_half_angle_deg))
        mu = np.sum(rel_hat * shell["wind_hat"][None, :], axis=1)
        sel_cone = sel_shell & np.isfinite(mu) & (mu >= cos_half)

        if require_inflow_in_cone and dwarf_vbulk_native is not None:
            gas_vel = to_nd(gas["vel"])
            vrel_cells = gas_vel - np.asarray(dwarf_vbulk_native)[None, :]
            mu_v = np.sum(vrel_cells * shell["wind_hat"][None, :], axis=1)
            sel_cone &= np.isfinite(mu_v) & (mu_v < 0.0)

        if np.sum(sel_cone) >= cone_min_cells:
            cone_tmp = summarize_environment(ds, gas, sel_cone, dwarf_vbulk_native)
            if cone_tmp is not None:
                cone = cone_tmp
                cone["n_cells"] = int(np.sum(sel_cone))

    preferred = cone if (
        estimator == "cone"
        and cone["n_cells"] >= cone_min_cells
        and np.isfinite(cone["rho_g_cm3"])
    ) else shell

    wind_hat = preferred.get("wind_hat")
    if wind_hat is None:
        wind_hat = shell.get("wind_hat")

    return {
        "rho_cgm_g_cm3": preferred.get("rho_g_cm3", np.nan),
        "vrel_km_s": preferred.get("vrel_km_s", np.nan),
        "pram_dyn_cm2": preferred.get("pram_dyn_cm2", np.nan),
        "n_env": int(preferred.get("n_cells", 0)),
        "cgm_host_id_restricted": 1.0 if (restrict_to_host_ids and host_gas_ids is not None and gas.get("id") is not None) else 0.0,
        "rho_cgm_shell_g_cm3": shell.get("rho_g_cm3", np.nan),
        "vrel_shell_km_s": shell.get("vrel_km_s", np.nan),
        "pram_shell_dyn_cm2": shell.get("pram_dyn_cm2", np.nan),
        "n_shell": int(shell.get("n_cells", 0)),
        "rho_cgm_upstream_g_cm3": cone.get("rho_g_cm3", np.nan),
        "vrel_upstream_km_s": cone.get("vrel_km_s", np.nan),
        "pram_upstream_dyn_cm2": cone.get("pram_dyn_cm2", np.nan),
        "n_upstream": int(cone.get("n_cells", 0)),
        "upstream_x": np.nan if wind_hat is None else float(wind_hat[0]),
        "upstream_y": np.nan if wind_hat is None else float(wind_hat[1]),
        "upstream_z": np.nan if wind_hat is None else float(wind_hat[2]),
        "upstream_hat": wind_hat,
    }


def measure_directional_diagnostics(
    ds: Any,
    center_native: Sequence[float],
    rhalf_star_native: float,
    aperture_factor: float,
    upstream_hat: Optional[Sequence[float]],
) -> Dict[str, float]:
    out = {
        "mgas_leading_frac": np.nan,
        "mgas_sf_leading_frac": np.nan,
        "sfr_leading_frac": np.nan,
        "rho_leading_over_trailing": np.nan,
        "gas_com_along_upstream_kpc": np.nan,
    }
    if upstream_hat is None:
        return out

    gas = get_gas_data(ds)
    if gas is None:
        return out

    center_q = ds.arr(center_native, gas["pos_units"])
    rap = ds.quan(aperture_factor * rhalf_star_native, gas["pos_units"])
    gas_pos = to_nd(gas["pos"].to(gas["pos_units"]))
    gas_mass = to_nd(gas["mass"])
    rgas = radius_3d(gas_pos, to_nd(center_q))
    sel_gas = rgas <= float(rap)
    if np.sum(sel_gas) < 5:
        return out

    rel = gas_pos - to_nd(center_q)
    mu = np.sum(rel * np.asarray(upstream_hat)[None, :], axis=1)
    sel_lead = sel_gas & np.isfinite(mu) & (mu >= 0.0)
    sel_trail = sel_gas & np.isfinite(mu) & (mu < 0.0)

    out["mgas_leading_frac"] = safe_div(np.sum(gas_mass[sel_lead]), np.sum(gas_mass[sel_gas]))

    if gas["dens"] is not None:
        try:
            dens_vals = to_nd(gas["dens"].to("g/cm**3"))
        except Exception:
            dens_vals = to_nd(gas["dens"])
        rho_lead = mass_weighted_average(dens_vals[sel_lead], gas_mass[sel_lead])
        rho_trail = mass_weighted_average(dens_vals[sel_trail], gas_mass[sel_trail])
        out["rho_leading_over_trailing"] = safe_div(rho_lead, rho_trail)

    gas_com = weighted_center(gas_pos[sel_gas], gas_mass[sel_gas])
    if gas_com is not None:
        try:
            proj = np.sum((np.asarray(gas_com) - to_nd(center_q)) * np.asarray(upstream_hat))
            out["gas_com_along_upstream_kpc"] = float(ds.quan(proj, gas["pos_units"]).to("kpc"))
        except Exception:
            out["gas_com_along_upstream_kpc"] = float(np.sum((np.asarray(gas_com) - to_nd(center_q)) * np.asarray(upstream_hat)))

    if gas["sfr"] is not None:
        try:
            gas_sfr = to_nd(gas["sfr"].to("Msun/yr"))
        except Exception:
            gas_sfr = to_nd(gas["sfr"])
        sel_sf = sel_gas & (gas_sfr > 0.0)
        out["mgas_sf_leading_frac"] = safe_div(np.sum(gas_mass[sel_sf & sel_lead]), np.sum(gas_mass[sel_sf]))
        out["sfr_leading_frac"] = safe_div(np.sum(gas_sfr[sel_lead]), np.sum(gas_sfr[sel_gas]))

    return out


def distance_kpc(ds: Any, x_native: Sequence[float], y_native: Sequence[float], native_units: Any) -> float:
    try:
        dx = ds.arr(np.asarray(x_native) - np.asarray(y_native), native_units).to("kpc")
        return float(np.sqrt(np.sum(to_nd(dx)**2)))
    except Exception:
        return float(np.sqrt(np.sum((np.asarray(x_native) - np.asarray(y_native))**2)))


def vector_and_distance_kpc(ds: Any, x_native: Sequence[float], x_units: Any, y_native: Sequence[float], y_units: Any) -> Tuple[np.ndarray, float]:
    try:
        xk = ds.arr(x_native, x_units).to("kpc").to_ndarray()
        yk = ds.arr(y_native, y_units).to("kpc").to_ndarray()
    except Exception:
        xk = np.asarray(x_native, dtype=float)
        yk = np.asarray(y_native, dtype=float)
    vec = np.asarray(xk, dtype=float) - np.asarray(yk, dtype=float)
    dist = float(np.sqrt(np.sum(vec**2)))
    return vec, dist


def enclosed_masses_by_radius(ds: Any, center_native: Sequence[float], center_units: Any, radii_kpc: Sequence[float]) -> np.ndarray:
    radii_kpc = np.asarray(radii_kpc, dtype=float)
    order = np.argsort(radii_kpc)
    r_sorted = radii_kpc[order]
    sums = np.zeros_like(r_sorted, dtype=float)

    for comp in get_component_catalog(ds):
        try:
            center_q = ds.arr(center_native, center_units).to(comp["pos_units"])
            pos = to_nd(comp["pos"].to(comp["pos_units"]))
            mass = to_nd(comp["mass"])
        except Exception:
            continue

        rr_native = radius_3d(pos, to_nd(center_q))
        try:
            rr_kpc = ds.arr(rr_native, comp["pos_units"]).to("kpc").to_ndarray()
        except Exception:
            rr_kpc = np.asarray(rr_native, dtype=float)

        good = np.isfinite(rr_kpc) & np.isfinite(mass)
        if not np.any(good):
            continue

        rr_use = rr_kpc[good]
        mass_use = mass[good]
        idx = np.argsort(rr_use)
        rr_use = rr_use[idx]
        mass_use = mass_use[idx]

        try:
            mass_use = ds.arr(mass_use, comp["mass_units"]).to("Msun").to_ndarray()
        except Exception:
            mass_use = np.asarray(mass_use, dtype=float)

        mcum = np.cumsum(mass_use)
        inds = np.searchsorted(rr_use, r_sorted, side="right") - 1
        partial = np.zeros_like(r_sorted, dtype=float)
        valid = inds >= 0
        partial[valid] = mcum[inds[valid]]
        sums += partial

    out = np.full_like(radii_kpc, np.nan, dtype=float)
    out[order] = sums
    return out


def measure_tidal_diagnostics(
    ds: Any,
    dwarf_center_native: Sequence[float],
    dwarf_units: Any,
    host_center_native: Optional[Sequence[float]],
    host_units: Optional[Any],
    dwarf_props: Dict[str, float],
    upstream_hat: Optional[Sequence[float]] = None,
    tidal_delta_kpc: float = 1.0,
) -> Dict[str, float]:
    out = {
        "host_dx_kpc": np.nan,
        "host_dy_kpc": np.nan,
        "host_dz_kpc": np.nan,
        "host_radial_x": np.nan,
        "host_radial_y": np.nan,
        "host_radial_z": np.nan,
        "cos_wind_host_radial": np.nan,
        "mhost_enclosed_msun": np.nan,
        "mhost_inner_msun": np.nan,
        "mhost_outer_msun": np.nan,
        "host_mean_density_msun_kpc3": np.nan,
        "tidal_strength_msun_kpc3": np.nan,
        "tidal_radius_kpc": np.nan,
        "rhalf_star_over_rtidal": np.nan,
        "rhalf_gas_sf_over_rtidal": np.nan,
        "rhalf_young_over_rtidal": np.nan,
        "rhalf_old_over_rtidal": np.nan,
        "delta_g_tidal_cms2": np.nan,
        "tidal_gradient_s2": np.nan,
        "a_tide_rhalf_star_cms2": np.nan,
        "a_tide_rhalf_gas_sf_cms2": np.nan,
        "a_self_star_proxy_cms2": np.nan,
        "a_self_gas_proxy_cms2": np.nan,
        "tidal_to_self_star": np.nan,
        "tidal_to_self_gas": np.nan,
        "tidal_delta_kpc": np.nan,
    }

    if host_center_native is None or host_units is None:
        return out

    vec_kpc, dist_kpc = vector_and_distance_kpc(ds, dwarf_center_native, dwarf_units, host_center_native, host_units)
    if not np.isfinite(dist_kpc) or dist_kpc <= 0:
        return out

    out["host_dx_kpc"] = float(vec_kpc[0])
    out["host_dy_kpc"] = float(vec_kpc[1])
    out["host_dz_kpc"] = float(vec_kpc[2])

    host_radial_hat = normalize_vector(vec_kpc)
    if host_radial_hat is not None:
        out["host_radial_x"] = float(host_radial_hat[0])
        out["host_radial_y"] = float(host_radial_hat[1])
        out["host_radial_z"] = float(host_radial_hat[2])
    if upstream_hat is not None and host_radial_hat is not None:
        out["cos_wind_host_radial"] = float(np.sum(np.asarray(upstream_hat) * host_radial_hat))

    rhalf_star_kpc = dwarf_props.get("rhalf_star_kpc", np.nan)
    rhalf_gas_sf_kpc = dwarf_props.get("rhalf_gas_sf_kpc", np.nan)
    rhalf_young_kpc = dwarf_props.get("rhalf_young_kpc", np.nan)
    rhalf_old_kpc = dwarf_props.get("rhalf_old_kpc", np.nan)
    mdwarf_proxy = dwarf_props.get("mdwarf_proxy_msun", np.nan)
    mbar_ap = dwarf_props.get("mbar_ap_msun", np.nan)

    delta_kpc = tidal_delta_kpc
    if np.isfinite(rhalf_star_kpc):
        delta_kpc = max(delta_kpc, rhalf_star_kpc)
    delta_kpc = min(max(delta_kpc, 0.2), max(0.5 * dist_kpc, delta_kpc))

    r_in = max(dist_kpc - delta_kpc, 0.2)
    r_mid = dist_kpc
    r_out = dist_kpc + delta_kpc
    m_in, m_mid, m_out = enclosed_masses_by_radius(ds, host_center_native, host_units, [r_in, r_mid, r_out])

    out["mhost_inner_msun"] = float(m_in) if np.isfinite(m_in) else np.nan
    out["mhost_enclosed_msun"] = float(m_mid) if np.isfinite(m_mid) else np.nan
    out["mhost_outer_msun"] = float(m_out) if np.isfinite(m_out) else np.nan
    out["tidal_delta_kpc"] = float(delta_kpc)

    if np.isfinite(m_mid) and dist_kpc > 0:
        out["host_mean_density_msun_kpc3"] = float(m_mid / ((4.0 / 3.0) * np.pi * dist_kpc**3))
        out["tidal_strength_msun_kpc3"] = float(m_mid / dist_kpc**3)

    if np.isfinite(mdwarf_proxy) and np.isfinite(m_mid) and m_mid > 0:
        rtidal = dist_kpc * (mdwarf_proxy / (3.0 * m_mid))**(1.0 / 3.0)
        out["tidal_radius_kpc"] = float(rtidal)
        out["rhalf_star_over_rtidal"] = safe_div(rhalf_star_kpc, rtidal)
        out["rhalf_gas_sf_over_rtidal"] = safe_div(rhalf_gas_sf_kpc, rtidal)
        out["rhalf_young_over_rtidal"] = safe_div(rhalf_young_kpc, rtidal)
        out["rhalf_old_over_rtidal"] = safe_div(rhalf_old_kpc, rtidal)

    if np.isfinite(m_in) and np.isfinite(m_out):
        g_in = G_CGS * m_in * MSUN_CGS / (r_in * KPC_CGS)**2
        g_out = G_CGS * m_out * MSUN_CGS / (r_out * KPC_CGS)**2
        delta_g = abs(g_out - g_in)
        grad = delta_g / (2.0 * delta_kpc * KPC_CGS)
        out["delta_g_tidal_cms2"] = float(delta_g)
        out["tidal_gradient_s2"] = float(grad)
        if np.isfinite(rhalf_star_kpc):
            out["a_tide_rhalf_star_cms2"] = float(grad * rhalf_star_kpc * KPC_CGS)
        if np.isfinite(rhalf_gas_sf_kpc):
            out["a_tide_rhalf_gas_sf_cms2"] = float(grad * rhalf_gas_sf_kpc * KPC_CGS)

    if np.isfinite(mdwarf_proxy) and np.isfinite(rhalf_star_kpc) and rhalf_star_kpc > 0:
        a_self_star = G_CGS * mdwarf_proxy * MSUN_CGS / (rhalf_star_kpc * KPC_CGS)**2
        out["a_self_star_proxy_cms2"] = float(a_self_star)
        out["tidal_to_self_star"] = safe_div(out["a_tide_rhalf_star_cms2"], a_self_star)

    if np.isfinite(mbar_ap) and np.isfinite(rhalf_gas_sf_kpc) and rhalf_gas_sf_kpc > 0:
        a_self_gas = G_CGS * mbar_ap * MSUN_CGS / (rhalf_gas_sf_kpc * KPC_CGS)**2
        out["a_self_gas_proxy_cms2"] = float(a_self_gas)
        out["tidal_to_self_gas"] = safe_div(out["a_tide_rhalf_gas_sf_cms2"], a_self_gas)

    return out


# =============================================================================
# Snapshot-directory analysis
# =============================================================================

def analyze_loaded_snapshot(
    ds: Any,
    snapshot_name: str,
    dwarf_center_guess_code: Sequence[float],
    cfg: TrackingConfig,
    host_center_guess_code: Optional[Sequence[float]] = None,
    origin_ids: Optional[OriginIDCatalog] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """Analyze one already-loaded yt dataset and return (record, new_dwarf_guess, new_host_guess)."""
    t = get_time_gyr(ds)
    z = get_redshift(ds)

    dwarf = track_object_from_stars(ds, dwarf_center_guess_code, cfg.search_radius)
    if dwarf is None:
        return {"snapshot": snapshot_name, "time_gyr": t, "redshift": z}, None, None

    dwarf_center = dwarf["center"]
    dwarf_units = dwarf["pos_units"]
    dwarf_rhalf_native = dwarf["rhalf_star_native"]
    dwarf_vbulk = dwarf["vbulk_star_native"]

    try:
        dwarf_center_kpc = ds.arr(dwarf_center, dwarf_units).to("kpc").to_ndarray()
    except Exception:
        dwarf_center_kpc = np.asarray(dwarf_center, dtype=float)

    props = measure_dwarf_properties(
        ds,
        center_native=dwarf_center,
        rhalf_star_native=dwarf_rhalf_native,
        aperture_factor=cfg.aperture_factor,
        young_age_myr=cfg.young_age_myr,
    ) or {}

    host_center_now: Optional[np.ndarray] = None
    host_units: Optional[Any] = None
    new_host_guess: Optional[np.ndarray] = None
    dist_host_kpc = np.nan

    if host_center_guess_code is not None:
        if cfg.track_host:
            host = track_object_from_stars(ds, host_center_guess_code, cfg.host_search_radius)
            if host is not None:
                host_center_now = host["center"]
                host_units = host["pos_units"]
                dist_host_kpc = distance_kpc(ds, dwarf_center, host_center_now, dwarf_units)
                try:
                    new_host_guess = ds.arr(host_center_now, host_units).to("code_length").to_ndarray()
                except Exception:
                    new_host_guess = np.asarray(host_center_now, dtype=float)
        else:
            host_center_now = np.asarray(host_center_guess_code, dtype=float)
            host_units = "code_length"
            dist_host_kpc = distance_kpc(ds, dwarf_center, host_center_now, dwarf_units)
            new_host_guess = np.asarray(host_center_guess_code, dtype=float)

    env = measure_local_cgm_and_pram(
        ds,
        dwarf_center,
        dwarf_vbulk,
        rin_kpc=cfg.cgm_rin,
        rout_kpc=cfg.cgm_rout,
        cgm_temp_min=cfg.cgm_temp_min,
        cgm_density_max=cfg.cgm_density_max,
        estimator=cfg.cgm_estimator,
        cone_half_angle_deg=cfg.cone_half_angle_deg,
        cone_min_cells=cfg.cone_min_cells,
        require_inflow_in_cone=cfg.require_inflow_in_cone,
        host_gas_ids=None if origin_ids is None else origin_ids.host_gas_ids,
        restrict_to_host_ids=bool(cfg.use_particle_ids and cfg.restrict_cgm_to_host_gas_ids),
    )
    if env is None:
        env = {
            "rho_cgm_g_cm3": np.nan, "vrel_km_s": np.nan, "pram_dyn_cm2": np.nan, "n_env": 0, "cgm_host_id_restricted": 0.0,
            "rho_cgm_shell_g_cm3": np.nan, "vrel_shell_km_s": np.nan, "pram_shell_dyn_cm2": np.nan, "n_shell": 0,
            "rho_cgm_upstream_g_cm3": np.nan, "vrel_upstream_km_s": np.nan, "pram_upstream_dyn_cm2": np.nan, "n_upstream": 0,
            "upstream_x": np.nan, "upstream_y": np.nan, "upstream_z": np.nan,
            "upstream_hat": None,
        }

    upstream_hat = env.get("upstream_hat")
    directional = measure_directional_diagnostics(ds, dwarf_center, dwarf_rhalf_native, cfg.aperture_factor, upstream_hat)
    tidal = measure_tidal_diagnostics(
        ds,
        dwarf_center,
        dwarf_units,
        host_center_now,
        host_units,
        props,
        upstream_hat=upstream_hat,
        tidal_delta_kpc=cfg.tidal_delta_kpc,
    )

    id_diag = {}
    if cfg.use_particle_ids and cfg.measure_id_stripping:
        id_diag = measure_origin_id_diagnostics(
            ds,
            dwarf_center,
            dwarf_units,
            dwarf_rhalf_native,
            cfg.aperture_factor,
            origin_ids,
            stripped_radius_factor=cfg.stripped_radius_factor,
            stripped_use_tidal_radius=cfg.stripped_use_tidal_radius,
            tidal_radius_kpc=tidal.get("tidal_radius_kpc", np.nan),
        )

    record: Dict[str, Any] = {
        "snapshot": snapshot_name,
        "time_gyr": t,
        "redshift": z,
        "x_kpc": float(dwarf_center_kpc[0]) if len(dwarf_center_kpc) > 0 else np.nan,
        "y_kpc": float(dwarf_center_kpc[1]) if len(dwarf_center_kpc) > 1 else np.nan,
        "z_kpc": float(dwarf_center_kpc[2]) if len(dwarf_center_kpc) > 2 else np.nan,
        "dist_host_kpc": dist_host_kpc,
    }
    record.update(props)
    record.update(id_diag)
    record.update({k: v for k, v in env.items() if k != "upstream_hat"})
    record.update(directional)
    record.update(tidal)

    try:
        new_dwarf_guess = ds.arr(dwarf_center, dwarf_units).to("code_length").to_ndarray()
    except Exception:
        new_dwarf_guess = np.asarray(dwarf_center, dtype=float)

    return record, new_dwarf_guess, new_host_guess


def analyze_snapshot_file(
    snapshot_path: str,
    dwarf_center_guess_code: Sequence[float],
    cfg: TrackingConfig,
    host_center_guess_code: Optional[Sequence[float]] = None,
    origin_ids: Optional[OriginIDCatalog] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    yt = try_import_yt()
    ds = yt.load(snapshot_path)
    return analyze_loaded_snapshot(
        ds,
        os.path.basename(snapshot_path),
        dwarf_center_guess_code,
        cfg,
        host_center_guess_code=host_center_guess_code,
        origin_ids=origin_ids,
    )


def analyze_snapshot_directory(
    snapdir: str,
    cfg: TrackingConfig,
    outdir: Optional[str] = None,
    table_name: str = "dwarf_ram_pressure_evolution_final.txt",
    write_table: bool = True,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    snapdir = os.path.abspath(snapdir)
    outdir = outdir or os.path.join(snapdir, "ram_pressure_analysis_final")
    ensure_dir(outdir)

    snapfiles = find_snapshots(snapdir)
    if cfg.max_snaps is not None:
        snapfiles = snapfiles[: cfg.max_snaps]
    if len(snapfiles) == 0:
        raise FileNotFoundError("No snapshot_XXX.hdf5 files found in {0}".format(snapdir))

    origin_ids = resolve_origin_id_catalog(snapfiles, cfg, outdir, verbose=verbose)

    dwarf_guess = np.asarray(cfg.dwarf_init_center, dtype=float)
    host_guess = np.asarray(cfg.host_center, dtype=float) if cfg.host_center is not None else None

    records: List[Dict[str, Any]] = []
    for i, snap in enumerate(snapfiles):
        if verbose:
            print("[{0:04d}/{1:04d}] {2}".format(i + 1, len(snapfiles), os.path.basename(snap)))
        try:
            rec, new_dwarf, new_host = analyze_snapshot_file(snap, dwarf_guess, cfg, host_guess, origin_ids=origin_ids)
        except Exception as exc:
            print("[WARN] Failed on {0}: {1}".format(os.path.basename(snap), exc))
            rec = {"snapshot": os.path.basename(snap), "time_gyr": np.nan, "redshift": np.nan}
            new_dwarf = None
            new_host = None

        records.append(rec)
        if new_dwarf is not None:
            dwarf_guess = np.asarray(new_dwarf, dtype=float)
        if new_host is not None:
            host_guess = np.asarray(new_host, dtype=float)

    table_path: Optional[str] = None
    if write_table:
        table_path = os.path.join(outdir, table_name)
        write_records_table(records, table_path, OUTPUT_COLUMNS)

    return records, table_path


# =============================================================================
# Run-table preparation and comparison metrics
# =============================================================================

def first_index_true(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size else None


def determine_peri_index(cols: Dict[str, np.ndarray]) -> int:
    if "dist_host_kpc" not in cols:
        return 0
    dist = np.asarray(cols["dist_host_kpc"], dtype=float)
    if not np.any(np.isfinite(dist)):
        return 0
    return int(np.nanargmin(dist))


def determine_entry_index(cols: Dict[str, np.ndarray], entry_radius_kpc: float = 250.0) -> int:
    if "dist_host_kpc" not in cols:
        return 0
    dist = np.asarray(cols["dist_host_kpc"], dtype=float)
    idx = first_index_true(np.isfinite(dist) & (dist <= entry_radius_kpc))
    return 0 if idx is None else idx


def determine_gas_loss_index(
    cols: Dict[str, np.ndarray],
    gas_loss_fraction: float = 0.05,
    gas_loss_mode: str = "total",
    min_gas_floor_abs: float = 1e5,
    ref_stop_idx: Optional[int] = None,
) -> Optional[int]:
    if gas_loss_mode == "sf" and "mgas_sf_msun" in cols:
        gas = np.asarray(cols["mgas_sf_msun"], dtype=float)
    elif "mgas_msun" in cols:
        gas = np.asarray(cols["mgas_msun"], dtype=float)
    else:
        return None

    if gas.size == 0 or not np.any(np.isfinite(gas)):
        return None

    if ref_stop_idx is None:
        ref_stop_idx = min(len(gas) - 1, max(2, len(gas) // 3))
    ref_stop_idx = max(0, min(int(ref_stop_idx), len(gas) - 1))

    ref = np.nanmax(gas[: ref_stop_idx + 1])
    if not np.isfinite(ref) or ref <= 0:
        return None

    threshold = max(gas_loss_fraction * ref, min_gas_floor_abs)
    return first_index_true(np.isfinite(gas) & (gas <= threshold))


def compaction_rate_between(t: Sequence[float], rhalf: Sequence[float], i0: int, i1: int, normalize_radius_at: str = "start") -> float:
    t = np.asarray(t, dtype=float)
    rhalf = np.asarray(rhalf, dtype=float)
    if i0 is None or i1 is None or i1 <= i0:
        return np.nan
    dt = float(t[i1] - t[i0])
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    r0 = float(rhalf[i0]) if normalize_radius_at == "start" else float(rhalf[i1])
    if not np.isfinite(r0) or r0 <= 0:
        return np.nan
    return float((rhalf[i1] - rhalf[i0]) / (r0 * dt))


def prepare_run(
    table_path: str,
    label: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
    smooth: int = 5,
    pre_window: Tuple[float, float] = (-0.5, -0.1),
    force_window: Tuple[float, float] = (-0.3, 0.3),
    response_window: Tuple[float, float] = (-0.1, 1.0),
    entry_radius_kpc: float = 250.0,
    gas_loss_fraction: float = 0.05,
    gas_loss_mode: str = "total",
    min_gas_floor_abs: float = 1e5,
) -> RunData:
    tab = read_table(table_path)
    cols = table_to_cols(tab)

    if "time_gyr" not in cols:
        raise RuntimeError("{0} does not contain time_gyr".format(table_path))

    t = np.asarray(cols["time_gyr"], dtype=float)
    if "dist_host_kpc" in cols and np.any(np.isfinite(cols["dist_host_kpc"])):
        i_peri = int(np.nanargmin(cols["dist_host_kpc"]))
    else:
        i_peri = 0
    t_peri = float(t[i_peri]) if len(t) else np.nan
    tau = t - t_peri

    label = label or (Path(table_path).parent.name or Path(table_path).stem)
    final_idx = len(t) - 1
    entry_idx = determine_entry_index(cols, entry_radius_kpc)
    peri_idx = determine_peri_index(cols)
    ref_stop_idx = max(entry_idx, min(peri_idx, final_idx))
    gas_loss_idx = determine_gas_loss_index(cols, gas_loss_fraction, gas_loss_mode, min_gas_floor_abs, ref_stop_idx)

    epochs: Dict[str, Any] = {
        "entry_idx": entry_idx,
        "peri_idx": peri_idx,
        "gas_loss_idx": -1 if gas_loss_idx is None else gas_loss_idx,
        "final_idx": final_idx,
        "entry_time_gyr": float(t[entry_idx]) if final_idx >= 0 else np.nan,
        "peri_time_gyr": float(t[peri_idx]) if final_idx >= 0 else np.nan,
        "gas_loss_time_gyr": np.nan if gas_loss_idx is None else float(t[gas_loss_idx]),
        "final_time_gyr": float(t[final_idx]) if final_idx >= 0 else np.nan,
        "entry_snapshot": str(cols["snapshot"][entry_idx]) if "snapshot" in cols and final_idx >= 0 else "",
        "peri_snapshot": str(cols["snapshot"][peri_idx]) if "snapshot" in cols and final_idx >= 0 else "",
        "gas_loss_snapshot": "" if gas_loss_idx is None or "snapshot" not in cols else str(cols["snapshot"][gas_loss_idx]),
        "final_snapshot": str(cols["snapshot"][final_idx]) if "snapshot" in cols and final_idx >= 0 else "",
    }

    # Smoothed and normalized columns for notebooks/plots.
    for name in [
        "dist_host_kpc", "pram_upstream_dyn_cm2", "pram_shell_dyn_cm2", "pram_dyn_cm2",
        "tidal_to_self_star", "tidal_to_self_gas", "f_sfr_central",
        "sfr_leading_frac", "rho_leading_over_trailing",
        "rhalf_young_kpc", "rhalf_old_kpc", "rhalf_star_kpc", "rhalf_gas_sf_kpc",
        "mstar_dwarf_origin_core_msun", "mstar_dwarf_origin_stripped_msun", "fstar_dwarf_origin_stripped",
        "mgas_dwarf_origin_core_msun", "mgas_dwarf_origin_stripped_msun", "fgas_dwarf_origin_stripped",
        "mgas_msun", "mgas_sf_msun", "gas_sf_to_star_size_ratio",
        "young_to_old_size_ratio", "rhalf_star_over_rtidal",
        "rhalf_old_over_rtidal", "rhalf_young_over_rtidal", "rhalf_gas_sf_over_rtidal",
        "gas_com_along_upstream_kpc", "sfr_msun_per_yr",
    ]:
        if name in cols:
            cols[name + "__smooth"] = running_nanmedian(cols[name], smooth)

    for name in [
        "rhalf_young_kpc", "rhalf_old_kpc", "rhalf_star_kpc", "rhalf_gas_sf_kpc",
        "mgas_msun", "mgas_sf_msun", "sfr_msun_per_yr", "f_sfr_central",
        "mstar_dwarf_origin_core_msun", "mstar_dwarf_origin_stripped_msun", "mgas_dwarf_origin_core_msun", "mgas_dwarf_origin_stripped_msun",
        "gas_sf_to_star_size_ratio", "young_to_old_size_ratio",
    ]:
        if name in cols:
            cols[name + "__norm"] = normalize_to_window(tau, cols[name], pre_window[0], pre_window[1])
            cols[name + "__norm__smooth"] = running_nanmedian(cols[name + "__norm"], smooth)

    metrics: Dict[str, float] = {
        "t_peri_gyr": t_peri,
        "d_peri_kpc": float(cols["dist_host_kpc"][i_peri]) if "dist_host_kpc" in cols else np.nan,
    }

    if "fstar_dwarf_origin_stripped" in cols:
        metrics["fstar_stripped_final"] = float(cols["fstar_dwarf_origin_stripped"][final_idx]) if final_idx >= 0 else np.nan
        metrics["fstar_stripped_peri"] = float(cols["fstar_dwarf_origin_stripped"][peri_idx]) if final_idx >= 0 else np.nan
    if "fgas_dwarf_origin_stripped" in cols:
        metrics["fgas_stripped_final"] = float(cols["fgas_dwarf_origin_stripped"][final_idx]) if final_idx >= 0 else np.nan
        metrics["fgas_stripped_peri"] = float(cols["fgas_dwarf_origin_stripped"][peri_idx]) if final_idx >= 0 else np.nan

    pram_col = "pram_upstream_dyn_cm2" if "pram_upstream_dyn_cm2" in cols else "pram_dyn_cm2"
    if pram_col in cols:
        metrics["pram_peak"], metrics["tau_pram_peak"] = argmax_in_window(tau, cols[pram_col], force_window[0], force_window[1])
    else:
        metrics["pram_peak"], metrics["tau_pram_peak"] = np.nan, np.nan

    if "tidal_to_self_star" in cols:
        metrics["tidal_peak"], metrics["tau_tidal_peak"] = argmax_in_window(tau, cols["tidal_to_self_star"], force_window[0], force_window[1])
    else:
        metrics["tidal_peak"], metrics["tau_tidal_peak"] = np.nan, np.nan

    for key, colname in [
        ("young_norm_min", "rhalf_young_kpc__norm"),
        ("old_norm_min", "rhalf_old_kpc__norm"),
        ("star_norm_min", "rhalf_star_kpc__norm"),
        ("gas_sf_norm_min", "rhalf_gas_sf_kpc__norm"),
        ("young_to_old_norm_min", "young_to_old_size_ratio__norm"),
        ("gas_sf_to_star_norm_min", "gas_sf_to_star_size_ratio__norm"),
    ]:
        if colname in cols:
            value, tau_at = argmin_in_window(tau, cols[colname], response_window[0], response_window[1])
            metrics[key] = value
            metrics["tau_" + key] = tau_at
        else:
            metrics[key] = np.nan
            metrics["tau_" + key] = np.nan

    # Gas and stellar channel metrics.
    for col, prefix in [
        ("mgas_msun", "gas_total"),
        ("mgas_sf_msun", "gas_sf"),
        ("rhalf_gas_sf_kpc", "rhalf_gas_sf"),
        ("f_sfr_central", "f_sfr_central"),
        ("gas_sf_to_star_size_ratio", "gas_sf_to_star_ratio"),
        ("rhalf_star_kpc", "rhalf_star"),
        ("mstar_msun", "mstar"),
    ]:
        if col in cols and final_idx >= 0:
            metrics[prefix + "_change_entry_to_peri"] = safe_frac_delta(cols[col][peri_idx], cols[col][entry_idx])
            metrics[prefix + "_change_entry_to_final"] = safe_frac_delta(cols[col][final_idx], cols[col][entry_idx])
        else:
            metrics[prefix + "_change_entry_to_peri"] = np.nan
            metrics[prefix + "_change_entry_to_final"] = np.nan

    if "rho_leading_over_trailing" in cols:
        lo = max(entry_idx, peri_idx - 3)
        hi = min(final_idx + 1, peri_idx + 4)
        metrics["rho_lead_trail_peak_near_peri"] = float(np.nanmax(cols["rho_leading_over_trailing"][lo:hi]))
    else:
        metrics["rho_lead_trail_peak_near_peri"] = np.nan

    if "rhalf_star_kpc" in cols:
        rhalf = np.asarray(cols["rhalf_star_kpc"], dtype=float)
        if gas_loss_idx is not None and gas_loss_idx > entry_idx:
            metrics["compaction_rate_entry_to_gasloss_gyr-1"] = compaction_rate_between(t, rhalf, entry_idx, gas_loss_idx)
            metrics["compaction_rate_nogas_gyr-1"] = compaction_rate_between(t, rhalf, gas_loss_idx, final_idx) if final_idx > gas_loss_idx else np.nan
        else:
            metrics["compaction_rate_entry_to_gasloss_gyr-1"] = compaction_rate_between(t, rhalf, entry_idx, final_idx)
            metrics["compaction_rate_nogas_gyr-1"] = np.nan
        metrics["delta_rhalf_over_rentry"] = safe_frac_delta(rhalf[final_idx], rhalf[entry_idx])
    else:
        metrics["compaction_rate_entry_to_gasloss_gyr-1"] = np.nan
        metrics["compaction_rate_nogas_gyr-1"] = np.nan
        metrics["delta_rhalf_over_rentry"] = np.nan

    return RunData(label, table_path, snapshot_dir, tab, cols, t, tau, t_peri, epochs, metrics)


def prepare_runs(
    table_paths: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    snapshot_dirs: Optional[Sequence[Optional[str]]] = None,
    **kwargs: Any,
) -> List[RunData]:
    if labels is not None and len(labels) != len(table_paths):
        raise ValueError("labels must have the same length as table_paths")
    if snapshot_dirs is not None and len(snapshot_dirs) != len(table_paths):
        raise ValueError("snapshot_dirs must have the same length as table_paths")

    out: List[RunData] = []
    for i, path in enumerate(table_paths):
        label = labels[i] if labels is not None else None
        snapdir = snapshot_dirs[i] if snapshot_dirs is not None else None
        out.append(prepare_run(path, label=label, snapshot_dir=snapdir, **kwargs))
    return out


def metrics_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {"label": run.label, "table_path": run.table_path}
        row.update(run.epochs)
        row.update(run.metrics)
        rows.append(row)
    return rows


# =============================================================================
# Snapshot-level profiles and mass budgets
# =============================================================================

def get_particle_block(ds: Any, ptype: str) -> Tuple[Optional[Any], Optional[Any]]:
    ad = ds.all_data()
    fields = set(ds.field_list) | set(ds.derived_field_list)

    if ptype == "stars":
        pos_candidates = [("PartType4", "Coordinates"), ("stars", "particle_position"), ("all", "particle_position")]
        mass_candidates = [("PartType4", "Masses"), ("PartType4", "particle_mass"), ("stars", "particle_mass"), ("all", "particle_mass")]
    elif ptype == "gas":
        pos_candidates = [("PartType0", "Coordinates"), ("gas", "particle_position"), ("all", "particle_position")]
        mass_candidates = [("PartType0", "Masses"), ("PartType0", "particle_mass"), ("gas", "particle_mass"), ("gas", "cell_mass"), ("all", "particle_mass")]
    else:
        raise ValueError("ptype must be 'stars' or 'gas'")

    pos = None
    mass = None
    for f in pos_candidates:
        if f in fields:
            pos = ad[f]
            break
    for f in mass_candidates:
        if f in fields:
            mass = ad[f]
            break
    return pos, mass


def load_particle_positions_masses(snapshot_path: str, ptype: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    yt = try_import_yt()
    ds = yt.load(str(snapshot_path))
    pos, mass = get_particle_block(ds, ptype)
    if pos is None or mass is None:
        return None
    try:
        pos_kpc = to_nd(pos.to("kpc"))
    except Exception:
        pos_kpc = to_nd(pos)
    try:
        mass_msun = to_nd(mass.to("Msun"))
    except Exception:
        mass_msun = to_nd(mass)
    if pos_kpc.ndim != 2 or pos_kpc.shape[1] < 3:
        return None
    return pos_kpc[:, :3], np.asarray(mass_msun, dtype=float)


def mass_inside_outside_from_snapshot(
    snapshot_path: str,
    center_kpc: Sequence[float],
    split_radius_kpc: float,
    ptype: str = "stars",
) -> Optional[Tuple[float, float]]:
    block = load_particle_positions_masses(snapshot_path, ptype)
    if block is None:
        return None
    pos_kpc, mass_msun = block
    r = radius_3d(pos_kpc, np.asarray(center_kpc, dtype=float))
    sel = np.isfinite(r) & np.isfinite(mass_msun) & (mass_msun > 0)
    if not np.any(sel):
        return None
    r = r[sel]
    m = mass_msun[sel]
    return float(np.sum(m[r <= split_radius_kpc])), float(np.sum(m[r > split_radius_kpc]))


def weighted_cumulative_profile(r: np.ndarray, m: np.ndarray, nbins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    sel = np.isfinite(r) & np.isfinite(m) & (r > 0) & (m > 0)
    if np.sum(sel) < 5:
        return np.array([]), np.array([])
    r = r[sel]
    m = m[sel]
    rb = np.logspace(np.log10(np.min(r)), np.log10(np.max(r)), nbins)
    cum = np.array([np.sum(m[r <= rr]) for rr in rb], dtype=float)
    return rb, cum


def binned_density_times_r2(r: np.ndarray, m: np.ndarray, nbins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    sel = np.isfinite(r) & np.isfinite(m) & (r > 0) & (m > 0)
    if np.sum(sel) < 10:
        return np.array([]), np.array([])
    r = r[sel]
    m = m[sel]
    edges = np.logspace(np.log10(np.min(r)), np.log10(np.max(r)), nbins + 1)
    mids = np.sqrt(edges[:-1] * edges[1:])
    prof = np.full(nbins, np.nan, dtype=float)
    for i in range(nbins):
        inbin = (r >= edges[i]) & (r < edges[i + 1])
        if np.any(inbin):
            shell_mass = np.sum(m[inbin])
            vol = (4.0 / 3.0) * math.pi * (edges[i + 1]**3 - edges[i]**3)
            prof[i] = shell_mass / vol * mids[i]**2
    return mids, prof


def extract_stellar_profile_from_snapshot(
    snapshot_path: str,
    center_kpc: Sequence[float],
    max_radius_kpc: Optional[float] = None,
    nbins_cum: int = 50,
    nbins_rho: int = 40,
) -> Optional[Dict[str, np.ndarray]]:
    block = load_particle_positions_masses(snapshot_path, "stars")
    if block is None:
        return None
    pos_kpc, mass_msun = block
    r = radius_3d(pos_kpc, np.asarray(center_kpc, dtype=float))
    if max_radius_kpc is not None and np.isfinite(max_radius_kpc):
        sel = r <= float(max_radius_kpc)
        r = r[sel]
        mass_msun = mass_msun[sel]

    rb_cum, cum = weighted_cumulative_profile(r, mass_msun, nbins=nbins_cum)
    rb_rho, rho_r2 = binned_density_times_r2(r, mass_msun, nbins=nbins_rho)

    return {
        "r_all_kpc": r,
        "m_all_msun": mass_msun,
        "r_cum_kpc": rb_cum,
        "cum_mass_msun": cum,
        "r_rho_kpc": rb_rho,
        "rho_r2_msun_kpc-1": rho_r2,
    }


def choose_split_radius(run: RunData, split_mode: str = "final_rhalf", split_radius_kpc: Optional[float] = None) -> Optional[float]:
    if split_mode == "fixed":
        return split_radius_kpc
    if "rhalf_star_kpc" not in run.cols:
        return None
    final_idx = int(run.epochs["final_idx"])
    r = float(run.cols["rhalf_star_kpc"][final_idx])
    return r if np.isfinite(r) and r > 0 else None


def mass_budget_for_run(
    run: RunData,
    split_mode: str = "final_rhalf",
    split_radius_kpc: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if run.snapshot_dir is None or "snapshot" not in run.cols:
        return None

    Rsplit = choose_split_radius(run, split_mode, split_radius_kpc)
    if Rsplit is None or not np.isfinite(Rsplit) or Rsplit <= 0:
        return None

    entry_idx = int(run.epochs["entry_idx"])
    peri_idx = int(run.epochs["peri_idx"])
    gas_loss_idx = int(run.epochs["gas_loss_idx"])
    final_idx = int(run.epochs["final_idx"])
    comp_idx = gas_loss_idx if gas_loss_idx >= 0 else final_idx
    phase_label = "gas_loss" if gas_loss_idx >= 0 else "final"

    idx_map = {"entry": entry_idx, "peri": peri_idx, "comp": comp_idx, "final": final_idx}
    results: Dict[str, Any] = {"label": run.label, "split_radius_kpc": float(Rsplit), "comp_epoch_kind": phase_label}

    for epoch_name, idx in idx_map.items():
        snap = Path(run.snapshot_dir) / str(run.cols["snapshot"][idx])
        if not all(k in run.cols for k in ["x_kpc", "y_kpc", "z_kpc"]):
            return None
        center = [float(run.cols["x_kpc"][idx]), float(run.cols["y_kpc"][idx]), float(run.cols["z_kpc"][idx])]

        gas_io = mass_inside_outside_from_snapshot(str(snap), center, Rsplit, "gas")
        star_io = mass_inside_outside_from_snapshot(str(snap), center, Rsplit, "stars")
        if gas_io is None or star_io is None:
            return None

        gin, gout = gas_io
        sin, sout = star_io
        results["gas_inner_{0}_msun".format(epoch_name)] = gin
        results["gas_outer_{0}_msun".format(epoch_name)] = gout
        results["star_inner_{0}_msun".format(epoch_name)] = sin
        results["star_outer_{0}_msun".format(epoch_name)] = sout

    mstar_entry = results["star_inner_entry_msun"] + results["star_outer_entry_msun"]
    mgas_entry = results["gas_inner_entry_msun"] + results["gas_outer_entry_msun"]

    results["dgas_inner_entry_to_peri_over_mgas_entry"] = safe_div(results["gas_inner_peri_msun"] - results["gas_inner_entry_msun"], mgas_entry)
    results["dgas_outer_entry_to_peri_over_mgas_entry"] = safe_div(results["gas_outer_peri_msun"] - results["gas_outer_entry_msun"], mgas_entry)
    results["dgas_inner_entry_to_comp_over_mgas_entry"] = safe_div(results["gas_inner_comp_msun"] - results["gas_inner_entry_msun"], mgas_entry)
    results["dgas_outer_entry_to_comp_over_mgas_entry"] = safe_div(results["gas_outer_comp_msun"] - results["gas_outer_entry_msun"], mgas_entry)

    results["dstar_inner_entry_to_comp_over_mstar_entry"] = safe_div(results["star_inner_comp_msun"] - results["star_inner_entry_msun"], mstar_entry)
    results["dstar_outer_entry_to_comp_over_mstar_entry"] = safe_div(results["star_outer_comp_msun"] - results["star_outer_entry_msun"], mstar_entry)
    results["dstar_outer_postcomp_over_mstar_entry"] = safe_div(results["star_outer_final_msun"] - results["star_outer_comp_msun"], mstar_entry)

    return results


def profile_pair_for_run(run: RunData, max_radius_kpc: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Return entry/final stellar profile arrays for a run with snapshots."""
    if run.snapshot_dir is None or "snapshot" not in run.cols:
        return None
    if not all(k in run.cols for k in ["x_kpc", "y_kpc", "z_kpc"]):
        return None

    entry_idx = int(run.epochs["entry_idx"])
    final_idx = int(run.epochs["final_idx"])

    entry_snap = str(Path(run.snapshot_dir) / str(run.cols["snapshot"][entry_idx]))
    final_snap = str(Path(run.snapshot_dir) / str(run.cols["snapshot"][final_idx]))

    center_entry = [float(run.cols["x_kpc"][entry_idx]), float(run.cols["y_kpc"][entry_idx]), float(run.cols["z_kpc"][entry_idx])]
    center_final = [float(run.cols["x_kpc"][final_idx]), float(run.cols["y_kpc"][final_idx]), float(run.cols["z_kpc"][final_idx])]

    prof_entry = extract_stellar_profile_from_snapshot(entry_snap, center_entry, max_radius_kpc=max_radius_kpc)
    prof_final = extract_stellar_profile_from_snapshot(final_snap, center_final, max_radius_kpc=max_radius_kpc)
    if prof_entry is None or prof_final is None:
        return None
    return {
        "label": run.label,
        "entry_profile": prof_entry,
        "final_profile": prof_final,
        "entry_snapshot": entry_snap,
        "final_snapshot": final_snap,
    }


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analysis utilities for controlled Gadget4 dwarf ram-pressure runs.")
    sub = p.add_subparsers(dest="command", required=True)

    p_track = sub.add_parser("track", help="Track one snapshot directory and write an evolution table.")
    p_track.add_argument("path", help="Directory containing snapshot_XXX.hdf5 files")
    p_track.add_argument("--outdir", default=None)
    p_track.add_argument("--table-name", default="dwarf_ram_pressure_evolution_final.txt")
    p_track.add_argument("--dwarf-init-center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p_track.add_argument("--host-center", nargs=3, type=float, default=None, metavar=("Xh", "Yh", "Zh"))
    p_track.add_argument("--track-host", action="store_true")
    p_track.add_argument("--search-radius", type=float, default=15.0)
    p_track.add_argument("--host-search-radius", type=float, default=30.0)
    p_track.add_argument("--aperture-factor", type=float, default=2.0)
    p_track.add_argument("--young-age-myr", type=float, default=100.0)
    p_track.add_argument("--cgm-rin", type=float, default=3.0)
    p_track.add_argument("--cgm-rout", type=float, default=10.0)
    p_track.add_argument("--cgm-temp-min", type=float, default=1e5)
    p_track.add_argument("--cgm-density-max", type=float, default=None)
    p_track.add_argument("--cgm-estimator", choices=["cone", "shell"], default="cone")
    p_track.add_argument("--cone-half-angle-deg", type=float, default=35.0)
    p_track.add_argument("--cone-min-cells", type=int, default=8)
    p_track.add_argument("--require-inflow-in-cone", action="store_true")
    p_track.add_argument("--tidal-delta-kpc", type=float, default=1.0)
    p_track.add_argument("--max-snaps", type=int, default=None)
    p_track.add_argument("--use-particle-ids", action="store_true", help="Use ParticleIDs to define particle origin and host-origin CGM.")
    p_track.add_argument("--origin-ids-path", default=None, help="Path to .npz origin ID catalog. If missing and auto-create is enabled, it is created from the first snapshot.")
    p_track.add_argument("--no-auto-create-origin-ids", action="store_true")
    p_track.add_argument("--origin-all-stars-are-dwarf", action="store_true", default=True)
    p_track.add_argument("--origin-dwarf-radius-kpc", type=float, default=None)
    p_track.add_argument("--origin-dwarf-aperture-factor", type=float, default=5.0)
    p_track.add_argument("--origin-min-radius-kpc", type=float, default=5.0)
    p_track.add_argument("--no-restrict-cgm-to-host-gas-ids", action="store_true")
    p_track.add_argument("--no-measure-id-stripping", action="store_true")
    p_track.add_argument("--stripped-radius-factor", type=float, default=2.0)
    p_track.add_argument("--no-stripped-use-tidal-radius", action="store_true")

    p_met = sub.add_parser("metrics", help="Prepare comparison metrics from existing evolution tables.")
    p_met.add_argument("tables", nargs="+")
    p_met.add_argument("--labels", nargs="*", default=None)
    p_met.add_argument("--snapshot-dirs", nargs="*", default=None)
    p_met.add_argument("--outdir", default="compare_metrics")
    p_met.add_argument("--smooth", type=int, default=5)
    p_met.add_argument("--pre-window", nargs=2, type=float, default=(-0.5, -0.1))
    p_met.add_argument("--force-window", nargs=2, type=float, default=(-0.3, 0.3))
    p_met.add_argument("--response-window", nargs=2, type=float, default=(-0.1, 1.0))
    p_met.add_argument("--entry-radius-kpc", type=float, default=250.0)
    p_met.add_argument("--gas-loss-fraction", type=float, default=0.05)
    p_met.add_argument("--gas-loss-mode", choices=["total", "sf"], default="total")
    p_met.add_argument("--min-gas-floor-abs", type=float, default=1e5)

    p_bud = sub.add_parser("budgets", help="Compute snapshot-level inner/outer gas and stellar mass budgets.")
    p_bud.add_argument("tables", nargs="+")
    p_bud.add_argument("--labels", nargs="*", default=None)
    p_bud.add_argument("--snapshot-dirs", nargs="+", required=True)
    p_bud.add_argument("--outdir", default="mass_budgets")
    p_bud.add_argument("--entry-radius-kpc", type=float, default=250.0)
    p_bud.add_argument("--gas-loss-fraction", type=float, default=0.05)
    p_bud.add_argument("--gas-loss-mode", choices=["total", "sf"], default="total")
    p_bud.add_argument("--min-gas-floor-abs", type=float, default=1e5)
    p_bud.add_argument("--split-mode", choices=["final_rhalf", "fixed"], default="final_rhalf")
    p_bud.add_argument("--split-radius-kpc", type=float, default=None)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "track":
        cfg = TrackingConfig(
            dwarf_init_center=args.dwarf_init_center,
            host_center=args.host_center,
            track_host=args.track_host,
            search_radius=args.search_radius,
            host_search_radius=args.host_search_radius,
            aperture_factor=args.aperture_factor,
            young_age_myr=args.young_age_myr,
            cgm_rin=args.cgm_rin,
            cgm_rout=args.cgm_rout,
            cgm_temp_min=args.cgm_temp_min,
            cgm_density_max=args.cgm_density_max,
            cgm_estimator=args.cgm_estimator,
            cone_half_angle_deg=args.cone_half_angle_deg,
            cone_min_cells=args.cone_min_cells,
            require_inflow_in_cone=args.require_inflow_in_cone,
            tidal_delta_kpc=args.tidal_delta_kpc,
            max_snaps=args.max_snaps,
            use_particle_ids=args.use_particle_ids,
            origin_ids_path=args.origin_ids_path,
            auto_create_origin_ids=not args.no_auto_create_origin_ids,
            origin_all_stars_are_dwarf=args.origin_all_stars_are_dwarf,
            origin_dwarf_radius_kpc=args.origin_dwarf_radius_kpc,
            origin_dwarf_aperture_factor=args.origin_dwarf_aperture_factor,
            origin_min_radius_kpc=args.origin_min_radius_kpc,
            restrict_cgm_to_host_gas_ids=not args.no_restrict_cgm_to_host_gas_ids,
            measure_id_stripping=not args.no_measure_id_stripping,
            stripped_radius_factor=args.stripped_radius_factor,
            stripped_use_tidal_radius=not args.no_stripped_use_tidal_radius,
        )
        records, table_path = analyze_snapshot_directory(args.path, cfg, outdir=args.outdir, table_name=args.table_name)
        print("[INFO] Analysed {0} snapshots".format(len(records)))
        print("[INFO] Wrote table: {0}".format(table_path))

    elif args.command == "metrics":
        ensure_dir(args.outdir)
        labels = args.labels if args.labels not in (None, []) else None
        snapdirs = args.snapshot_dirs if args.snapshot_dirs not in (None, []) else None
        runs = prepare_runs(
            args.tables,
            labels=labels,
            snapshot_dirs=snapdirs,
            smooth=args.smooth,
            pre_window=tuple(args.pre_window),
            force_window=tuple(args.force_window),
            response_window=tuple(args.response_window),
            entry_radius_kpc=args.entry_radius_kpc,
            gas_loss_fraction=args.gas_loss_fraction,
            gas_loss_mode=args.gas_loss_mode,
            min_gas_floor_abs=args.min_gas_floor_abs,
        )
        outcsv = os.path.join(args.outdir, "summary_metrics.csv")
        write_csv(metrics_rows(runs), outcsv)
        print("[INFO] Wrote: {0}".format(outcsv))

    elif args.command == "budgets":
        ensure_dir(args.outdir)
        labels = args.labels if args.labels not in (None, []) else None
        runs = prepare_runs(
            args.tables,
            labels=labels,
            snapshot_dirs=args.snapshot_dirs,
            entry_radius_kpc=args.entry_radius_kpc,
            gas_loss_fraction=args.gas_loss_fraction,
            gas_loss_mode=args.gas_loss_mode,
            min_gas_floor_abs=args.min_gas_floor_abs,
        )
        rows: List[Dict[str, Any]] = []
        for run in runs:
            row = mass_budget_for_run(run, split_mode=args.split_mode, split_radius_kpc=args.split_radius_kpc)
            if row is not None:
                rows.append(row)
        outcsv = os.path.join(args.outdir, "epoch_mass_budget.csv")
        write_csv(rows, outcsv)
        print("[INFO] Wrote: {0}".format(outcsv))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
