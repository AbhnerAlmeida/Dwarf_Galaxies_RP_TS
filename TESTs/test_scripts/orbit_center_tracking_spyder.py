#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_center_tracking_spyder.py

Spyder-friendly analysis script for tracking the centre of a satellite galaxy in
GADGET/GADGET-4 HDF5 snapshots.

Main purpose
------------
This version is focused on robust centre tracking. It was designed for the case
where:

  - PartType4 = stars of the satellite;
  - PartType1 = satellite DM, or satellite+host DM depending on your ICs;
  - PartType0 = gas, possibly including both satellite gas and host CGM gas;
  - the first snapshot has a well-defined ideal stellar COM, e.g. z=0 and vz=0.

The recommended centre mode is:

    center_mode = "inner_ids_shrinking"

This means:

  1. in the first snapshot, compute the stellar centre of mass, stars_com;
  2. define a central stellar region around this initial COM;
  3. store the ParticleIDs of stars initially inside that central region;
  4. in later snapshots, track those same central IDs;
  5. use a shrinking-sphere centre on these central IDs, seeded by the previous
     centre, to follow the satellite core.

This avoids the main weakness of stars_com_all: after tidal stripping, the COM of
all stars can be biased by tidal tails. It is also faster than using all stars,
because the centre is computed from a smaller, persistent core-ID subset.

Outputs per label
-----------------
For each label, the script saves:

  output_dir/[label]/center_tracking_timeseries.csv
  output_dir/[label]/center_tracking_summary.png
  output_dir/[label]/orbit_xy.png
  output_dir/[label]/initial_id_catalogue.npz
  output_dir/[label]/initial_id_summary.json

It also leaves DataFrames in memory when run in Spyder:

  results[label]
  combined_df

No argparse is used. Edit the USER SETTINGS section and press Run in Spyder.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import warnings
import copy

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

G_KPC_KMS2_MSUN = 4.30091e-6
ArrayLike3 = Union[Sequence[float], np.ndarray]


# =============================================================================
# Configuration classes
# =============================================================================

@dataclass
class HostHaloConfig:
    host_center_kpc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    host_velocity_kms: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    m200_msun: float = 1.0e12
    r200_kpc: float = 210.0
    concentration: float = 10.0
    tidal_factor: float = 3.0
    truncate_mass_at_r200: bool = False

    def f_nfw(self, x: np.ndarray | float) -> np.ndarray | float:
        return np.log1p(x) - x / (1.0 + x)

    def mass_enclosed_msun(self, radius_kpc: np.ndarray | float) -> np.ndarray | float:
        r = np.asarray(radius_kpc, dtype=float)
        r_safe = np.maximum(r, 1e-12)
        c = self.concentration
        x = c * r_safe / self.r200_kpc
        mass = self.m200_msun * self.f_nfw(x) / self.f_nfw(c)
        if self.truncate_mass_at_r200:
            mass = np.minimum(mass, self.m200_msun)
        if np.isscalar(radius_kpc):
            return float(mass)
        return mass


@dataclass
class CenterTrackingConfig:
    # Paths and labels.
    root: str = "SIMULATIONS/ORBIT/HigherRes"
    labels: Optional[List[str]] = None
    output_dir: str = "orbit_center_tracking_outputs"
    snapshot_glob: str = "snapshot_*.hdf5"

    # GADGET particle types.
    gas_ptype: int = 0
    dm_ptype: int = 1
    star_ptype: int = 4

    # Unit conversion from code units to physical units.
    length_unit_to_kpc: float = 1.0
    velocity_unit_to_kms: float = 1.0
    mass_unit_to_msun: float = 1.0e10
    time_unit_to_gyr: float = 0.977792221

    host: HostHaloConfig = field(default_factory=HostHaloConfig)

    # -------------------------------------------------------------------------
    # Centre modes.
    # -------------------------------------------------------------------------
    # Options:
    #   "stars_com_all"
    #       COM of all tracked stars. Good for checking ICs, not robust to tails.
    #   "stellar_core"
    #       shrinking sphere using all tracked stars near previous centre.
    #       This mimics the robust method in your previous script.
    #   "inner_ids_com"
    #       COM of the stars that were initially inside the central region.
    #   "inner_ids_shrinking"
    #       shrinking sphere using the initial central-ID subset. Recommended.
    #   "hybrid_inner_bound"
    #       first use inner IDs, then recompute using member stars near the core.
    center_mode: str = "inner_ids_shrinking"

    # Velocity modes:
    #   "inner_ids"       velocity COM of initial central-ID subset.
    #   "same_as_center"  for most modes, same particles used for centre.
    #   "member_stars"    velocity COM of currently associated/member stars.
    #   "all_stars"       velocity COM of all tracked stars.
    velocity_mode: str = "inner_ids"

    # Initial central stellar region definition.
    # The selected central radius is:
    #   max(inner_min_radius_kpc, inner_radius_factor_rhalf * rhalf0)
    # but it can be limited by inner_max_radius_kpc.
    inner_radius_factor_rhalf: float = 1.0
    inner_min_radius_kpc: float = 0.5
    inner_max_radius_kpc: float = 5.0
    inner_min_particles: int = 50

    # If True, use stars_com_all as the first-snapshot centre even if center_mode
    # is a different method. This is useful when the IC was explicitly set to an
    # ideal COM, e.g. z=0 and vz=0.
    use_stars_com_for_initial_core: bool = True

    # Shrinking sphere parameters.
    shrink_initial_radius_kpc: Optional[float] = None
    shrink_factor: float = 0.75
    shrink_min_particles: int = 100
    shrink_min_radius_kpc: float = 0.2
    center_search_radius_kpc: float = 40.0

    # Initial gas and DM selection.
    initial_satellite_gas_radius_kpc: Optional[float] = None
    default_initial_gas_radius_kpc: float = 30.0
    gas_radius_factor_rhalf: float = 8.0

    # If PartType1 is only the satellite DM, use "all".
    # If PartType1 includes host + satellite DM, use "radius".
    dm_selection_mode: str = "all"  # "all" or "radius"
    initial_satellite_dm_radius_kpc: Optional[float] = None
    default_initial_dm_radius_kpc: float = 80.0
    dm_radius_factor_rhalf: float = 20.0

    # Tidal-radius solver.
    tidal_max_iterations: int = 40
    tidal_tolerance: float = 1e-3
    tidal_initial_fraction_of_R: float = 0.15
    tidal_max_fraction_of_R: float = 0.5
    tidal_min_radius_kpc: float = 0.05

    # Stellar membership / stripping diagnostic.
    # member_mode options:
    #   "tidal"            member if r <= member_tidal_factor * rt
    #   "tidal_kinematic"  member if inside tidal radius and v_rel <= f*vesc_sat
    #   "loose_or_bound"   member if inside rt OR bound inside stripped_factor*rt
    member_mode: str = "tidal_kinematic"
    member_tidal_factor: float = 1.0
    stripped_tidal_factor: float = 1.5
    v_escape_factor: float = 1.25
    stripped_consecutive_snapshots: int = 2
    enclosed_mass_softening_kpc: float = 0.05

    # Plotting and verbosity.
    make_plots: bool = True
    verbose: bool = True


# =============================================================================
# General helpers
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_snapshot_number(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if not digits:
        return -1
    return int(digits[-1])


def discover_labels(root: Union[str, Path]) -> List[str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "output").exists()])


def find_snapshots_for_label(cfg: CenterTrackingConfig, label: str) -> List[Path]:
    out = Path(cfg.root) / label / "output"
    snaps = sorted(out.glob(cfg.snapshot_glob), key=natural_snapshot_number)
    if not snaps:
        raise FileNotFoundError(f"No snapshots found for {label} in {out}")
    return snaps


def dataset_first_available(group: h5py.Group, candidates: Sequence[str]) -> Optional[np.ndarray]:
    for name in candidates:
        if name in group:
            return group[name][:]
    return None


def header_attr(h5: h5py.File, name: str, default=None):
    if "Header" not in h5:
        return default
    return h5["Header"].attrs.get(name, default)


def inspect_snapshot_structure(snapshot_file: Union[str, Path]) -> None:
    snapshot_file = Path(snapshot_file)
    print(f"\n=== {snapshot_file} ===")
    with h5py.File(snapshot_file, "r") as f:
        print("\nTop-level groups:")
        for key in f.keys():
            print(f"  {key}")
        if "Header" in f:
            print("\nHeader attrs:")
            for k, v in f["Header"].attrs.items():
                print(f"  {k}: {v}")
        print("\nParticle groups:")
        for key in f.keys():
            if key.startswith("PartType"):
                print(f"\n{key}:")
                for dset in f[key].keys():
                    arr = f[key][dset]
                    print(f"  {dset}: shape={arr.shape}, dtype={arr.dtype}")


def read_snapshot_time_gyr(snapshot_file: Union[str, Path], cfg: CenterTrackingConfig) -> float:
    with h5py.File(snapshot_file, "r") as f:
        t_code = header_attr(f, "Time", np.nan)
    try:
        return float(t_code) * cfg.time_unit_to_gyr
    except Exception:
        return np.nan


def read_particle_type(
    snapshot_file: Union[str, Path],
    ptype: int,
    cfg: CenterTrackingConfig,
) -> Optional[Dict[str, np.ndarray]]:
    snapshot_file = Path(snapshot_file)
    group_name = f"PartType{ptype}"
    with h5py.File(snapshot_file, "r") as f:
        if group_name not in f:
            return None
        g = f[group_name]
        pos = dataset_first_available(g, ["Coordinates", "Position", "Positions"])
        vel = dataset_first_available(g, ["Velocities", "Velocity"])
        ids = dataset_first_available(g, ["ParticleIDs", "ParticleID", "IDs", "ID"])
        mass = dataset_first_available(g, ["Masses", "Mass", "ParticleMasses"])
        sfr = dataset_first_available(g,["StarFormationRate", "StarFormationRates", "SFR", "sfr"])
        
        if pos is None or vel is None:
            raise KeyError(f"Missing Coordinates/Velocities in {snapshot_file}:{group_name}")
        if ids is None:
            ids = np.arange(1, len(pos) + 1, dtype=np.int64)
        if mass is None:
            mass_table = header_attr(f, "MassTable", None)
            if mass_table is None:
                raise KeyError(f"No Masses and no Header/MassTable for {snapshot_file}:{group_name}")
            m_code = float(mass_table[ptype])
            if m_code <= 0:
                raise ValueError(f"MassTable[{ptype}] is zero and no Masses dataset exists")
            mass = np.full(len(ids), m_code, dtype=float)
            

    return {
        "pos": np.asarray(pos, dtype=float) * cfg.length_unit_to_kpc,
        "vel": np.asarray(vel, dtype=float) * cfg.velocity_unit_to_kms,
        "ids": np.asarray(ids, dtype=np.int64),
        "mass": np.asarray(mass, dtype=float) * cfg.mass_unit_to_msun,
        "sfr" : np.asarray(sfr, dtype=float)
    }


def empty_particle_dict() -> Dict[str, np.ndarray]:
    return {
        "pos": np.empty((0, 3), dtype=float),
        "vel": np.empty((0, 3), dtype=float),
        "ids": np.empty(0, dtype=np.int64),
        "mass": np.empty(0, dtype=float),
        "sfr": np.empty(0, dtype=float),
    }


def mask_ids(ids: np.ndarray, selected_ids: Optional[np.ndarray]) -> np.ndarray:
    if selected_ids is None:
        return np.ones(len(ids), dtype=bool)
    if len(ids) == 0 or len(selected_ids) == 0:
        return np.zeros(len(ids), dtype=bool)
    return np.isin(ids, selected_ids, assume_unique=False)


def filter_particle_dict(
    pdata: Optional[Dict[str, np.ndarray]],
    selected_ids: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    if pdata is None:
        return empty_particle_dict()
    m = mask_ids(pdata["ids"], selected_ids)
    return {k: v[m] if isinstance(v, np.ndarray) and len(v) == len(pdata["ids"]) else v for k, v in pdata.items()}


def distances(pos: np.ndarray, center: ArrayLike3) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    return np.linalg.norm(pos - center[None, :], axis=1)


def mass_weighted_mean(values: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(values.shape[1] if values.ndim > 1 else 1, np.nan)
    if masses is None or np.sum(masses) <= 0:
        return np.nanmean(values, axis=0)
    masses = np.asarray(masses, dtype=float)
    good = np.isfinite(masses) & (masses > 0)
    if values.ndim == 2:
        good &= np.all(np.isfinite(values), axis=1)
    else:
        good &= np.isfinite(values)
    if np.sum(good) == 0:
        return np.nanmean(values, axis=0)
    return np.average(values[good], axis=0, weights=masses[good])


def half_mass_radius(pos: np.ndarray, masses: np.ndarray, center: ArrayLike3) -> float:
    if len(pos) == 0 or np.sum(masses) <= 0:
        return np.nan
    r = distances(pos, center)
    order = np.argsort(r)
    rs = r[order]
    ms = masses[order]
    cum = np.cumsum(ms)
    idx = np.searchsorted(cum, 0.5 * cum[-1])
    return float(rs[min(idx, len(rs) - 1)])


def enclosed_radius_fraction(pos: np.ndarray, masses: np.ndarray, center: ArrayLike3, fraction: float) -> float:
    if len(pos) == 0 or np.sum(masses) <= 0:
        return np.nan
    r = distances(pos, center)
    order = np.argsort(r)
    rs = r[order]
    ms = masses[order]
    cum = np.cumsum(ms)
    idx = np.searchsorted(cum, fraction * cum[-1])
    return float(rs[min(idx, len(rs) - 1)])


def shrinking_sphere_center(
    pos: np.ndarray,
    masses: np.ndarray,
    initial_center: Optional[ArrayLike3] = None,
    initial_radius: Optional[float] = None,
    shrink_factor: float = 0.75,
    min_particles: int = 100,
    min_radius_kpc: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return centre and a boolean mask of the final particles used.
    """
    if len(pos) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float), np.zeros(0, dtype=bool)

    center = mass_weighted_mean(pos, masses) if initial_center is None else np.asarray(initial_center, dtype=float)
    r = distances(pos, center)

    if initial_radius is None:
        radius = float(np.nanpercentile(r, 90))
        if not np.isfinite(radius) or radius <= 0:
            radius = float(np.nanmax(r))
    else:
        radius = float(initial_radius)
    radius = max(radius, min_radius_kpc)

    current = np.arange(len(pos))
    while True:
        rr = distances(pos[current], center)
        inside = current[rr <= radius]
        if len(inside) < max(min_particles, 10) or radius <= min_radius_kpc:
            if len(inside) >= 5:
                current = inside
            break
        center = mass_weighted_mean(pos[inside], masses[inside])
        current = inside
        radius *= shrink_factor

    if len(current) > 0:
        center = mass_weighted_mean(pos[current], masses[current])

    mask = np.zeros(len(pos), dtype=bool)
    mask[current] = True
    return np.asarray(center, dtype=float), mask


# =============================================================================
# Satellite ID catalogue and central-ID selection
# =============================================================================

def select_initial_core_star_ids(
    stars: Dict[str, np.ndarray],
    cfg: CenterTrackingConfig,
) -> Dict[str, object]:
    """
    Select initial central stellar IDs around the first-snapshot stars_com.
    """
    stars_com = mass_weighted_mean(stars["pos"], stars["mass"])
    v_stars_com = mass_weighted_mean(stars["vel"], stars["mass"])
    rhalf0 = half_mass_radius(stars["pos"], stars["mass"], stars_com)

    inner_radius = cfg.inner_min_radius_kpc
    if np.isfinite(rhalf0):
        inner_radius = max(inner_radius, cfg.inner_radius_factor_rhalf * rhalf0)
    inner_radius = min(cfg.inner_max_radius_kpc, inner_radius)

    r = distances(stars["pos"], stars_com)
    inner = r <= inner_radius

    # Guarantee a minimum number of core stars if the initial radius is too small.
    if np.sum(inner) < min(cfg.inner_min_particles, len(stars["ids"])):
        order = np.argsort(r)
        n = min(len(order), max(cfg.inner_min_particles, 5))
        inner = np.zeros(len(r), dtype=bool)
        inner[order[:n]] = True
        inner_radius = float(np.nanmax(r[inner]))

    return {
        "initial_stars_com_kpc": stars_com,
        "initial_stars_com_velocity_kms": v_stars_com,
        "initial_rhalf_star_kpc": float(rhalf0),
        "initial_inner_radius_kpc": float(inner_radius),
        "inner_star_ids": np.unique(stars["ids"][inner]),
        "N_inner_star_ids": int(np.sum(inner)),
        "M_inner_star_ids_msun": float(np.sum(stars["mass"][inner])),
    }


def build_initial_id_catalogue(
    first_snapshot: Union[str, Path],
    cfg: CenterTrackingConfig,
) -> Dict[str, object]:
    stars = read_particle_type(first_snapshot, cfg.star_ptype, cfg)
    if stars is None or len(stars["ids"]) == 0:
        raise RuntimeError(f"No star particles found in {first_snapshot}")

    core = select_initial_core_star_ids(stars, cfg)
    initial_center = core["initial_stars_com_kpc"] if cfg.use_stars_com_for_initial_core else core["initial_stars_com_kpc"]
    rhalf0 = float(core["initial_rhalf_star_kpc"])

    gas_radius = cfg.initial_satellite_gas_radius_kpc
    if gas_radius is None:
        gas_radius = max(cfg.default_initial_gas_radius_kpc, cfg.gas_radius_factor_rhalf * rhalf0) if np.isfinite(rhalf0) else cfg.default_initial_gas_radius_kpc

    dm_radius = cfg.initial_satellite_dm_radius_kpc
    if dm_radius is None:
        dm_radius = max(cfg.default_initial_dm_radius_kpc, cfg.dm_radius_factor_rhalf * rhalf0) if np.isfinite(rhalf0) else cfg.default_initial_dm_radius_kpc

    gas_all = read_particle_type(first_snapshot, cfg.gas_ptype, cfg)
    if gas_all is not None and len(gas_all["ids"]) > 0:
        rg = distances(gas_all["pos"], initial_center)
        gas_ids = np.unique(gas_all["ids"][rg <= gas_radius])
    else:
        gas_ids = np.array([], dtype=np.int64)

    dm_all = read_particle_type(first_snapshot, cfg.dm_ptype, cfg)
    if dm_all is not None and len(dm_all["ids"]) > 0:
        if cfg.dm_selection_mode.lower() == "all":
            dm_ids = np.unique(dm_all["ids"])
        elif cfg.dm_selection_mode.lower() == "radius":
            rd = distances(dm_all["pos"], initial_center)
            dm_ids = np.unique(dm_all["ids"][rd <= dm_radius])
        else:
            raise ValueError("cfg.dm_selection_mode must be 'all' or 'radius'")
    else:
        dm_ids = np.array([], dtype=np.int64)

    return {
        "star_ids": np.unique(stars["ids"]),
        "inner_star_ids": np.asarray(core["inner_star_ids"], dtype=np.int64),
        "gas_ids": gas_ids,
        "dm_ids": dm_ids,
        "initial_stars_com_kpc": np.asarray(core["initial_stars_com_kpc"], dtype=float),
        "initial_stars_com_velocity_kms": np.asarray(core["initial_stars_com_velocity_kms"], dtype=float),
        "initial_rhalf_star_kpc": rhalf0,
        "initial_inner_radius_kpc": float(core["initial_inner_radius_kpc"]),
        "N_inner_star_ids": int(core["N_inner_star_ids"]),
        "M_inner_star_ids_msun": float(core["M_inner_star_ids_msun"]),
        "initial_gas_selection_radius_kpc": float(gas_radius),
        "initial_dm_selection_radius_kpc": float(dm_radius),
    }


def save_id_catalogue(outdir: Union[str, Path], idcat: Dict[str, object], cfg: CenterTrackingConfig) -> None:
    outdir = ensure_dir(outdir)
    np.savez_compressed(
        outdir / "initial_id_catalogue.npz",
        star_ids=np.asarray(idcat["star_ids"], dtype=np.int64),
        inner_star_ids=np.asarray(idcat["inner_star_ids"], dtype=np.int64),
        gas_ids=np.asarray(idcat["gas_ids"], dtype=np.int64),
        dm_ids=np.asarray(idcat["dm_ids"], dtype=np.int64),
    )

    summary = {
        "config": asdict(cfg),
        "N_star_ids": int(len(idcat["star_ids"])),
        "N_inner_star_ids": int(len(idcat["inner_star_ids"])),
        "N_gas_ids": int(len(idcat["gas_ids"])),
        "N_dm_ids": int(len(idcat["dm_ids"])),
        "initial_stars_com_kpc": np.asarray(idcat["initial_stars_com_kpc"]).tolist(),
        "initial_stars_com_velocity_kms": np.asarray(idcat["initial_stars_com_velocity_kms"]).tolist(),
        "initial_rhalf_star_kpc": float(idcat["initial_rhalf_star_kpc"]),
        "initial_inner_radius_kpc": float(idcat["initial_inner_radius_kpc"]),
        "initial_gas_selection_radius_kpc": float(idcat["initial_gas_selection_radius_kpc"]),
        "initial_dm_selection_radius_kpc": float(idcat["initial_dm_selection_radius_kpc"]),
    }
    with open(outdir / "initial_id_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# Tidal radius and membership/stripping diagnostics
# =============================================================================

def satellite_mass_within_radius(
    component_dicts: Sequence[Dict[str, np.ndarray]],
    center: ArrayLike3,
    radius_kpc: float,
) -> float:
    total = 0.0
    for comp in component_dicts:
        if len(comp["mass"]) == 0:
            continue
        r = distances(comp["pos"], center)
        total += float(np.sum(comp["mass"][r <= radius_kpc]))
    return total


def solve_tidal_radius_kpc(
    component_dicts: Sequence[Dict[str, np.ndarray]],
    center_kpc: ArrayLike3,
    host: HostHaloConfig,
    cfg: CenterTrackingConfig,
) -> float:
    center = np.asarray(center_kpc, dtype=float)
    host_center = np.asarray(host.host_center_kpc, dtype=float)
    R = float(np.linalg.norm(center - host_center))
    if not np.isfinite(R) or R <= 0:
        return np.nan
    M_host = host.mass_enclosed_msun(R)
    if not np.isfinite(M_host) or M_host <= 0:
        return np.nan

    rt = max(cfg.tidal_min_radius_kpc, cfg.tidal_initial_fraction_of_R * R)
    rt = min(rt, cfg.tidal_max_fraction_of_R * R)

    for _ in range(cfg.tidal_max_iterations):
        m_sat = satellite_mass_within_radius(component_dicts, center, rt)
        if m_sat <= 0:
            return cfg.tidal_min_radius_kpc
        rt_new = R * (m_sat / (host.tidal_factor * M_host)) ** (1.0 / 3.0)
        rt_new = max(cfg.tidal_min_radius_kpc, min(rt_new, cfg.tidal_max_fraction_of_R * R))
        if abs(rt_new - rt) / max(rt, cfg.tidal_min_radius_kpc) < cfg.tidal_tolerance:
            return float(rt_new)
        rt = rt_new
    return float(rt)


def enclosed_mass_at_radii(
    component_dicts: Sequence[Dict[str, np.ndarray]],
    center: ArrayLike3,
    query_radii: np.ndarray,
) -> np.ndarray:
    all_r = []
    all_m = []
    for comp in component_dicts:
        if len(comp["mass"]) == 0:
            continue
        all_r.append(distances(comp["pos"], center))
        all_m.append(comp["mass"])
    if not all_r:
        return np.zeros_like(query_radii, dtype=float)
    r = np.concatenate(all_r)
    m = np.concatenate(all_m)
    order = np.argsort(r)
    r_sorted = r[order]
    m_cum = np.cumsum(m[order])
    idx = np.searchsorted(r_sorted, query_radii, side="right") - 1
    out = np.zeros_like(query_radii, dtype=float)
    ok = idx >= 0
    out[ok] = m_cum[idx[ok]]
    return out


def stellar_membership_and_stripping_flags(
    stars: Dict[str, np.ndarray],
    component_dicts: Sequence[Dict[str, np.ndarray]],
    center: ArrayLike3,
    vcenter: ArrayLike3,
    rt: float,
    cfg: CenterTrackingConfig,
) -> Dict[str, np.ndarray]:
    n = len(stars["ids"])
    if n == 0:
        return {
            "r": np.empty(0),
            "vrel": np.empty(0),
            "vesc": np.empty(0),
            "inside_rt": np.empty(0, dtype=bool),
            "kinematic_bound": np.empty(0, dtype=bool),
            "member": np.empty(0, dtype=bool),
            "stripped_candidate": np.empty(0, dtype=bool),
        }

    center = np.asarray(center, dtype=float)
    vcenter = np.asarray(vcenter, dtype=float)
    r = distances(stars["pos"], center)
    vrel = np.linalg.norm(stars["vel"] - vcenter[None, :], axis=1)

    m_enc = enclosed_mass_at_radii(component_dicts, center, np.maximum(r, cfg.enclosed_mass_softening_kpc))
    vesc = np.sqrt(2.0 * G_KPC_KMS2_MSUN * m_enc / np.maximum(r, cfg.enclosed_mass_softening_kpc))

    inside_rt = r <= cfg.member_tidal_factor * rt if np.isfinite(rt) else np.zeros(n, dtype=bool)
    near_rt = r <= cfg.stripped_tidal_factor * rt if np.isfinite(rt) else np.zeros(n, dtype=bool)
    kinematic_bound = vrel <= cfg.v_escape_factor * vesc

    mode = cfg.member_mode.lower()
    if mode == "tidal":
        member = inside_rt
    elif mode == "tidal_kinematic":
        member = inside_rt & kinematic_bound
    elif mode == "loose_or_bound":
        member = inside_rt | (near_rt & kinematic_bound)
    else:
        raise ValueError("cfg.member_mode must be 'tidal', 'tidal_kinematic', or 'loose_or_bound'")

    stripped_candidate = (~near_rt) | ((~inside_rt) & (~kinematic_bound))

    return {
        "r": r,
        "vrel": vrel,
        "vesc": vesc,
        "inside_rt": inside_rt,
        "kinematic_bound": kinematic_bound,
        "member": member,
        "stripped_candidate": stripped_candidate,
    }


# =============================================================================
# Centre methods
# =============================================================================

def select_stars_by_ids(stars: Dict[str, np.ndarray], ids: np.ndarray) -> Dict[str, np.ndarray]:
    return filter_particle_dict(stars, ids)


def compute_center_and_velocity(
    stars: Dict[str, np.ndarray],
    inner_stars: Dict[str, np.ndarray],
    member_mask_previous: Optional[np.ndarray],
    cfg: CenterTrackingConfig,
    previous_center: Optional[np.ndarray],
    previous_rhalf: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Compute centre according to cfg.center_mode.

    Returns centre, velocity centre, and metadata.
    """
    mode = cfg.center_mode.lower()
    meta: Dict[str, object] = {}

    if mode == "stars_com_all":
        center = mass_weighted_mean(stars["pos"], stars["mass"])
        center_mask_all = np.ones(len(stars["ids"]), dtype=bool)
        center_source = stars
        center_source_mask = center_mask_all

    elif mode == "stellar_core":
        if previous_center is not None and np.all(np.isfinite(previous_center)):
            rprev = distances(stars["pos"], previous_center)
            search_radius = cfg.center_search_radius_kpc
            if previous_rhalf is not None and np.isfinite(previous_rhalf):
                search_radius = max(search_radius, 10.0 * previous_rhalf)
            near = rprev <= search_radius
            if np.sum(near) >= max(20, cfg.shrink_min_particles // 4):
                pos_in = stars["pos"][near]
                mass_in = stars["mass"][near]
                center_guess = previous_center
                initial_radius = min(search_radius, float(np.nanmax(rprev)) if len(rprev) else search_radius)
                center, local_mask = shrinking_sphere_center(
                    pos_in, mass_in, initial_center=center_guess, initial_radius=initial_radius,
                    shrink_factor=cfg.shrink_factor, min_particles=cfg.shrink_min_particles,
                    min_radius_kpc=cfg.shrink_min_radius_kpc,
                )
                center_mask_all = np.zeros(len(stars["ids"]), dtype=bool)
                idx_near = np.where(near)[0]
                center_mask_all[idx_near[local_mask]] = True
            else:
                center, center_mask_all = shrinking_sphere_center(
                    stars["pos"], stars["mass"], initial_center=previous_center,
                    initial_radius=cfg.shrink_initial_radius_kpc,
                    shrink_factor=cfg.shrink_factor, min_particles=cfg.shrink_min_particles,
                    min_radius_kpc=cfg.shrink_min_radius_kpc,
                )
        else:
            center, center_mask_all = shrinking_sphere_center(
                stars["pos"], stars["mass"], initial_radius=cfg.shrink_initial_radius_kpc,
                shrink_factor=cfg.shrink_factor, min_particles=cfg.shrink_min_particles,
                min_radius_kpc=cfg.shrink_min_radius_kpc,
            )
        center_source = stars
        center_source_mask = center_mask_all

    elif mode == "inner_ids_com":
        if len(inner_stars["ids"]) >= 5:
            center = mass_weighted_mean(inner_stars["pos"], inner_stars["mass"])
            center_source = inner_stars
            center_source_mask = np.ones(len(inner_stars["ids"]), dtype=bool)
        else:
            warnings.warn("Too few inner IDs found; falling back to all-stars COM")
            center = mass_weighted_mean(stars["pos"], stars["mass"])
            center_source = stars
            center_source_mask = np.ones(len(stars["ids"]), dtype=bool)

    elif mode == "inner_ids_shrinking":
        if len(inner_stars["ids"]) >= 5:
            init_center = previous_center if previous_center is not None and np.all(np.isfinite(previous_center)) else None
            init_radius = cfg.center_search_radius_kpc
            if previous_rhalf is not None and np.isfinite(previous_rhalf):
                init_radius = max(init_radius, 10.0 * previous_rhalf)
            center, center_source_mask = shrinking_sphere_center(
                inner_stars["pos"], inner_stars["mass"], initial_center=init_center,
                initial_radius=init_radius, shrink_factor=cfg.shrink_factor,
                min_particles=min(cfg.shrink_min_particles, max(10, len(inner_stars["ids"]) // 2)),
                min_radius_kpc=cfg.shrink_min_radius_kpc,
            )
            center_source = inner_stars
        else:
            warnings.warn("Too few inner IDs found; falling back to stellar_core")
            center, center_mask_all = shrinking_sphere_center(
                stars["pos"], stars["mass"], initial_center=previous_center,
                initial_radius=cfg.shrink_initial_radius_kpc,
                shrink_factor=cfg.shrink_factor, min_particles=cfg.shrink_min_particles,
                min_radius_kpc=cfg.shrink_min_radius_kpc,
            )
            center_source = stars
            center_source_mask = center_mask_all

    elif mode == "hybrid_inner_bound":
        # First get a core estimate from inner IDs.
        if len(inner_stars["ids"]) >= 5:
            center0 = mass_weighted_mean(inner_stars["pos"], inner_stars["mass"])
        else:
            center0 = mass_weighted_mean(stars["pos"], stars["mass"])

        # If we already have a previous member mask, use those stars near center0.
        if member_mask_previous is not None and len(member_mask_previous) == len(stars["ids"]) and np.sum(member_mask_previous) >= 20:
            pos_in = stars["pos"][member_mask_previous]
            mass_in = stars["mass"][member_mask_previous]
            center, local_mask = shrinking_sphere_center(
                pos_in, mass_in, initial_center=center0,
                initial_radius=cfg.center_search_radius_kpc,
                shrink_factor=cfg.shrink_factor, min_particles=min(cfg.shrink_min_particles, max(10, np.sum(member_mask_previous)//2)),
                min_radius_kpc=cfg.shrink_min_radius_kpc,
            )
            center_source = {"pos": pos_in, "vel": stars["vel"][member_mask_previous], "mass": mass_in, "ids": stars["ids"][member_mask_previous]}
            center_source_mask = local_mask
        else:
            center = center0
            center_source = inner_stars if len(inner_stars["ids"]) >= 5 else stars
            center_source_mask = np.ones(len(center_source["ids"]), dtype=bool)

    else:
        raise ValueError(
            "Unknown center_mode. Use 'stars_com_all', 'stellar_core', 'inner_ids_com', "
            "'inner_ids_shrinking', or 'hybrid_inner_bound'."
        )

    # Velocity centre.
    vmode = cfg.velocity_mode.lower()
    if vmode == "all_stars":
        vcenter = mass_weighted_mean(stars["vel"], stars["mass"])
        n_vel = len(stars["ids"])
    elif vmode == "inner_ids":
        if len(inner_stars["ids"]) >= 5:
            vcenter = mass_weighted_mean(inner_stars["vel"], inner_stars["mass"])
            n_vel = len(inner_stars["ids"])
        else:
            vcenter = mass_weighted_mean(stars["vel"], stars["mass"])
            n_vel = len(stars["ids"])
    elif vmode == "same_as_center":
        vcenter = mass_weighted_mean(center_source["vel"][center_source_mask], center_source["mass"][center_source_mask])
        n_vel = int(np.sum(center_source_mask))
    elif vmode == "member_stars":
        if member_mask_previous is not None and len(member_mask_previous) == len(stars["ids"]) and np.sum(member_mask_previous) >= 5:
            vcenter = mass_weighted_mean(stars["vel"][member_mask_previous], stars["mass"][member_mask_previous])
            n_vel = int(np.sum(member_mask_previous))
        else:
            vcenter = mass_weighted_mean(inner_stars["vel"], inner_stars["mass"]) if len(inner_stars["ids"]) >= 5 else mass_weighted_mean(stars["vel"], stars["mass"])
            n_vel = len(inner_stars["ids"]) if len(inner_stars["ids"]) >= 5 else len(stars["ids"])
    else:
        raise ValueError("Unknown velocity_mode. Use 'inner_ids', 'same_as_center', 'member_stars', or 'all_stars'.")

    meta["N_center_source"] = int(len(center_source["ids"]))
    meta["N_center_final"] = int(np.sum(center_source_mask))
    meta["N_velocity_source"] = int(n_vel)
    return np.asarray(center, dtype=float), np.asarray(vcenter, dtype=float), meta


# =============================================================================
# Snapshot and label analysis
# =============================================================================

def component_masses(comp: Dict[str, np.ndarray], center: ArrayLike3, mask: Optional[np.ndarray] = None) -> float:
    if len(comp["mass"]) == 0:
        return 0.0
    if mask is None:
        return float(np.sum(comp["mass"]))
    return float(np.sum(comp["mass"][mask]))

def sum_particle_field(
    comp: Dict[str, np.ndarray],
    field: str,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Sum an optional particle field, e.g. gas SFR.

    Returns NaN if the field is absent.
    """

    if field not in comp:
        return np.nan

    values = np.asarray(comp[field], dtype=float)

    if len(values) == 0:
        return np.nan

    if mask is None:
        return float(np.nansum(values))

    if len(mask) != len(values):
        return np.nan

    return float(np.nansum(values[mask]))

def analyse_one_snapshot(
    snapshot_file: Union[str, Path],
    snapshot_index: int,
    cfg: CenterTrackingConfig,
    idcat: Dict[str, object],
    previous_center: Optional[np.ndarray],
    previous_rhalf: Optional[float],
    previous_member_mask: Optional[np.ndarray],
    stripped_counter: Dict[int, int],
) -> Tuple[Dict[str, object], np.ndarray, float, np.ndarray, Dict[int, int]]:
    snapshot_file = Path(snapshot_file)

    stars_all = read_particle_type(snapshot_file, cfg.star_ptype, cfg)
    gas_all = read_particle_type(snapshot_file, cfg.gas_ptype, cfg)
    dm_all = read_particle_type(snapshot_file, cfg.dm_ptype, cfg)

    stars = filter_particle_dict(stars_all, np.asarray(idcat["star_ids"], dtype=np.int64))
    inner_stars = filter_particle_dict(stars_all, np.asarray(idcat["inner_star_ids"], dtype=np.int64))
    gas = filter_particle_dict(gas_all, np.asarray(idcat["gas_ids"], dtype=np.int64))
    dm = filter_particle_dict(dm_all, np.asarray(idcat["dm_ids"], dtype=np.int64))

    if len(stars["ids"]) == 0:
        raise RuntimeError(f"No tracked star IDs found in {snapshot_file}")

    center, vcenter, center_meta = compute_center_and_velocity(
        stars=stars,
        inner_stars=inner_stars,
        member_mask_previous=previous_member_mask,
        cfg=cfg,
        previous_center=previous_center,
        previous_rhalf=previous_rhalf,
    )

    # Tidal radius and membership.
    rt = solve_tidal_radius_kpc([stars, gas, dm], center, cfg.host, cfg)
    flags = stellar_membership_and_stripping_flags(stars, [stars, gas, dm], center, vcenter, rt, cfg)
    member = flags["member"]
    stripped_candidate = flags["stripped_candidate"]

    # Update persistent stripping counter.
    new_counter: Dict[int, int] = {}
    ids = stars["ids"]
    for sid, cand in zip(ids, stripped_candidate):
        sid_int = int(sid)
        old = stripped_counter.get(sid_int, 0)
        new_counter[sid_int] = old + 1 if bool(cand) else 0
    stripped_definitive = np.array([new_counter.get(int(sid), 0) >= cfg.stripped_consecutive_snapshots for sid in ids], dtype=bool)

    # Sizes.
    rhalf_all = half_mass_radius(stars["pos"], stars["mass"], center)
    r90_all = enclosed_radius_fraction(stars["pos"], stars["mass"], center, 0.90)
    if np.sum(member) >= 5:
        rhalf_member = half_mass_radius(stars["pos"][member], stars["mass"][member], center)
        r90_member = enclosed_radius_fraction(stars["pos"][member], stars["mass"][member], center, 0.90)
    else:
        rhalf_member = np.nan
        r90_member = np.nan

    # Orbit.
    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    host_vel = np.asarray(cfg.host.host_velocity_kms, dtype=float)
    rel_pos = center - host_center
    rel_vel = vcenter - host_vel
    R = float(np.linalg.norm(rel_pos))
    V = float(np.linalg.norm(rel_vel))
    if R > 0:
        rhat = rel_pos / R
        V_rad = float(np.dot(rel_vel, rhat))
        V_tan = float(np.sqrt(max(V * V - V_rad * V_rad, 0.0)))
    else:
        V_rad = np.nan
        V_tan = np.nan

    Lvec = np.cross(rel_pos, rel_vel)
    L = float(np.linalg.norm(Lvec))

    # Masses.
    r_star = flags["r"]
    stars_inside_rt = r_star <= rt if np.isfinite(rt) else np.zeros(len(r_star), dtype=bool)
    gas_inside_rt = r_gas <= rt if np.isfinite(rt) else np.zeros(len(r_star), dtype=bool)

    stars_member = member
    stars_def_bound = ~stripped_definitive

    row: Dict[str, object] = {
        "snapshot_index": int(snapshot_index),
        "snapshot_number": int(natural_snapshot_number(snapshot_file)),
        "snapshot_file": str(snapshot_file),
        "time_gyr": read_snapshot_time_gyr(snapshot_file, cfg),
        "center_mode": cfg.center_mode,
        "velocity_mode": cfg.velocity_mode,
        "member_mode": cfg.member_mode,
        "x_sat_kpc": float(center[0]),
        "y_sat_kpc": float(center[1]),
        "z_sat_kpc": float(center[2]),
        "vx_sat_kms": float(vcenter[0]),
        "vy_sat_kms": float(vcenter[1]),
        "vz_sat_kms": float(vcenter[2]),
        "R_host_kpc": R,
        "V_3d_kms": V,
        "V_rad_kms": V_rad,
        "V_tan_kms": V_tan,
        "L_kpc_kms": L,
        "r_tidal_kpc": float(rt),
        "rhalf_star_all_kpc": float(rhalf_all),
        "r90_star_all_kpc": float(r90_all),
        "rhalf_star_member_kpc": float(rhalf_member),
        "r90_star_member_kpc": float(r90_member),
        "Nstar_tracked": int(len(stars["ids"])),
        "Nstar_inner_ids_found": int(len(inner_stars["ids"])),
        "Nstar_member": int(np.sum(stars_member)),
        "Nstar_inside_rt": int(np.sum(stars_inside_rt)),
        "Nstar_stripped_candidate": int(np.sum(stripped_candidate)),
        "Nstar_stripped_definitive": int(np.sum(stripped_definitive)),
        "fstar_member": float(np.sum(stars_member) / len(stars["ids"])),
        "fstar_stripped_candidate": float(np.sum(stripped_candidate) / len(stars["ids"])),
        "fstar_stripped_definitive": float(np.sum(stripped_definitive) / len(stars["ids"])),
        "Mstar_tracked_msun": component_masses(stars, center),
        "Mstar_inside_rt_msun": component_masses(stars, center, stars_inside_rt),
        "Mstar_member_msun": component_masses(stars, center, stars_member),
        "Mstar_not_definitively_stripped_msun": component_masses(stars, center, stars_def_bound),
        "Mgas_tracked_msun": component_masses(gas, center),
        "SFR_gas_tracked_msun_yr": sum_particle_field(gas, "sfr"),
        "SFR_gas_inside_rt_msun_yr": sum_particle_field(gas, "sfr", gas_inside_rt),
        "Mdm_tracked_msun": component_masses(dm, center),
        "median_rstar_over_rt": float(np.nanmedian(r_star / rt)) if np.isfinite(rt) and rt > 0 else np.nan,
        "p90_rstar_over_rt": float(np.nanpercentile(r_star / rt, 90)) if np.isfinite(rt) and rt > 0 else np.nan,
        "median_vrel_over_vesc": float(np.nanmedian(flags["vrel"] / flags["vesc"])),
    }
    row.update(center_meta)

    # Useful diagnostic: how far the all-stars COM is from the adopted centre.
    all_com = mass_weighted_mean(stars["pos"], stars["mass"])
    inner_com = mass_weighted_mean(inner_stars["pos"], inner_stars["mass"]) if len(inner_stars["ids"]) else np.full(3, np.nan)
    row["offset_all_stars_com_from_center_kpc"] = float(np.linalg.norm(all_com - center))
    row["offset_inner_ids_com_from_center_kpc"] = float(np.linalg.norm(inner_com - center)) if np.all(np.isfinite(inner_com)) else np.nan
    row["x_all_stars_com_kpc"] = float(all_com[0])
    row["y_all_stars_com_kpc"] = float(all_com[1])
    row["z_all_stars_com_kpc"] = float(all_com[2])

    return row, center, rhalf_member if np.isfinite(rhalf_member) else rhalf_all, member, new_counter


def analyse_label(label: str, cfg: CenterTrackingConfig) -> pd.DataFrame:
    snaps = find_snapshots_for_label(cfg, label)
    outdir = ensure_dir(Path(cfg.output_dir) / label)

    if cfg.verbose:
        print(f"\n[{label}] {len(snaps)} snapshots")
        print(f"First snapshot: {snaps[0]}")

    idcat = build_initial_id_catalogue(snaps[0], cfg)
    save_id_catalogue(outdir, idcat, cfg)

    if cfg.verbose:
        print("Initial stellar COM [kpc]:", np.asarray(idcat["initial_stars_com_kpc"]))
        print("Initial stellar COM velocity [km/s]:", np.asarray(idcat["initial_stars_com_velocity_kms"]))
        print("Initial rhalf_star [kpc]:", float(idcat["initial_rhalf_star_kpc"]))
        print("Initial inner radius [kpc]:", float(idcat["initial_inner_radius_kpc"]))
        print("N inner star IDs:", int(len(idcat["inner_star_ids"])))

    rows = []
    prev_center = None
    prev_rhalf = None
    prev_member_mask = None
    stripped_counter: Dict[int, int] = {}

    for i, snap in enumerate(snaps):
        row, prev_center, prev_rhalf, prev_member_mask, stripped_counter = analyse_one_snapshot(
            snapshot_file=snap,
            snapshot_index=i,
            cfg=cfg,
            idcat=idcat,
            previous_center=prev_center,
            previous_rhalf=prev_rhalf,
            previous_member_mask=prev_member_mask,
            stripped_counter=stripped_counter,
        )
        rows.append(row)
        if cfg.verbose and (i == 0 or i == len(snaps) - 1 or i % 10 == 0):
            print(
                f"  snap={row['snapshot_number']:04d} "
                f"R={row['R_host_kpc']:.2f} kpc "
                f"rt={row['r_tidal_kpc']:.2f} kpc "
                f"f_member={row['fstar_member']:.3f} "
                f"f_stripped={row['fstar_stripped_definitive']:.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "center_tracking_timeseries.csv", index=False)

    if cfg.make_plots:
        plot_label_summary(df, outdir, label)

    return df


def analyse_all_labels(cfg: CenterTrackingConfig) -> Dict[str, pd.DataFrame]:
    labels = cfg.labels if cfg.labels is not None else discover_labels(cfg.root)
    results: Dict[str, pd.DataFrame] = {}
    for label in labels:
        results[label] = analyse_label(label, cfg)
    combined = pd.concat([df.assign(label=label) for label, df in results.items()], ignore_index=True)
    ensure_dir(cfg.output_dir)
    combined.to_csv(Path(cfg.output_dir) / "combined_center_tracking_timeseries.csv", index=False)
    return results


# =============================================================================
# Plots
# =============================================================================

def get_time_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "time_gyr" in df.columns and np.all(np.isfinite(df["time_gyr"].values)):
        return df["time_gyr"].values, "Time [Gyr]"
    return df["snapshot_number"].values, "Snapshot"


def plot_label_summary(df: pd.DataFrame, outdir: Union[str, Path], label: str) -> None:
    outdir = ensure_dir(outdir)
    x, xlabel = get_time_axis(df)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), facecolor="white")
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(x, df["R_host_kpc"], marker="o", lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$R_{\rm host}$ [kpc]")
    ax.set_title("Orbital radius")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, df["V_3d_kms"], marker="o", lw=1.5, label=r"$|v|$")
    ax.plot(x, df["V_rad_kms"], marker="o", lw=1.5, label=r"$v_{rad}$")
    ax.plot(x, df["V_tan_kms"], marker="o", lw=1.5, label=r"$v_{tan}$")
    ax.axhline(0, lw=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title("Velocity decomposition")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, df["rhalf_star_all_kpc"], marker="o", lw=1.5, label="all stars")
    ax.plot(x, df["rhalf_star_member_kpc"], marker="o", lw=1.5, label="member stars")
    ax.plot(x, df["r_tidal_kpc"], marker="o", lw=1.5, label=r"$r_t$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Radius [kpc]")
    ax.set_title("Size and tidal radius")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(x, df["fstar_member"], marker="o", lw=1.5, label="member")
    ax.plot(x, df["fstar_stripped_candidate"], marker="o", lw=1.5, label="stripped candidate")
    ax.plot(x, df["fstar_stripped_definitive"], marker="o", lw=1.5, label="stripped definitive")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of tracked stars")
    ax.set_title("Membership/stripping fractions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[4]
    ax.plot(x, df["offset_all_stars_com_from_center_kpc"], marker="o", lw=1.5, label="all-stars COM offset")
    ax.plot(x, df["offset_inner_ids_com_from_center_kpc"], marker="o", lw=1.5, label="inner-IDs COM offset")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Offset [kpc]")
    ax.set_title("Centre diagnostics")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[5]
    ax.plot(x, np.log10(np.where(df["Mstar_member_msun"].values > 0, df["Mstar_member_msun"].values, np.nan)), marker="o", lw=1.5, label="member stars")
    ax.plot(x, np.log10(np.where(df["Mstar_inside_rt_msun"].values > 0, df["Mstar_inside_rt_msun"].values, np.nan)), marker="o", lw=1.5, label="inside rt")
    ax.plot(x, np.log10(np.where(df["Mstar_tracked_msun"].values > 0, df["Mstar_tracked_msun"].values, np.nan)), marker="o", lw=1.5, label="tracked")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\log(M_\star/M_\odot)$")
    ax.set_title("Stellar mass diagnostics")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{label}: {df['center_mode'].iloc[0]} centre tracking", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "center_tracking_summary.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.plot(df["x_sat_kpc"], df["y_sat_kpc"], marker="o", lw=1.5, label="satellite centre")
    ax.scatter([0], [0], marker="+", s=120, label="host")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"{label}: orbit in xy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "orbit_xy.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def compare_center_modes_for_first_label(base_cfg: CenterTrackingConfig, modes: Sequence[str]) -> pd.DataFrame:
    """
    Convenience function for Spyder: run the first label with different centre
    modes and return a compact comparison table.
    """
    label = base_cfg.labels[0] if base_cfg.labels else discover_labels(base_cfg.root)[0]
    rows = []
    for mode in modes:
        cfg = copy.deepcopy(base_cfg)
        cfg.center_mode = mode
        cfg.output_dir = str(Path(base_cfg.output_dir) / f"compare_{mode}")
        cfg.make_plots = False
        cfg.verbose = False
        df = analyse_label(label, cfg)
        first = df.iloc[0]
        last = df.iloc[-1]
        rows.append({
            "label": label,
            "center_mode": mode,
            "x0": first["x_sat_kpc"],
            "y0": first["y_sat_kpc"],
            "z0": first["z_sat_kpc"],
            "vx0": first["vx_sat_kms"],
            "vy0": first["vy_sat_kms"],
            "vz0": first["vz_sat_kms"],
            "R0": first["R_host_kpc"],
            "R_last": last["R_host_kpc"],
            "f_member_last": last["fstar_member"],
            "f_stripped_def_last": last["fstar_stripped_definitive"],
            "offset_all_com_last": last["offset_all_stars_com_from_center_kpc"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# USER SETTINGS FOR SPYDER
# =============================================================================

# In Spyder, edit only this block first, then press Run.

cfg = CenterTrackingConfig(
    root="./../SIMULATIONS/ORBIT/HigherRes",
    labels=[
        "E_mid_L_radial",
        "E_mid_L_mid",
        "E_mid_L_high",
    ],
    output_dir="orbit_center_tracking_outputs_inner_ids",

    length_unit_to_kpc=1.0,
    velocity_unit_to_kms=1.0,
    mass_unit_to_msun=1.0e10,
    time_unit_to_gyr=0.977792221,

    host=HostHaloConfig(
        host_center_kpc=(0.0, 0.0, 0.0),
        host_velocity_kms=(0.0, 0.0, 0.0),
        m200_msun=1.0e12,
        r200_kpc=210.0,
        concentration=10.0,
        tidal_factor=3.0,
    ),

    # Recommended for your current question.
    center_mode="inner_ids_shrinking",
    velocity_mode="inner_ids",

    # Define the initial central stellar region around first-snapshot stars_com.
    inner_radius_factor_rhalf=1.0,
    inner_min_radius_kpc=0.5,
    inner_max_radius_kpc=5.0,
    inner_min_particles=50,

    # Use radius if PartType1 contains both host and satellite DM.
    dm_selection_mode="all",

    # More conservative member criterion.
    member_mode="tidal_kinematic",
    member_tidal_factor=1.0,
    stripped_tidal_factor=1.5,
    v_escape_factor=1.25,
    stripped_consecutive_snapshots=2,

    make_plots=True,
    verbose=True,
)

#%%

# =============================================================================
# RUN BLOCKS FOR SPYDER
# =============================================================================

# Set to True when you want the script to run immediately after pressing Run.
RUN_ANALYSIS = True
RUN_STRUCTURE_INSPECTION = True
RUN_CENTER_MODE_COMPARISON = True

if RUN_STRUCTURE_INSPECTION:
    first_label = cfg.labels[0] if cfg.labels else discover_labels(cfg.root)[0]
    first_snapshot = find_snapshots_for_label(cfg, first_label)[0]
    inspect_snapshot_structure(first_snapshot)

if RUN_ANALYSIS:
    results = analyse_all_labels(cfg)
    combined_df = pd.concat([df.assign(label=label) for label, df in results.items()], ignore_index=True)

    cols_show = [
        "label", "snapshot_number", "time_gyr", "center_mode", "x_sat_kpc", "y_sat_kpc", "z_sat_kpc",
        "vx_sat_kms", "vy_sat_kms", "vz_sat_kms", 
        "r_tidal_kpc", "rhalf_star_member_kpc", "fstar_member", "fstar_stripped_definitive"
    ]
    cols_show = [c for c in cols_show if c in combined_df.columns]
    print("\nCombined summary, first rows:")
    print(combined_df[cols_show].head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nSaved combined file:")
    print(Path(cfg.output_dir) / "combined_center_tracking_timeseries.csv")

if RUN_CENTER_MODE_COMPARISON:
    mode_comparison_df = compare_center_modes_for_first_label(
        cfg,
        modes=["stars_com_all", "stellar_core", "inner_ids_com", "inner_ids_shrinking", "hybrid_inner_bound"],
    )
    print("\nCentre-mode comparison:")
    print(mode_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

#%%

make_plots=True
RUN_ANALYSIS=True

cfg = CenterTrackingConfig(
    root="./../SIMULATIONS/ORBIT/HigherRes",
    labels=[
        "E_mid_L_radial",
        "E_mid_L_mid",
        "E_mid_L_high",
    ],
    output_dir="orbit_center_tracking_outputs_inner_ids",

    center_mode="inner_ids_shrinking",
    velocity_mode="inner_ids",

    make_plots=True,
    verbose=True,
)

#%%

# %%
# PRINT PERICENTRES AND APOCENTRES

def find_orbital_extrema(df, r_col="R_host_kpc"):
    """
    Identify pericentres and apocentres from local extrema of R_host_kpc.

    Pericentre = local minimum of R_host_kpc
    Apocentre  = local maximum of R_host_kpc
    """

    if len(df) == 0 or r_col not in df.columns:
        return pd.DataFrame()

    r = np.asarray(df[r_col], dtype=float)

    rows = []

    for i in range(1, len(df) - 1):
        if not np.isfinite(r[i - 1]) or not np.isfinite(r[i]) or not np.isfinite(r[i + 1]):
            continue

        # Local minimum: pericentre
        if r[i] < r[i - 1] and r[i] < r[i + 1]:
            rows.append({
                "event_type": "pericentre",
                "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else int(i),
                "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else int(i),
                "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
                "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
                "V_3d_kms": float(df.iloc[i]["V_3d_kms"]) if "V_3d_kms" in df.columns else np.nan,
                "V_rad_kms": float(df.iloc[i]["V_rad_kms"]) if "V_rad_kms" in df.columns else np.nan,
                "V_tan_kms": float(df.iloc[i]["V_tan_kms"]) if "V_tan_kms" in df.columns else np.nan,
                "r_tidal_kpc": float(df.iloc[i]["r_tidal_kpc"]) if "r_tidal_kpc" in df.columns else np.nan,
                "rhalf_star_member_kpc": float(df.iloc[i]["rhalf_star_member_kpc"]) if "rhalf_star_member_kpc" in df.columns else np.nan,
            })

        # Local maximum: apocentre
        if r[i] > r[i - 1] and r[i] > r[i + 1]:
            rows.append({
                "event_type": "apocentre",
                "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else int(i),
                "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else int(i),
                "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
                "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
                "V_3d_kms": float(df.iloc[i]["V_3d_kms"]) if "V_3d_kms" in df.columns else np.nan,
                "V_rad_kms": float(df.iloc[i]["V_rad_kms"]) if "V_rad_kms" in df.columns else np.nan,
                "V_tan_kms": float(df.iloc[i]["V_tan_kms"]) if "V_tan_kms" in df.columns else np.nan,
                "r_tidal_kpc": float(df.iloc[i]["r_tidal_kpc"]) if "r_tidal_kpc" in df.columns else np.nan,
                "rhalf_star_member_kpc": float(df.iloc[i]["rhalf_star_member_kpc"]) if "rhalf_star_member_kpc" in df.columns else np.nan,
            })

    # If no pericentre was bracketed, use global minimum as candidate.
    has_peri = any(row["event_type"] == "pericentre" for row in rows)

    if not has_peri and np.any(np.isfinite(r)):
        i = int(np.nanargmin(r))
        rows.append({
            "event_type": "pericentre_candidate_global_minimum",
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else int(i),
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else int(i),
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
            "V_3d_kms": float(df.iloc[i]["V_3d_kms"]) if "V_3d_kms" in df.columns else np.nan,
            "V_rad_kms": float(df.iloc[i]["V_rad_kms"]) if "V_rad_kms" in df.columns else np.nan,
            "V_tan_kms": float(df.iloc[i]["V_tan_kms"]) if "V_tan_kms" in df.columns else np.nan,
            "r_tidal_kpc": float(df.iloc[i]["r_tidal_kpc"]) if "r_tidal_kpc" in df.columns else np.nan,
            "rhalf_star_member_kpc": float(df.iloc[i]["rhalf_star_member_kpc"]) if "rhalf_star_member_kpc" in df.columns else np.nan,
        })

    return pd.DataFrame(rows)

#%%

# %%
# PRINT EVENTS FOR EACH SIMULATION

for label, df in results.items():

    extrema_df = find_orbital_extrema(df)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    if len(extrema_df) == 0:
        print("No pericentre or apocentre detected.")
        continue

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
    ]

    cols = [c for c in cols if c in extrema_df.columns]

    print(
        extrema_df[cols]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    
#%%

results = analyse_all_labels(cfg)
combined_df = pd.concat(
    [df.assign(label=label) for label, df in results.items()],
    ignore_index=True,
)

#%%

from orbit_comparison_tools_spyder_v2 import run_all_comparison_plots

comparison_outputs = run_all_comparison_plots(
    results,
    cfg,
    output_subdir="comparison_plots_t_minus_tperi",
    x_axis_mode="time_since_first_pericentre",
    annotate_extrema=True,
)