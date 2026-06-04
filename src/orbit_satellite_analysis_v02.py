#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:07:13 2026

@author: abhner

orbit_satellite_analysis.py

Analysis pipeline for a satellite galaxy orbiting inside an analytic Milky-Way-like
dark matter halo + particle CGM, using GADGET/GADGET-4 HDF5 snapshots.

Main idea
---------
For each simulation label with snapshots in

    SIMULATIONS/ORBIT/HigherRes/[LABEL]/output/snapshot_XXX.hdf5

the code:
  1. builds an initial satellite-ID catalogue;
  2. tracks the satellite stellar core through time;
  3. computes orbital radius, velocity, radial/tangential velocity;
  4. estimates an instantaneous tidal radius using an analytic NFW host halo;
  5. measures stellar/gas/DM masses from tracked satellite IDs, both total tracked
     and inside the tidal radius;
  6. measures stellar half-mass radius and masses inside the stellar half-mass radius;
  7. measures masses inside fixed physical apertures such as 1 and 3 kpc;
  8. optionally tracks gas SFR, star-forming gas mass, and young stellar mass;
  9. identifies pericentres/apocentres and adds t - t_peri phase columns;
 10. saves time-evolution tables and plots;
 11. optionally makes yt projection maps and GIFs.

Designed for use in a Jupyter notebook or as a standalone script in Spyder/terminal.

Important assumptions
---------------------
Default GADGET particle types:
    PartType0 = gas
    PartType1 = dark matter
    PartType4 = stars

Your setup:
    - all star particles belong to the satellite;
    - PartType1 DM particles are assumed to belong to the satellite by default;
    - gas has two origins: satellite gas + host CGM gas.
      The initial satellite gas IDs are selected around the initial stellar core.
      After that, gas is tracked by ParticleIDs.

Unit defaults are common for isolated GADGET runs:
    length_unit_to_kpc = 1
    velocity_unit_to_kms = 1
    mass_unit_to_msun = 1e10
    time_unit_to_gyr = 0.977792221

If your snapshot masses are already in Msun, set:
    mass_unit_to_msun = 1

Dependencies
------------
Required:
    numpy, pandas, h5py, matplotlib

Optional:
    yt, imageio
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import math
import warnings

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Physical constant in useful units:
# G = 4.30091e-6 kpc (km/s)^2 Msun^-1
G_KPC_KMS2_MSUN = 4.30091e-6
KPC_IN_CM = 3.0856775814913673e21
MSUN_IN_G = 1.98847e33


ArrayLike3 = Union[Sequence[float], np.ndarray]


@dataclass
class HostHaloConfig:
    """
    Analytic host-halo model.

    The default is an NFW halo roughly appropriate for a Milky-Way-like system.
    Adjust m200_msun, r200_kpc, and concentration for your adopted model.
    """

    host_center_kpc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    host_velocity_kms: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    m200_msun: float = 1.0e12
    r200_kpc: float = 210.0
    concentration: float = 10.0

    # Tidal-radius prefactor:
    # r_t = R * [m_sat(<r_t)/(tidal_factor * M_host(<R))]^(1/3)
    # 3 is common for circular-orbit approximation in a point-mass-like potential.
    tidal_factor: float = 3.0

    # If True, M_host(<R) is capped at M200 for R > R200.
    truncate_mass_at_r200: bool = False

    def f_nfw(self, x: np.ndarray | float) -> np.ndarray | float:
        return np.log1p(x) - x / (1.0 + x)

    def mass_enclosed_msun(self, radius_kpc: np.ndarray | float) -> np.ndarray | float:
        """NFW enclosed mass M(<r) in Msun."""
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
class AnalysisConfig:
    """
    Configuration for the full analysis.

    Change the unit conversion factors to match your IC/snapshot units.
    """

    root: str = "./../SIMULATIONS/ORBIT/HigherRes"
    labels: Optional[List[str]] = None
    output_dir: str = "orbit_analysis_outputs"

    snapshot_glob: str = "snapshot_*.hdf5"

    # GADGET particle types
    gas_ptype: int = 0
    dm_ptype: int = 1
    star_ptype: int = 4

    # Unit conversions from snapshot code units to physical units.
    length_unit_to_kpc: float = 1.0
    velocity_unit_to_kms: float = 1.0
    mass_unit_to_msun: float = 1.0e10

    # If code length is kpc and velocity is km/s, 1 code-time = kpc/(km/s)
    # = 0.977792221 Gyr.
    time_unit_to_gyr: float = 0.977792221

    host: HostHaloConfig = field(default_factory=HostHaloConfig)

    # Initial satellite gas selection.
    # If None, the code uses max(default_initial_gas_radius_kpc, gas_radius_factor_rhalf * rhalf_star_initial).
    initial_satellite_gas_radius_kpc: Optional[float] = None
    default_initial_gas_radius_kpc: float = 30.0
    gas_radius_factor_rhalf: float = 8.0

    # Initial satellite DM selection.
    # "all": all PartType1 particles are the satellite DM.
    # "radius": only PartType1 particles initially close to the satellite are tracked.
    dm_selection_mode: str = "all"
    initial_satellite_dm_radius_kpc: Optional[float] = None
    default_initial_dm_radius_kpc: float = 80.0
    dm_radius_factor_rhalf: float = 20.0

    # Center finding with shrinking spheres.
    shrink_initial_radius_kpc: Optional[float] = None
    shrink_factor: float = 0.75
    shrink_min_particles: int = 100
    shrink_min_radius_kpc: float = 0.2
    center_search_radius_kpc: float = 40.0
    velocity_center_radius_factor_rhalf: float = 1.0
    velocity_center_min_radius_kpc: float = 1.0

    # Tidal-radius solver.
    tidal_max_iterations: int = 40
    tidal_tolerance: float = 1e-3
    tidal_initial_fraction_of_R: float = 0.15
    tidal_max_fraction_of_R: float = 0.5
    tidal_min_radius_kpc: float = 0.05

    # Plots and maps.
    make_maps: bool = True
    make_gifs: bool = True
    gif_fps: int = 5
    gif_stride: int = 1

    # Plot settings. If True, masses and radii are plotted as log(quantity/unit)
    # instead of using a logarithmic y-axis. Labels use "log" for readability.
    plot_log10_masses_and_radii: bool = True
    annotate_orbital_extrema: bool = True
    event_window_snapshots: int = 1

    # Orbital phase alignment. The table will receive t_minus_tperi_* columns.
    pericentre_reference: str = "first"  # "first" or "deepest"

    # Additional aperture diagnostics. These are physical 3D radii around the
    # tracked stellar-core centre. Column names use tags such as 1kpc and 3kpc.
    fixed_aperture_radii_kpc: Tuple[float, ...] = (1.0, 3.0)

    # Star-formation diagnostics. SFR is read from gas fields such as
    # StarFormationRate/SFR, assumed to be in Msun/yr unless the factor below
    # is changed. Star-forming gas is defined as SFR > threshold.
    compute_star_formation: bool = True
    sfr_unit_to_msun_per_yr: float = 1.0
    star_forming_sfr_threshold_msun_per_yr: float = 0.0

    # Young-star diagnostics. For isolated Gadget runs, star birth times are
    # usually stored in code-time units, so the default mode is "code_time".
    # Other possible modes: "gyr", "scale_factor", "auto".
    young_star_age_myr: float = 100.0
    stellar_birth_time_mode: str = "code_time"

    # Your setup has no host stellar component, so all PartType4 particles can
    # be interpreted as satellite stars. This also captures newly formed stars
    # whose ParticleIDs were not present in the initial stellar ID catalogue.
    include_all_stars_as_satellite: bool = True

    # Map/GIF settings. For the most consistent GIFs, the default backend is
    # a direct matplotlib particle projection. yt can still be used by setting
    # map_backend="yt", but global vmin/vmax are most robust with matplotlib.
    map_backend: str = "matplotlib"  # "matplotlib", "yt", or "auto"
    map_auto_width_include_orbit: bool = True
    map_margin_kpc: float = 20.0
    map_vmin: Optional[float] = None
    map_vmax: Optional[float] = None
    map_vlim_percentiles: Tuple[float, float] = (1.0, 99.5)
    map_cmap: str = "viridis"
    map_white_background: bool = True

    # Ram-pressure estimate. The code estimates the local CGM density around the
    # satellite excluding the initially identified satellite gas IDs. If the
    # snapshot has a gas Density field, it uses a mass-weighted local density;
    # otherwise it estimates density from CGM gas mass inside a spherical aperture.
    compute_ram_pressure: bool = True
    ram_pressure_density_radius_kpc: float = 10.0
    ram_pressure_max_density_radius_kpc: float = 30.0
    ram_pressure_min_cgm_particles: int = 16
    ram_pressure_velocity_mode: str = "local_cgm"  # "local_cgm" or "host_frame"
    ram_pressure_use_cgm_only: bool = True

    # yt projection settings.
    map_axis: str = "z"
    map_width_kpc: float = 250.0
    map_bbox_kpc: float = 400.0
    map_field_candidates: Tuple[Tuple[str, str], ...] = (
        ("gas", "density"),
        ("PartType0", "Density"),
        ("PartType0", "density"),
    )

    # Matplotlib fallback map settings.
    fallback_map_npix: int = 600

    # Save per-snapshot diagnostic JSON?
    save_snapshot_json: bool = False

    # Verbose console output.
    verbose: bool = True


def natural_snapshot_number(path: Path) -> int:
    """Extract the XXX number from snapshot_XXX.hdf5. Returns -1 if absent."""
    stem = path.stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if not digits:
        return -1
    return int(digits[-1])


def discover_labels(root: Union[str, Path]) -> List[str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    labels = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "output").exists():
            labels.append(p.name)
    return labels


def find_snapshots_for_label(cfg: AnalysisConfig, label: str) -> List[Path]:
    output_dir = Path(cfg.root) / label / "output"
    snaps = sorted(output_dir.glob(cfg.snapshot_glob), key=natural_snapshot_number)
    if len(snaps) == 0:
        raise FileNotFoundError(f"No snapshots found for label={label} in {output_dir}")
    return snaps


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    """
    Print groups, datasets, shapes, and header attributes.
    Useful for checking whether your snapshots use standard GADGET field names.
    """
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


def read_particle_type(
    snapshot_file: Union[str, Path],
    ptype: int,
    cfg: AnalysisConfig,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Read positions, velocities, IDs, and masses for a given GADGET PartType.

    Returns None if the particle type is absent.
    """
    snapshot_file = Path(snapshot_file)
    group_name = f"PartType{ptype}"

    with h5py.File(snapshot_file, "r") as f:
        if group_name not in f:
            return None

        g = f[group_name]

        pos = dataset_first_available(g, ["Coordinates", "Position", "Positions"])
        vel = dataset_first_available(g, ["Velocities", "Velocity"])
        ids = dataset_first_available(g, ["ParticleIDs", "ParticleID", "IDs", "ID"])

        if pos is None or vel is None or ids is None:
            raise KeyError(
                f"Missing Coordinates/Velocities/ParticleIDs in {snapshot_file}:{group_name}. "
                "Run inspect_snapshot_structure() to check field names."
            )

        masses = dataset_first_available(g, ["Masses", "Mass", "ParticleMasses"])

        if masses is None:
            mass_table = header_attr(f, "MassTable", None)
            if mass_table is None:
                raise KeyError(
                    f"No Masses dataset and no Header/MassTable found for {group_name} in {snapshot_file}"
                )
            m_code = float(mass_table[ptype])
            if m_code <= 0:
                raise ValueError(
                    f"MassTable[{ptype}] is zero but no individual Masses dataset exists in {snapshot_file}"
                )
            masses = np.full(len(ids), m_code, dtype=float)

        # Optional gas-density field. GADGET stores density in code units
        # of mass / length^3. We convert to Msun/kpc^3.
        density = dataset_first_available(g, ["Density", "Densities", "rho", "Rho"])

        # Optional gas star-formation-rate field. Usually Msun/yr in TNG-like
        # files, but cfg.sfr_unit_to_msun_per_yr can be used to rescale it.
        sfr = dataset_first_available(
            g,
            ["StarFormationRate", "StarFormationRates", "SFR", "Sfr", "sfr"],
        )

        # Optional stellar birth/formation time field. In TNG this is often
        # GFM_StellarFormationTime, while in isolated Gadget runs it may be a
        # code time. Conversion is handled later because it depends on the
        # snapshot time and cfg.stellar_birth_time_mode.
        formation_time = dataset_first_available(
            g,
            [
                "GFM_StellarFormationTime", "StellarFormationTime",
                "FormationTime", "BirthTime", "BirthTimes",
                "StarFormationTime", "StellarAge",
            ],
        )

        # Convert units.
        pos = np.asarray(pos, dtype=float) * cfg.length_unit_to_kpc
        vel = np.asarray(vel, dtype=float) * cfg.velocity_unit_to_kms
        ids = np.asarray(ids, dtype=np.int64)
        masses = np.asarray(masses, dtype=float) * cfg.mass_unit_to_msun

        data = {
            "pos": pos,
            "vel": vel,
            "ids": ids,
            "mass": masses,
        }

        if density is not None:
            density = np.asarray(density, dtype=float)
            density *= cfg.mass_unit_to_msun / (cfg.length_unit_to_kpc ** 3)
            data["density"] = density

        if sfr is not None:
            data["sfr_msun_per_yr"] = np.asarray(sfr, dtype=float) * cfg.sfr_unit_to_msun_per_yr

        if formation_time is not None:
            data["formation_time_raw"] = np.asarray(formation_time, dtype=float)

        return data


def read_snapshot_time_gyr(snapshot_file: Union[str, Path], cfg: AnalysisConfig) -> float:
    with h5py.File(snapshot_file, "r") as f:
        t_code = header_attr(f, "Time", np.nan)
    try:
        return float(t_code) * cfg.time_unit_to_gyr
    except Exception:
        return np.nan


def mask_ids(ids: np.ndarray, selected_ids: Optional[np.ndarray]) -> np.ndarray:
    if selected_ids is None:
        return np.ones(len(ids), dtype=bool)
    if len(ids) == 0 or len(selected_ids) == 0:
        return np.zeros(len(ids), dtype=bool)
    return np.isin(ids, selected_ids, assume_unique=False)


def mass_weighted_mean(values: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    if values.size == 0:
        return np.full(values.shape[1] if values.ndim > 1 else 1, np.nan)
    if masses is None or np.sum(masses) <= 0:
        return np.mean(values, axis=0)
    return np.average(values, axis=0, weights=masses)


def distances(pos: np.ndarray, center: ArrayLike3) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    return np.linalg.norm(pos - center[None, :], axis=1)


def half_mass_radius(pos: np.ndarray, masses: np.ndarray, center: ArrayLike3) -> float:
    """
    3D half-mass radius.
    """
    if len(pos) == 0 or np.sum(masses) <= 0:
        return np.nan

    r = distances(pos, center)
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = masses[order]
    cum = np.cumsum(m_sorted)
    half = 0.5 * cum[-1]
    idx = np.searchsorted(cum, half)
    idx = min(idx, len(r_sorted) - 1)
    return float(r_sorted[idx])


def enclosed_radius_fraction(
    pos: np.ndarray,
    masses: np.ndarray,
    center: ArrayLike3,
    fraction: float = 0.9,
) -> float:
    if len(pos) == 0 or np.sum(masses) <= 0:
        return np.nan
    r = distances(pos, center)
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = masses[order]
    cum = np.cumsum(m_sorted)
    target = fraction * cum[-1]
    idx = np.searchsorted(cum, target)
    idx = min(idx, len(r_sorted) - 1)
    return float(r_sorted[idx])


def shrinking_sphere_center(
    pos: np.ndarray,
    masses: np.ndarray,
    initial_center: Optional[ArrayLike3] = None,
    initial_radius: Optional[float] = None,
    shrink_factor: float = 0.75,
    min_particles: int = 100,
    min_radius: float = 0.2,
) -> np.ndarray:
    """
    Find a robust stellar-core center using a shrinking-sphere method.

    This is useful when stripped stars form tidal tails and would bias a simple COM.
    """
    if len(pos) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    if initial_center is None:
        center = mass_weighted_mean(pos, masses)
    else:
        center = np.asarray(initial_center, dtype=float)

    r = distances(pos, center)

    if initial_radius is None:
        # Start from a robust large radius instead of the maximum, which can be
        # dominated by far stripped particles.
        initial_radius = float(np.nanpercentile(r, 90))
        if not np.isfinite(initial_radius) or initial_radius <= 0:
            initial_radius = float(np.nanmax(r))

    radius = max(float(initial_radius), min_radius)

    current = np.arange(len(pos))
    while True:
        r_current = distances(pos[current], center)
        inside = current[r_current <= radius]

        if len(inside) < max(min_particles, 10) or radius <= min_radius:
            if len(inside) >= 5:
                current = inside
            break

        center = mass_weighted_mean(pos[inside], masses[inside])
        current = inside
        radius *= shrink_factor

    if len(current) == 0:
        return center

    return mass_weighted_mean(pos[current], masses[current])


def select_initial_satellite_ids(
    first_snapshot: Union[str, Path],
    cfg: AnalysisConfig,
) -> Dict[str, np.ndarray | float | List[float]]:
    """
    Build initial ID catalogues for satellite stars, gas, and dark matter.

    Since all star particles are from the satellite, all PartType4 IDs are used.
    For gas, we select particles initially close to the stellar core.
    For DM, default is all PartType1 particles; this matches your setup if there is
    no live host DM halo.
    """
    stars = read_particle_type(first_snapshot, cfg.star_ptype, cfg)
    if stars is None or len(stars["ids"]) == 0:
        raise RuntimeError(
            f"No star particles found in {first_snapshot}. "
            "The center tracking in this script assumes satellite stars exist."
        )

    initial_center = shrinking_sphere_center(
        stars["pos"],
        stars["mass"],
        initial_radius=cfg.shrink_initial_radius_kpc,
        shrink_factor=cfg.shrink_factor,
        min_particles=cfg.shrink_min_particles,
        min_radius=cfg.shrink_min_radius_kpc,
    )
    rhalf0 = half_mass_radius(stars["pos"], stars["mass"], initial_center)

    gas_radius = cfg.initial_satellite_gas_radius_kpc
    if gas_radius is None:
        if np.isfinite(rhalf0):
            gas_radius = max(cfg.default_initial_gas_radius_kpc, cfg.gas_radius_factor_rhalf * rhalf0)
        else:
            gas_radius = cfg.default_initial_gas_radius_kpc

    dm_radius = cfg.initial_satellite_dm_radius_kpc
    if dm_radius is None:
        if np.isfinite(rhalf0):
            dm_radius = max(cfg.default_initial_dm_radius_kpc, cfg.dm_radius_factor_rhalf * rhalf0)
        else:
            dm_radius = cfg.default_initial_dm_radius_kpc

    gas = read_particle_type(first_snapshot, cfg.gas_ptype, cfg)
    if gas is not None and len(gas["ids"]) > 0:
        rg = distances(gas["pos"], initial_center)
        gas_ids = gas["ids"][rg <= gas_radius]
    else:
        gas_ids = np.array([], dtype=np.int64)

    dm = read_particle_type(first_snapshot, cfg.dm_ptype, cfg)
    if dm is not None and len(dm["ids"]) > 0:
        if cfg.dm_selection_mode.lower() == "all":
            dm_ids = dm["ids"]
        elif cfg.dm_selection_mode.lower() == "radius":
            rd = distances(dm["pos"], initial_center)
            dm_ids = dm["ids"][rd <= dm_radius]
        else:
            raise ValueError("cfg.dm_selection_mode must be 'all' or 'radius'")
    else:
        dm_ids = np.array([], dtype=np.int64)

    return {
        "star_ids": np.unique(stars["ids"]),
        "gas_ids": np.unique(gas_ids),
        "dm_ids": np.unique(dm_ids),
        "initial_center_kpc": initial_center.tolist(),
        "initial_rhalf_star_kpc": float(rhalf0),
        "initial_gas_selection_radius_kpc": float(gas_radius),
        "initial_dm_selection_radius_kpc": float(dm_radius),
    }


def empty_particle_dict() -> Dict[str, np.ndarray]:
    return {
        "pos": np.empty((0, 3), dtype=float),
        "vel": np.empty((0, 3), dtype=float),
        "ids": np.empty(0, dtype=np.int64),
        "mass": np.empty(0, dtype=float),
        "density": np.empty(0, dtype=float),
        "sfr_msun_per_yr": np.empty(0, dtype=float),
        "formation_time_raw": np.empty(0, dtype=float),
    }


def filter_particle_dict(
    pdata: Optional[Dict[str, np.ndarray]],
    selected_ids: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Return only particles with IDs in selected_ids, preserving optional fields such as density.
    """
    if pdata is None:
        return empty_particle_dict()

    m = mask_ids(pdata["ids"], selected_ids)
    out = {}
    for k, v in pdata.items():
        if isinstance(v, np.ndarray) and len(v) == len(pdata["ids"]):
            out[k] = v[m]
        else:
            out[k] = v
    if "density" not in out:
        out["density"] = np.empty(0, dtype=float)
    if "sfr_msun_per_yr" not in out:
        out["sfr_msun_per_yr"] = np.empty(0, dtype=float)
    if "formation_time_raw" not in out:
        out["formation_time_raw"] = np.empty(0, dtype=float)
    return out


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
    cfg: AnalysisConfig,
) -> float:
    """
    Iterative tidal-radius estimate.

    Uses:
        r_t = R * [m_sat(<r_t)/(tidal_factor * M_host(<R))]^(1/3)

    This is an approximate diagnostic, not a full bound-particle finder.
    """
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


def component_masses(
    comp: Dict[str, np.ndarray],
    center: ArrayLike3,
    tidal_radius_kpc: float,
    rhalf_star_kpc: float,
) -> Dict[str, float]:
    """
    Compute total tracked mass, mass inside tidal radius, and mass inside stellar half-mass radius.
    """
    if len(comp["mass"]) == 0:
        return {
            "tracked": 0.0,
            "inside_rt": 0.0,
            "inside_rhalf_star": 0.0,
            "n_tracked": 0,
            "n_inside_rt": 0,
            "n_inside_rhalf_star": 0,
        }

    r = distances(comp["pos"], center)

    inside_rt = r <= tidal_radius_kpc if np.isfinite(tidal_radius_kpc) else np.zeros(len(r), dtype=bool)
    inside_rh = r <= rhalf_star_kpc if np.isfinite(rhalf_star_kpc) else np.zeros(len(r), dtype=bool)

    return {
        "tracked": float(np.sum(comp["mass"])),
        "inside_rt": float(np.sum(comp["mass"][inside_rt])),
        "inside_rhalf_star": float(np.sum(comp["mass"][inside_rh])),
        "n_tracked": int(len(comp["mass"])),
        "n_inside_rt": int(np.sum(inside_rt)),
        "n_inside_rhalf_star": int(np.sum(inside_rh)),
    }




def radius_tag_kpc(radius_kpc: float) -> str:
    """Return a compact column-name tag for a physical radius in kpc."""
    r = float(radius_kpc)
    if abs(r - round(r)) < 1e-8:
        return f"{int(round(r))}kpc"
    return (f"{r:.3g}".replace(".", "p").replace("-", "m") + "kpc")


def mass_count_within_radius(
    comp: Dict[str, np.ndarray],
    center: ArrayLike3,
    radius_kpc: float,
) -> Tuple[float, int]:
    """Mass and particle count inside a fixed 3D radius."""
    if comp is None or len(comp.get("mass", [])) == 0 or not np.isfinite(radius_kpc):
        return 0.0, 0
    r = distances(comp["pos"], center)
    inside = r <= radius_kpc
    return float(np.sum(comp["mass"][inside])), int(np.sum(inside))


def fixed_aperture_component_masses(
    comp: Dict[str, np.ndarray],
    center: ArrayLike3,
    radii_kpc: Sequence[float],
) -> Dict[str, float]:
    """Return mass/count entries inside each requested fixed physical aperture."""
    out: Dict[str, float] = {}
    for radius in radii_kpc:
        tag = radius_tag_kpc(radius)
        mass, n = mass_count_within_radius(comp, center, float(radius))
        out[f"inside_{tag}"] = mass
        out[f"n_inside_{tag}"] = n
    return out


def gas_star_formation_quantities(
    gas: Dict[str, np.ndarray],
    center: ArrayLike3,
    tidal_radius_kpc: float,
    rhalf_star_kpc: float,
    fixed_radii_kpc: Sequence[float],
    cfg: AnalysisConfig,
) -> Dict[str, float]:
    """
    Compute SFR and star-forming gas mass for tracked satellite gas.

    The gas must contain an optional array `sfr_msun_per_yr`. If absent, the
    output is filled with NaNs for SFR and zeros for star-forming gas mass.
    Star-forming gas is defined as SFR > cfg.star_forming_sfr_threshold_msun_per_yr.
    """
    out: Dict[str, float] = {
        "SFR_tracked_msun_per_yr": np.nan,
        "SFR_inside_rt_msun_per_yr": np.nan,
        "SFR_inside_rhalf_msun_per_yr": np.nan,
        "Mgas_SF_tracked_msun": np.nan,
        "Mgas_SF_inside_rt_msun": np.nan,
        "Mgas_SF_inside_rhalf_msun": np.nan,
        "Ngas_SF_tracked": 0,
        "Ngas_SF_inside_rt": 0,
        "Ngas_SF_inside_rhalf": 0,
    }

    for radius in fixed_radii_kpc:
        tag = radius_tag_kpc(radius)
        out[f"SFR_inside_{tag}_msun_per_yr"] = np.nan
        out[f"Mgas_SF_inside_{tag}_msun"] = np.nan
        out[f"Ngas_SF_inside_{tag}"] = 0

    if gas is None or len(gas.get("mass", [])) == 0:
        return out

    sfr = gas.get("sfr_msun_per_yr", np.empty(0, dtype=float))
    if not isinstance(sfr, np.ndarray) or len(sfr) != len(gas["mass"]):
        return out

    sfr = np.asarray(sfr, dtype=float)
    sfr_finite = np.where(np.isfinite(sfr), sfr, 0.0)
    sf = sfr_finite > float(cfg.star_forming_sfr_threshold_msun_per_yr)
    r = distances(gas["pos"], center)

    def fill(prefix: str, mask: np.ndarray):
        out[f"SFR_{prefix}_msun_per_yr"] = float(np.sum(sfr_finite[mask]))
        out[f"Mgas_SF_{prefix}_msun"] = float(np.sum(gas["mass"][mask & sf]))
        out[f"Ngas_SF_{prefix}"] = int(np.sum(mask & sf))

    all_mask = np.ones(len(r), dtype=bool)
    fill("tracked", all_mask)

    if np.isfinite(tidal_radius_kpc):
        fill("inside_rt", r <= tidal_radius_kpc)
    if np.isfinite(rhalf_star_kpc):
        fill("inside_rhalf", r <= rhalf_star_kpc)

    for radius in fixed_radii_kpc:
        tag = radius_tag_kpc(radius)
        mask = r <= float(radius)
        out[f"SFR_inside_{tag}_msun_per_yr"] = float(np.sum(sfr_finite[mask]))
        out[f"Mgas_SF_inside_{tag}_msun"] = float(np.sum(gas["mass"][mask & sf]))
        out[f"Ngas_SF_inside_{tag}"] = int(np.sum(mask & sf))

    return out


def stellar_birth_time_to_gyr(
    formation_time_raw: np.ndarray,
    snapshot_time_gyr: float,
    cfg: AnalysisConfig,
) -> np.ndarray:
    """
    Convert a stellar formation/birth-time field to Gyr in the simulation clock.

    For isolated Gadget runs, use cfg.stellar_birth_time_mode='code_time'.
    For fields already in Gyr, use 'gyr'. The 'scale_factor' mode is included
    for TNG-like initial-star metadata but cannot define recent star formation
    in an isolated run without a cosmological time conversion, so it returns NaN.
    """
    ft = np.asarray(formation_time_raw, dtype=float)
    mode = str(cfg.stellar_birth_time_mode).lower()

    if mode == "code_time":
        return ft * cfg.time_unit_to_gyr
    if mode == "gyr":
        return ft
    if mode == "scale_factor":
        return np.full_like(ft, np.nan, dtype=float)
    if mode == "auto":
        # Heuristic: if values are within the isolated runtime scale, treat as
        # code time. If they look like cosmological scale factors near 0-1 while
        # the isolated time is small, do not classify young stars from them.
        good = ft[np.isfinite(ft)]
        if len(good) == 0:
            return np.full_like(ft, np.nan, dtype=float)
        if np.nanmax(good) <= max(10.0, snapshot_time_gyr / max(cfg.time_unit_to_gyr, 1e-30) + 1.0):
            return ft * cfg.time_unit_to_gyr
        return np.full_like(ft, np.nan, dtype=float)

    raise ValueError("cfg.stellar_birth_time_mode must be 'code_time', 'gyr', 'scale_factor', or 'auto'")


def young_star_quantities(
    stars: Dict[str, np.ndarray],
    center: ArrayLike3,
    snapshot_time_gyr: float,
    tidal_radius_kpc: float,
    rhalf_star_kpc: float,
    fixed_radii_kpc: Sequence[float],
    cfg: AnalysisConfig,
) -> Dict[str, float]:
    """Compute mass/count of stars younger than cfg.young_star_age_myr."""
    age_limit_gyr = float(cfg.young_star_age_myr) / 1000.0

    out: Dict[str, float] = {
        "Mstar_young_tracked_msun": np.nan,
        "Mstar_young_inside_rt_msun": np.nan,
        "Mstar_young_inside_rhalf_msun": np.nan,
        "Nstar_young_tracked": 0,
        "Nstar_young_inside_rt": 0,
        "Nstar_young_inside_rhalf": 0,
    }
    for radius in fixed_radii_kpc:
        tag = radius_tag_kpc(radius)
        out[f"Mstar_young_inside_{tag}_msun"] = np.nan
        out[f"Nstar_young_inside_{tag}"] = 0

    if stars is None or len(stars.get("mass", [])) == 0:
        return out

    ft = stars.get("formation_time_raw", np.empty(0, dtype=float))
    if not isinstance(ft, np.ndarray) or len(ft) != len(stars["mass"]):
        return out

    birth_gyr = stellar_birth_time_to_gyr(ft, snapshot_time_gyr, cfg)
    age = snapshot_time_gyr - birth_gyr
    young = np.isfinite(age) & (age >= 0.0) & (age <= age_limit_gyr)
    r = distances(stars["pos"], center)

    def fill(prefix: str, mask: np.ndarray):
        out[f"Mstar_young_{prefix}_msun"] = float(np.sum(stars["mass"][mask & young]))
        out[f"Nstar_young_{prefix}"] = int(np.sum(mask & young))

    all_mask = np.ones(len(r), dtype=bool)
    fill("tracked", all_mask)

    if np.isfinite(tidal_radius_kpc):
        fill("inside_rt", r <= tidal_radius_kpc)
    if np.isfinite(rhalf_star_kpc):
        fill("inside_rhalf", r <= rhalf_star_kpc)

    for radius in fixed_radii_kpc:
        tag = radius_tag_kpc(radius)
        mask = r <= float(radius)
        out[f"Mstar_young_inside_{tag}_msun"] = float(np.sum(stars["mass"][mask & young]))
        out[f"Nstar_young_inside_{tag}"] = int(np.sum(mask & young))

    return out



def log10_safe_values(values: np.ndarray) -> np.ndarray:
    """
    Return log10(values), with non-positive values mapped to NaN.
    """
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, np.log10(values), np.nan)


def pressure_unit_msun_kpc3_kms2_to_dyne_cm2() -> float:
    """
    Conversion factor from (Msun/kpc^3) (km/s)^2 to dyne/cm^2.
    """
    return (MSUN_IN_G / (KPC_IN_CM ** 3)) * (1.0e5 ** 2)


def compute_local_cgm_ram_pressure(
    gas_all: Optional[Dict[str, np.ndarray]],
    satellite_gas_ids: np.ndarray,
    center_kpc: ArrayLike3,
    v_sat_kms: ArrayLike3,
    cfg: AnalysisConfig,
) -> Dict[str, float]:
    """
    Estimate ram pressure at the satellite position.

    P_ram = rho_CGM * |v_sat - v_CGM|^2.

    rho_CGM is estimated around the satellite using host/CGM gas particles.
    By default, particles whose IDs were tagged as initial satellite gas are
    excluded, so the estimate is based on the ambient CGM, not on the satellite ISM.

    If the gas particle dictionary contains a Density field, the local density is
    the mass-weighted mean density of CGM gas inside the aperture. If not, the code
    estimates density as M_CGM(<R_ap) / (4*pi*R_ap^3/3).
    """
    keys = {
        "rho_cgm_msun_kpc3": np.nan,
        "rho_cgm_aperture_msun_kpc3": np.nan,
        "rho_cgm_density_field_msun_kpc3": np.nan,
        "N_cgm_for_ram": 0,
        "ram_pressure_radius_kpc": np.nan,
        "V_rel_cgm_kms": np.nan,
        "P_ram_msun_kpc3_kms2": np.nan,
        "P_ram_dyne_cm2": np.nan,
        "log10_P_ram_dyne_cm2": np.nan,
    }

    if (not cfg.compute_ram_pressure) or gas_all is None or len(gas_all.get("ids", [])) == 0:
        return keys

    pos = gas_all["pos"]
    vel = gas_all["vel"]
    mass = gas_all["mass"]
    ids = gas_all["ids"]

    if cfg.ram_pressure_use_cgm_only:
        is_cgm = ~np.isin(ids, satellite_gas_ids, assume_unique=False)
    else:
        is_cgm = np.ones(len(ids), dtype=bool)

    if np.sum(is_cgm) == 0:
        return keys

    center = np.asarray(center_kpc, dtype=float)
    v_sat = np.asarray(v_sat_kms, dtype=float)

    pos_cgm = pos[is_cgm]
    vel_cgm = vel[is_cgm]
    mass_cgm = mass[is_cgm]
    r = distances(pos_cgm, center)

    aperture = float(cfg.ram_pressure_density_radius_kpc)
    max_ap = float(cfg.ram_pressure_max_density_radius_kpc)
    min_n = int(cfg.ram_pressure_min_cgm_particles)

    while aperture < max_ap and np.sum(r <= aperture) < min_n:
        aperture = min(max_ap, aperture * 1.5)
        if aperture >= max_ap:
            break

    use = r <= aperture
    n_use = int(np.sum(use))
    keys["N_cgm_for_ram"] = n_use
    keys["ram_pressure_radius_kpc"] = aperture

    if n_use == 0:
        # No nearby CGM particles: fall back to zero density but keep velocity relative to host.
        v_cgm = np.asarray(cfg.host.host_velocity_kms, dtype=float)
        v_rel = float(np.linalg.norm(v_sat - v_cgm))
        keys["V_rel_cgm_kms"] = v_rel
        return keys

    volume = (4.0 / 3.0) * np.pi * aperture ** 3
    rho_ap = float(np.sum(mass_cgm[use]) / volume) if volume > 0 else np.nan
    keys["rho_cgm_aperture_msun_kpc3"] = rho_ap

    rho_field = np.nan
    density_all = gas_all.get("density", np.empty(0, dtype=float))
    if isinstance(density_all, np.ndarray) and len(density_all) == len(ids):
        density_cgm = density_all[is_cgm]
        finite_density = np.isfinite(density_cgm[use]) & (density_cgm[use] > 0)
        if np.any(finite_density):
            rho_field = float(
                np.average(
                    density_cgm[use][finite_density],
                    weights=mass_cgm[use][finite_density],
                )
            )
    keys["rho_cgm_density_field_msun_kpc3"] = rho_field

    if np.isfinite(rho_field) and rho_field > 0:
        rho = rho_field
    else:
        rho = rho_ap
    keys["rho_cgm_msun_kpc3"] = rho

    if cfg.ram_pressure_velocity_mode.lower() == "host_frame":
        v_cgm = np.asarray(cfg.host.host_velocity_kms, dtype=float)
    else:
        v_cgm = mass_weighted_mean(vel_cgm[use], mass_cgm[use])
        if not np.all(np.isfinite(v_cgm)):
            v_cgm = np.asarray(cfg.host.host_velocity_kms, dtype=float)

    v_rel = float(np.linalg.norm(v_sat - v_cgm))
    keys["V_rel_cgm_kms"] = v_rel

    if np.isfinite(rho) and rho > 0 and np.isfinite(v_rel):
        p_code = rho * v_rel ** 2
        p_cgs = p_code * pressure_unit_msun_kpc3_kms2_to_dyne_cm2()
        keys["P_ram_msun_kpc3_kms2"] = float(p_code)
        keys["P_ram_dyne_cm2"] = float(p_cgs)
        keys["log10_P_ram_dyne_cm2"] = float(np.log10(p_cgs)) if p_cgs > 0 else np.nan

    return keys

def analyze_one_snapshot(
    snapshot_file: Union[str, Path],
    snapshot_index: int,
    cfg: AnalysisConfig,
    idcat: Dict[str, np.ndarray],
    previous_center_kpc: Optional[np.ndarray] = None,
    previous_rhalf_kpc: Optional[float] = None,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Analyze one snapshot and return:
        row dict, current satellite center, current stellar rhalf.
    """
    snapshot_file = Path(snapshot_file)

    # Read particles.
    stars_all = read_particle_type(snapshot_file, cfg.star_ptype, cfg)
    gas_all = read_particle_type(snapshot_file, cfg.gas_ptype, cfg)
    dm_all = read_particle_type(snapshot_file, cfg.dm_ptype, cfg)

    stars_initial = filter_particle_dict(stars_all, idcat["star_ids"])
    gas = filter_particle_dict(gas_all, idcat["gas_ids"])
    dm = filter_particle_dict(dm_all, idcat["dm_ids"])

    if len(stars_initial["ids"]) == 0:
        raise RuntimeError(f"No tracked initial star IDs found in {snapshot_file}")

    # If the run forms new PartType4 particles, their IDs were not present in
    # the initial catalogue. Since your setup has no host stellar component, the
    # safest way to include newly formed stars in mass/young-star diagnostics is
    # to treat all current PartType4 particles as satellite stars. The tracked
    # initial stars remain available as stars_initial if needed.
    if cfg.include_all_stars_as_satellite and stars_all is not None:
        stars = stars_all
        if "density" not in stars:
            stars["density"] = np.empty(0, dtype=float)
        if "sfr_msun_per_yr" not in stars:
            stars["sfr_msun_per_yr"] = np.empty(0, dtype=float)
        if "formation_time_raw" not in stars:
            stars["formation_time_raw"] = np.empty(0, dtype=float)
    else:
        stars = stars_initial

    # Center finding.
    if previous_center_kpc is not None and np.all(np.isfinite(previous_center_kpc)):
        # Restrict to stars close to previous center if possible, to avoid being pulled by stripped tails.
        rprev = distances(stars["pos"], previous_center_kpc)
        search_radius = cfg.center_search_radius_kpc
        if previous_rhalf_kpc is not None and np.isfinite(previous_rhalf_kpc):
            search_radius = max(search_radius, 10.0 * previous_rhalf_kpc)
        near = rprev <= search_radius

        if np.sum(near) >= max(20, cfg.shrink_min_particles // 4):
            center_input_pos = stars["pos"][near]
            center_input_mass = stars["mass"][near]
        else:
            center_input_pos = stars["pos"]
            center_input_mass = stars["mass"]

        center = shrinking_sphere_center(
            center_input_pos,
            center_input_mass,
            initial_center=previous_center_kpc,
            initial_radius=min(search_radius, np.nanmax(rprev) if len(rprev) else search_radius),
            shrink_factor=cfg.shrink_factor,
            min_particles=cfg.shrink_min_particles,
            min_radius=cfg.shrink_min_radius_kpc,
        )
    else:
        center = shrinking_sphere_center(
            stars["pos"],
            stars["mass"],
            initial_radius=cfg.shrink_initial_radius_kpc,
            shrink_factor=cfg.shrink_factor,
            min_particles=cfg.shrink_min_particles,
            min_radius=cfg.shrink_min_radius_kpc,
        )

    # Preliminary rhalf using all tracked stars around the robust center.
    rhalf_pre = half_mass_radius(stars["pos"], stars["mass"], center)

    # Center velocity from inner stars.
    rv = distances(stars["pos"], center)
    v_radius = cfg.velocity_center_min_radius_kpc
    if np.isfinite(rhalf_pre):
        v_radius = max(v_radius, cfg.velocity_center_radius_factor_rhalf * rhalf_pre)
    inner_vel = rv <= v_radius
    if np.sum(inner_vel) < 5:
        # Fallback: use the innermost 10% stars, at least 5.
        order = np.argsort(rv)
        n_inner = max(5, min(len(order), int(0.1 * len(order))))
        inner_vel = np.zeros(len(rv), dtype=bool)
        inner_vel[order[:n_inner]] = True

    vcenter = mass_weighted_mean(stars["vel"][inner_vel], stars["mass"][inner_vel])

    # Tidal radius based on tracked material.
    rt = solve_tidal_radius_kpc([stars, gas, dm], center, cfg.host, cfg)

    # Bound/apparently associated stars are those inside rt.
    rstar = distances(stars["pos"], center)
    bound_stars = rstar <= rt if np.isfinite(rt) else np.zeros(len(rstar), dtype=bool)

    if np.sum(bound_stars) >= 5:
        rhalf_star = half_mass_radius(stars["pos"][bound_stars], stars["mass"][bound_stars], center)
        r90_star = enclosed_radius_fraction(stars["pos"][bound_stars], stars["mass"][bound_stars], center, 0.90)
    else:
        rhalf_star = rhalf_pre
        r90_star = enclosed_radius_fraction(stars["pos"], stars["mass"], center, 0.90)

    # Recompute masses using final rhalf.
    mstar = component_masses(stars, center, rt, rhalf_star)
    mgas = component_masses(gas, center, rt, rhalf_star)
    mdm = component_masses(dm, center, rt, rhalf_star)

    # Additional masses inside fixed physical apertures, e.g. 1 and 3 kpc.
    fixed_radii = tuple(float(r) for r in cfg.fixed_aperture_radii_kpc)
    mstar_fixed = fixed_aperture_component_masses(stars, center, fixed_radii)
    mgas_fixed = fixed_aperture_component_masses(gas, center, fixed_radii)
    mdm_fixed = fixed_aperture_component_masses(dm, center, fixed_radii)

    # Orbital quantities.
    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    host_vel = np.asarray(cfg.host.host_velocity_kms, dtype=float)
    rel_pos = center - host_center
    rel_vel = vcenter - host_vel
    R = float(np.linalg.norm(rel_pos))
    V = float(np.linalg.norm(rel_vel))

    if R > 0:
        rhat = rel_pos / R
        v_rad = float(np.dot(rel_vel, rhat))
        v_tan = float(np.sqrt(max(V * V - v_rad * v_rad, 0.0)))
    else:
        v_rad = np.nan
        v_tan = np.nan

    time_gyr = read_snapshot_time_gyr(snapshot_file, cfg)

    if cfg.compute_star_formation:
        sf_gas = gas_star_formation_quantities(
            gas, center, rt, rhalf_star, fixed_radii, cfg
        )
        young_stars = young_star_quantities(
            stars, center, time_gyr, rt, rhalf_star, fixed_radii, cfg
        )
    else:
        sf_gas = {}
        young_stars = {}

    ram = compute_local_cgm_ram_pressure(
        gas_all=gas_all,
        satellite_gas_ids=idcat["gas_ids"],
        center_kpc=center,
        v_sat_kms=vcenter,
        cfg=cfg,
    )

    row = {
        "snapshot_index": int(snapshot_index),
        "snapshot_number": int(natural_snapshot_number(snapshot_file)),
        "snapshot_file": str(snapshot_file),
        "time_gyr": float(time_gyr),

        "x_sat_kpc": float(center[0]),
        "y_sat_kpc": float(center[1]),
        "z_sat_kpc": float(center[2]),
        "vx_sat_kms": float(vcenter[0]),
        "vy_sat_kms": float(vcenter[1]),
        "vz_sat_kms": float(vcenter[2]),

        "R_host_kpc": R,
        "V_3d_kms": V,
        "V_rad_kms": v_rad,
        "V_tan_kms": v_tan,

        "r_tidal_kpc": float(rt),
        "rhalf_star_kpc": float(rhalf_star),
        "r90_star_kpc": float(r90_star),

        "Mstar_tracked_msun": mstar["tracked"],
        "Mstar_inside_rt_msun": mstar["inside_rt"],
        "Mstar_inside_rhalf_msun": mstar["inside_rhalf_star"],
        "Nstar_tracked": mstar["n_tracked"],
        "Nstar_inside_rt": mstar["n_inside_rt"],
        "Nstar_inside_rhalf": mstar["n_inside_rhalf_star"],

        "Mgas_tracked_msun": mgas["tracked"],
        "Mgas_inside_rt_msun": mgas["inside_rt"],
        "Mgas_inside_rhalf_msun": mgas["inside_rhalf_star"],
        "Ngas_tracked": mgas["n_tracked"],
        "Ngas_inside_rt": mgas["n_inside_rt"],
        "Ngas_inside_rhalf": mgas["n_inside_rhalf_star"],

        "Mdm_tracked_msun": mdm["tracked"],
        "Mdm_inside_rt_msun": mdm["inside_rt"],
        "Mdm_inside_rhalf_msun": mdm["inside_rhalf_star"],
        "Ndm_tracked": mdm["n_tracked"],
        "Ndm_inside_rt": mdm["n_inside_rt"],
        "Ndm_inside_rhalf": mdm["n_inside_rhalf_star"],

        "rho_cgm_msun_kpc3": ram["rho_cgm_msun_kpc3"],
        "rho_cgm_aperture_msun_kpc3": ram["rho_cgm_aperture_msun_kpc3"],
        "rho_cgm_density_field_msun_kpc3": ram["rho_cgm_density_field_msun_kpc3"],
        "N_cgm_for_ram": ram["N_cgm_for_ram"],
        "ram_pressure_radius_kpc": ram["ram_pressure_radius_kpc"],
        "V_rel_cgm_kms": ram["V_rel_cgm_kms"],
        "P_ram_msun_kpc3_kms2": ram["P_ram_msun_kpc3_kms2"],
        "P_ram_dyne_cm2": ram["P_ram_dyne_cm2"],
        "log10_P_ram_dyne_cm2": ram["log10_P_ram_dyne_cm2"],
    }

    # Add fixed-aperture mass columns.
    for key, val in mstar_fixed.items():
        if key.startswith("n_"):
            row["Nstar_" + key[2:]] = val
        else:
            row["Mstar_" + key + "_msun"] = val
    for key, val in mgas_fixed.items():
        if key.startswith("n_"):
            row["Ngas_" + key[2:]] = val
        else:
            row["Mgas_" + key + "_msun"] = val
    for key, val in mdm_fixed.items():
        if key.startswith("n_"):
            row["Ndm_" + key[2:]] = val
        else:
            row["Mdm_" + key + "_msun"] = val

    # Add SFR/star-forming-gas and young-star diagnostics.
    row.update(sf_gas)
    row.update(young_stars)

    # Total satellite masses, including fixed apertures.
    row["Msat_tracked_msun"] = row["Mstar_tracked_msun"] + row["Mgas_tracked_msun"] + row["Mdm_tracked_msun"]
    row["Msat_inside_rt_msun"] = row["Mstar_inside_rt_msun"] + row["Mgas_inside_rt_msun"] + row["Mdm_inside_rt_msun"]
    row["Msat_inside_rhalf_msun"] = (
        row["Mstar_inside_rhalf_msun"] + row["Mgas_inside_rhalf_msun"] + row["Mdm_inside_rhalf_msun"]
    )

    for radius in fixed_radii:
        tag = radius_tag_kpc(radius)
        row[f"Msat_inside_{tag}_msun"] = (
            row.get(f"Mstar_inside_{tag}_msun", 0.0)
            + row.get(f"Mgas_inside_{tag}_msun", 0.0)
            + row.get(f"Mdm_inside_{tag}_msun", 0.0)
        )

    if row["Mstar_inside_rt_msun"] > 0:
        row["gas_to_star_inside_rt"] = row["Mgas_inside_rt_msun"] / row["Mstar_inside_rt_msun"]
        row["dm_to_star_inside_rt"] = row["Mdm_inside_rt_msun"] / row["Mstar_inside_rt_msun"]
    else:
        row["gas_to_star_inside_rt"] = np.nan
        row["dm_to_star_inside_rt"] = np.nan

    return row, center, rhalf_star


def save_config_and_idcat(
    outdir: Union[str, Path],
    cfg: AnalysisConfig,
    idcat: Dict[str, np.ndarray | float | List[float]],
) -> None:
    outdir = ensure_dir(outdir)

    cfg_dict = asdict(cfg)
    with open(outdir / "analysis_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Save potentially large ID arrays in npz and a lightweight JSON summary.
    np.savez_compressed(
        outdir / "initial_satellite_ids.npz",
        star_ids=np.asarray(idcat["star_ids"], dtype=np.int64),
        gas_ids=np.asarray(idcat["gas_ids"], dtype=np.int64),
        dm_ids=np.asarray(idcat["dm_ids"], dtype=np.int64),
    )

    summary = {}
    for k, v in idcat.items():
        if isinstance(v, np.ndarray):
            summary[k] = {
                "n": int(len(v)),
                "min": int(np.min(v)) if len(v) else None,
                "max": int(np.max(v)) if len(v) else None,
            }
        else:
            summary[k] = v

    with open(outdir / "initial_satellite_id_summary.json", "w") as f:
        json.dump(summary, f, indent=2)



def safe_positive_values(y: np.ndarray, floor: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, floor)


def get_time_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    x = df["time_gyr"].values if "time_gyr" in df.columns else np.full(len(df), np.nan)
    if not np.all(np.isfinite(x)):
        return df["snapshot_number"].values, "Snapshot"
    return x, "Time [Gyr]"


def add_extrema_lines(ax, df: pd.DataFrame, extrema_df: Optional[pd.DataFrame] = None) -> None:
    """Add vertical markers for pericentres and apocentres."""
    if extrema_df is None or len(extrema_df) == 0:
        return
    x, _ = get_time_axis(df)
    first_peri = True
    first_apo = True
    for _, ev in extrema_df.iterrows():
        idx = int(ev["snapshot_index"])
        if idx < 0 or idx >= len(x):
            continue
        if ev["event_type"] == "pericentre":
            ax.axvline(x[idx], ls="--", lw=0.9, alpha=0.7, label="pericentre" if first_peri else None)
            first_peri = False
        elif ev["event_type"] == "apocentre":
            ax.axvline(x[idx], ls=":", lw=0.9, alpha=0.7, label="apocentre" if first_apo else None)
            first_apo = False


def plot_evolution(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    label: str,
    cfg: Optional[AnalysisConfig] = None,
    extrema_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Save standard diagnostic plots.

    If cfg.plot_log10_masses_and_radii is True, mass and radius panels are plotted
    as log(quantity/unit), not as log-scaled axes. This makes the values easy to
    compare directly across panels and labels.
    """
    outdir = ensure_dir(outdir)
    paths = {}

    use_log = True if cfg is None else cfg.plot_log10_masses_and_radii
    annotate = True if cfg is None else cfg.annotate_orbital_extrema

    x, xlabel = get_time_axis(df)

    def savefig(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=180, facecolor="white", transparent=False)
        plt.close(fig)

    # Orbit.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    if use_log:
        ax.plot(x, log10_safe_values(df["R_host_kpc"]), marker="o", lw=1.5)
        ax.set_ylabel(r"$\log(R_{\rm host}/{\rm kpc})$")
    else:
        ax.plot(x, df["R_host_kpc"], marker="o", lw=1.5)
        ax.set_ylabel(r"$R_{\rm host}$ [kpc]")
    if annotate:
        add_extrema_lines(ax, df, extrema_df)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{label}: satellite orbital radius")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False) if annotate and extrema_df is not None and len(extrema_df) else None
    p = outdir / "orbit_radius.png"
    savefig(fig, p)
    paths["orbit_radius"] = p

    # Velocities.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.plot(x, df["V_3d_kms"], marker="o", lw=1.5, label=r"$|v|$")
    ax.plot(x, df["V_rad_kms"], marker="o", lw=1.5, label=r"$v_{\rm rad}$")
    ax.plot(x, df["V_tan_kms"], marker="o", lw=1.5, label=r"$v_{\rm tan}$")
    ax.axhline(0, lw=0.8, alpha=0.5)
    if annotate:
        add_extrema_lines(ax, df, extrema_df)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title(f"{label}: satellite velocity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    p = outdir / "orbit_velocity.png"
    savefig(fig, p)
    paths["orbit_velocity"] = p

    # Masses inside tidal radius.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    if use_log:
        ax.plot(x, log10_safe_values(df["Mstar_inside_rt_msun"]), marker="o", lw=1.5, label="stars")
        ax.plot(x, log10_safe_values(df["Mgas_inside_rt_msun"]), marker="o", lw=1.5, label="satellite gas")
        ax.plot(x, log10_safe_values(df["Mdm_inside_rt_msun"]), marker="o", lw=1.5, label="satellite DM")
        ax.plot(x, log10_safe_values(df["Msat_inside_rt_msun"]), marker="o", lw=1.5, label="total", alpha=0.8)
        ax.set_ylabel(r"$\log(M(<r_t)/M_\odot)$")
    else:
        ax.plot(x, safe_positive_values(df["Mstar_inside_rt_msun"]), marker="o", lw=1.5, label="stars")
        ax.plot(x, safe_positive_values(df["Mgas_inside_rt_msun"]), marker="o", lw=1.5, label="satellite gas")
        ax.plot(x, safe_positive_values(df["Mdm_inside_rt_msun"]), marker="o", lw=1.5, label="satellite DM")
        ax.plot(x, safe_positive_values(df["Msat_inside_rt_msun"]), marker="o", lw=1.5, label="total", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylabel(r"Mass inside $r_t$ [$M_\odot$]")
    if annotate:
        add_extrema_lines(ax, df, extrema_df)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{label}: associated mass inside tidal radius")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    p = outdir / "masses_inside_tidal_radius.png"
    savefig(fig, p)
    paths["masses_inside_tidal_radius"] = p

    # Tracked vs inside rt ratios.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    for comp, pretty in [
        ("Mstar", "stars"),
        ("Mgas", "satellite gas"),
        ("Mdm", "satellite DM"),
    ]:
        tracked = df[f"{comp}_tracked_msun"].replace(0, np.nan)
        frac = df[f"{comp}_inside_rt_msun"] / tracked
        ax.plot(x, frac, marker="o", lw=1.5, label=pretty)
    if annotate:
        add_extrema_lines(ax, df, extrema_df)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$M(<r_t)/M_{\rm tracked}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{label}: retained fraction inside tidal radius")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    p = outdir / "retained_fraction_inside_tidal_radius.png"
    savefig(fig, p)
    paths["retained_fraction_inside_tidal_radius"] = p

    # Sizes.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    if use_log:
        ax.plot(x, log10_safe_values(df["rhalf_star_kpc"]), marker="o", lw=1.5, label=r"$r_{1/2,\star}$")
        ax.plot(x, log10_safe_values(df["r90_star_kpc"]), marker="o", lw=1.5, label=r"$r_{90,\star}$")
        ax.plot(x, log10_safe_values(df["r_tidal_kpc"]), marker="o", lw=1.5, label=r"$r_t$")
        ax.set_ylabel(r"$\log(r/{\rm kpc})$")
    else:
        ax.plot(x, df["rhalf_star_kpc"], marker="o", lw=1.5, label=r"$r_{1/2,\star}$")
        ax.plot(x, df["r90_star_kpc"], marker="o", lw=1.5, label=r"$r_{90,\star}$")
        ax.plot(x, df["r_tidal_kpc"], marker="o", lw=1.5, label=r"$r_t$")
        ax.set_yscale("log")
        ax.set_ylabel("Radius [kpc]")
    if annotate:
        add_extrema_lines(ax, df, extrema_df)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{label}: satellite size and tidal radius")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    p = outdir / "sizes_and_tidal_radius.png"
    savefig(fig, p)
    paths["sizes_and_tidal_radius"] = p

    # Masses inside stellar half-mass radius.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    if use_log:
        ax.plot(x, log10_safe_values(df["Mstar_inside_rhalf_msun"]), marker="o", lw=1.5, label="stars")
        ax.plot(x, log10_safe_values(df["Mgas_inside_rhalf_msun"]), marker="o", lw=1.5, label="satellite gas")
        ax.plot(x, log10_safe_values(df["Mdm_inside_rhalf_msun"]), marker="o", lw=1.5, label="satellite DM")
        ax.set_ylabel(r"$\log(M(<r_{1/2,\star})/M_\odot)$")
    else:
        ax.plot(x, safe_positive_values(df["Mstar_inside_rhalf_msun"]), marker="o", lw=1.5, label="stars")
        ax.plot(x, safe_positive_values(df["Mgas_inside_rhalf_msun"]), marker="o", lw=1.5, label="satellite gas")
        ax.plot(x, safe_positive_values(df["Mdm_inside_rhalf_msun"]), marker="o", lw=1.5, label="satellite DM")
        ax.set_yscale("log")
        ax.set_ylabel(r"Mass inside $r_{1/2,\star}$ [$M_\odot$]")
    if annotate:
        add_extrema_lines(ax, df, extrema_df)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{label}: central mass evolution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    p = outdir / "masses_inside_stellar_half_mass_radius.png"
    savefig(fig, p)
    paths["masses_inside_stellar_half_mass_radius"] = p

    # Ram pressure.
    if "P_ram_dyne_cm2" in df.columns and np.any(np.isfinite(df["P_ram_dyne_cm2"])):
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        ax.plot(x, log10_safe_values(df["P_ram_dyne_cm2"]), marker="o", lw=1.5)
        if annotate:
            add_extrema_lines(ax, df, extrema_df)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$")
        ax.set_title(f"{label}: ram-pressure evolution")
        ax.grid(alpha=0.3)
        p = outdir / "ram_pressure.png"
        savefig(fig, p)
        paths["ram_pressure"] = p



    # Star formation and star-forming gas.
    if "SFR_tracked_msun_per_yr" in df.columns and np.any(np.isfinite(df["SFR_tracked_msun_per_yr"])):
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        ax.plot(x, log10_safe_values(df["SFR_tracked_msun_per_yr"]), marker="o", lw=1.5, label="tracked gas")
        if "SFR_inside_rt_msun_per_yr" in df.columns:
            ax.plot(x, log10_safe_values(df["SFR_inside_rt_msun_per_yr"]), marker="o", lw=1.5, label=r"inside $r_t$")
        if "SFR_inside_rhalf_msun_per_yr" in df.columns:
            ax.plot(x, log10_safe_values(df["SFR_inside_rhalf_msun_per_yr"]), marker="o", lw=1.5, label=r"inside $r_{1/2,\star}$")
        if annotate:
            add_extrema_lines(ax, df, extrema_df)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
        ax.set_title(f"{label}: star-formation rate")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        p = outdir / "star_formation_rate.png"
        savefig(fig, p)
        paths["star_formation_rate"] = p

    if "Mgas_SF_tracked_msun" in df.columns and np.any(np.isfinite(df["Mgas_SF_tracked_msun"])):
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        ax.plot(x, log10_safe_values(df["Mgas_SF_tracked_msun"]), marker="o", lw=1.5, label="tracked gas")
        if "Mgas_SF_inside_rt_msun" in df.columns:
            ax.plot(x, log10_safe_values(df["Mgas_SF_inside_rt_msun"]), marker="o", lw=1.5, label=r"inside $r_t$")
        if "Mgas_SF_inside_rhalf_msun" in df.columns:
            ax.plot(x, log10_safe_values(df["Mgas_SF_inside_rhalf_msun"]), marker="o", lw=1.5, label=r"inside $r_{1/2,\star}$")
        if annotate:
            add_extrema_lines(ax, df, extrema_df)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log(M_{\rm gas,SF}/M_\odot)$")
        ax.set_title(f"{label}: star-forming gas mass")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        p = outdir / "star_forming_gas_mass.png"
        savefig(fig, p)
        paths["star_forming_gas_mass"] = p

    if "Mstar_young_tracked_msun" in df.columns and np.any(np.isfinite(df["Mstar_young_tracked_msun"])):
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        ax.plot(x, log10_safe_values(df["Mstar_young_tracked_msun"]), marker="o", lw=1.5, label="all satellite stars")
        if "Mstar_young_inside_rt_msun" in df.columns:
            ax.plot(x, log10_safe_values(df["Mstar_young_inside_rt_msun"]), marker="o", lw=1.5, label=r"inside $r_t$")
        if "Mstar_young_inside_rhalf_msun" in df.columns:
            ax.plot(x, log10_safe_values(df["Mstar_young_inside_rhalf_msun"]), marker="o", lw=1.5, label=r"inside $r_{1/2,\star}$")
        if annotate:
            add_extrema_lines(ax, df, extrema_df)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log(M_{\star,{\rm young}}/M_\odot)$")
        ax.set_title(f"{label}: young stellar mass")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        p = outdir / "young_stellar_mass.png"
        savefig(fig, p)
        paths["young_stellar_mass"] = p

    return paths



def make_parameter_gif(
    df: pd.DataFrame,
    outdir: Union[str, Path],
    label: str,
    fps: int = 5,
    stride: int = 1,
) -> Optional[Path]:
    """
    Make a GIF where the time-series curves grow with time.
    Masses, radii, and ram pressure are displayed as log(quantity/unit).
    """
    try:
        import imageio.v2 as imageio
    except Exception:
        warnings.warn("imageio is not installed; skipping parameter GIF.")
        return None

    outdir = ensure_dir(outdir)
    frame_dir = ensure_dir(outdir / "_parameter_gif_frames")

    x_all, xlabel = get_time_axis(df)

    frames = []
    indices = list(range(0, len(df), max(1, stride)))
    if indices and indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)

    mass_cols = ["Mstar_inside_rt_msun", "Mgas_inside_rt_msun", "Mdm_inside_rt_msun"]
    size_cols = ["rhalf_star_kpc", "r_tidal_kpc"]

    y_mass_all = log10_safe_values(df[mass_cols].values)
    y_size_all = log10_safe_values(df[size_cols].values)
    y_ram_all = log10_safe_values(df["P_ram_dyne_cm2"].values) if "P_ram_dyne_cm2" in df.columns else np.array([np.nan])

    def finite_limits(arr, pad=0.15, default=(0.0, 1.0)):
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return default
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        if ymin == ymax:
            return ymin - 0.5, ymax + 0.5
        delta = ymax - ymin
        return ymin - pad * delta, ymax + pad * delta

    ylims = {
        "R": finite_limits(log10_safe_values(df["R_host_kpc"].values)),
        "mass": finite_limits(y_mass_all),
        "size": finite_limits(y_size_all),
        "vel": finite_limits(df[["V_rad_kms", "V_tan_kms", "V_3d_kms"]].values),
        "ram": finite_limits(y_ram_all, default=(-15.0, -10.0)),
    }

    for n, i in enumerate(indices):
        sub = df.iloc[: i + 1]
        x = x_all[: i + 1]

        fig, axes = plt.subplots(3, 2, figsize=(11, 11), facecolor="white")
        axes = axes.ravel()

        ax = axes[0]
        ax.plot(x, log10_safe_values(sub["R_host_kpc"]), marker="o", lw=1.5)
        ax.set_ylabel(r"$\log(R_{\rm host}/{\rm kpc})$")
        ax.set_ylim(*ylims["R"])
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(x, log10_safe_values(sub["Mstar_inside_rt_msun"]), marker="o", lw=1.5, label="stars")
        ax.plot(x, log10_safe_values(sub["Mgas_inside_rt_msun"]), marker="o", lw=1.5, label="gas")
        ax.plot(x, log10_safe_values(sub["Mdm_inside_rt_msun"]), marker="o", lw=1.5, label="DM")
        ax.set_ylim(*ylims["mass"])
        ax.set_ylabel(r"$\log(M(<r_t)/M_\odot)$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(x, log10_safe_values(sub["rhalf_star_kpc"]), marker="o", lw=1.5, label=r"$r_{1/2,\star}$")
        ax.plot(x, log10_safe_values(sub["r_tidal_kpc"]), marker="o", lw=1.5, label=r"$r_t$")
        ax.set_ylim(*ylims["size"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log(r/{\rm kpc})$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3]
        ax.plot(x, sub["V_3d_kms"], marker="o", lw=1.5, label=r"$|v|$")
        ax.plot(x, sub["V_rad_kms"], marker="o", lw=1.5, label=r"$v_{\rm rad}$")
        ax.plot(x, sub["V_tan_kms"], marker="o", lw=1.5, label=r"$v_{\rm tan}$")
        ax.axhline(0, lw=0.8, alpha=0.5)
        ax.set_ylim(*ylims["vel"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Velocity [km/s]")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[4]
        if "P_ram_dyne_cm2" in sub.columns:
            ax.plot(x, log10_safe_values(sub["P_ram_dyne_cm2"]), marker="o", lw=1.5)
        ax.set_ylim(*ylims["ram"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$")
        ax.grid(alpha=0.3)

        ax = axes[5]
        retained = sub["Mstar_inside_rt_msun"] / sub["Mstar_tracked_msun"].replace(0, np.nan)
        ax.plot(x, retained, marker="o", lw=1.5, label="stars")
        retained = sub["Mgas_inside_rt_msun"] / sub["Mgas_tracked_msun"].replace(0, np.nan)
        ax.plot(x, retained, marker="o", lw=1.5, label="gas")
        retained = sub["Mdm_inside_rt_msun"] / sub["Mdm_tracked_msun"].replace(0, np.nan)
        ax.plot(x, retained, marker="o", lw=1.5, label="DM")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$M(<r_t)/M_{\rm tracked}$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

        if np.isfinite(df.iloc[i]["time_gyr"]):
            fig.suptitle(f"{label} | snapshot {int(df.iloc[i]['snapshot_number'])} | t={df.iloc[i]['time_gyr']:.3f} Gyr")
        else:
            fig.suptitle(f"{label} | snapshot {int(df.iloc[i]['snapshot_number'])}")
        fig.tight_layout()

        frame_path = frame_dir / f"frame_{n:04d}.png"
        fig.savefig(frame_path, dpi=140, facecolor="white", transparent=False)
        plt.close(fig)
        frames.append(frame_path)

    if not frames:
        return None

    gif_path = outdir / "parameter_evolution.gif"
    images = [imageio.imread(p) for p in frames]
    imageio.mimsave(gif_path, images, fps=fps)
    return gif_path



def projection_axes(map_axis: str) -> Tuple[int, int, int, str, str]:
    """
    Return projected plane axes for a line-of-sight axis.
    """
    axis = map_axis.lower()
    if axis == "x":
        return 1, 2, 0, "y - y_host [kpc]", "z - z_host [kpc]"
    if axis == "y":
        return 0, 2, 1, "x - x_host [kpc]", "z - z_host [kpc]"
    return 0, 1, 2, "x - x_host [kpc]", "y - y_host [kpc]"


def map_width_for_orbit(df: pd.DataFrame, cfg: AnalysisConfig) -> float:
    """
    Choose a constant map width centered on the host.

    If map_auto_width_include_orbit=True, the width is enlarged, if necessary,
    so that the satellite orbit projected on the chosen map plane, including the
    initial position, is inside the frame with cfg.map_margin_kpc margin.
    """
    width = float(cfg.map_width_kpc)
    if not cfg.map_auto_width_include_orbit or len(df) == 0:
        return width

    i, j, _, _, _ = projection_axes(cfg.map_axis)
    host = np.asarray(cfg.host.host_center_kpc, dtype=float)
    coords = df[["x_sat_kpc", "y_sat_kpc", "z_sat_kpc"]].values - host[None, :]
    max_abs = np.nanmax(np.abs(coords[:, [i, j]]))
    if np.isfinite(max_abs):
        width = max(width, 2.0 * (max_abs + cfg.map_margin_kpc))
    return float(width)


def gas_projection_histogram(
    snapshot_file: Union[str, Path],
    cfg: AnalysisConfig,
    map_width_kpc: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Host-centered projected gas-mass map used by the matplotlib backend.
    Returns histogram, xedges, yedges.
    """
    gas = read_particle_type(snapshot_file, cfg.gas_ptype, cfg)
    if gas is None or len(gas["pos"]) == 0:
        raise RuntimeError(f"No gas particles available for map in {snapshot_file}")

    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    rel = gas["pos"] - host_center[None, :]
    i, j, _, _, _ = projection_axes(cfg.map_axis)
    x = rel[:, i]
    y = rel[:, j]
    m = gas["mass"]

    half_width = 0.5 * map_width_kpc
    hist, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=cfg.fallback_map_npix,
        range=[[-half_width, half_width], [-half_width, half_width]],
        weights=m,
    )
    return hist.T, xedges, yedges


def compute_global_map_limits(
    snapshots: Sequence[Union[str, Path]],
    cfg: AnalysisConfig,
    map_width_kpc: float,
) -> Tuple[float, float]:
    """
    Compute global vmin/vmax for the matplotlib gas projection maps.
    User-provided cfg.map_vmin/map_vmax take precedence.
    """
    if cfg.map_vmin is not None and cfg.map_vmax is not None:
        return float(cfg.map_vmin), float(cfg.map_vmax)

    positive_values = []
    for snap in snapshots:
        try:
            hist, _, _ = gas_projection_histogram(snap, cfg, map_width_kpc)
            vals = hist[np.isfinite(hist) & (hist > 0)]
            if len(vals):
                # Store a subsample if the map is large, keeping memory modest.
                if len(vals) > 20000:
                    vals = np.random.default_rng(12345).choice(vals, size=20000, replace=False)
                positive_values.append(vals)
        except Exception as exc:
            warnings.warn(f"Could not compute map limits for {snap}: {exc}")

    if len(positive_values) == 0:
        return 1.0, 10.0

    vals = np.concatenate(positive_values)
    pmin, pmax = cfg.map_vlim_percentiles

    vmin = float(cfg.map_vmin) if cfg.map_vmin is not None else float(np.nanpercentile(vals, pmin))
    vmax = float(cfg.map_vmax) if cfg.map_vmax is not None else float(np.nanpercentile(vals, pmax))

    if not np.isfinite(vmin) or vmin <= 0:
        vmin = float(np.nanmin(vals[vals > 0]))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin * 10.0

    return vmin, vmax


def make_fallback_particle_map(
    snapshot_file: Union[str, Path],
    outpath: Union[str, Path],
    cfg: AnalysisConfig,
    sat_center_kpc: ArrayLike3,
    title: str = "",
    map_width_kpc: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """
    Matplotlib particle-projection map with fixed frame and fixed color scale.

    The map is host-centered, has a white background, and is saved without
    transparency so it is GIF-friendly.
    """
    outpath = Path(outpath)
    map_width = float(map_width_kpc if map_width_kpc is not None else cfg.map_width_kpc)
    hist, xedges, yedges = gas_projection_histogram(snapshot_file, cfg, map_width)

    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    sat_center = np.asarray(sat_center_kpc, dtype=float)
    i, j, _, xlabel, ylabel = projection_axes(cfg.map_axis)

    half_width = 0.5 * map_width

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.set_facecolor("white")

    if vmin is None or vmax is None:
        vals = hist[np.isfinite(hist) & (hist > 0)]
        if len(vals):
            vmin = float(np.nanpercentile(vals, 1.0)) if vmin is None else vmin
            vmax = float(np.nanpercentile(vals, 99.5)) if vmax is None else vmax
        else:
            vmin, vmax = 1.0, 10.0

    image_data = np.ma.masked_where(hist <= 0, hist)
    im = ax.imshow(
        image_data,
        origin="lower",
        extent=[-half_width, half_width, -half_width, half_width],
        norm=LogNorm(vmin=max(float(vmin), 1e-300), vmax=float(vmax)),
        cmap=cfg.map_cmap,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Projected gas mass [$M_\odot$ per pixel]")

    ax.scatter(0, 0, marker="+", s=100, label="host center")
    ax.scatter(
        sat_center[i] - host_center[i],
        sat_center[j] - host_center[j],
        marker="x",
        s=80,
        label="satellite center",
    )

    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or Path(snapshot_file).name)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, facecolor="white", transparent=False)
    plt.close(fig)
    return outpath


def make_yt_map(
    snapshot_file: Union[str, Path],
    outpath: Union[str, Path],
    cfg: AnalysisConfig,
    sat_center_kpc: ArrayLike3,
    title: str = "",
    map_width_kpc: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """
    Make a map using the requested backend.

    For reproducible GIFs, map_backend='matplotlib' is recommended because it
    guarantees a fixed frame, fixed color scale, and white non-transparent output.
    With map_backend='yt', the code tries to set the same zlim, but exact units
    depend on yt's frontend interpretation of the GADGET fields.
    """
    outpath = Path(outpath)
    snapshot_file = Path(snapshot_file)
    map_width = float(map_width_kpc if map_width_kpc is not None else cfg.map_width_kpc)

    if cfg.map_backend.lower() == "matplotlib":
        return make_fallback_particle_map(
            snapshot_file, outpath, cfg, sat_center_kpc, title=title,
            map_width_kpc=map_width, vmin=vmin, vmax=vmax,
        )

    try:
        import yt
    except Exception:
        if cfg.map_backend.lower() == "yt":
            warnings.warn("yt is not installed; using matplotlib particle map.")
        return make_fallback_particle_map(
            snapshot_file, outpath, cfg, sat_center_kpc, title=title,
            map_width_kpc=map_width, vmin=vmin, vmax=vmax,
        )

    unit_base = {
        "UnitLength_in_cm": cfg.length_unit_to_kpc * KPC_IN_CM,
        "UnitMass_in_g": cfg.mass_unit_to_msun * MSUN_IN_G,
        "UnitVelocity_in_cm_per_s": cfg.velocity_unit_to_kms * 1.0e5,
    }

    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    half_bbox = max(cfg.map_bbox_kpc, map_width)

    bbox = np.array([
        [host_center[0] - half_bbox, host_center[0] + half_bbox],
        [host_center[1] - half_bbox, host_center[1] + half_bbox],
        [host_center[2] - half_bbox, host_center[2] + half_bbox],
    ])

    try:
        ds = yt.load(str(snapshot_file), unit_base=unit_base, bounding_box=bbox)
    except TypeError:
        ds = yt.load(str(snapshot_file), unit_base=unit_base, bbox=bbox)

    center = ds.arr(host_center, "kpc")
    width = (map_width, "kpc")

    last_error = None
    for field in cfg.map_field_candidates:
        try:
            p = yt.ProjectionPlot(ds, cfg.map_axis, field, center=center, width=width)
            p.set_log(field, True)
            try:
                p.set_cmap(field, cfg.map_cmap)
            except Exception:
                pass
            if vmin is not None and vmax is not None:
                try:
                    p.set_zlim(field, float(vmin), float(vmax))
                except Exception:
                    pass
            if title:
                p.annotate_title(title)

            try:
                p.annotate_marker(
                    ds.arr(np.asarray(sat_center_kpc, dtype=float), "kpc"),
                    coord_system="data",
                    plot_args={"marker": "x", "s": 80},
                )
            except Exception:
                pass

            p.save(str(outpath.with_suffix("")))
            candidates = sorted(outpath.parent.glob(outpath.stem + "*.png"), key=lambda pth: pth.stat().st_mtime)
            if len(candidates) > 0:
                candidates[-1].rename(outpath)
            return outpath
        except Exception as e:
            last_error = e
            continue

    warnings.warn(
        f"yt projection failed for {snapshot_file}. Last error: {last_error}. "
        "Using matplotlib particle map."
    )
    return make_fallback_particle_map(
        snapshot_file, outpath, cfg, sat_center_kpc, title=title,
        map_width_kpc=map_width, vmin=vmin, vmax=vmax,
    )


def make_map_gif(
    map_paths: Sequence[Union[str, Path]],
    gif_path: Union[str, Path],
    fps: int = 5,
) -> Optional[Path]:
    try:
        import imageio.v2 as imageio
    except Exception:
        warnings.warn("imageio is not installed; skipping map GIF.")
        return None

    paths = [Path(p) for p in map_paths if Path(p).exists()]
    if len(paths) == 0:
        return None

    images = [imageio.imread(p) for p in paths]
    gif_path = Path(gif_path)
    imageio.mimsave(gif_path, images, fps=fps)
    return gif_path




def quadratic_extremum_from_three_points(
    x: np.ndarray,
    y: np.ndarray,
    i: int,
) -> Tuple[float, float]:
    """
    Quadratic interpolation of an extremum using points i-1, i, i+1.
    Returns (x_ext, y_ext). Falls back to the central point if interpolation fails.
    """
    if i <= 0 or i >= len(x) - 1:
        return float(x[i]), float(y[i])
    xs = np.asarray(x[i - 1:i + 2], dtype=float)
    ys = np.asarray(y[i - 1:i + 2], dtype=float)
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)) or len(np.unique(xs)) < 3:
        return float(x[i]), float(y[i])
    try:
        a, b, c = np.polyfit(xs, ys, 2)
        if a == 0:
            return float(x[i]), float(y[i])
        x0 = -b / (2.0 * a)
        if x0 < np.min(xs) or x0 > np.max(xs):
            return float(x[i]), float(y[i])
        y0 = a * x0 ** 2 + b * x0 + c
        return float(x0), float(y0)
    except Exception:
        return float(x[i]), float(y[i])


def compute_orbital_extrema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify pericentres and apocentres from local extrema of R_host_kpc.

    This is snapshot-based. If time_gyr is available, a simple quadratic fit to
    the three nearest points is also stored as time_refined_gyr and R_refined_kpc.
    Sparse sampling near pericentre should be interpreted carefully.
    """
    if len(df) == 0 or "R_host_kpc" not in df.columns:
        return pd.DataFrame()

    r = np.asarray(df["R_host_kpc"], dtype=float)
    time = np.asarray(df["time_gyr"], dtype=float) if "time_gyr" in df.columns else np.arange(len(df), dtype=float)
    rows = []

    for i in range(1, len(df) - 1):
        if not np.all(np.isfinite(r[i-1:i+2])):
            continue

        event_type = None
        if r[i] <= r[i - 1] and r[i] <= r[i + 1]:
            event_type = "pericentre"
        elif r[i] >= r[i - 1] and r[i] >= r[i + 1]:
            event_type = "apocentre"

        if event_type is None:
            continue

        tref, rref = quadratic_extremum_from_three_points(time, r, i)
        rows.append({
            "event_id": len(rows),
            "event_type": event_type,
            "snapshot_index": int(df.iloc[i]["snapshot_index"]),
            "snapshot_number": int(df.iloc[i]["snapshot_number"]),
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
            "time_refined_gyr": tref,
            "R_refined_kpc": rref,
            "V_3d_kms": float(df.iloc[i]["V_3d_kms"]) if "V_3d_kms" in df.columns else np.nan,
            "V_rad_kms": float(df.iloc[i]["V_rad_kms"]) if "V_rad_kms" in df.columns else np.nan,
            "V_tan_kms": float(df.iloc[i]["V_tan_kms"]) if "V_tan_kms" in df.columns else np.nan,
        })

    # If the simulation only captures one inbound passage and no local minimum is
    # bracketed, keep the global minimum as a pericentre candidate.
    if len(rows) == 0 and len(df) > 0:
        i = int(np.nanargmin(r))
        rows.append({
            "event_id": 0,
            "event_type": "pericentre_candidate_global_minimum",
            "snapshot_index": int(df.iloc[i]["snapshot_index"]),
            "snapshot_number": int(df.iloc[i]["snapshot_number"]),
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
            "time_refined_gyr": float(time[i]) if np.isfinite(time[i]) else np.nan,
            "R_refined_kpc": float(r[i]),
            "V_3d_kms": float(df.iloc[i]["V_3d_kms"]) if "V_3d_kms" in df.columns else np.nan,
            "V_rad_kms": float(df.iloc[i]["V_rad_kms"]) if "V_rad_kms" in df.columns else np.nan,
            "V_tan_kms": float(df.iloc[i]["V_tan_kms"]) if "V_tan_kms" in df.columns else np.nan,
        })

    return number_orbital_extrema(pd.DataFrame(rows))


def number_orbital_extrema(extrema_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-kind event numbers to the extrema table."""
    if extrema_df is None or len(extrema_df) == 0:
        return pd.DataFrame() if extrema_df is None else extrema_df
    out = extrema_df.copy()
    out["event_number"] = -1
    for kind in ["pericentre", "apocentre"]:
        mask = out["event_type"].astype(str).str.contains(kind)
        out.loc[mask, "event_number"] = np.arange(np.sum(mask)) + 1
    return out


def add_orbital_phase_columns(
    df: pd.DataFrame,
    extrema_df: pd.DataFrame,
    reference: str = "first",
) -> pd.DataFrame:
    """
    Add t_minus_tperi columns based on the selected reference pericentre.

    Added columns:\n
        t_minus_first_peri_gyr\n
        t_minus_deepest_peri_gyr\n
        t_minus_tperi_gyr  # chosen by cfg.pericentre_reference\n
    The reference time uses time_refined_gyr when available; otherwise it uses
    the nearest snapshot time_gyr.
    """
    out = df.copy()
    for col in ["t_minus_first_peri_gyr", "t_minus_deepest_peri_gyr", "t_minus_tperi_gyr"]:
        out[col] = np.nan

    if extrema_df is None or len(extrema_df) == 0 or "time_gyr" not in out.columns:
        return out

    peris = extrema_df[extrema_df["event_type"].astype(str).str.contains("pericentre")].copy()
    if len(peris) == 0:
        return out

    def event_time(row):
        tref = row.get("time_refined_gyr", np.nan)
        if np.isfinite(tref):
            return float(tref)
        return float(row.get("time_gyr", np.nan))

    peris = peris.copy()
    peris["_t_ref"] = [event_time(row) for _, row in peris.iterrows()]
    peris = peris[np.isfinite(peris["_t_ref"])]
    if len(peris) == 0:
        return out

    first = peris.sort_values("_t_ref").iloc[0]
    deepest = peris.sort_values("R_refined_kpc" if "R_refined_kpc" in peris.columns else "R_host_kpc").iloc[0]

    t = np.asarray(out["time_gyr"], dtype=float)
    out["t_minus_first_peri_gyr"] = t - float(first["_t_ref"])
    out["t_minus_deepest_peri_gyr"] = t - float(deepest["_t_ref"])

    ref = str(reference).lower()
    if ref == "deepest":
        out["t_minus_tperi_gyr"] = out["t_minus_deepest_peri_gyr"]
    else:
        out["t_minus_tperi_gyr"] = out["t_minus_first_peri_gyr"]

    out["tperi_first_gyr"] = float(first["_t_ref"])
    out["tperi_deepest_gyr"] = float(deepest["_t_ref"])
    return out



def add_orbital_event_column(df: pd.DataFrame, extrema_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with orbital_event labels at extrema snapshots."""
    out = df.copy()
    out["orbital_event"] = ""
    if extrema_df is None or len(extrema_df) == 0:
        return out
    for _, ev in extrema_df.iterrows():
        idx = int(ev["snapshot_index"])
        if idx in out.index:
            current = out.loc[idx, "orbital_event"]
            label = str(ev["event_type"])
            out.loc[idx, "orbital_event"] = label if current == "" else current + ";" + label
    return out


def summarize_extrema_windows(
    df: pd.DataFrame,
    extrema_df: pd.DataFrame,
    window_snapshots: int = 1,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Summarize parameter values around each pericentre/apocentre.

    This is especially useful for quantities like tidal radius and ram pressure,
    where the exact snapshot at pericentre may not capture the full extremum.
    The table stores central-snapshot values plus min/median/max over +/- N snapshots.
    """
    if extrema_df is None or len(extrema_df) == 0:
        return pd.DataFrame()

    if columns is None:
        columns = [
            "R_host_kpc", "r_tidal_kpc", "rhalf_star_kpc", "r90_star_kpc",
            "Mstar_inside_rt_msun", "Mgas_inside_rt_msun", "Mdm_inside_rt_msun",
            "Msat_inside_rt_msun", "Mstar_inside_rhalf_msun",
            "Mgas_inside_rhalf_msun", "Mdm_inside_rhalf_msun",
            "P_ram_dyne_cm2", "rho_cgm_msun_kpc3", "V_rel_cgm_kms",
            "SFR_tracked_msun_per_yr", "SFR_inside_rt_msun_per_yr",
            "Mgas_SF_tracked_msun", "Mgas_SF_inside_rt_msun",
            "Mstar_young_tracked_msun", "Mstar_young_inside_rt_msun",
            "V_3d_kms", "V_rad_kms", "V_tan_kms",
        ]

    rows = []
    nwin = int(max(0, window_snapshots))
    for _, ev in extrema_df.iterrows():
        idx = int(ev["snapshot_index"])
        lo = max(0, idx - nwin)
        hi = min(len(df), idx + nwin + 1)
        sub = df.iloc[lo:hi]

        row = ev.to_dict()
        row["window_snapshot_index_min"] = int(lo)
        row["window_snapshot_index_max"] = int(hi - 1)
        row["window_n_snapshots"] = int(len(sub))

        for col in columns:
            if col not in df.columns:
                continue
            vals = np.asarray(sub[col], dtype=float)
            row[f"{col}_at_event"] = float(df.iloc[idx][col]) if idx < len(df) and np.isfinite(df.iloc[idx][col]) else np.nan
            row[f"{col}_window_min"] = float(np.nanmin(vals)) if np.any(np.isfinite(vals)) else np.nan
            row[f"{col}_window_median"] = float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan
            row[f"{col}_window_max"] = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)




def analyze_label(label: str, cfg: AnalysisConfig) -> pd.DataFrame:
    """
    Analyze a single simulation label.
    """
    snaps = find_snapshots_for_label(cfg, label)

    label_outdir = ensure_dir(Path(cfg.output_dir) / label)
    maps_outdir = ensure_dir(label_outdir / "maps")

    if cfg.verbose:
        print(f"\n=== Analyzing label: {label} ===")
        print(f"Found {len(snaps)} snapshots.")
        print(f"Output: {label_outdir}")

    idcat = select_initial_satellite_ids(snaps[0], cfg)
    save_config_and_idcat(label_outdir, cfg, idcat)

    if cfg.verbose:
        print("Initial satellite ID catalogue:")
        print(f"  stars: {len(idcat['star_ids'])}")
        print(f"  gas:   {len(idcat['gas_ids'])}")
        print(f"  DM:    {len(idcat['dm_ids'])}")
        print(f"  initial center [kpc]: {idcat['initial_center_kpc']}")
        print(f"  initial stellar rhalf [kpc]: {idcat['initial_rhalf_star_kpc']:.3f}")
        print(f"  initial gas selection radius [kpc]: {idcat['initial_gas_selection_radius_kpc']:.3f}")

    rows = []
    prev_center = None
    prev_rhalf = None

    # First pass: compute all scalar quantities. Maps are made after the table is
    # complete, so all frames can share a fixed size and global vmin/vmax.
    for i, snap in enumerate(snaps):
        if cfg.verbose:
            print(f"[{label}] {i + 1}/{len(snaps)}: {snap.name}")

        row, center, rhalf = analyze_one_snapshot(
            snap,
            snapshot_index=i,
            cfg=cfg,
            idcat=idcat,
            previous_center_kpc=prev_center,
            previous_rhalf_kpc=prev_rhalf,
        )

        rows.append(row)
        prev_center = center
        prev_rhalf = rhalf

        if cfg.save_snapshot_json:
            with open(label_outdir / f"snapshot_{row['snapshot_number']:03d}_diagnostics.json", "w") as f:
                json.dump(row, f, indent=2)

    df = pd.DataFrame(rows)

    extrema_df = compute_orbital_extrema(df)
    df = add_orbital_event_column(df, extrema_df)
    df = add_orbital_phase_columns(df, extrema_df, reference=cfg.pericentre_reference)
    extrema_window_df = summarize_extrema_windows(
        df,
        extrema_df,
        window_snapshots=cfg.event_window_snapshots,
    )

    df_path = label_outdir / "satellite_evolution.csv"
    df.to_csv(df_path, index=False)

    extrema_path = label_outdir / "orbital_extrema.csv"
    extrema_df.to_csv(extrema_path, index=False)

    extrema_window_path = label_outdir / "orbital_extrema_window_summary.csv"
    extrema_window_df.to_csv(extrema_window_path, index=False)

    # Backward-compatible pericentre summary, now based on extrema table when possible.
    if len(df) > 0:
        if len(extrema_df) > 0 and np.any(extrema_df["event_type"].astype(str).str.contains("pericentre")):
            peri_rows = extrema_df[extrema_df["event_type"].astype(str).str.contains("pericentre")]
            peri = peri_rows.iloc[0]
            i_peri = int(peri["snapshot_index"])
        else:
            i_peri = int(np.nanargmin(df["R_host_kpc"].values))
            peri = df.iloc[i_peri]
        peri_summary = {
            "pericentre_snapshot_number": int(df.iloc[i_peri]["snapshot_number"]),
            "pericentre_time_gyr": float(df.iloc[i_peri]["time_gyr"]),
            "pericentre_R_kpc": float(df.iloc[i_peri]["R_host_kpc"]),
            "pericentre_V_3d_kms": float(df.iloc[i_peri]["V_3d_kms"]),
            "pericentre_r_tidal_kpc": float(df.iloc[i_peri]["r_tidal_kpc"]),
            "pericentre_P_ram_dyne_cm2": float(df.iloc[i_peri]["P_ram_dyne_cm2"]) if "P_ram_dyne_cm2" in df.columns else np.nan,
        }
        with open(label_outdir / "pericentre_summary.json", "w") as f:
            json.dump(peri_summary, f, indent=2)

    plot_evolution(df, label_outdir, label, cfg=cfg, extrema_df=extrema_df)

    # Second pass: make maps with constant frame and global color scale.
    map_paths = []
    if cfg.make_maps:
        map_indices = list(range(0, len(snaps), max(1, cfg.gif_stride)))
        if map_indices and map_indices[-1] != len(snaps) - 1:
            map_indices.append(len(snaps) - 1)
        elif not map_indices and len(snaps) > 0:
            map_indices = [len(snaps) - 1]

        map_width = map_width_for_orbit(df, cfg)
        map_snaps = [snaps[i] for i in map_indices]

        if cfg.map_backend.lower() in ("matplotlib", "auto"):
            map_vmin, map_vmax = compute_global_map_limits(map_snaps, cfg, map_width)
        else:
            map_vmin, map_vmax = cfg.map_vmin, cfg.map_vmax

        with open(label_outdir / "map_settings.json", "w") as f:
            json.dump(
                {
                    "map_backend": cfg.map_backend,
                    "map_axis": cfg.map_axis,
                    "map_width_kpc": map_width,
                    "map_vmin": None if map_vmin is None else float(map_vmin),
                    "map_vmax": None if map_vmax is None else float(map_vmax),
                    "map_cmap": cfg.map_cmap,
                    "map_white_background": cfg.map_white_background,
                },
                f,
                indent=2,
            )

        for i in map_indices:
            snap = snaps[i]
            row = df.iloc[i]
            center = np.array([row["x_sat_kpc"], row["y_sat_kpc"], row["z_sat_kpc"]], dtype=float)
            map_path = maps_outdir / f"map_snapshot_{int(row['snapshot_number']):03d}.png"
            title = f"{label} | snapshot {int(row['snapshot_number'])} | R={row['R_host_kpc']:.1f} kpc"
            try:
                make_yt_map(
                    snap,
                    map_path,
                    cfg,
                    center,
                    title=title,
                    map_width_kpc=map_width,
                    vmin=map_vmin,
                    vmax=map_vmax,
                )
                map_paths.append(map_path)
            except Exception as e:
                warnings.warn(f"Map failed for {snap}: {e}")

    if cfg.make_gifs:
        make_parameter_gif(
            df,
            label_outdir,
            label,
            fps=cfg.gif_fps,
            stride=cfg.gif_stride,
        )
        if len(map_paths) > 0:
            make_map_gif(
                map_paths,
                label_outdir / "gas_map_evolution.gif",
                fps=cfg.gif_fps,
            )

    if cfg.verbose:
        print(f"Saved table: {df_path}")
        print(f"Saved extrema table: {extrema_path}")
        print(f"Saved extrema-window table: {extrema_window_path}")

    return df


def analyze_all(cfg: AnalysisConfig) -> Dict[str, pd.DataFrame]:
    """
    Analyze all labels in cfg.labels. If cfg.labels is None, all directories
    under cfg.root with an output/ subdirectory are analyzed.
    """
    labels = cfg.labels or discover_labels(cfg.root)
    if len(labels) == 0:
        raise RuntimeError(f"No simulation labels found under {cfg.root}")

    results = {}
    for label in labels:
        results[label] = analyze_label(label, cfg)

    return results



def plot_compare_log_quantity(
    results: Dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    output_dir: Union[str, Path],
    filename: str,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Compare a quantity across labels using log(quantity/unit) on the y-axis."""
    if len(results) == 0:
        return None
    output_dir = ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    plotted = False
    for label, df in results.items():
        if column not in df.columns:
            continue
        x, xlabel = get_time_axis(df)
        ax.plot(x, log10_safe_values(df[column]), marker="o", lw=1.5, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    return path


def plot_retained_fraction_comparison(
    results: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    filename: str = "comparison_retained_fractions.png",
) -> Optional[Path]:
    """Compare retained mass fractions M(<r_t)/M_tracked across labels."""
    if len(results) == 0:
        return None
    output_dir = ensure_dir(output_dir)

    pairs = [
        ("Mstar_inside_rt_msun", "Mstar_tracked_msun", "stars"),
        ("Mgas_inside_rt_msun", "Mgas_tracked_msun", "gas"),
        ("Mdm_inside_rt_msun", "Mdm_tracked_msun", "DM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, facecolor="white")
    for ax, (inside_col, total_col, name) in zip(axes, pairs):
        for label, df in results.items():
            if inside_col not in df.columns or total_col not in df.columns:
                continue
            x, xlabel = get_time_axis(df)
            total = np.asarray(df[total_col], dtype=float)
            inside = np.asarray(df[inside_col], dtype=float)
            frac = np.where(total > 0, inside / total, np.nan)
            ax.plot(x, frac, marker="o", lw=1.5, label=label)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$M(<r_t)/M_{\rm tracked}$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    return path


def compare_labels(results: Dict[str, pd.DataFrame], output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create comparison plots across labels.

    Masses and radii are shown as log(quantity/unit), with labels written as
    'log' instead of 'log_10' for cleaner figures.
    """
    output_dir = ensure_dir(output_dir)
    paths: Dict[str, Path] = {}

    # Orbital-radius comparison.
    p = plot_compare_log_quantity(
        results,
        column="R_host_kpc",
        ylabel=r"$\log(R_{\rm host}/{\rm kpc})$",
        output_dir=output_dir,
        filename="comparison_log_Rhost.png",
        title="Orbital-radius comparison",
    )
    if p is not None:
        paths["R_host"] = p

    comparisons = [
        ("r_tidal_kpc", r"$\log(r_t/{\rm kpc})$", "comparison_log_rt.png", "Tidal-radius comparison"),
        ("rhalf_star_kpc", r"$\log(r_{1/2,\star}/{\rm kpc})$", "comparison_log_rhalf_star.png", "Stellar half-mass radius"),
        ("Mstar_inside_rt_msun", r"$\log(M_\star(<r_t)/M_\odot)$", "comparison_log_Mstar_inside_rt.png", "Stellar mass inside tidal radius"),
        ("Mgas_inside_rt_msun", r"$\log(M_{\rm gas}(<r_t)/M_\odot)$", "comparison_log_Mgas_inside_rt.png", "Gas mass inside tidal radius"),
        ("Mdm_inside_rt_msun", r"$\log(M_{\rm DM}(<r_t)/M_\odot)$", "comparison_log_Mdm_inside_rt.png", "DM mass inside tidal radius"),
        ("Msat_inside_rt_msun", r"$\log(M_{\rm sat}(<r_t)/M_\odot)$", "comparison_log_Msat_inside_rt.png", "Total mass inside tidal radius"),
        ("P_ram_dyne_cm2", r"$\log(P_{\rm ram}/{\rm dyn\ cm^{-2}})$", "comparison_log_Pram.png", "Ram-pressure comparison"),
    ]

    for column, ylabel, filename, title in comparisons:
        p = plot_compare_log_quantity(results, column, ylabel, output_dir, filename, title)
        if p is not None:
            paths[column] = p

    p = plot_retained_fraction_comparison(results, output_dir)
    if p is not None:
        paths["retained_fractions"] = p

    return paths

#%%
# ---------------------------------------------------------------------------
# Example configuration for direct execution.
# In a notebook, import this file and create your own cfg instead.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = AnalysisConfig(
        root="./../SIMULATIONS/ORBIT/HigherRes",
        labels=None,  # e.g. ["E_mid_L_mid"]; None analyzes all labels found.
        output_dir="orbit_analysis_outputs",

        # Adjust if needed.
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
        ),

        # If gas selection is too broad/narrow, tune this.
        initial_satellite_gas_radius_kpc=None,
        default_initial_gas_radius_kpc=30.0,

        # In your setup, PartType1 is usually only satellite DM.
        dm_selection_mode="all",

        make_maps=True,
        make_gifs=True,
        gif_fps=5,
        gif_stride=1,

        verbose=True,
    )

    results = analyze_all(cfg)
    compare_labels(results, cfg.output_dir)
