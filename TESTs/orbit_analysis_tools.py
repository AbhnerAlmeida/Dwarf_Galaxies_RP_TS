"""orbit_analysis_tools.py

Core analysis tools for Gadget/Gadget-4 satellite-orbit HDF5 snapshots.

This file is the consolidated analysis module produced from the original
Spyder scripts supplied by Abhner.  It intentionally contains no figure-making
routines and no hard-coded execution block.  Use it from a Spyder workflow script
or import it in notebooks.

Main responsibilities
---------------------
1. Discover simulation labels and snapshot files with layout:
       root / LABEL / output / snapshot_*.hdf5

2. Read HDF5 particle data for gas, dark matter, and stars:
       PartType0 = gas
       PartType1 = dark matter
       PartType4 = stars
   The particle-type numbers and unit conversions remain configurable.

3. Build the initial satellite ID catalogue:
       star_ids
       inner_star_ids
       gas_ids
       dm_ids
   The central stellar IDs are selected in the first snapshot around the initial
   stellar COM and are later used for robust centre tracking.

4. Track the satellite centre through time.  Recommended:
       center_mode = "inner_ids_shrinking"
       velocity_mode = "inner_ids"
   This follows a persistent central stellar population and avoids the all-stars
   COM being pulled by tidal tails.

5. Compute physical/orbital diagnostics:
       R_host, V_3d, V_rad, V_tan
       NFW host enclosed mass and tidal-field proxy
       instantaneous tidal radius
       stellar membership/stripping diagnostics
       stellar/gas/DM masses and sizes
       SFR, sSFR, SFE and depletion time when SFR fields exist
       local CGM density and ram pressure
       pericentres/apocentres and event-window summaries

Typical use
-----------
    from orbit_analysis_tools import *

    cfg = OrbitAnalysisConfig(...)
    inspect_snapshot_structure(find_snapshots_for_label(cfg, cfg.labels[0])[0])
    results = analyze_all(cfg)

The high-level function `analyze_all(cfg)` saves per-label CSV/NPZ/JSON outputs
and returns:
       results[label] = pandas.DataFrame

Companion plotting code lives in:
       orbit_science_plots.py

A Spyder step-by-step workflow lives in:
       orbit_spyder_workflow.py"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import warnings

import h5py
import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

G_KPC_KMS2_MSUN = 4.30091e-6
KPC_IN_CM = 3.0856775814913673e21
MSUN_IN_G = 1.98847e33

ArrayLike3 = Union[Sequence[float], np.ndarray]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HostHaloConfig:
    """Analytic NFW host halo."""

    host_center_kpc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    host_velocity_kms: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    m200_msun: float = 1.0e12
    r200_kpc: float = 210.0
    concentration: float = 10.0
    truncate_mass_at_r200: bool = False

    # Tidal radius approximation:
    # r_t = R [ m_sat(<r_t) / (tidal_factor * M_host(<R)) ]^(1/3)
    tidal_factor: float = 3.0

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

    def tidal_field_kms2_kpc2(self, radius_kpc: float) -> float:
        """Simple scalar tidal-field proxy G M(<R) / R^3."""
        if not np.isfinite(radius_kpc) or radius_kpc <= 0:
            return np.nan
        return float(G_KPC_KMS2_MSUN * self.mass_enclosed_msun(radius_kpc) / radius_kpc**3)


@dataclass
class OrbitAnalysisConfig:
    """
    Main analysis configuration.

    Edit these values in the USER SETTINGS block.
    """

    root: str = "./../SIMULATIONS/ORBIT/HigherRes"
    labels: Optional[List[str]] = None
    output_dir: str = "orbit_full_analysis_outputs"
    snapshot_glob: str = "snapshot_*.hdf5"

    # Gadget particle types.
    gas_ptype: int = 0
    dm_ptype: int = 1
    star_ptype: int = 4

    # Unit conversions from code units to physical units.
    length_unit_to_kpc: float = 1.0
    velocity_unit_to_kms: float = 1.0
    mass_unit_to_msun: float = 1.0e10
    time_unit_to_gyr: float = 0.977792221

    host: HostHaloConfig = field(default_factory=HostHaloConfig)

    # -------------------------------------------------------------------------
    # Initial ID catalogue
    # -------------------------------------------------------------------------
    # "stars_com" is useful for initial IC validation and for defining inner IDs.
    initial_center_mode: str = "stars_com"  # "stars_com" or "stellar_core"

    # Central stellar IDs are selected at the initial snapshot around stars_com.
    inner_radius_factor_rhalf: float = 1.0
    inner_min_radius_kpc: float = 0.5
    inner_max_radius_kpc: float = 5.0
    inner_min_particles: int = 50

    # Gas selection around initial centre.
    initial_satellite_gas_radius_kpc: Optional[float] = None
    default_initial_gas_radius_kpc: float = 30.0
    gas_radius_factor_rhalf: float = 8.0

    # DM selection around initial centre, or all PartType1.
    dm_selection_mode: str = "all"  # "all" or "radius"
    initial_satellite_dm_radius_kpc: Optional[float] = None
    default_initial_dm_radius_kpc: float = 80.0
    dm_radius_factor_rhalf: float = 20.0

    # -------------------------------------------------------------------------
    # Centre and velocity modes
    # -------------------------------------------------------------------------
    # Recommended: "inner_ids_shrinking"
    center_mode: str = "inner_ids_shrinking"
    # Options:
    #   "inner_ids_shrinking"      robust shrinking-sphere using initial central IDs
    #   "inner_ids_com"            COM of initial central IDs
    #   "stellar_core"             shrinking-sphere using all tracked stars
    #   "stars_com_all"            COM of all tracked stars
    #   "stars_com_near_previous"  local COM of stars near previous centre

    velocity_mode: str = "inner_ids"
    # Options:
    #   "inner_ids"      velocity of initial central stellar IDs
    #   "all_stars"      velocity COM of all tracked stars
    #   "inner_stars"    velocity of stars within max(min_radius, factor*rhalf)
    #   "same_as_center" velocity of particles used for centre

    # Shrinking-sphere parameters.
    shrink_initial_radius_kpc: Optional[float] = None
    shrink_factor: float = 0.75
    shrink_min_particles: int = 100
    shrink_min_radius_kpc: float = 0.2
    center_search_radius_kpc: float = 40.0

    # Inner-stars velocity aperture.
    velocity_center_radius_factor_rhalf: float = 1.0
    velocity_center_min_radius_kpc: float = 1.0

    # -------------------------------------------------------------------------
    # Membership / stripping
    # -------------------------------------------------------------------------
    member_mode: str = "tidal_kinematic"
    # Options:
    #   "tidal"           r <= member_tidal_factor * r_tidal
    #   "tidal_kinematic" tidal cut + v_rel <= v_escape_factor * vesc_sat(r)
    #   "none"            all tracked stars are members

    member_tidal_factor: float = 1.0
    stripped_tidal_factor: float = 1.5
    v_escape_factor: float = 1.25
    stripped_consecutive_snapshots: int = 2

    # -------------------------------------------------------------------------
    # Tidal radius
    # -------------------------------------------------------------------------
    tidal_max_iterations: int = 40
    tidal_tolerance: float = 1.0e-3
    tidal_initial_fraction_of_R: float = 0.15
    tidal_max_fraction_of_R: float = 0.5
    tidal_min_radius_kpc: float = 0.05

    # -------------------------------------------------------------------------
    # SFR and density field names
    # -------------------------------------------------------------------------
    sfr_field_candidates: Tuple[str, ...] = (
        "StarFormationRate",
        "StarFormationRates",
        "SFR",
        "Sfr",
        "sfr",
    )

    density_field_candidates: Tuple[str, ...] = (
        "Density",
        "Densities",
        "rho",
        "Rho",
        "density",
    )

    # -------------------------------------------------------------------------
    # Ram pressure
    # -------------------------------------------------------------------------
    compute_ram_pressure: bool = True
    ram_pressure_density_radius_kpc: float = 10.0
    ram_pressure_max_density_radius_kpc: float = 30.0
    ram_pressure_min_cgm_particles: int = 16
    ram_pressure_velocity_mode: str = "local_cgm"  # "local_cgm" or "host_frame"
    ram_pressure_use_cgm_only: bool = True

    # -------------------------------------------------------------------------
    # Output / Spyder
    # -------------------------------------------------------------------------
    event_window_snapshots: int = 1
    save_snapshot_json: bool = False
    verbose: bool = True


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_snapshot_number(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    return int(digits[-1]) if digits else -1


def discover_labels(root: Union[str, Path]) -> List[str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "output").exists()])


def find_snapshots_for_label(cfg: OrbitAnalysisConfig, label: str) -> List[Path]:
    out = Path(cfg.root) / label / "output"
    snaps = sorted(out.glob(cfg.snapshot_glob), key=natural_snapshot_number)
    if len(snaps) == 0:
        raise FileNotFoundError(f"No snapshots found for label={label} in {out}")
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
    """Print HDF5 groups, datasets, shapes, and header attributes."""
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


def read_snapshot_time_gyr(snapshot_file: Union[str, Path], cfg: OrbitAnalysisConfig) -> float:
    with h5py.File(snapshot_file, "r") as f:
        t_code = header_attr(f, "Time", np.nan)
    try:
        return float(t_code) * cfg.time_unit_to_gyr
    except Exception:
        return np.nan


def read_particle_type(
    snapshot_file: Union[str, Path],
    ptype: int,
    cfg: OrbitAnalysisConfig,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Read positions, velocities, IDs, masses, optional density, and optional SFR.

    SFR is stored as data["sfr"] if any candidate field is found in the HDF5
    particle group. The script assumes the SFR field is already in Msun/yr.
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
        masses = dataset_first_available(g, ["Masses", "Mass", "ParticleMasses"])

        if pos is None or vel is None:
            raise KeyError(f"Missing Coordinates/Velocities in {snapshot_file}:{group_name}")

        n = len(pos)

        if ids is None:
            warnings.warn(f"No IDs in {snapshot_file}:{group_name}; generating sequential IDs.")
            ids = np.arange(1, n + 1, dtype=np.int64)

        if masses is None:
            mass_table = header_attr(f, "MassTable", None)
            if mass_table is None:
                raise KeyError(f"No Masses dataset and no Header/MassTable for {group_name}")
            m_code = float(mass_table[ptype])
            if m_code <= 0:
                raise ValueError(f"MassTable[{ptype}] <= 0 but no Masses dataset exists.")
            masses = np.full(n, m_code, dtype=float)

        density = dataset_first_available(g, cfg.density_field_candidates)
        sfr = dataset_first_available(g, cfg.sfr_field_candidates)

    data = {
        "pos": np.asarray(pos, dtype=float) * cfg.length_unit_to_kpc,
        "vel": np.asarray(vel, dtype=float) * cfg.velocity_unit_to_kms,
        "ids": np.asarray(ids, dtype=np.int64),
        "mass": np.asarray(masses, dtype=float) * cfg.mass_unit_to_msun,
    }

    if density is not None:
        data["density"] = (
            np.asarray(density, dtype=float)
            * cfg.mass_unit_to_msun
            / cfg.length_unit_to_kpc**3
        )
    else:
        data["density"] = np.empty(0, dtype=float)

    if sfr is not None:
        data["sfr"] = np.asarray(sfr, dtype=float)
    else:
        data["sfr"] = np.empty(0, dtype=float)

    return data


def empty_particle_dict() -> Dict[str, np.ndarray]:
    return {
        "pos": np.empty((0, 3), dtype=float),
        "vel": np.empty((0, 3), dtype=float),
        "ids": np.empty(0, dtype=np.int64),
        "mass": np.empty(0, dtype=float),
        "density": np.empty(0, dtype=float),
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

    mask = mask_ids(pdata["ids"], selected_ids)
    out: Dict[str, np.ndarray] = {}
    for key, value in pdata.items():
        if isinstance(value, np.ndarray) and len(value) == len(pdata["ids"]):
            out[key] = value[mask]
        else:
            out[key] = value

    for key in ["density", "sfr"]:
        if key not in out:
            out[key] = np.empty(0, dtype=float)

    return out


def distances(pos: np.ndarray, center: ArrayLike3) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    if len(pos) == 0:
        return np.empty(0, dtype=float)
    return np.linalg.norm(pos - center[None, :], axis=1)


def mass_weighted_mean(values: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(values.shape[1] if values.ndim > 1 else 1, np.nan)

    if masses is None or len(masses) == 0 or np.nansum(masses) <= 0:
        return np.nanmean(values, axis=0)

    masses = np.asarray(masses, dtype=float)
    good = np.isfinite(masses) & (masses > 0)
    if values.ndim == 2:
        good &= np.all(np.isfinite(values), axis=1)
    else:
        good &= np.isfinite(values)

    if np.count_nonzero(good) == 0:
        return np.nanmean(values, axis=0)

    return np.average(values[good], axis=0, weights=masses[good])


def safe_divide(num: float, den: float) -> float:
    if np.isfinite(num) and np.isfinite(den) and den > 0:
        return float(num / den)
    return np.nan


def half_mass_radius(pos: np.ndarray, masses: np.ndarray, center: ArrayLike3) -> float:
    if len(pos) == 0 or np.nansum(masses) <= 0:
        return np.nan

    r = distances(pos, center)
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = masses[order]
    cum = np.cumsum(m_sorted)
    idx = np.searchsorted(cum, 0.5 * cum[-1])
    return float(r_sorted[min(idx, len(r_sorted) - 1)])


def enclosed_radius_fraction(
    pos: np.ndarray,
    masses: np.ndarray,
    center: ArrayLike3,
    fraction: float = 0.9,
) -> float:
    if len(pos) == 0 or np.nansum(masses) <= 0:
        return np.nan

    r = distances(pos, center)
    order = np.argsort(r)
    r_sorted = r[order]
    m_sorted = masses[order]
    cum = np.cumsum(m_sorted)
    idx = np.searchsorted(cum, fraction * cum[-1])
    return float(r_sorted[min(idx, len(r_sorted) - 1)])


def shrinking_sphere_center(
    pos: np.ndarray,
    masses: np.ndarray,
    initial_center: Optional[ArrayLike3] = None,
    initial_radius: Optional[float] = None,
    shrink_factor: float = 0.75,
    min_particles: int = 100,
    min_radius: float = 0.2,
) -> np.ndarray:
    """Robust centre estimator, less sensitive to tidal tails than all-particle COM."""
    if len(pos) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    center = mass_weighted_mean(pos, masses) if initial_center is None else np.asarray(initial_center, dtype=float)
    r = distances(pos, center)

    if initial_radius is None:
        radius = float(np.nanpercentile(r, 90))
        if not np.isfinite(radius) or radius <= 0:
            radius = float(np.nanmax(r))
    else:
        radius = float(initial_radius)

    radius = max(radius, min_radius)
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

    if len(current) > 0:
        center = mass_weighted_mean(pos[current], masses[current])

    return np.asarray(center, dtype=float)


# =============================================================================
# INITIAL SATELLITE CATALOGUE
# =============================================================================

def select_initial_satellite_catalogue(
    first_snapshot: Union[str, Path],
    cfg: OrbitAnalysisConfig,
) -> Dict[str, object]:
    """
    Build initial satellite particle ID catalogue.

    All stars are assumed to belong to the satellite.
    Initial central stellar IDs are selected around stars_com in the first snapshot.
    Satellite gas IDs are selected around the initial centre.
    DM IDs are either all PartType1 or selected around the initial centre.
    """
    stars = read_particle_type(first_snapshot, cfg.star_ptype, cfg)
    if stars is None or len(stars["ids"]) == 0:
        raise RuntimeError(f"No star particles found in {first_snapshot}.")

    stars_com0 = mass_weighted_mean(stars["pos"], stars["mass"])
    stellar_core0 = shrinking_sphere_center(
        stars["pos"],
        stars["mass"],
        initial_radius=cfg.shrink_initial_radius_kpc,
        shrink_factor=cfg.shrink_factor,
        min_particles=cfg.shrink_min_particles,
        min_radius=cfg.shrink_min_radius_kpc,
    )

    initial_center = stars_com0 if cfg.initial_center_mode.lower() == "stars_com" else stellar_core0

    rhalf0_com = half_mass_radius(stars["pos"], stars["mass"], stars_com0)
    rhalf0_core = half_mass_radius(stars["pos"], stars["mass"], stellar_core0)
    rhalf_ref = rhalf0_com if np.isfinite(rhalf0_com) else rhalf0_core

    inner_radius = cfg.inner_min_radius_kpc
    if np.isfinite(rhalf_ref):
        inner_radius = cfg.inner_radius_factor_rhalf * rhalf_ref
    inner_radius = min(cfg.inner_max_radius_kpc, max(cfg.inner_min_radius_kpc, inner_radius))

    r_inner = distances(stars["pos"], stars_com0)
    inner_mask = r_inner <= inner_radius

    if np.count_nonzero(inner_mask) < cfg.inner_min_particles:
        order = np.argsort(r_inner)
        n = min(len(order), max(cfg.inner_min_particles, 5))
        inner_mask = np.zeros(len(r_inner), dtype=bool)
        inner_mask[order[:n]] = True

    inner_star_ids = np.unique(stars["ids"][inner_mask])
    star_ids = np.unique(stars["ids"])

    gas_radius = cfg.initial_satellite_gas_radius_kpc
    if gas_radius is None:
        if np.isfinite(rhalf_ref):
            gas_radius = max(cfg.default_initial_gas_radius_kpc, cfg.gas_radius_factor_rhalf * rhalf_ref)
        else:
            gas_radius = cfg.default_initial_gas_radius_kpc

    dm_radius = cfg.initial_satellite_dm_radius_kpc
    if dm_radius is None:
        if np.isfinite(rhalf_ref):
            dm_radius = max(cfg.default_initial_dm_radius_kpc, cfg.dm_radius_factor_rhalf * rhalf_ref)
        else:
            dm_radius = cfg.default_initial_dm_radius_kpc

    gas = read_particle_type(first_snapshot, cfg.gas_ptype, cfg)
    if gas is not None and len(gas["ids"]) > 0:
        rg = distances(gas["pos"], initial_center)
        gas_ids = np.unique(gas["ids"][rg <= gas_radius])
    else:
        gas_ids = np.array([], dtype=np.int64)

    dm = read_particle_type(first_snapshot, cfg.dm_ptype, cfg)
    if dm is not None and len(dm["ids"]) > 0:
        if cfg.dm_selection_mode.lower() == "all":
            dm_ids = np.unique(dm["ids"])
        elif cfg.dm_selection_mode.lower() == "radius":
            rd = distances(dm["pos"], initial_center)
            dm_ids = np.unique(dm["ids"][rd <= dm_radius])
        else:
            raise ValueError("cfg.dm_selection_mode must be 'all' or 'radius'")
    else:
        dm_ids = np.array([], dtype=np.int64)

    return {
        "star_ids": star_ids,
        "inner_star_ids": inner_star_ids,
        "gas_ids": gas_ids,
        "dm_ids": dm_ids,
        "initial_center_kpc": np.asarray(initial_center, dtype=float).tolist(),
        "initial_stars_com_kpc": np.asarray(stars_com0, dtype=float).tolist(),
        "initial_stellar_core_kpc": np.asarray(stellar_core0, dtype=float).tolist(),
        "initial_rhalf_star_com_kpc": float(rhalf0_com),
        "initial_rhalf_star_core_kpc": float(rhalf0_core),
        "initial_inner_radius_kpc": float(inner_radius),
        "initial_gas_selection_radius_kpc": float(gas_radius),
        "initial_dm_selection_radius_kpc": float(dm_radius),
        "initial_Nstar": int(len(star_ids)),
        "initial_Ninner_star": int(len(inner_star_ids)),
        "initial_Ngas": int(len(gas_ids)),
        "initial_Ndm": int(len(dm_ids)),
    }


def save_config_and_catalogue(
    outdir: Union[str, Path],
    cfg: OrbitAnalysisConfig,
    idcat: Dict[str, object],
) -> None:
    outdir = ensure_dir(outdir)

    cfg_dict = asdict(cfg)
    with open(outdir / "analysis_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    np.savez_compressed(
        outdir / "initial_satellite_ids.npz",
        star_ids=np.asarray(idcat["star_ids"], dtype=np.int64),
        inner_star_ids=np.asarray(idcat["inner_star_ids"], dtype=np.int64),
        gas_ids=np.asarray(idcat["gas_ids"], dtype=np.int64),
        dm_ids=np.asarray(idcat["dm_ids"], dtype=np.int64),
    )

    summary = {}
    for key, value in idcat.items():
        if isinstance(value, np.ndarray):
            summary[key] = {
                "n": int(len(value)),
                "min": int(np.min(value)) if len(value) else None,
                "max": int(np.max(value)) if len(value) else None,
            }
        else:
            summary[key] = value

    with open(outdir / "initial_satellite_id_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# CENTRE / VELOCITY TRACKING
# =============================================================================

def compute_satellite_center(
    stars: Dict[str, np.ndarray],
    idcat: Dict[str, object],
    cfg: OrbitAnalysisConfig,
    previous_center_kpc: Optional[np.ndarray] = None,
    previous_rhalf_kpc: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return satellite centre, mask of stars used, and method label.
    """
    mode = cfg.center_mode.lower()
    nstar = len(stars["ids"])
    all_mask = np.ones(nstar, dtype=bool)

    if nstar == 0:
        return np.full(3, np.nan), np.zeros(0, dtype=bool), mode

    inner_ids = np.asarray(idcat["inner_star_ids"], dtype=np.int64)
    inner_mask = np.isin(stars["ids"], inner_ids, assume_unique=False)

    if mode in ["stars_com_all", "stellar_com_all", "stars_com"]:
        return mass_weighted_mean(stars["pos"], stars["mass"]), all_mask, mode

    if mode in ["inner_ids_com", "inner_com"]:
        if np.count_nonzero(inner_mask) >= 5:
            return mass_weighted_mean(stars["pos"][inner_mask], stars["mass"][inner_mask]), inner_mask, mode
        return mass_weighted_mean(stars["pos"], stars["mass"]), all_mask, mode + "_fallback_all"

    if mode in ["inner_ids_shrinking", "inner_shrinking"]:
        if np.count_nonzero(inner_mask) >= 5:
            pos = stars["pos"][inner_mask]
            mass = stars["mass"][inner_mask]
            init_center = previous_center_kpc if previous_center_kpc is not None else None
            center = shrinking_sphere_center(
                pos,
                mass,
                initial_center=init_center,
                initial_radius=cfg.shrink_initial_radius_kpc,
                shrink_factor=cfg.shrink_factor,
                min_particles=max(10, min(cfg.shrink_min_particles, np.count_nonzero(inner_mask) // 2)),
                min_radius=cfg.shrink_min_radius_kpc,
            )
            return center, inner_mask, mode

        return shrinking_sphere_center(
            stars["pos"],
            stars["mass"],
            initial_center=previous_center_kpc,
            initial_radius=cfg.shrink_initial_radius_kpc,
            shrink_factor=cfg.shrink_factor,
            min_particles=cfg.shrink_min_particles,
            min_radius=cfg.shrink_min_radius_kpc,
        ), all_mask, mode + "_fallback_all"

    if mode in ["stellar_core", "star_core", "shrinking_sphere"]:
        if previous_center_kpc is not None and np.all(np.isfinite(previous_center_kpc)):
            rprev = distances(stars["pos"], previous_center_kpc)
            search_radius = cfg.center_search_radius_kpc
            if previous_rhalf_kpc is not None and np.isfinite(previous_rhalf_kpc):
                search_radius = max(search_radius, 10.0 * previous_rhalf_kpc)

            near = rprev <= search_radius
            if np.count_nonzero(near) >= max(20, cfg.shrink_min_particles // 4):
                pos = stars["pos"][near]
                mass = stars["mass"][near]
                used_mask = near
            else:
                pos = stars["pos"]
                mass = stars["mass"]
                used_mask = all_mask

            center = shrinking_sphere_center(
                pos,
                mass,
                initial_center=previous_center_kpc,
                initial_radius=min(search_radius, np.nanmax(rprev) if len(rprev) else search_radius),
                shrink_factor=cfg.shrink_factor,
                min_particles=cfg.shrink_min_particles,
                min_radius=cfg.shrink_min_radius_kpc,
            )
            return center, used_mask, mode

        center = shrinking_sphere_center(
            stars["pos"],
            stars["mass"],
            initial_radius=cfg.shrink_initial_radius_kpc,
            shrink_factor=cfg.shrink_factor,
            min_particles=cfg.shrink_min_particles,
            min_radius=cfg.shrink_min_radius_kpc,
        )
        return center, all_mask, mode

    if mode in ["stars_com_near_previous", "local_stars_com"]:
        if previous_center_kpc is not None and np.all(np.isfinite(previous_center_kpc)):
            rprev = distances(stars["pos"], previous_center_kpc)
            search_radius = cfg.center_search_radius_kpc
            if previous_rhalf_kpc is not None and np.isfinite(previous_rhalf_kpc):
                search_radius = max(search_radius, 10.0 * previous_rhalf_kpc)

            near = rprev <= search_radius
            if np.count_nonzero(near) >= max(20, cfg.shrink_min_particles // 4):
                return mass_weighted_mean(stars["pos"][near], stars["mass"][near]), near, mode

        return mass_weighted_mean(stars["pos"], stars["mass"]), all_mask, mode + "_fallback_all"

    raise ValueError(
        "Unknown cfg.center_mode. Use one of: "
        "inner_ids_shrinking, inner_ids_com, stellar_core, stars_com_all, stars_com_near_previous"
    )


def compute_satellite_velocity(
    stars: Dict[str, np.ndarray],
    idcat: Dict[str, object],
    center: np.ndarray,
    center_mask: np.ndarray,
    rhalf_kpc: float,
    cfg: OrbitAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return bulk velocity, mask of stars used for velocity, and method label.
    """
    mode = cfg.velocity_mode.lower()
    nstar = len(stars["ids"])

    if nstar == 0:
        return np.full(3, np.nan), np.zeros(0, dtype=bool), mode

    all_mask = np.ones(nstar, dtype=bool)

    if mode in ["all_stars", "stars_com", "stellar_com"]:
        return mass_weighted_mean(stars["vel"], stars["mass"]), all_mask, mode

    if mode in ["same_as_center", "center_particles"]:
        if len(center_mask) == nstar and np.count_nonzero(center_mask) >= 5:
            return mass_weighted_mean(stars["vel"][center_mask], stars["mass"][center_mask]), center_mask, mode
        return mass_weighted_mean(stars["vel"], stars["mass"]), all_mask, mode + "_fallback_all"

    if mode in ["inner_ids", "inner"]:
        inner_ids = np.asarray(idcat["inner_star_ids"], dtype=np.int64)
        inner_mask = np.isin(stars["ids"], inner_ids, assume_unique=False)
        if np.count_nonzero(inner_mask) >= 5:
            return mass_weighted_mean(stars["vel"][inner_mask], stars["mass"][inner_mask]), inner_mask, mode
        return mass_weighted_mean(stars["vel"], stars["mass"]), all_mask, mode + "_fallback_all"

    if mode in ["inner_stars", "stellar_core"]:
        r = distances(stars["pos"], center)
        aperture = cfg.velocity_center_min_radius_kpc
        if np.isfinite(rhalf_kpc):
            aperture = max(aperture, cfg.velocity_center_radius_factor_rhalf * rhalf_kpc)

        mask = r <= aperture
        if np.count_nonzero(mask) < 5:
            order = np.argsort(r)
            n = max(5, min(len(order), int(0.1 * len(order))))
            mask = np.zeros(nstar, dtype=bool)
            mask[order[:n]] = True

        return mass_weighted_mean(stars["vel"][mask], stars["mass"][mask]), mask, mode

    raise ValueError("Unknown cfg.velocity_mode.")


# =============================================================================
# MASSES, TIDAL RADIUS, MEMBERSHIP, SFR, RAM PRESSURE
# =============================================================================

def satellite_mass_within_radius(
    components: Sequence[Dict[str, np.ndarray]],
    center: ArrayLike3,
    radius_kpc: float,
) -> float:
    total = 0.0
    for comp in components:
        if len(comp["mass"]) == 0:
            continue
        r = distances(comp["pos"], center)
        total += float(np.nansum(comp["mass"][r <= radius_kpc]))
    return total


def solve_tidal_radius_kpc(
    components: Sequence[Dict[str, np.ndarray]],
    center_kpc: ArrayLike3,
    cfg: OrbitAnalysisConfig,
) -> float:
    center = np.asarray(center_kpc, dtype=float)
    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    R = float(np.linalg.norm(center - host_center))

    if not np.isfinite(R) or R <= 0:
        return np.nan

    M_host = cfg.host.mass_enclosed_msun(R)
    if not np.isfinite(M_host) or M_host <= 0:
        return np.nan

    rt = max(cfg.tidal_min_radius_kpc, cfg.tidal_initial_fraction_of_R * R)
    rt = min(rt, cfg.tidal_max_fraction_of_R * R)

    for _ in range(cfg.tidal_max_iterations):
        m_sat = satellite_mass_within_radius(components, center, rt)
        if m_sat <= 0:
            return cfg.tidal_min_radius_kpc

        rt_new = R * (m_sat / (cfg.host.tidal_factor * M_host)) ** (1.0 / 3.0)
        rt_new = max(cfg.tidal_min_radius_kpc, min(rt_new, cfg.tidal_max_fraction_of_R * R))

        if abs(rt_new - rt) / max(rt, cfg.tidal_min_radius_kpc) < cfg.tidal_tolerance:
            return float(rt_new)

        rt = rt_new

    return float(rt)


def concatenate_components(components: Sequence[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    pos_list = []
    mass_list = []
    for comp in components:
        if len(comp["mass"]) == 0:
            continue
        pos_list.append(comp["pos"])
        mass_list.append(comp["mass"])

    if not pos_list:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    return np.vstack(pos_list), np.concatenate(mass_list)


def enclosed_mass_profile_at_radii(
    components: Sequence[Dict[str, np.ndarray]],
    center: ArrayLike3,
    radii: np.ndarray,
) -> np.ndarray:
    """Approximate M_sat(<r) for arbitrary radii using tracked particles."""
    pos, mass = concatenate_components(components)
    if len(mass) == 0:
        return np.zeros_like(radii, dtype=float)

    r_all = distances(pos, center)
    order = np.argsort(r_all)
    r_sorted = r_all[order]
    m_cum = np.cumsum(mass[order])

    radii = np.asarray(radii, dtype=float)
    return np.interp(radii, r_sorted, m_cum, left=0.0, right=m_cum[-1])


def compute_star_membership(
    stars: Dict[str, np.ndarray],
    gas: Dict[str, np.ndarray],
    dm: Dict[str, np.ndarray],
    center: np.ndarray,
    vcenter: np.ndarray,
    rt: float,
    cfg: OrbitAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      member_mask,
      stripped_candidate_mask,
      v_escape_profile_kms
    """
    nstar = len(stars["ids"])
    if nstar == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.empty(0, dtype=float)

    mode = cfg.member_mode.lower()
    rstar = distances(stars["pos"], center)

    if mode == "none":
        member = np.ones(nstar, dtype=bool)
        stripped = np.zeros(nstar, dtype=bool)
        vesc = np.full(nstar, np.nan)
        return member, stripped, vesc

    if not np.isfinite(rt):
        member = np.ones(nstar, dtype=bool)
        stripped = np.zeros(nstar, dtype=bool)
        vesc = np.full(nstar, np.nan)
        return member, stripped, vesc

    tidal_member = rstar <= cfg.member_tidal_factor * rt
    stripped_candidate = rstar > cfg.stripped_tidal_factor * rt

    components = [stars, gas, dm]
    Menc = enclosed_mass_profile_at_radii(components, center, np.maximum(rstar, cfg.tidal_min_radius_kpc))
    vesc = np.sqrt(np.maximum(2.0 * G_KPC_KMS2_MSUN * Menc / np.maximum(rstar, cfg.tidal_min_radius_kpc), 0.0))

    if mode == "tidal":
        member = tidal_member

    elif mode == "tidal_kinematic":
        vrel = np.linalg.norm(stars["vel"] - vcenter[None, :], axis=1)
        kin_member = vrel <= cfg.v_escape_factor * vesc
        member = tidal_member & kin_member
        stripped_candidate = stripped_candidate | ((~tidal_member) & (~kin_member))

    else:
        raise ValueError("cfg.member_mode must be 'none', 'tidal', or 'tidal_kinematic'.")

    return member, stripped_candidate, vesc


def component_summary(
    comp: Dict[str, np.ndarray],
    center: ArrayLike3,
    rt: float,
    rhalf: float,
    prefix: str,
) -> Dict[str, float]:
    out = {
        f"M{prefix}_tracked_msun": 0.0,
        f"M{prefix}_inside_rt_msun": 0.0,
        f"M{prefix}_inside_rhalf_msun": 0.0,
        f"N{prefix}_tracked": 0,
        f"N{prefix}_inside_rt": 0,
        f"N{prefix}_inside_rhalf": 0,
    }

    if len(comp["mass"]) == 0:
        return out

    r = distances(comp["pos"], center)
    inside_rt = r <= rt if np.isfinite(rt) else np.zeros(len(r), dtype=bool)
    inside_rhalf = r <= rhalf if np.isfinite(rhalf) else np.zeros(len(r), dtype=bool)

    out[f"M{prefix}_tracked_msun"] = float(np.nansum(comp["mass"]))
    out[f"M{prefix}_inside_rt_msun"] = float(np.nansum(comp["mass"][inside_rt]))
    out[f"M{prefix}_inside_rhalf_msun"] = float(np.nansum(comp["mass"][inside_rhalf]))
    out[f"N{prefix}_tracked"] = int(len(comp["mass"]))
    out[f"N{prefix}_inside_rt"] = int(np.count_nonzero(inside_rt))
    out[f"N{prefix}_inside_rhalf"] = int(np.count_nonzero(inside_rhalf))
    return out


def sum_particle_field(comp: Dict[str, np.ndarray], field: str, mask: Optional[np.ndarray] = None) -> float:
    if field not in comp:
        return np.nan
    arr = np.asarray(comp[field], dtype=float)
    if len(arr) == 0:
        return np.nan
    if mask is None:
        return float(np.nansum(arr))
    if len(mask) != len(arr):
        return np.nan
    return float(np.nansum(arr[mask]))


def pressure_unit_msun_kpc3_kms2_to_dyne_cm2() -> float:
    return (MSUN_IN_G / KPC_IN_CM**3) * (1.0e5**2)


def compute_local_cgm_ram_pressure(
    gas_all: Optional[Dict[str, np.ndarray]],
    satellite_gas_ids: np.ndarray,
    center_kpc: ArrayLike3,
    v_sat_kms: ArrayLike3,
    cfg: OrbitAnalysisConfig,
) -> Dict[str, float]:
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

    if np.count_nonzero(is_cgm) == 0:
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

    while aperture < max_ap and np.count_nonzero(r <= aperture) < min_n:
        aperture = min(max_ap, aperture * 1.5)
        if aperture >= max_ap:
            break

    use = r <= aperture
    n_use = int(np.count_nonzero(use))
    keys["N_cgm_for_ram"] = n_use
    keys["ram_pressure_radius_kpc"] = aperture

    if n_use == 0:
        v_cgm = np.asarray(cfg.host.host_velocity_kms, dtype=float)
        keys["V_rel_cgm_kms"] = float(np.linalg.norm(v_sat - v_cgm))
        return keys

    volume = (4.0 / 3.0) * np.pi * aperture**3
    rho_ap = float(np.nansum(mass_cgm[use]) / volume) if volume > 0 else np.nan
    keys["rho_cgm_aperture_msun_kpc3"] = rho_ap

    rho_field = np.nan
    density_all = gas_all.get("density", np.empty(0, dtype=float))
    if isinstance(density_all, np.ndarray) and len(density_all) == len(ids):
        density_cgm = density_all[is_cgm]
        finite_density = np.isfinite(density_cgm[use]) & (density_cgm[use] > 0)
        if np.any(finite_density):
            rho_field = float(np.average(density_cgm[use][finite_density], weights=mass_cgm[use][finite_density]))

    keys["rho_cgm_density_field_msun_kpc3"] = rho_field
    rho = rho_field if np.isfinite(rho_field) and rho_field > 0 else rho_ap
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
        p_code = rho * v_rel**2
        p_cgs = p_code * pressure_unit_msun_kpc3_kms2_to_dyne_cm2()
        keys["P_ram_msun_kpc3_kms2"] = float(p_code)
        keys["P_ram_dyne_cm2"] = float(p_cgs)
        keys["log10_P_ram_dyne_cm2"] = float(np.log10(p_cgs)) if p_cgs > 0 else np.nan

    return keys


# =============================================================================
# ORBITAL EXTREMA
# =============================================================================

def compute_orbital_extrema(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or "R_host_kpc" not in df.columns:
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
        row = {
            "event_type": "pericentre_candidate_global_minimum",
            "snapshot_index": int(df.iloc[i]["snapshot_index"]) if "snapshot_index" in df.columns else i,
            "snapshot_number": int(df.iloc[i]["snapshot_number"]) if "snapshot_number" in df.columns else i,
            "time_gyr": float(df.iloc[i]["time_gyr"]) if "time_gyr" in df.columns else np.nan,
            "R_host_kpc": float(df.iloc[i]["R_host_kpc"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def add_event_and_time_columns(df: pd.DataFrame, extrema_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["orbital_event"] = ""

    if len(extrema_df) > 0:
        for _, ev in extrema_df.iterrows():
            idx = int(ev["snapshot_index"])
            if 0 <= idx < len(out):
                out.loc[out.index[idx], "orbital_event"] = ev["event_type"]

    tperi = np.nan
    if len(extrema_df) > 0 and "time_gyr" in extrema_df.columns:
        peri = extrema_df[extrema_df["event_type"] == "pericentre"]
        if len(peri) == 0:
            peri = extrema_df[extrema_df["event_type"].astype(str).str.startswith("pericentre")]
        if len(peri) > 0:
            tperi = float(peri.iloc[0]["time_gyr"])

    out["t_first_pericentre_gyr"] = tperi
    out["time_since_first_pericentre_gyr"] = out["time_gyr"] - tperi if np.isfinite(tperi) else np.nan

    return out


def summarize_extrema_windows(
    df: pd.DataFrame,
    extrema_df: pd.DataFrame,
    window_snapshots: int = 1,
) -> pd.DataFrame:
    if extrema_df is None or len(extrema_df) == 0:
        return pd.DataFrame()

    columns = [
        "R_host_kpc", "r_tidal_kpc", "rhalf_star_member_kpc", "rhalf_star_all_kpc",
        "Mstar_member_msun", "Mstar_inside_rt_msun", "Mgas_inside_rt_msun", "Mdm_inside_rt_msun",
        "SFR_gas_tracked_msun_yr", "SFR_gas_inside_rt_msun_yr",
        "sSFR_inside_rt_yr", "SFE_inside_rt_yr",
        "P_ram_dyne_cm2", "rho_cgm_msun_kpc3", "V_rel_cgm_kms",
        "tidal_field_kms2_kpc2", "tidal_accel_across_rhalf_kms2_kpc",
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


# =============================================================================
# ONE SNAPSHOT / ONE LABEL / ALL LABELS
# =============================================================================

def analyze_one_snapshot(
    snapshot_file: Union[str, Path],
    snapshot_index: int,
    cfg: OrbitAnalysisConfig,
    idcat: Dict[str, object],
    stripped_counter: np.ndarray,
    previous_center_kpc: Optional[np.ndarray] = None,
    previous_rhalf_kpc: Optional[float] = None,
) -> Tuple[Dict[str, float], np.ndarray, float, np.ndarray]:
    snapshot_file = Path(snapshot_file)

    stars_all = read_particle_type(snapshot_file, cfg.star_ptype, cfg)
    gas_all = read_particle_type(snapshot_file, cfg.gas_ptype, cfg)
    dm_all = read_particle_type(snapshot_file, cfg.dm_ptype, cfg)

    stars = filter_particle_dict(stars_all, np.asarray(idcat["star_ids"], dtype=np.int64))
    gas = filter_particle_dict(gas_all, np.asarray(idcat["gas_ids"], dtype=np.int64))
    dm = filter_particle_dict(dm_all, np.asarray(idcat["dm_ids"], dtype=np.int64))

    if len(stars["ids"]) == 0:
        raise RuntimeError(f"No tracked star IDs found in {snapshot_file}")

    center, center_mask, center_method_used = compute_satellite_center(
        stars,
        idcat,
        cfg,
        previous_center_kpc=previous_center_kpc,
        previous_rhalf_kpc=previous_rhalf_kpc,
    )

    rhalf_all = half_mass_radius(stars["pos"], stars["mass"], center)

    vcenter, velocity_mask, velocity_method_used = compute_satellite_velocity(
        stars,
        idcat,
        center,
        center_mask,
        rhalf_all,
        cfg,
    )

    rt = solve_tidal_radius_kpc([stars, gas, dm], center, cfg)

    star_member, star_stripped_candidate, vesc_star = compute_star_membership(
        stars,
        gas,
        dm,
        center,
        vcenter,
        rt,
        cfg,
    )

    # Update persistent stripped counters.
    star_ids_all = np.asarray(idcat["star_ids"], dtype=np.int64)
    loc = np.searchsorted(star_ids_all, stars["ids"])
    valid = (loc >= 0) & (loc < len(star_ids_all)) & (star_ids_all[loc] == stars["ids"])

    loc_valid = loc[valid]
    cand_valid = star_stripped_candidate[valid]
    stripped_counter[loc_valid] = np.where(cand_valid, stripped_counter[loc_valid] + 1, 0)

    current_definitive = np.zeros(len(stars["ids"]), dtype=bool)
    current_definitive[valid] = stripped_counter[loc_valid] >= cfg.stripped_consecutive_snapshots

    if np.count_nonzero(star_member) >= 5:
        rhalf_member = half_mass_radius(stars["pos"][star_member], stars["mass"][star_member], center)
        r90_member = enclosed_radius_fraction(stars["pos"][star_member], stars["mass"][star_member], center, 0.9)
    else:
        rhalf_member = rhalf_all
        r90_member = enclosed_radius_fraction(stars["pos"], stars["mass"], center, 0.9)

    r90_all = enclosed_radius_fraction(stars["pos"], stars["mass"], center, 0.9)

    # Gas masks for SFR.
    if len(gas["ids"]) > 0:
        rgas = distances(gas["pos"], center)
    else:
        rgas = np.empty(0, dtype=float)

    gas_inside_rt = rgas <= rt if np.isfinite(rt) else np.zeros(len(rgas), dtype=bool)
    gas_inside_rhalf = rgas <= rhalf_member if np.isfinite(rhalf_member) else np.zeros(len(rgas), dtype=bool)

    # Component summaries.
    row: Dict[str, float] = {}
    row.update(component_summary(stars, center, rt, rhalf_member, "star"))
    row.update(component_summary(gas, center, rt, rhalf_member, "gas"))
    row.update(component_summary(dm, center, rt, rhalf_member, "dm"))

    # Star member mass.
    row["Mstar_member_msun"] = float(np.nansum(stars["mass"][star_member]))
    row["Nstar_member"] = int(np.count_nonzero(star_member))
    row["Mstar_stripped_candidate_msun"] = float(np.nansum(stars["mass"][star_stripped_candidate]))
    row["Nstar_stripped_candidate"] = int(np.count_nonzero(star_stripped_candidate))
    row["Mstar_stripped_definitive_msun"] = float(np.nansum(stars["mass"][current_definitive]))
    row["Nstar_stripped_definitive"] = int(np.count_nonzero(current_definitive))

    row["fstar_member"] = safe_divide(row["Nstar_member"], row["Nstar_tracked"])
    row["fstar_stripped_candidate"] = safe_divide(row["Nstar_stripped_candidate"], row["Nstar_tracked"])
    row["fstar_stripped_definitive"] = safe_divide(row["Nstar_stripped_definitive"], row["Nstar_tracked"])

    # SFR diagnostics from tracked satellite gas.
    sfr_tracked = sum_particle_field(gas, "sfr")
    sfr_inside_rt = sum_particle_field(gas, "sfr", gas_inside_rt)
    sfr_inside_rhalf = sum_particle_field(gas, "sfr", gas_inside_rhalf)

    row["SFR_gas_tracked_msun_yr"] = sfr_tracked
    row["SFR_gas_inside_rt_msun_yr"] = sfr_inside_rt
    row["SFR_gas_inside_rhalf_msun_yr"] = sfr_inside_rhalf
    row["has_sfr_field"] = bool(len(gas.get("sfr", [])) == len(gas.get("ids", [])) and len(gas.get("sfr", [])) > 0)

    # Derived SFR quantities.
    row["sSFR_tracked_yr"] = safe_divide(sfr_tracked, row["Mstar_tracked_msun"])
    row["sSFR_inside_rt_yr"] = safe_divide(sfr_inside_rt, row["Mstar_inside_rt_msun"])
    row["sSFR_member_yr"] = safe_divide(sfr_tracked, row["Mstar_member_msun"])

    row["SFE_tracked_yr"] = safe_divide(sfr_tracked, row["Mgas_tracked_msun"])
    row["SFE_inside_rt_yr"] = safe_divide(sfr_inside_rt, row["Mgas_inside_rt_msun"])
    row["SFE_inside_rhalf_yr"] = safe_divide(sfr_inside_rhalf, row["Mgas_inside_rhalf_msun"])

    row["tdep_tracked_gyr"] = safe_divide(row["Mgas_tracked_msun"], sfr_tracked) / 1.0e9 if np.isfinite(safe_divide(row["Mgas_tracked_msun"], sfr_tracked)) else np.nan
    row["tdep_inside_rt_gyr"] = safe_divide(row["Mgas_inside_rt_msun"], sfr_inside_rt) / 1.0e9 if np.isfinite(safe_divide(row["Mgas_inside_rt_msun"], sfr_inside_rt)) else np.nan

    # Orbital quantities.
    host_center = np.asarray(cfg.host.host_center_kpc, dtype=float)
    host_vel = np.asarray(cfg.host.host_velocity_kms, dtype=float)
    rel_pos = center - host_center
    rel_vel = vcenter - host_vel
    R = float(np.linalg.norm(rel_pos))
    V = float(np.linalg.norm(rel_vel))

    if R > 0:
        rhat = rel_pos / R
        Vrad = float(np.dot(rel_vel, rhat))
        Vtan = float(np.sqrt(max(V * V - Vrad * Vrad, 0.0)))
    else:
        Vrad = np.nan
        Vtan = np.nan

    time_gyr = read_snapshot_time_gyr(snapshot_file, cfg)

    # Ram pressure.
    ram = compute_local_cgm_ram_pressure(
        gas_all=gas_all,
        satellite_gas_ids=np.asarray(idcat["gas_ids"], dtype=np.int64),
        center_kpc=center,
        v_sat_kms=vcenter,
        cfg=cfg,
    )

    # Tidal/environment.
    Mhost_R = cfg.host.mass_enclosed_msun(R) if np.isfinite(R) and R > 0 else np.nan
    tidal_field = cfg.host.tidal_field_kms2_kpc2(R)
    tidal_accel_rhalf = tidal_field * rhalf_member if np.isfinite(tidal_field) and np.isfinite(rhalf_member) else np.nan

    all_stars_com = mass_weighted_mean(stars["pos"], stars["mass"])

    row.update({
        "snapshot_index": int(snapshot_index),
        "snapshot_number": int(natural_snapshot_number(snapshot_file)),
        "snapshot_file": str(snapshot_file),
        "time_gyr": float(time_gyr),

        "center_mode_requested": cfg.center_mode,
        "center_method_used": center_method_used,
        "velocity_mode_requested": cfg.velocity_mode,
        "velocity_method_used": velocity_method_used,

        "x_sat_kpc": float(center[0]),
        "y_sat_kpc": float(center[1]),
        "z_sat_kpc": float(center[2]),
        "vx_sat_kms": float(vcenter[0]),
        "vy_sat_kms": float(vcenter[1]),
        "vz_sat_kms": float(vcenter[2]),

        "R_host_kpc": R,
        "V_3d_kms": V,
        "V_rad_kms": Vrad,
        "V_tan_kms": Vtan,

        "r_tidal_kpc": float(rt),
        "rhalf_star_all_kpc": float(rhalf_all),
        "r90_star_all_kpc": float(r90_all),
        "rhalf_star_member_kpc": float(rhalf_member),
        "r90_star_member_kpc": float(r90_member),
        "rhalf_over_rt": safe_divide(rhalf_member, rt),

        "x_all_stars_com_kpc": float(all_stars_com[0]),
        "y_all_stars_com_kpc": float(all_stars_com[1]),
        "z_all_stars_com_kpc": float(all_stars_com[2]),
        "offset_all_stars_com_from_center_kpc": float(np.linalg.norm(all_stars_com - center)),

        "N_center_particles": int(np.count_nonzero(center_mask)),
        "N_velocity_particles": int(np.count_nonzero(velocity_mask)),
        "median_vesc_star_kms": float(np.nanmedian(vesc_star)) if len(vesc_star) else np.nan,

        "Mhost_enclosed_msun": Mhost_R,
        "tidal_field_kms2_kpc2": tidal_field,
        "tidal_field_proxy_msun_kpc3": safe_divide(Mhost_R, R**3) if np.isfinite(R) and R > 0 else np.nan,
        "tidal_accel_across_rhalf_kms2_kpc": tidal_accel_rhalf,
    })

    row.update(ram)

    if row["Mstar_inside_rt_msun"] > 0:
        row["gas_to_star_inside_rt"] = row["Mgas_inside_rt_msun"] / row["Mstar_inside_rt_msun"]
        row["dm_to_star_inside_rt"] = row["Mdm_inside_rt_msun"] / row["Mstar_inside_rt_msun"]
    else:
        row["gas_to_star_inside_rt"] = np.nan
        row["dm_to_star_inside_rt"] = np.nan

    return row, center, rhalf_member, stripped_counter


def analyze_label(label: str, cfg: OrbitAnalysisConfig) -> pd.DataFrame:
    snaps = find_snapshots_for_label(cfg, label)

    label_outdir = ensure_dir(Path(cfg.output_dir) / label)

    if cfg.verbose:
        print(f"\n=== Analyzing label: {label} ===")
        print(f"Found {len(snaps)} snapshots.")
        print(f"Output: {label_outdir}")

    idcat = select_initial_satellite_catalogue(snaps[0], cfg)
    save_config_and_catalogue(label_outdir, cfg, idcat)

    if cfg.verbose:
        print("Initial satellite ID catalogue:")
        print(f"  stars:       {len(idcat['star_ids'])}")
        print(f"  inner stars: {len(idcat['inner_star_ids'])}")
        print(f"  gas:         {len(idcat['gas_ids'])}")
        print(f"  DM:          {len(idcat['dm_ids'])}")
        print(f"  initial stars_com [kpc]: {idcat['initial_stars_com_kpc']}")
        print(f"  initial inner radius [kpc]: {idcat['initial_inner_radius_kpc']:.3f}")

    rows = []
    prev_center = None
    prev_rhalf = None

    star_ids_all = np.asarray(idcat["star_ids"], dtype=np.int64)
    stripped_counter = np.zeros(len(star_ids_all), dtype=int)

    for i, snap in enumerate(snaps):
        if cfg.verbose:
            print(f"[{label}] {i + 1}/{len(snaps)}: {snap.name}")

        row, center, rhalf, stripped_counter = analyze_one_snapshot(
            snap,
            snapshot_index=i,
            cfg=cfg,
            idcat=idcat,
            stripped_counter=stripped_counter,
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
    df = add_event_and_time_columns(df, extrema_df)
    extrema_window_df = summarize_extrema_windows(df, extrema_df, cfg.event_window_snapshots)

    df.to_csv(label_outdir / "orbit_full_timeseries.csv", index=False)
    extrema_df.to_csv(label_outdir / "orbital_extrema.csv", index=False)
    extrema_window_df.to_csv(label_outdir / "orbital_extrema_window_summary.csv", index=False)

    if cfg.verbose:
        print(f"Saved: {label_outdir / 'orbit_full_timeseries.csv'}")
        if len(extrema_df):
            cols = ["event_type", "snapshot_number", "time_gyr", "R_host_kpc", "V_3d_kms", "V_rad_kms", "V_tan_kms", "r_tidal_kpc", "rhalf_star_member_kpc"]
            cols = [c for c in cols if c in extrema_df.columns]
            print("\nOrbital extrema:")
            print(extrema_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


def analyze_all(cfg: OrbitAnalysisConfig) -> Dict[str, pd.DataFrame]:
    outdir = ensure_dir(cfg.output_dir)

    labels = cfg.labels if cfg.labels is not None else discover_labels(cfg.root)

    results: Dict[str, pd.DataFrame] = {}
    for label in labels:
        results[label] = analyze_label(label, cfg)

    combined_df = pd.concat([df.assign(label=label) for label, df in results.items()], ignore_index=True)
    combined_df.to_csv(outdir / "combined_orbit_full_timeseries.csv", index=False)

    extrema_all = []
    for label, df in results.items():
        ev = compute_orbital_extrema(df)
        if len(ev):
            extrema_all.append(ev.assign(label=label))

    if extrema_all:
        extrema_all_df = pd.concat(extrema_all, ignore_index=True)
    else:
        extrema_all_df = pd.DataFrame()

    extrema_all_df.to_csv(outdir / "combined_orbital_extrema.csv", index=False)

    if cfg.verbose:
        print("\nSaved combined outputs:")
        print(" ", outdir / "combined_orbit_full_timeseries.csv")
        print(" ", outdir / "combined_orbital_extrema.csv")

    return results


def load_results_from_output(output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load saved per-label CSVs if you want to plot without rerunning HDF5 analysis."""
    output_dir = Path(output_dir)
    results = {}
    for path in sorted(output_dir.glob("*/orbit_full_timeseries.csv")):
        label = path.parent.name
        results[label] = pd.read_csv(path)
    return results