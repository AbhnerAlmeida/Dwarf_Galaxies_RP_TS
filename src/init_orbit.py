#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initial_orbit_diagnostics.py

Compute initial orbital diagnostics from the first Gadget/Gadget4 snapshot of
several dwarf-galaxy orbit runs.

For each label, the script reads the first snapshot and measures, for several
possible definitions of the satellite centre:

  - x0, y0, z0 and vx0, vy0, vz0
  - R0, V0, Vrad0, Vtan0
  - specific angular momentum vector L = r x v and |L|
  - specific orbital energy E = 0.5 V^2 + Phi_NFW(R)
  - local Vcirc, Vesc, L/Lcirc(R)
  - predicted turning points r_peri and r_apo from the initial E and L

Recommended use:

python initial_orbit_diagnostics.py \
    --root ./../SIMULATIONS/ORBIT/HigherRes \
    --labels E_mid_L_radial E_mid_L_mid E_mid_L_high \
    --output-dir orbit_initial_diagnostics \
    --preferred-center-mode inner_stars_dm \
    --mass-unit-to-msun 1e10 \
    --length-unit-to-kpc 1.0 \
    --velocity-unit-to-kms 1.0 \
    --time-unit-to-gyr 0.977792221 \
    --m200-msun 1e12 \
    --r200-kpc 210 \
    --concentration 10

Outputs:
  initial_orbit_diagnostics_all_centers.csv
  initial_orbit_diagnostics_preferred.csv
  initial_orbit_diagnostics_summary.png
  initial_orbit_vectors_xy.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import brentq
except Exception:
    brentq = None

G_KPC_KMS2_MSUN = 4.30091e-6


@dataclass
class Units:
    length_unit_to_kpc: float = 1.0
    velocity_unit_to_kms: float = 1.0
    mass_unit_to_msun: float = 1.0e10
    time_unit_to_gyr: float = 0.977792221


@dataclass
class Host:
    center_kpc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_kms: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    m200_msun: float = 1.0e12
    r200_kpc: float = 210.0
    concentration: float = 10.0
    truncate_mass_at_r200: bool = False


def natural_snapshot_number(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def discover_labels(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "output").exists()])


def first_snapshot(root: Path, label: str, snapshot_glob: str) -> Path:
    out = root / label / "output"
    snaps = sorted(out.glob(snapshot_glob), key=natural_snapshot_number)
    if not snaps:
        raise FileNotFoundError(f"No snapshots found for {label} in {out} with {snapshot_glob}")
    return snaps[0]


def header_attr(f: h5py.File, name: str, default=None):
    if "Header" not in f:
        return default
    return f["Header"].attrs.get(name, default)


def first_dataset(g: h5py.Group, names: Sequence[str]):
    for n in names:
        if n in g:
            return g[n][:]
    return None


def read_time(snapshot: Path, units: Units) -> float:
    with h5py.File(snapshot, "r") as f:
        t = header_attr(f, "Time", np.nan)
    return float(t) * units.time_unit_to_gyr if np.isfinite(t) else np.nan


def read_ptype(snapshot: Path, ptype: int, units: Units) -> Optional[Dict[str, np.ndarray]]:
    gname = f"PartType{ptype}"
    with h5py.File(snapshot, "r") as f:
        if gname not in f:
            return None
        g = f[gname]
        pos = first_dataset(g, ["Coordinates", "Position", "Positions"])
        vel = first_dataset(g, ["Velocities", "Velocity"])
        ids = first_dataset(g, ["ParticleIDs", "ParticleID", "IDs", "ID"])
        mass = first_dataset(g, ["Masses", "Mass", "ParticleMasses"])
        if pos is None or vel is None:
            raise KeyError(f"Missing Coordinates or Velocities in {snapshot}:{gname}")
        if ids is None:
            ids = np.arange(1, len(pos) + 1, dtype=np.int64)
        if mass is None:
            mt = header_attr(f, "MassTable", None)
            if mt is None or float(mt[ptype]) <= 0:
                raise KeyError(f"No Masses and no usable MassTable for {snapshot}:{gname}")
            mass = np.full(len(pos), float(mt[ptype]), dtype=float)
    return {
        "pos": np.asarray(pos, dtype=float) * units.length_unit_to_kpc,
        "vel": np.asarray(vel, dtype=float) * units.velocity_unit_to_kms,
        "ids": np.asarray(ids, dtype=np.int64),
        "mass": np.asarray(mass, dtype=float) * units.mass_unit_to_msun,
    }


def distances(pos: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos - np.asarray(center, dtype=float)[None, :], axis=1)


def mwmean(x: np.ndarray, m: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.full(x.shape[1] if x.ndim == 2 else 1, np.nan)
    if m is None or np.sum(m) <= 0:
        return np.nanmean(x, axis=0)
    m = np.asarray(m, dtype=float)
    good = np.isfinite(m) & (m > 0)
    if x.ndim == 2:
        good &= np.all(np.isfinite(x), axis=1)
    else:
        good &= np.isfinite(x)
    if np.count_nonzero(good) == 0:
        return np.nanmean(x, axis=0)
    return np.average(x[good], axis=0, weights=m[good])


def half_mass_radius(pos: np.ndarray, mass: np.ndarray, center: np.ndarray) -> float:
    if len(pos) == 0 or np.sum(mass) <= 0:
        return np.nan
    r = distances(pos, center)
    order = np.argsort(r)
    rs = r[order]
    ms = mass[order]
    cum = np.cumsum(ms)
    idx = np.searchsorted(cum, 0.5 * cum[-1])
    return float(rs[min(idx, len(rs) - 1)])


def shrinking_sphere_center(
    pos: np.ndarray,
    mass: np.ndarray,
    initial_radius: Optional[float] = None,
    shrink_factor: float = 0.75,
    min_particles: int = 100,
    min_radius_kpc: float = 0.2,
) -> np.ndarray:
    if len(pos) == 0:
        return np.array([np.nan, np.nan, np.nan])
    center = mwmean(pos, mass)
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
        center = mwmean(pos[inside], mass[inside])
        current = inside
        radius *= shrink_factor
    if len(current) > 0:
        center = mwmean(pos[current], mass[current])
    return np.asarray(center, dtype=float)


def empty_comp() -> Dict[str, np.ndarray]:
    return {"pos": np.empty((0, 3)), "vel": np.empty((0, 3)), "mass": np.empty(0), "ids": np.empty(0, dtype=np.int64)}


def select_within(comp: Optional[Dict[str, np.ndarray]], center: np.ndarray, radius: float) -> Dict[str, np.ndarray]:
    if comp is None or len(comp["mass"]) == 0:
        return empty_comp()
    r = distances(comp["pos"], center)
    mask = r <= radius
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(r) else v) for k, v in comp.items()}


def concat(comps: Sequence[Optional[Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
    ps, vs, ms, ids = [], [], [], []
    for c in comps:
        if c is None or len(c.get("mass", [])) == 0:
            continue
        ps.append(c["pos"]); vs.append(c["vel"]); ms.append(c["mass"]); ids.append(c.get("ids", np.arange(len(c["mass"]))))
    if not ps:
        return empty_comp()
    return {"pos": np.vstack(ps), "vel": np.vstack(vs), "mass": np.concatenate(ms), "ids": np.concatenate(ids)}


# ---------------- NFW dynamics ----------------

def f_nfw(x):
    return np.log1p(x) - x / (1.0 + x)


def nfw_mass(R: float, host: Host) -> float:
    R = max(float(R), 1e-12)
    c = host.concentration
    x = c * R / host.r200_kpc
    m = host.m200_msun * f_nfw(x) / f_nfw(c)
    if host.truncate_mass_at_r200:
        m = min(m, host.m200_msun)
    return float(m)


def nfw_phi(R: float, host: Host) -> float:
    R = float(R)
    c = host.concentration
    rs = host.r200_kpc / c
    A = host.m200_msun / f_nfw(c)
    if R <= 1e-12:
        return float(-G_KPC_KMS2_MSUN * A / rs)
    return float(-G_KPC_KMS2_MSUN * A * np.log1p(R / rs) / R)


def vcirc(R: float, host: Host) -> float:
    if not np.isfinite(R) or R <= 0:
        return np.nan
    return float(np.sqrt(G_KPC_KMS2_MSUN * nfw_mass(R, host) / R))


def vesc(R: float, host: Host) -> float:
    phi = nfw_phi(R, host)
    return float(np.sqrt(-2.0 * phi)) if phi < 0 else np.nan


def turning_points(E: float, L: float, host: Host, R0: float) -> Tuple[float, float]:
    if brentq is None or not np.isfinite(E) or not np.isfinite(L) or not np.isfinite(R0) or R0 <= 0:
        return np.nan, np.nan
    if E >= 0:
        return np.nan, np.nan
    if L <= 0:
        return 0.0, np.nan

    def fturn(r):
        return E - nfw_phi(r, host) - L * L / (2.0 * r * r)

    rmin = 1e-3
    rmax = max(20.0 * R0, 20.0 * host.r200_kpc)
    grid = np.logspace(np.log10(rmin), np.log10(rmax), 4096)
    val = np.array([fturn(r) for r in grid])
    roots = []
    for i in range(len(grid) - 1):
        if not np.isfinite(val[i]) or not np.isfinite(val[i + 1]):
            continue
        if val[i] == 0:
            roots.append(grid[i])
        elif val[i] * val[i + 1] < 0:
            try:
                roots.append(brentq(fturn, grid[i], grid[i + 1], maxiter=100))
            except Exception:
                pass
    roots = sorted(set(round(float(r), 9) for r in roots))
    if len(roots) == 0:
        return np.nan, np.nan
    if len(roots) == 1:
        return roots[0], np.nan
    return roots[0], roots[-1]


def center_method(
    method: str,
    stars: Dict[str, np.ndarray],
    dm: Dict[str, np.ndarray],
    gas: Dict[str, np.ndarray],
    star_core: np.ndarray,
    star_core_vel: np.ndarray,
    rhalf: float,
    inner_factor: float,
    inner_min: float,
    inner_max: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    method = method.lower()
    if method == "star_core":
        return star_core, star_core_vel, {"center_radius_kpc": np.nan, "N_center_particles": len(stars["mass"]), "M_center_msun": float(np.sum(stars["mass"]))}
    if method == "stars_com":
        return mwmean(stars["pos"], stars["mass"]), mwmean(stars["vel"], stars["mass"]), {"center_radius_kpc": np.nan, "N_center_particles": len(stars["mass"]), "M_center_msun": float(np.sum(stars["mass"]))}
    if method == "all_tracked":
        c = concat([stars, dm, gas])
        return mwmean(c["pos"], c["mass"]), mwmean(c["vel"], c["mass"]), {"center_radius_kpc": np.nan, "N_center_particles": len(c["mass"]), "M_center_msun": float(np.sum(c["mass"]))}

    radius = inner_min if not np.isfinite(rhalf) else inner_factor * rhalf
    radius = min(inner_max, max(inner_min, radius))
    if method == "inner_stars_dm":
        c = concat([stars, dm])
    elif method == "inner_stars_dm_gas":
        c = concat([stars, dm, gas])
    else:
        raise ValueError(f"Unknown center method: {method}")

    if len(c["mass"]) == 0:
        return np.full(3, np.nan), np.full(3, np.nan), {"center_radius_kpc": radius, "N_center_particles": 0, "M_center_msun": 0.0}
    r = distances(c["pos"], star_core)
    inside = r <= radius
    if np.count_nonzero(inside) < 20:
        c = stars
        r = distances(c["pos"], star_core)
        inside = r <= radius
    if np.count_nonzero(inside) < 5:
        inside = np.ones(len(c["mass"]), dtype=bool)
    return mwmean(c["pos"][inside], c["mass"][inside]), mwmean(c["vel"][inside], c["mass"][inside]), {"center_radius_kpc": float(radius), "N_center_particles": int(np.count_nonzero(inside)), "M_center_msun": float(np.sum(c["mass"][inside]))}


def orbit_row(label: str, snapshot: Path, time_gyr: float, method: str, center: np.ndarray, vel: np.ndarray, host: Host, extra: Dict[str, float]) -> Dict[str, float]:
    hcen = np.asarray(host.center_kpc, dtype=float)
    hvel = np.asarray(host.velocity_kms, dtype=float)
    rvec = center - hcen
    vvec = vel - hvel
    R = float(np.linalg.norm(rvec))
    V = float(np.linalg.norm(vvec))
    if R > 0:
        rhat = rvec / R
        Vrad = float(np.dot(vvec, rhat))
        Vtan = float(np.sqrt(max(V * V - Vrad * Vrad, 0.0)))
    else:
        Vrad = np.nan; Vtan = np.nan
    Lvec = np.cross(rvec, vvec)
    L = float(np.linalg.norm(Lvec))
    phi = nfw_phi(R, host) if np.isfinite(R) else np.nan
    E = 0.5 * V * V + phi if np.isfinite(V) and np.isfinite(phi) else np.nan
    vc = vcirc(R, host)
    ve = vesc(R, host)
    Lcirc = R * vc if np.isfinite(R) and np.isfinite(vc) else np.nan
    rperi, rapo = turning_points(E, L, host, R)
    inc = float(np.degrees(np.arccos(np.clip(Lvec[2] / L, -1, 1)))) if L > 0 else np.nan
    row = {
        "label": label,
        "snapshot_file": str(snapshot),
        "snapshot_number": natural_snapshot_number(snapshot),
        "time_gyr": float(time_gyr),
        "center_method": method,
        "x0_kpc": float(center[0]), "y0_kpc": float(center[1]), "z0_kpc": float(center[2]),
        "vx0_kms": float(vel[0]), "vy0_kms": float(vel[1]), "vz0_kms": float(vel[2]),
        "R0_kpc": R, "V0_kms": V, "Vrad0_kms": Vrad, "Vtan0_kms": Vtan,
        "Vtan0_over_V0": float(Vtan / V) if V > 0 else np.nan,
        "Lx0_kpc_kms": float(Lvec[0]), "Ly0_kpc_kms": float(Lvec[1]), "Lz0_kpc_kms": float(Lvec[2]),
        "L0_kpc_kms": L,
        "orbital_plane_inclination_deg": inc,
        "Mhost_R0_msun": nfw_mass(R, host) if np.isfinite(R) else np.nan,
        "Phi_NFW_R0_km2_s2": phi,
        "E0_km2_s2": E,
        "is_bound_E0_lt_0": bool(E < 0) if np.isfinite(E) else False,
        "Vcirc_R0_kms": vc,
        "Vesc_R0_kms": ve,
        "V0_over_Vcirc": float(V / vc) if np.isfinite(vc) and vc > 0 else np.nan,
        "Vtan0_over_Vcirc": float(Vtan / vc) if np.isfinite(vc) and vc > 0 else np.nan,
        "L0_over_Lcirc_local": float(L / Lcirc) if np.isfinite(Lcirc) and Lcirc > 0 else np.nan,
        "z0_over_R0": float(center[2] / R) if R > 0 else np.nan,
        "vz0_over_V0": float(vel[2] / V) if V > 0 else np.nan,
        "rperi_pred_kpc": rperi,
        "rapo_pred_kpc": rapo,
    }
    row.update(extra)
    return row


def analyze_label(label: str, snapshot: Path, units: Units, host: Host, center_methods: Sequence[str], args) -> List[Dict[str, float]]:
    stars = read_ptype(snapshot, args.star_ptype, units)
    gas_all = read_ptype(snapshot, args.gas_ptype, units)
    dm_all = read_ptype(snapshot, args.dm_ptype, units)
    if stars is None or len(stars["mass"]) == 0:
        raise RuntimeError(f"No stars in {snapshot}")

    time_gyr = read_time(snapshot, units)
    star_core = shrinking_sphere_center(stars["pos"], stars["mass"], args.shrink_initial_radius_kpc, args.shrink_factor, args.shrink_min_particles, args.shrink_min_radius_kpc)
    rhalf = half_mass_radius(stars["pos"], stars["mass"], star_core)

    rs = distances(stars["pos"], star_core)
    v_radius = args.velocity_center_min_radius_kpc
    if np.isfinite(rhalf):
        v_radius = max(v_radius, args.velocity_center_radius_factor_rhalf * rhalf)
    inner = rs <= v_radius
    if np.count_nonzero(inner) < 5:
        order = np.argsort(rs)
        inner = np.zeros(len(rs), dtype=bool)
        inner[order[:max(5, min(len(order), int(0.1 * len(order))))]] = True
    star_core_vel = mwmean(stars["vel"][inner], stars["mass"][inner])

    gas_radius = args.gas_selection_radius_kpc
    if gas_radius is None:
        gas_radius = max(args.default_gas_radius_kpc, args.gas_radius_factor_rhalf * rhalf) if np.isfinite(rhalf) else args.default_gas_radius_kpc
    gas = select_within(gas_all, star_core, gas_radius)

    dm_radius = args.dm_selection_radius_kpc
    if dm_radius is None:
        dm_radius = max(args.default_dm_radius_kpc, args.dm_radius_factor_rhalf * rhalf) if np.isfinite(rhalf) else args.default_dm_radius_kpc
    if dm_all is None:
        dm = empty_comp()
    elif args.dm_selection_mode == "radius":
        dm = select_within(dm_all, star_core, dm_radius)
    else:
        dm = dm_all

    base_extra = {
        "Nstar_total": int(len(stars["mass"])),
        "Mstar_total_msun": float(np.sum(stars["mass"])),
        "Ngas_selected": int(len(gas["mass"])),
        "Mgas_selected_msun": float(np.sum(gas["mass"])),
        "Ndm_selected": int(len(dm["mass"])),
        "Mdm_selected_msun": float(np.sum(dm["mass"])),
        "rhalf_star_initial_kpc": float(rhalf),
        "star_core_x_kpc": float(star_core[0]),
        "star_core_y_kpc": float(star_core[1]),
        "star_core_z_kpc": float(star_core[2]),
        "star_core_vx_kms": float(star_core_vel[0]),
        "star_core_vy_kms": float(star_core_vel[1]),
        "star_core_vz_kms": float(star_core_vel[2]),
        "gas_selection_radius_kpc": float(gas_radius),
        "dm_selection_radius_kpc": float(dm_radius),
    }

    rows = []
    for method in center_methods:
        center, vel, meta = center_method(method, stars, dm, gas, star_core, star_core_vel, rhalf, args.inner_radius_factor_rhalf, args.inner_min_radius_kpc, args.inner_max_radius_kpc)
        extra = dict(base_extra)
        extra.update(meta)
        extra["offset_center_from_star_core_kpc"] = float(np.linalg.norm(center - star_core)) if np.all(np.isfinite(center)) else np.nan
        rows.append(orbit_row(label, snapshot, time_gyr, method, center, vel, host, extra))
    return rows


def parse_methods(s: str) -> List[str]:
    if s.lower() == "all":
        return ["star_core", "stars_com", "inner_stars_dm", "inner_stars_dm_gas", "all_tracked"]
    return [x.strip() for x in s.split(",") if x.strip()]


def save_summary_plot(df: pd.DataFrame, outdir: Path) -> None:
    if len(df) == 0:
        return
    labels = df["label"].values
    x = np.arange(len(labels))
    panels = [
        ("R0_kpc", r"$R_0$ [kpc]"),
        ("Vrad0_kms", r"$v_{rad,0}$ [km/s]"),
        ("Vtan0_kms", r"$v_{tan,0}$ [km/s]"),
        ("L0_kpc_kms", r"$|L_0|$ [kpc km/s]"),
        ("E0_km2_s2", r"$E_0$ [(km/s)$^2$]"),
        ("rperi_pred_kpc", r"predicted $r_{peri}$ [kpc]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    for ax, (col, ylabel) in zip(axes.ravel(), panels):
        ax.plot(x, df[col].values, marker="o", lw=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    
    fig.savefig(outdir / "initial_orbit_diagnostics_summary.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    for _, row in df.iterrows():
        ax.scatter(row["x0_kpc"], row["y0_kpc"], s=50, label=row["label"])
        ax.quiver(row["x0_kpc"], row["y0_kpc"], row["vx0_kms"], row["vy0_kms"], angles="xy", scale_units="xy", scale=5.0, width=0.004)
    ax.scatter(0, 0, marker="+", s=150, color="black", label="host")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x_0$ [kpc]")
    ax.set_ylabel(r"$y_0$ [kpc]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "initial_orbit_vectors_xy.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.show()

    plt.close(fig)

#%%

# ============================================================
# SPYDER USER SETTINGS
# ============================================================

from types import SimpleNamespace

# Main paths
root = Path("./../SIMULATIONS/ORBIT/HigherRes")
output_dir = Path("orbit_initial_diagnostics")

# Labels to analyse.
# Use None to automatically discover all folders inside root.
labels = [
    "E_mid_L_radial",
    "E_mid_L_mid",
    "E_mid_L_high",
]

# If you want automatic discovery instead, use:
# labels = None

snapshot_glob = "snapshot_*.hdf5"

# Units
units = Units(
    length_unit_to_kpc=1.0,
    velocity_unit_to_kms=1.0,
    mass_unit_to_msun=1.0e10,
    time_unit_to_gyr=0.977792221,
)

# Host halo model
host = Host(
    center_kpc=(0.0, 0.0, 0.0),
    velocity_kms=(0.0, 0.0, 0.0),
    m200_msun=1.0e12,
    r200_kpc=210.0,
    concentration=10.0,
    truncate_mass_at_r200=False,
)

# Centre methods
center_methods = parse_methods("all")
preferred_center_mode = "inner_stars_dm"

if preferred_center_mode not in center_methods:
    center_methods.append(preferred_center_mode)

# Parameters formerly handled by argparse
args = SimpleNamespace(
    gas_ptype=0,
    dm_ptype=1,
    star_ptype=4,

    shrink_initial_radius_kpc=None,
    shrink_factor=0.75,
    shrink_min_particles=100,
    shrink_min_radius_kpc=0.2,

    velocity_center_radius_factor_rhalf=1.0,
    velocity_center_min_radius_kpc=1.0,

    gas_selection_radius_kpc=None,
    default_gas_radius_kpc=30.0,
    gas_radius_factor_rhalf=8.0,

    dm_selection_mode="all",  # options: "all" or "radius"
    dm_selection_radius_kpc=None,
    default_dm_radius_kpc=80.0,
    dm_radius_factor_rhalf=20.0,

    inner_radius_factor_rhalf=2.0,
    inner_min_radius_kpc=1.0,
    inner_max_radius_kpc=5.0,
)

# ============================================================
# RUN ANALYSIS
# ============================================================

output_dir.mkdir(parents=True, exist_ok=True)

if labels is None:
    labels = discover_labels(root)

rows = []

for label in labels:
    snap = first_snapshot(root, label, snapshot_glob)
    print(f"[{label}] {snap}")

    rows.extend(
        analyze_label(
            label=label,
            snapshot=snap,
            units=units,
            host=host,
            center_methods=center_methods,
            args=args,
        )
    )

df = pd.DataFrame(rows)

all_path = output_dir / "initial_orbit_diagnostics_all_centers.csv"
df.to_csv(all_path, index=False)

pref = (
    df[df["center_method"] == preferred_center_mode]
    .copy()
    .sort_values("L0_kpc_kms")
)

pref_path = output_dir / "initial_orbit_diagnostics_preferred.csv"
pref.to_csv(pref_path, index=False)

cols = [
    "label",
    "x0_kpc",
    "y0_kpc",
    "z0_kpc",
    "vx0_kms",
    "vy0_kms",
    "vz0_kms",
    "R0_kpc",
    "Vrad0_kms",
    "Vtan0_kms",
    "L0_kpc_kms",
    "E0_km2_s2",
    "offset_center_from_star_core_kpc",
]

print("\nPreferred center mode:", preferred_center_mode)
print(pref[[c for c in cols if c in pref.columns]].to_string(index=False))

save_summary_plot(pref, output_dir)

print("\nSaved:")
print(" ", all_path)
print(" ", pref_path)
print(" ", output_dir / "initial_orbit_diagnostics_summary.png")
print(" ", output_dir / "initial_orbit_vectors_xy.png")

#%%
def inspect_hdf5_file(path):
    path = Path(path)

    print("=" * 80)
    print(path)
    print("=" * 80)

    with h5py.File(path, "r") as f:
        print("Top-level keys:")
        for key in f.keys():
            print("  ", key)

        if "Header" in f:
            print("\nHeader attributes:")
            for k, v in f["Header"].attrs.items():
                print(f"  {k}: {v}")

        for ptype in [0, 1, 2, 3, 4, 5]:
            gname = f"PartType{ptype}"

            if gname not in f:
                continue

            print(f"\n{gname}:")
            g = f[gname]

            for dset in g.keys():
                obj = g[dset]
                if hasattr(obj, "shape"):
                    print(f"  {dset}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"  {dset}: {type(obj)}")

#%%


snapshots = {
    "E_mid_L_radial": Path("./../utils/test_ics/G001_DG001_E_mid_L_radial_HigherRes.hdf5"),
    "E_mid_L_mid": Path("./../utils/test_ics/G001_DG001_E_mid_L_mid_HigherRes.hdf5"),
    "E_mid_L_high": Path("./../utils/test_ics/G001_DG001_E_mid_L_high_HigherRes.hdf5"),
}

rows = []

for label, snapshot in snapshots.items():
    print(f"[{label}] {snapshot}")

    inspect_hdf5_file(snapshot)

    rows.extend(
        analyze_label(
            label=label,
            snapshot=snapshot,
            units=units,
            host=host,
            center_methods=center_methods,
            args=args,
        )
    )

df = pd.DataFrame(rows)


#%%

cols = [
    "label",
    "x0_kpc",
    "y0_kpc",
    "z0_kpc",
    "vx0_kms",
    "vy0_kms",
    "vz0_kms",
    "Vrad0_kms",
    "Vtan0_kms",
    "L0_kpc_kms",
    "E0_km2_s2"
]

for pref_method in [ 'star_core', 'stars_com', 'inner_stars_dm', 'inner_stars_dm_gas', 'all_tracked']:
    pref = (
        df[df["center_method"] == pref_method]
        .copy()
        .sort_values("L0_kpc_kms")
    )

    pref_path = output_dir / "initial_orbit_diagnostics_preferred.csv"
    pref.to_csv(pref_path, index=False)

    print("\nPreferred center mode:", pref_method)
    print(
        pref[[c for c in cols if c in pref.columns]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    save_summary_plot(pref, output_dir)

    print("\nSaved:")
    print(" ", all_path)
    print(" ", pref_path)
    print(" ", output_dir / "initial_orbit_diagnostics_summary.png")
    print(" ", output_dir / "initial_orbit_vectors_xy.png")
