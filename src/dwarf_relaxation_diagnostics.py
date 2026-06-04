#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dwarf_relaxation_diagnostics.py

Check whether an isolated Gadget/Gadget4 dwarf-galaxy IC relaxed stably.

It reads snapshot_*.hdf5 files and computes:
  - centre drift and bulk-velocity drift;
  - mass conservation by component;
  - half-mass and r90 radii;
  - masses inside fixed apertures;
  - velocity dispersions and angular momentum;
  - approximate energy and virial diagnostics;
  - gas internal energy, density, SFR and star-forming gas mass when present;
  - initial-vs-final radial density profiles;
  - fixed-scale maps of stars/gas/DM and optional GIFs.

Example for one relaxation run:
python dwarf_relaxation_diagnostics.py \
  --snapshot-dir ./DG001_RELAX/output \
  --output-dir DG001_relax_diagnostics \
  --mass-unit-to-msun 1e10 \
  --length-unit-to-kpc 1.0 \
  --velocity-unit-to-kms 1.0 \
  --time-unit-to-gyr 0.977792221 \
  --center-mode inner_stars_dm \
  --map-width-kpc 30 \
  --apertures-kpc 0.5 1 2 3 5 \
  --make-gif

Example for several labels:
python dwarf_relaxation_diagnostics.py \
  --root ./SIMULATIONS/RELAX \
  --labels DG001 DG001_ZOOM \
  --output-dir relaxation_diagnostics \
  --make-gif
"""
from __future__ import annotations

import argparse, json, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

G_KPC_KMS2_MSUN = 4.30091e-6
KM_S_TO_KPC_GYR = 1.0227121650537077


@dataclass
class Units:
    length_unit_to_kpc: float = 1.0
    velocity_unit_to_kms: float = 1.0
    mass_unit_to_msun: float = 1.0e10
    time_unit_to_gyr: float = 0.977792221


@dataclass
class Config:
    gas_ptype: int = 0
    dm_ptype: int = 1
    star_ptype: int = 4
    center_mode: str = "inner_stars_dm"  # inner_stars_dm, star_core, stars_com, all_tracked
    inner_center_radius_kpc: float = 2.0
    shrink_factor: float = 0.75
    shrink_min_particles: int = 100
    shrink_min_radius_kpc: float = 0.05
    apertures_kpc: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    softening_kpc: float = 0.02
    gamma_gas: float = 5.0 / 3.0
    sfr_threshold_msun_per_yr: float = 0.0
    map_width_kpc: float = 30.0
    map_npix: int = 450
    map_axis: str = "z"
    map_every: int = 1
    make_gif: bool = False
    gif_fps: int = 5
    profile_bins: int = 40
    profile_rmin_kpc: float = 0.03
    profile_rmax_kpc: Optional[float] = None


def natural_snapshot_number(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def find_snapshots(path: Path, glob: str) -> List[Path]:
    snaps = sorted(path.glob(glob), key=natural_snapshot_number)
    if not snaps:
        raise FileNotFoundError(f"No snapshots matching {glob} in {path}")
    return snaps


def discover_labels(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "output").exists()])


def header_attr(f: h5py.File, name: str, default=None):
    return f["Header"].attrs.get(name, default) if "Header" in f else default


def first_dataset(g: h5py.Group, names: Sequence[str]):
    for n in names:
        if n in g:
            return g[n][:]
    return None


def read_time(snapshot: Path, units: Units) -> float:
    with h5py.File(snapshot, "r") as f:
        t = header_attr(f, "Time", np.nan)
    try:
        return float(t) * units.time_unit_to_gyr
    except Exception:
        return np.nan


def read_ptype(snapshot: Path, ptype: int, units: Units) -> Optional[Dict[str, np.ndarray]]:
    gname = f"PartType{ptype}"
    with h5py.File(snapshot, "r") as f:
        if gname not in f:
            return None
        g = f[gname]
        pos = first_dataset(g, ["Coordinates", "Position", "Positions"])
        vel = first_dataset(g, ["Velocities", "Velocity"])
        ids = first_dataset(g, ["ParticleIDs", "ParticleID", "IDs", "ID"])
        if pos is None or vel is None:
            raise KeyError(f"{snapshot}:{gname} missing Coordinates or Velocities")
        if ids is None:
            ids = np.arange(1, len(pos) + 1, dtype=np.int64)
        mass = first_dataset(g, ["Masses", "Mass", "ParticleMasses"])
        if mass is None:
            mt = header_attr(f, "MassTable", None)
            if mt is None or float(mt[ptype]) <= 0:
                raise KeyError(f"{snapshot}:{gname} has no Masses and no valid MassTable")
            mass = np.full(len(ids), float(mt[ptype]), dtype=float)
        out = {
            "pos": np.asarray(pos, dtype=float) * units.length_unit_to_kpc,
            "vel": np.asarray(vel, dtype=float) * units.velocity_unit_to_kms,
            "ids": np.asarray(ids, dtype=np.int64),
            "mass": np.asarray(mass, dtype=float) * units.mass_unit_to_msun,
        }
        optional = {
            "u": ["InternalEnergy", "U"],
            "density": ["Density", "Densities", "rho", "Rho"],
            "sfr": ["StarFormationRate", "StarFormationRates", "SFR", "sfr"],
            "hsml": ["SmoothingLength", "Hsml", "SmoothingLengths"],
            "birth_time": ["GFM_StellarFormationTime", "StellarFormationTime", "FormationTime", "BirthTime", "BirthTimes"],
        }
        for key, names in optional.items():
            arr = first_dataset(g, names)
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                if key == "density":
                    arr = arr * units.mass_unit_to_msun / units.length_unit_to_kpc**3
                elif key == "u":
                    arr = arr * units.velocity_unit_to_kms**2
                out[key] = arr
        return out


def empty_comp():
    return {"pos": np.empty((0, 3)), "vel": np.empty((0, 3)), "ids": np.empty(0, dtype=np.int64), "mass": np.empty(0)}


def select_ids(comp: Optional[Dict[str, np.ndarray]], ids: Optional[np.ndarray]):
    if comp is None:
        return empty_comp()
    if ids is None:
        return comp
    m = np.isin(comp["ids"], ids, assume_unique=False)
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in comp.items()}


def dist(pos, center):
    return np.linalg.norm(pos - np.asarray(center)[None, :], axis=1)


def mwmean(x, m=None):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.full(x.shape[1] if x.ndim > 1 else 1, np.nan)
    if m is None:
        return np.nanmean(x, axis=0)
    m = np.asarray(m, dtype=float)
    good = np.isfinite(m) & (m > 0)
    good = good & (np.all(np.isfinite(x), axis=1) if x.ndim == 2 else np.isfinite(x))
    if np.count_nonzero(good) == 0:
        return np.nanmean(x, axis=0)
    return np.average(x[good], axis=0, weights=m[good])


def shrinking_center(pos, mass, cfg: Config, previous=None):
    if len(pos) == 0:
        return np.full(3, np.nan)
    c = mwmean(pos, mass) if previous is None else np.asarray(previous, dtype=float)
    r = dist(pos, c)
    radius = np.nanpercentile(r, 90)
    if not np.isfinite(radius) or radius <= 0:
        radius = np.nanmax(r)
    radius = max(float(radius), cfg.shrink_min_radius_kpc)
    idx = np.arange(len(pos))
    while True:
        rr = dist(pos[idx], c)
        inside = idx[rr <= radius]
        if len(inside) < max(10, cfg.shrink_min_particles) or radius <= cfg.shrink_min_radius_kpc:
            if len(inside) >= 5:
                idx = inside
            break
        c = mwmean(pos[inside], mass[inside])
        idx = inside
        radius *= cfg.shrink_factor
    return mwmean(pos[idx], mass[idx]) if len(idx) else c


def enclosed_radius(pos, mass, center, frac):
    if len(pos) == 0 or np.sum(mass) <= 0:
        return np.nan
    r = dist(pos, center)
    o = np.argsort(r)
    rs, ms = r[o], mass[o]
    cs = np.cumsum(ms)
    j = np.searchsorted(cs, frac * cs[-1])
    return float(rs[min(j, len(rs) - 1)])


def concat(comps):
    pos, vel, mass, ids = [], [], [], []
    for c in comps:
        if c is None or len(c.get("mass", [])) == 0:
            continue
        pos.append(c["pos"]); vel.append(c["vel"]); mass.append(c["mass"]); ids.append(c.get("ids", np.arange(len(c["mass"]))))
    if not pos:
        return empty_comp()
    return {"pos": np.vstack(pos), "vel": np.vstack(vel), "mass": np.concatenate(mass), "ids": np.concatenate(ids)}


def choose_center(stars, dm, gas, cfg: Config, previous=None):
    star_core = shrinking_center(stars["pos"], stars["mass"], cfg, previous)
    rhalf_star = enclosed_radius(stars["pos"], stars["mass"], star_core, 0.5)
    if cfg.center_mode == "star_core":
        r = dist(stars["pos"], star_core)
        rad = max(1.0, rhalf_star if np.isfinite(rhalf_star) else 1.0)
        inner = r <= rad
        if np.count_nonzero(inner) < 5:
            inner = np.argsort(r)[:max(5, min(len(r), len(r)//10))]
        v = mwmean(stars["vel"][inner], stars["mass"][inner])
        return star_core, v, {"rhalf_star_center_kpc": rhalf_star, "center_radius_kpc": rad, "N_center_particles": int(np.count_nonzero(inner))}
    if cfg.center_mode == "stars_com":
        return mwmean(stars["pos"], stars["mass"]), mwmean(stars["vel"], stars["mass"]), {"rhalf_star_center_kpc": rhalf_star, "center_radius_kpc": np.nan, "N_center_particles": len(stars["mass"])}
    if cfg.center_mode == "all_tracked":
        allc = concat([stars, dm, gas])
        return mwmean(allc["pos"], allc["mass"]), mwmean(allc["vel"], allc["mass"]), {"rhalf_star_center_kpc": rhalf_star, "center_radius_kpc": np.nan, "N_center_particles": len(allc["mass"])}
    # inner_stars_dm: recommended for relaxation orbital/frame diagnostics
    allc = concat([stars, dm])
    r = dist(allc["pos"], star_core)
    inside = r <= cfg.inner_center_radius_kpc
    if np.count_nonzero(inside) < 20:
        allc = stars
        r = dist(allc["pos"], star_core)
        inside = r <= cfg.inner_center_radius_kpc
    if np.count_nonzero(inside) < 5:
        inside = np.ones(len(allc["mass"]), dtype=bool)
    c = mwmean(allc["pos"][inside], allc["mass"][inside])
    v = mwmean(allc["vel"][inside], allc["mass"][inside])
    return c, v, {"rhalf_star_center_kpc": rhalf_star, "center_radius_kpc": cfg.inner_center_radius_kpc, "N_center_particles": int(np.count_nonzero(inside))}


def component_diag(comp, name, center, vcenter, apertures, sfr_threshold):
    out = {}
    if len(comp["mass"]) == 0:
        out[f"N{name}"] = 0; out[f"M{name}_msun"] = 0.0
        return out
    p, v, m = comp["pos"], comp["vel"], comp["mass"]
    r = dist(p, center)
    dv = v - vcenter[None, :]
    speed2 = np.sum(dv * dv, axis=1)
    mtot = float(np.sum(m))
    out[f"N{name}"] = int(len(m))
    out[f"M{name}_msun"] = mtot
    out[f"rhalf_{name}_kpc"] = enclosed_radius(p, m, center, 0.5)
    out[f"r90_{name}_kpc"] = enclosed_radius(p, m, center, 0.9)
    out[f"sigma3d_{name}_kms"] = float(np.sqrt(np.average(speed2, weights=m))) if mtot > 0 else np.nan
    out[f"sigma1d_{name}_kms"] = out[f"sigma3d_{name}_kms"] / np.sqrt(3) if np.isfinite(out[f"sigma3d_{name}_kms"]) else np.nan
    out[f"K_{name}_Msun_kms2"] = float(0.5 * np.sum(m * speed2))
    L = np.sum(np.cross(p - center[None, :], dv) * m[:, None], axis=0)
    out[f"L_{name}_Msun_kpc_kms"] = float(np.linalg.norm(L))
    out[f"Lx_{name}_Msun_kpc_kms"] = float(L[0]); out[f"Ly_{name}_Msun_kpc_kms"] = float(L[1]); out[f"Lz_{name}_Msun_kpc_kms"] = float(L[2])
    for ap in apertures:
        key = f"{ap:g}".replace(".", "p")
        inside = r <= ap
        out[f"M{name}_inside_{key}kpc_msun"] = float(np.sum(m[inside]))
        out[f"N{name}_inside_{key}kpc"] = int(np.count_nonzero(inside))
    if name == "gas":
        if "u" in comp:
            u = comp["u"]; good = np.isfinite(u) & (u > 0)
            out["U_gas_Msun_kms2"] = float(np.sum(m[good] * u[good])) if np.any(good) else np.nan
            out["median_u_gas_kms2"] = float(np.nanmedian(u[good])) if np.any(good) else np.nan
        else:
            out["U_gas_Msun_kms2"] = np.nan; out["median_u_gas_kms2"] = np.nan
        if "density" in comp:
            rho = comp["density"]; good = np.isfinite(rho) & (rho > 0)
            out["median_rho_gas_msun_kpc3"] = float(np.nanmedian(rho[good])) if np.any(good) else np.nan
            out["p90_rho_gas_msun_kpc3"] = float(np.nanpercentile(rho[good], 90)) if np.any(good) else np.nan
        if "sfr" in comp:
            sfr = comp["sfr"]; sf = np.isfinite(sfr) & (sfr > sfr_threshold)
            out["SFR_total_msun_per_yr"] = float(np.nansum(sfr))
            out["Mgas_SF_msun"] = float(np.sum(m[sf]))
            out["Ngas_SF"] = int(np.count_nonzero(sf))
        else:
            out["SFR_total_msun_per_yr"] = np.nan; out["Mgas_SF_msun"] = np.nan; out["Ngas_SF"] = 0
    return out


def spherical_W(comps, center, eps):
    c = concat(comps)
    if len(c["mass"]) < 2:
        return np.nan
    r = dist(c["pos"], center); m = c["mass"]
    o = np.argsort(r); rs, ms = r[o], m[o]
    Minside = np.cumsum(ms) - 0.5 * ms
    return float(-G_KPC_KMS2_MSUN * np.sum(ms * Minside / np.sqrt(rs * rs + eps * eps)))


def radial_profile(comp, center, bins):
    if len(comp["mass"]) == 0:
        return pd.DataFrame()
    r, m = dist(comp["pos"], center), comp["mass"]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (r >= lo) & (r < hi)
        vol = 4/3 * np.pi * (hi**3 - lo**3)
        rows.append({"r_mid_kpc": np.sqrt(lo*hi), "r_lo_kpc": lo, "r_hi_kpc": hi, "mass_msun": float(np.sum(m[sel])), "density_msun_kpc3": float(np.sum(m[sel])/vol), "N": int(np.count_nonzero(sel))})
    return pd.DataFrame(rows)


def projection_axes(axis):
    if axis == "x": return 1, 2, r"$y$ [kpc]", r"$z$ [kpc]"
    if axis == "y": return 0, 2, r"$x$ [kpc]", r"$z$ [kpc]"
    return 0, 1, r"$x$ [kpc]", r"$y$ [kpc]"


def hist2d(comp, center, cfg):
    i, j, _, _ = projection_axes(cfg.map_axis)
    rel = comp["pos"] - center[None, :]
    h = 0.5 * cfg.map_width_kpc
    hist, xe, ye = np.histogram2d(rel[:, i], rel[:, j], bins=cfg.map_npix, range=[[-h, h], [-h, h]], weights=comp["mass"])
    return hist.T


def estimate_map_limits(snaps, units, cfg, ids):
    vals = {"stars": [], "gas": [], "dm": []}
    prev = None
    step = max(1, len(snaps)//15)
    for snap in snaps[::step]:
        s = select_ids(read_ptype(snap, cfg.star_ptype, units), ids["stars"])
        d = select_ids(read_ptype(snap, cfg.dm_ptype, units), ids["dm"])
        g = select_ids(read_ptype(snap, cfg.gas_ptype, units), ids["gas"])
        c, v, _ = choose_center(s, d, g, cfg, prev); prev = c
        for name, comp in {"stars": s, "gas": g, "dm": d}.items():
            if len(comp["mass"]) == 0: continue
            h = hist2d(comp, c, cfg); x = h[np.isfinite(h) & (h > 0)]
            if len(x): vals[name].append(x)
    lim = {}
    for name, arrs in vals.items():
        if not arrs:
            lim[name] = (1.0, 10.0); continue
        x = np.concatenate(arrs)
        vmin, vmax = np.nanpercentile(x, [1, 99.7])
        if not np.isfinite(vmin) or vmin <= 0: vmin = np.nanmin(x[x > 0])
        if not np.isfinite(vmax) or vmax <= vmin: vmax = np.nanmax(x)
        lim[name] = (float(vmin), float(vmax))
    return lim


def make_map(path, row, comps, center, cfg, limits):
    names = [("stars", "stellar mass"), ("gas", "gas mass"), ("dm", "DM mass")]
    _, _, xlabel, ylabel = projection_axes(cfg.map_axis)
    hwidth = 0.5 * cfg.map_width_kpc
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True, facecolor="white")
    for ax, (name, title) in zip(axes, names):
        comp = comps[name]
        if len(comp["mass"]) == 0:
            ax.set_title(f"{title}: absent"); ax.axis("off"); continue
        img = np.ma.masked_where((hist2d(comp, center, cfg) <= 0), hist2d(comp, center, cfg))
        vmin, vmax = limits.get(name, (1.0, 10.0))
        im = ax.imshow(img, origin="lower", extent=[-hwidth, hwidth, -hwidth, hwidth], norm=LogNorm(vmin=max(vmin, 1e-300), vmax=vmax), cmap="viridis", aspect="equal")
        ax.scatter(0, 0, marker="+", s=90, color="white", linewidth=1.5)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"t={row.get('time_gyr', np.nan):.3f} Gyr | rhalf,*={row.get('rhalf_stars_kpc', np.nan):.3f} kpc | Qeff={row.get('Q_virial_eff', np.nan):.3f}", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return path


def analyze_one(snap, units, cfg, ids, prev_center):
    stars = select_ids(read_ptype(snap, cfg.star_ptype, units), ids["stars"])
    dm = select_ids(read_ptype(snap, cfg.dm_ptype, units), ids["dm"])
    gas = select_ids(read_ptype(snap, cfg.gas_ptype, units), ids["gas"])
    if len(stars["mass"]) == 0: raise RuntimeError(f"No tracked stars in {snap}")
    c, vc, meta = choose_center(stars, dm, gas, cfg, prev_center)
    row = {"snapshot_file": str(snap), "snapshot_number": natural_snapshot_number(snap), "time_gyr": read_time(snap, units),
           "x_center_kpc": float(c[0]), "y_center_kpc": float(c[1]), "z_center_kpc": float(c[2]),
           "vx_center_kms": float(vc[0]), "vy_center_kms": float(vc[1]), "vz_center_kms": float(vc[2])}
    row.update(meta)
    for name, comp in [("stars", stars), ("gas", gas), ("dm", dm)]:
        row.update(component_diag(comp, name, c, vc, cfg.apertures_kpc, cfg.sfr_threshold_msun_per_yr))
    K = sum(row.get(k, 0.0) for k in ["K_stars_Msun_kms2", "K_gas_Msun_kms2", "K_dm_Msun_kms2"] if np.isfinite(row.get(k, np.nan)))
    U = row.get("U_gas_Msun_kms2", np.nan)
    W = spherical_W([stars, dm, gas], c, cfg.softening_kpc)
    row["K_total_Msun_kms2"] = float(K); row["W_spherical_Msun_kms2"] = W
    row["E_total_approx_Msun_kms2"] = float(K + (U if np.isfinite(U) else 0.0) + W) if np.isfinite(W) else np.nan
    row["Q_virial_collisionless"] = float(2*K/abs(W)) if np.isfinite(W) and W != 0 else np.nan
    row["Q_virial_eff"] = float((2*K + 2*U)/abs(W)) if np.isfinite(W) and W != 0 and np.isfinite(U) else np.nan
    row["Mtotal_msun"] = row.get("Mstars_msun",0)+row.get("Mgas_msun",0)+row.get("Mdm_msun",0)
    rh = row.get("rhalf_stars_kpc", np.nan)
    if np.isfinite(rh) and rh > 0:
        Menc = 0.0
        for comp in [stars, gas, dm]:
            if len(comp["mass"]): Menc += np.sum(comp["mass"][dist(comp["pos"], c) <= rh])
        row["Mtotal_inside_rhalf_star_msun"] = float(Menc)
        row["t_dyn_rhalf_gyr"] = float(np.sqrt(rh**3/(G_KPC_KMS2_MSUN*max(Menc,1e-300))) / KM_S_TO_KPC_GYR)
    return row, {"stars": stars, "gas": gas, "dm": dm}, c


def add_relative(df):
    df = df.copy()
    x0 = df[["x_center_kpc","y_center_kpc","z_center_kpc"]].iloc[0].values.astype(float)
    v0 = df[["vx_center_kms","vy_center_kms","vz_center_kms"]].iloc[0].values.astype(float)
    x = df[["x_center_kpc","y_center_kpc","z_center_kpc"]].values.astype(float)
    v = df[["vx_center_kms","vy_center_kms","vz_center_kms"]].values.astype(float)
    df["center_drift_kpc"] = np.linalg.norm(x-x0[None,:], axis=1)
    df["bulk_velocity_drift_kms"] = np.linalg.norm(v-v0[None,:], axis=1)
    cols = ["Mtotal_msun","Mstars_msun","Mgas_msun","Mdm_msun","rhalf_stars_kpc","rhalf_gas_kpc","rhalf_dm_kpc","sigma1d_stars_kms","sigma1d_dm_kms","Q_virial_eff","E_total_approx_Msun_kms2"]
    for col in cols:
        if col in df:
            y0 = df[col].iloc[0]
            df[f"{col}_frac_change_from_initial"] = (df[col]-y0)/abs(y0) if np.isfinite(y0) and y0 != 0 else np.nan
    return df


def quality_summary(df, path):
    def maxabs(col):
        return float(np.nanmax(np.abs(df[col]))) if col in df and np.any(np.isfinite(df[col])) else np.nan
    s = {
        "n_snapshots": int(len(df)),
        "time_start_gyr": float(np.nanmin(df["time_gyr"])),
        "time_end_gyr": float(np.nanmax(df["time_gyr"])),
        "max_center_drift_kpc": maxabs("center_drift_kpc"),
        "max_bulk_velocity_drift_kms": maxabs("bulk_velocity_drift_kms"),
        "max_frac_change_Mtotal": maxabs("Mtotal_msun_frac_change_from_initial"),
        "max_frac_change_Mstars": maxabs("Mstars_msun_frac_change_from_initial"),
        "max_frac_change_Mgas": maxabs("Mgas_msun_frac_change_from_initial"),
        "max_frac_change_Mdm": maxabs("Mdm_msun_frac_change_from_initial"),
        "max_frac_change_rhalf_stars": maxabs("rhalf_stars_kpc_frac_change_from_initial"),
        "max_frac_change_sigma1d_stars": maxabs("sigma1d_stars_kms_frac_change_from_initial"),
        "max_frac_change_Etotal_approx": maxabs("E_total_approx_Msun_kms2_frac_change_from_initial"),
        "median_Q_virial_eff": float(np.nanmedian(df["Q_virial_eff"])) if "Q_virial_eff" in df else np.nan,
        "median_Q_virial_collisionless": float(np.nanmedian(df["Q_virial_collisionless"])) if "Q_virial_collisionless" in df else np.nan,
    }
    warn = []
    if np.isfinite(s["max_frac_change_Mstars"]) and s["max_frac_change_Mstars"] > 0.01: warn.append("stellar mass changes by >1%")
    if np.isfinite(s["max_frac_change_rhalf_stars"]) and s["max_frac_change_rhalf_stars"] > 0.10: warn.append("stellar half-mass radius changes by >10%")
    r0 = df["rhalf_stars_kpc"].iloc[0] if "rhalf_stars_kpc" in df else np.nan
    if np.isfinite(r0) and r0 > 0 and np.isfinite(s["max_center_drift_kpc"]) and s["max_center_drift_kpc"] > 0.2*r0: warn.append("centre drifts by >0.2 initial stellar rhalf")
    if np.isfinite(s["median_Q_virial_eff"]) and not (0.7 <= s["median_Q_virial_eff"] <= 1.3): warn.append("effective virial ratio is far from unity; check gas/softening caveats")
    s["warnings"] = warn
    path.write_text(json.dumps(s, indent=2))
    return s


def safe_log(y):
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, np.log10(y), np.nan)


def plot_times(df, outdir):
    t = df["time_gyr"].values if "time_gyr" in df else np.arange(len(df))
    panels = [("center_drift_kpc","Centre drift [kpc]",False),("bulk_velocity_drift_kms","Bulk velocity drift [km/s]",False),("rhalf_stars_kpc",r"$r_{1/2,*}$ [kpc]",False),("rhalf_gas_kpc",r"$r_{1/2,gas}$ [kpc]",False),("rhalf_dm_kpc",r"$r_{1/2,DM}$ [kpc]",False),("Mstars_msun",r"$\log M_*$",True),("Mgas_msun",r"$\log M_{gas}$",True),("Mdm_msun",r"$\log M_{DM}$",True),("sigma1d_stars_kms",r"$\sigma_{*,1D}$ [km/s]",False),("sigma1d_dm_kms",r"$\sigma_{DM,1D}$ [km/s]",False),("Q_virial_eff",r"$Q_{eff}$",False),("E_total_approx_Msun_kms2_frac_change_from_initial",r"$\Delta E/E_0$",False)]
    fig, axes = plt.subplots(4,3,figsize=(15,12),sharex=True,facecolor="white"); axes=axes.ravel()
    for ax,(col,lab,lg) in zip(axes,panels):
        if col not in df: ax.axis("off"); continue
        y = safe_log(df[col]) if lg else df[col].values
        ax.plot(t,y,marker="o",lw=1.4); ax.set_ylabel(lab); ax.grid(alpha=.3)
        if col == "Q_virial_eff": ax.axhline(1,ls="--",lw=1,color="0.4")
    for ax in axes[-3:]: ax.set_xlabel("Time [Gyr]")
    fig.tight_layout(); fig.savefig(outdir/"relaxation_time_series.png",dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)


def plot_profiles(profs, outdir):
    fig, axes = plt.subplots(1,3,figsize=(14,4.2),sharey=True,facecolor="white")
    for ax,name in zip(axes,["stars","gas","dm"]):
        if name not in profs: ax.axis("off"); continue
        p0,p1=profs[name]
        ax.plot(p0.r_mid_kpc,p0.density_msun_kpc3,marker="o",lw=1.4,label="initial")
        ax.plot(p1.r_mid_kpc,p1.density_msun_kpc3,marker="s",lw=1.4,label="final")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("r [kpc]"); ax.set_title(name); ax.grid(alpha=.3,which="both"); ax.legend(frameon=False)
    axes[0].set_ylabel(r"$\rho$ [$M_\odot\,kpc^{-3}$]")
    fig.tight_layout(); fig.savefig(outdir/"initial_vs_final_density_profiles.png",dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)


def analyze_sequence(snaps, outdir, units, cfg):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"maps").mkdir(exist_ok=True)
    first = snaps[0]
    ids = {"stars": np.unique(read_ptype(first,cfg.star_ptype,units)["ids"]),
           "dm": np.unique(read_ptype(first,cfg.dm_ptype,units)["ids"]) if read_ptype(first,cfg.dm_ptype,units) is not None else np.array([],dtype=np.int64),
           "gas": np.unique(read_ptype(first,cfg.gas_ptype,units)["ids"]) if read_ptype(first,cfg.gas_ptype,units) is not None else np.array([],dtype=np.int64)}
    np.savez_compressed(outdir/"initial_particle_ids.npz", **ids)
    (outdir/"relax_config.json").write_text(json.dumps({"units": asdict(units), "config": asdict(cfg)}, indent=2))
    limits = estimate_map_limits(snaps, units, cfg, ids)
    rows=[]; prev=None; map_paths=[]; profs={}
    for i,snap in enumerate(snaps):
        print(f"[{i+1}/{len(snaps)}] {snap.name}")
        row, comps, center = analyze_one(snap, units, cfg, ids, prev)
        rows.append(row); prev=center
        if i in (0, len(snaps)-1):
            rmax = cfg.profile_rmax_kpc or max(cfg.map_width_kpc/2, 5.0)
            bins = np.logspace(np.log10(cfg.profile_rmin_kpc), np.log10(rmax), cfg.profile_bins+1)
            for cname in ["stars","gas","dm"]:
                pr = radial_profile(comps[cname], center, bins)
                if i==0: profs[cname]=[pr,None]
                elif cname in profs: profs[cname][1]=pr
        if i % max(1,cfg.map_every)==0:
            mp = outdir/"maps"/f"map_{natural_snapshot_number(snap):03d}.png"
            map_paths.append(make_map(mp,row,comps,center,cfg,limits))
    df=add_relative(pd.DataFrame(rows)); df.to_csv(outdir/"relaxation_diagnostics.csv", index=False)
    quality_summary(df,outdir/"relaxation_quality_summary.json")
    plot_times(df,outdir)
    profs_clean={k:(v[0],v[1]) for k,v in profs.items() if v[0] is not None and v[1] is not None}
    if profs_clean:
        for k,(p0,p1) in profs_clean.items(): p0.to_csv(outdir/f"profile_initial_{k}.csv",index=False); p1.to_csv(outdir/f"profile_final_{k}.csv",index=False)
        plot_profiles(profs_clean,outdir)
    if cfg.make_gif and map_paths:
        if imageio is None: warnings.warn("imageio not installed; skipping GIF")
        else: imageio.mimsave(outdir/"relaxation_maps.gif", [imageio.imread(p) for p in map_paths], fps=cfg.gif_fps)
    return df


def compare(labels, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    panels=[("center_drift_kpc","Centre drift [kpc]",False),("rhalf_stars_kpc",r"$r_{1/2,*}$ [kpc]",False),("Mstars_msun",r"$\log M_*$",True),("Mgas_msun",r"$\log M_{gas}$",True),("Mdm_msun",r"$\log M_{DM}$",True),("Q_virial_eff",r"$Q_{eff}$",False),("E_total_approx_Msun_kms2_frac_change_from_initial",r"$\Delta E/E_0$",False),("sigma1d_stars_kms",r"$\sigma_{*,1D}$ [km/s]",False)]
    fig,axes=plt.subplots(4,2,figsize=(12,13),facecolor="white"); axes=axes.ravel()
    for ax,(col,lab,lg) in zip(axes,panels):
        for name,df in labels.items():
            if col not in df: continue
            t=df["time_gyr"].values; y=safe_log(df[col]) if lg else df[col].values
            ax.plot(t,y,marker="o",lw=1.3,label=name)
        ax.set_ylabel(lab); ax.grid(alpha=.3)
        if col=="Q_virial_eff": ax.axhline(1,ls="--",lw=1,color="0.4")
    axes[0].legend(frameon=False,fontsize=8)
    for ax in axes[-2:]: ax.set_xlabel("Time [Gyr]")
    fig.tight_layout(); fig.savefig(outdir/"comparison_relaxation_diagnostics.png",dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)


def parse():
    p=argparse.ArgumentParser(description="Diagnostics for isolated dwarf-galaxy relaxation snapshots")
    g=p.add_mutually_exclusive_group(required=True); g.add_argument("--snapshot-dir"); g.add_argument("--root")
    p.add_argument("--labels", nargs="*", default=None); p.add_argument("--snapshot-glob", default="snapshot_*.hdf5"); p.add_argument("--output-dir", default="relaxation_diagnostics")
    p.add_argument("--length-unit-to-kpc", type=float, default=1.0); p.add_argument("--velocity-unit-to-kms", type=float, default=1.0); p.add_argument("--mass-unit-to-msun", type=float, default=1.0e10); p.add_argument("--time-unit-to-gyr", type=float, default=0.977792221)
    p.add_argument("--gas-ptype", type=int, default=0); p.add_argument("--dm-ptype", type=int, default=1); p.add_argument("--star-ptype", type=int, default=4)
    p.add_argument("--center-mode", choices=["inner_stars_dm","star_core","stars_com","all_tracked"], default="inner_stars_dm"); p.add_argument("--inner-center-radius-kpc", type=float, default=2.0)
    p.add_argument("--apertures-kpc", nargs="+", type=float, default=[0.5,1,2,3,5]); p.add_argument("--softening-kpc", type=float, default=0.02); p.add_argument("--sfr-threshold-msun-per-yr", type=float, default=0.0)
    p.add_argument("--map-width-kpc", type=float, default=30.0); p.add_argument("--map-npix", type=int, default=450); p.add_argument("--map-axis", choices=["x","y","z"], default="z"); p.add_argument("--map-every", type=int, default=1); p.add_argument("--make-gif", action="store_true"); p.add_argument("--gif-fps", type=int, default=5)
    p.add_argument("--profile-rmin-kpc", type=float, default=0.03); p.add_argument("--profile-rmax-kpc", type=float, default=None); p.add_argument("--profile-bins", type=int, default=40)
    return p.parse_args()


def main():
    a=parse(); out=Path(a.output_dir)
    units=Units(a.length_unit_to_kpc,a.velocity_unit_to_kms,a.mass_unit_to_msun,a.time_unit_to_gyr)
    cfg=Config(a.gas_ptype,a.dm_ptype,a.star_ptype,a.center_mode,a.inner_center_radius_kpc,apertures_kpc=tuple(a.apertures_kpc),softening_kpc=a.softening_kpc,sfr_threshold_msun_per_yr=a.sfr_threshold_msun_per_yr,map_width_kpc=a.map_width_kpc,map_npix=a.map_npix,map_axis=a.map_axis,map_every=a.map_every,make_gif=a.make_gif,gif_fps=a.gif_fps,profile_bins=a.profile_bins,profile_rmin_kpc=a.profile_rmin_kpc,profile_rmax_kpc=a.profile_rmax_kpc)
    if a.snapshot_dir:
        analyze_sequence(find_snapshots(Path(a.snapshot_dir),a.snapshot_glob), out, units, cfg); print(f"Saved diagnostics to {out}"); return
    root=Path(a.root); labels=a.labels if a.labels else discover_labels(root); tables={}
    for lab in labels:
        tables[lab]=analyze_sequence(find_snapshots(root/lab/"output",a.snapshot_glob), out/lab, units, cfg)
    compare(tables,out/"comparison"); print(f"Saved diagnostics to {out}")

if __name__ == "__main__": main()
