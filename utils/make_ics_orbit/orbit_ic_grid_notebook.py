# -*- coding: utf-8 -*-
"""
orbit_ic_grid_notebook_v5.py

Notebook-friendly generator of controlled orbital initial conditions for dwarf
satellite simulations in a Hernquist host potential.

Version 5 keeps the v4 recommended workflow and adds manual-IC grid helpers.

Version 4 changes the recommended workflow. Instead of using a rigid grid in
specific orbital energy and angular momentum, which can fail for a fixed initial
position, it uses a robust IC-first grid:

    r0 = (x0, y0, 0)
    v0 = v_r e_r + v_t e_t

with

    f_E = v_total / v_escape(r0)
    f_T = v_t / v_total

The code then computes the corresponding Cartesian velocity (vx0, vy0), specific
energy eps, and specific angular momentum h. This preserves the physical
interpretation of energy and angular momentum, but guarantees real velocities as
long as 0 < f_E < 1 and 0 <= f_T < 1.

Available velocity modes
------------------------
1. velocity_mode="velocity_fraction_grid"  [recommended]
   Keep x0 and y0 fixed. Build ICs from vtotal/vesc and vt/vtotal.

2. velocity_mode="fixed_position_vxy"
   Keep x0 and y0 fixed. For each target eps and eta=h/h_circ(eps), solve
   radial/tangential velocity components at r0. Some combinations have no
   velocity solution.

3. velocity_mode="vx_only_solve_y"  [legacy]
   Preserve the old restriction v0=(vx0,0,0). For each eps and h, solve for y0
   and vx0. Some E-L combinations have no solution in this restricted geometry.

The main notebook workflow is:

    from orbit_ic_grid_notebook_v4 import *

    cfg = OrbitGridConfig(...)
    df = run_orbit_grid(cfg)
    accepted_ics_table(df)
    plot_3x3_radius_vrad_grid(df, cfg, mass_label="intermediate")

Units
-----
Internal distance: kpc
Internal time:     Gyr
Internal mass:     Msun
Internal velocity: kpc/Gyr
Input/output velocity: km/s
Specific energy eps is reported in (km/s)^2.
Specific angular momentum h is reported in kpc km/s.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from scipy.optimize import brentq

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# =============================================================================
# Constants and unit conversions
# =============================================================================

KMS_TO_KPCGYR = 1.022712165045695
KPCGYR_TO_KMS = 1.0 / KMS_TO_KPCGYR
G = 4.30091e-6 * (KMS_TO_KPCGYR ** 2)  # kpc^3 / (Msun Gyr^2)


def kms_to_kpcgyr(v_kms: float) -> float:
    return float(v_kms) * KMS_TO_KPCGYR


def kpcgyr_to_kms(v_kpcgyr: float) -> float:
    return float(v_kpcgyr) * KPCGYR_TO_KMS


def eps_int_to_kms2(eps: float) -> float:
    return float(eps) * (KPCGYR_TO_KMS ** 2)


def eps_kms2_to_int(eps_kms2: float) -> float:
    return float(eps_kms2) * (KMS_TO_KPCGYR ** 2)


def h_int_to_kpckms(h: float) -> float:
    return float(h) * KPCGYR_TO_KMS


def h_kpckms_to_int(h_kpckms: float) -> float:
    return float(h_kpckms) * KMS_TO_KPCGYR


# =============================================================================
# Hernquist host potential, accelerations, and optional dynamical friction
# =============================================================================


def phi_hernquist(r: float, M: float, a: float) -> float:
    """Hernquist potential in kpc^2/Gyr^2."""
    return -G * M / (r + a)


def rho_hernquist(r: float, M: float, a: float) -> float:
    if r <= 0:
        return np.inf
    return M * a / (2.0 * math.pi * r * (r + a) ** 3)


def v_circ_hernquist(r: float, M: float, a: float) -> float:
    """Circular speed in kpc/Gyr."""
    if r <= 0:
        return 0.0
    return math.sqrt(G * M * r / ((r + a) ** 2))


def eps_circ_hernquist(r: float, M: float, a: float) -> float:
    vc = v_circ_hernquist(r, M, a)
    return 0.5 * vc * vc + phi_hernquist(r, M, a)


def h_circ_hernquist(r: float, M: float, a: float) -> float:
    return r * v_circ_hernquist(r, M, a)


def rcirc_from_eps_hernquist(
    eps: float,
    M: float,
    a: float,
    r_min: float = 1.0e-4,
    r_max: float = 1.0e6,
    maxiter: int = 200,
) -> float:
    """Return the circular-orbit radius with the same specific energy.

    For the Hernquist potential, eps_circ(r) is monotonic increasing from a
    very negative value at small radius to 0 from below at large radius. Bound
    orbits have eps < 0. If eps is outside this circular-orbit range, return NaN.
    """
    if not np.isfinite(eps) or eps >= 0:
        return np.nan

    lo = float(r_min)
    hi = float(r_max)
    flo = eps_circ_hernquist(lo, M, a) - eps
    fhi = eps_circ_hernquist(hi, M, a) - eps

    # eps_circ(lo) is usually more negative than eps, so flo < 0.
    # eps_circ(hi) is close to 0, so fhi > 0 for bound eps.
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = eps_circ_hernquist(mid, M, a) - eps
        if not np.isfinite(fm):
            return np.nan
        if abs(fm) < 1.0e-10:
            return mid
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def sigma_local(r: float, M: float, a: float) -> float:
    return v_circ_hernquist(r, M, a) / math.sqrt(2.0)


def accel_gravity(x: float, y: float, M: float, a: float) -> Tuple[float, float]:
    r2 = x * x + y * y
    r = math.sqrt(r2)
    if r == 0:
        return 0.0, 0.0
    fac = -G * M / (r * (r + a) ** 2)
    return fac * x, fac * y


def chandra_F(X: float) -> float:
    if X <= 0:
        return 0.0
    return math.erf(X) - (2.0 * X / math.sqrt(math.pi)) * math.exp(-X * X)


def accel_df(
    x: float,
    y: float,
    vx: float,
    vy: float,
    M_host: float,
    a_host: float,
    M_sat: float,
    lnLambda: float,
    sigma_mode: str,
    sigma_const_kms: Optional[float],
) -> Tuple[float, float]:
    """Chandrasekhar dynamical-friction acceleration in kpc/Gyr^2."""
    v2 = vx * vx + vy * vy
    v = math.sqrt(v2)
    if v == 0 or M_sat <= 0 or lnLambda <= 0:
        return 0.0, 0.0

    r = math.sqrt(x * x + y * y)
    rho = rho_hernquist(r, M_host, a_host)

    if sigma_mode == "constant":
        if sigma_const_kms is None or sigma_const_kms <= 0:
            raise ValueError("sigma_mode='constant' requires sigma_const_kms > 0.")
        sigma = kms_to_kpcgyr(sigma_const_kms)
    elif sigma_mode == "local":
        sigma = sigma_local(r, M_host, a_host)
    else:
        raise ValueError("sigma_mode must be 'local' or 'constant'.")

    if sigma <= 0:
        return 0.0, 0.0

    X = v / (math.sqrt(2.0) * sigma)
    FX = chandra_F(X)
    coef = -4.0 * math.pi * G * G * M_sat * rho * lnLambda * FX / (v ** 3)
    return coef * vx, coef * vy


def derivs(
    state: np.ndarray,
    M_host: float,
    a_host: float,
    M_sat: float,
    lnLambda: float,
    use_df: bool,
    sigma_mode: str,
    sigma_const_kms: Optional[float],
) -> np.ndarray:
    x, y, vx, vy = state
    axg, ayg = accel_gravity(x, y, M_host, a_host)
    if use_df:
        axd, ayd = accel_df(
            x, y, vx, vy, M_host, a_host, M_sat, lnLambda, sigma_mode, sigma_const_kms
        )
    else:
        axd, ayd = 0.0, 0.0
    return np.array([vx, vy, axg + axd, ayg + ayd], dtype=float)


def rk4_step(
    state: np.ndarray,
    dt: float,
    M_host: float,
    a_host: float,
    M_sat: float,
    lnLambda: float,
    use_df: bool,
    sigma_mode: str,
    sigma_const_kms: Optional[float],
) -> np.ndarray:
    f = lambda s: derivs(s, M_host, a_host, M_sat, lnLambda, use_df, sigma_mode, sigma_const_kms)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# Orbital quantities and integration
# =============================================================================


def radius_xy(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def radial_velocity(x: float, y: float, vx: float, vy: float) -> float:
    r = radius_xy(x, y)
    if r == 0:
        return np.nan
    return (x * vx + y * vy) / r


def angular_momentum_z(x: float, y: float, vx: float, vy: float) -> float:
    """Specific angular momentum z-component in internal units: kpc^2/Gyr."""
    return x * vy - y * vx


def eps_from_state(x: float, y: float, vx: float, vy: float, M_host: float, a_host: float) -> float:
    r = radius_xy(x, y)
    return 0.5 * (vx * vx + vy * vy) + phi_hernquist(r, M_host, a_host)


def interp_turn(t0: float, r0: float, vr0: float, t1: float, r1: float, vr1: float) -> Tuple[float, float]:
    dv = vr1 - vr0
    f = 0.5 if dv == 0 else -vr0 / dv
    f = max(0.0, min(1.0, f))
    return t0 + f * (t1 - t0), r0 + f * (r1 - r0)


def integrate_orbit(
    x0: float,
    y0: float,
    vx0_kms: float,
    vy0_kms: float = 0.0,
    M_host: float = 1.0e12,
    a_host: float = 30.0,
    M_sat: float = 1.0e9,
    lnLambda: float = 3.0,
    use_df: bool = False,
    sigma_mode: str = "local",
    sigma_const_kms: Optional[float] = None,
    tmax: float = 8.0,
    dt: float = 0.001,
    stop_after_apo1: bool = True,
    store_trajectory: bool = False,
) -> Dict[str, object]:
    """Integrate one orbit and detect first pericenter and first post-pericenter apocenter."""
    state = np.array([x0, y0, kms_to_kpcgyr(vx0_kms), kms_to_kpcgyr(vy0_kms)], dtype=float)

    events: List[Dict[str, float]] = []
    trajectory: List[Dict[str, float]] = []

    t = 0.0
    x, y, vx, vy = state
    r_prev = radius_xy(x, y)
    vr_prev = radial_velocity(x, y, vx, vy)

    if store_trajectory:
        trajectory.append({
            "t_gyr": t,
            "x_kpc": x,
            "y_kpc": y,
            "vx_kms": kpcgyr_to_kms(vx),
            "vy_kms": kpcgyr_to_kms(vy),
            "r_kpc": r_prev,
            "vr_kms": kpcgyr_to_kms(vr_prev),
            "vt_kms": kpcgyr_to_kms(abs(angular_momentum_z(x, y, vx, vy)) / max(r_prev, 1e-300)),
        })

    nsteps = int(math.ceil(tmax / dt))
    for _ in range(nsteps):
        state_new = rk4_step(
            state, dt, M_host, a_host, M_sat, lnLambda, use_df, sigma_mode, sigma_const_kms
        )
        t_new = t + dt
        x1, y1, vx1, vy1 = state_new
        r_new = radius_xy(x1, y1)
        vr_new = radial_velocity(x1, y1, vx1, vy1)

        if np.isfinite(vr_prev) and np.isfinite(vr_new):
            if vr_prev < 0.0 and vr_new >= 0.0:
                te, re = interp_turn(t, r_prev, vr_prev, t_new, r_new, vr_new)
                events.append({"type": "peri", "t_gyr": te, "r_kpc": re})
            if vr_prev > 0.0 and vr_new <= 0.0:
                te, re = interp_turn(t, r_prev, vr_prev, t_new, r_new, vr_new)
                events.append({"type": "apo", "t_gyr": te, "r_kpc": re})

        state = state_new
        t = t_new
        r_prev = r_new
        vr_prev = vr_new

        if store_trajectory:
            trajectory.append({
                "t_gyr": t,
                "x_kpc": x1,
                "y_kpc": y1,
                "vx_kms": kpcgyr_to_kms(vx1),
                "vy_kms": kpcgyr_to_kms(vy1),
                "r_kpc": r_new,
                "vr_kms": kpcgyr_to_kms(vr_new),
                "vt_kms": kpcgyr_to_kms(abs(angular_momentum_z(x1, y1, vx1, vy1)) / max(r_new, 1e-300)),
            })

        if stop_after_apo1:
            peris = [ev for ev in events if ev["type"] == "peri"]
            apos = [ev for ev in events if ev["type"] == "apo"]
            if len(peris) >= 1:
                apos_after_p1 = [ev for ev in apos if ev["t_gyr"] > peris[0]["t_gyr"]]
                if len(apos_after_p1) >= 1:
                    break

    out: Dict[str, object] = {"events": events, "final_state": state, "t_final_gyr": t}
    if store_trajectory:
        out["trajectory"] = pd.DataFrame(trajectory)
    return out


def summarize_events(events: List[Dict[str, float]]) -> Dict[str, float]:
    peris = [ev for ev in events if ev["type"] == "peri"]
    apos = [ev for ev in events if ev["type"] == "apo"]

    out: Dict[str, float] = {}
    out["rperi1_kpc"] = peris[0]["r_kpc"] if len(peris) >= 1 else np.nan
    out["t_peri1_gyr"] = peris[0]["t_gyr"] if len(peris) >= 1 else np.nan

    apos_after_p1 = [ev for ev in apos if len(peris) >= 1 and ev["t_gyr"] > peris[0]["t_gyr"]]
    out["rapo1_after_p1_kpc"] = apos_after_p1[0]["r_kpc"] if len(apos_after_p1) >= 1 else np.nan
    out["t_apo1_after_p1_gyr"] = apos_after_p1[0]["t_gyr"] if len(apos_after_p1) >= 1 else np.nan
    out["dt_peri1_to_apo1_gyr"] = (
        out["t_apo1_after_p1_gyr"] - out["t_peri1_gyr"]
        if np.isfinite(out["t_apo1_after_p1_gyr"]) and np.isfinite(out["t_peri1_gyr"])
        else np.nan
    )
    out["rapo1_over_rperi1"] = (
        out["rapo1_after_p1_kpc"] / out["rperi1_kpc"]
        if np.isfinite(out["rapo1_after_p1_kpc"])
        and np.isfinite(out["rperi1_kpc"])
        and out["rperi1_kpc"] > 0
        else np.nan
    )
    return out


# =============================================================================
# Velocity solvers
# =============================================================================


@dataclass
class GeometrySolution:
    y0: float
    vx0_kms: float
    vy0_kms: float
    eps_internal: float
    h_internal: float
    r0: float
    vr0_kms: float
    vt0_kms: float
    score: float = np.nan


def solve_vxy_from_energy_angular_momentum(
    x0: float,
    y0: float,
    eps: float,
    h: float,
    M_host: float,
    a_host: float,
    radial_sign: float = -1.0,
    tangential_sign: float = 1.0,
) -> Optional[GeometrySolution]:
    """Solve vx0 and vy0 at fixed position from eps and |h|.

    The decomposition is

        v = v_r e_r + v_t e_t,

    with e_r=(x/r,y/r) and e_t=(-y/r,x/r). tangential_sign=+1 gives positive
    angular momentum z. radial_sign=-1 gives initial infall.
    """
    r0 = radius_xy(x0, y0)
    if r0 <= 0:
        return None

    phi0 = phi_hernquist(r0, M_host, a_host)
    v2 = 2.0 * (eps - phi0)
    if not np.isfinite(v2) or v2 <= 0:
        return None

    vt_abs = abs(h) / r0
    vr2 = v2 - vt_abs * vt_abs
    if vr2 < -1e-10:
        return None
    vr_abs = math.sqrt(max(0.0, vr2))

    vr = math.copysign(vr_abs, radial_sign)
    vt = math.copysign(vt_abs, tangential_sign)

    erx, ery = x0 / r0, y0 / r0
    etx, ety = -y0 / r0, x0 / r0

    vx = vr * erx + vt * etx
    vy = vr * ery + vt * ety

    h_signed = angular_momentum_z(x0, y0, vx, vy)
    vt_check = abs(h_signed) / r0

    return GeometrySolution(
        y0=y0,
        vx0_kms=kpcgyr_to_kms(vx),
        vy0_kms=kpcgyr_to_kms(vy),
        eps_internal=eps,
        h_internal=abs(h_signed),
        r0=r0,
        vr0_kms=kpcgyr_to_kms(radial_velocity(x0, y0, vx, vy)),
        vt0_kms=kpcgyr_to_kms(vt_check),
    )


def solve_vxy_from_velocity_fractions(
    x0: float,
    y0: float,
    f_energy: float,
    f_tangential: float,
    M_host: float,
    a_host: float,
    radial_sign: float = -1.0,
    tangential_sign: float = 1.0,
) -> Tuple[Optional[GeometrySolution], Dict[str, float]]:
    """Build a velocity solution from vtotal/vesc and vt/vtotal.

    Parameters
    ----------
    f_energy:
        v_total / v_escape(r0). Use 0 < f_energy < 1 for initially bound ICs.
        Larger values are less bound / higher energy.
    f_tangential:
        v_t / v_total. Use 0 for purely radial and values close to 1 for high-L
        orbits. If require_initial_infall is True, keep f_tangential < 1.

    Returns
    -------
    solution, metadata
        solution is a GeometrySolution or None. Metadata includes vesc, vcirc,
        vtotal, f_energy, f_tangential, eps and h in output units where possible.
    """
    meta: Dict[str, float] = {
        "f_energy_vtot_over_vesc": float(f_energy),
        "f_tangential_vt_over_vtot": float(f_tangential),
    }

    r0 = radius_xy(x0, y0)
    if r0 <= 0:
        meta["invalid_reason_velocity_fraction"] = "Initial radius must be > 0."
        return None, meta

    if not (np.isfinite(f_energy) and 0.0 < f_energy < 1.0):
        meta["invalid_reason_velocity_fraction"] = "Use 0 < f_energy < 1 for bound ICs."
        return None, meta
    if not (np.isfinite(f_tangential) and 0.0 <= f_tangential < 1.0):
        meta["invalid_reason_velocity_fraction"] = "Use 0 <= f_tangential < 1."
        return None, meta

    phi0 = phi_hernquist(r0, M_host, a_host)
    vesc = math.sqrt(max(0.0, -2.0 * phi0))
    vcirc = v_circ_hernquist(r0, M_host, a_host)

    vtot = float(f_energy) * vesc
    vt_abs = float(f_tangential) * vtot
    vr2 = vtot * vtot - vt_abs * vt_abs
    if vr2 < -1e-10:
        meta["invalid_reason_velocity_fraction"] = "Invalid velocity fractions produced negative vr^2."
        return None, meta

    vr_abs = math.sqrt(max(0.0, vr2))
    vr = math.copysign(vr_abs, radial_sign)
    vt = math.copysign(vt_abs, tangential_sign)

    erx, ery = x0 / r0, y0 / r0
    etx, ety = -y0 / r0, x0 / r0

    vx = vr * erx + vt * etx
    vy = vr * ery + vt * ety

    eps = eps_from_state(x0, y0, vx, vy, M_host, a_host)
    h_signed = angular_momentum_z(x0, y0, vx, vy)
    h_abs = abs(h_signed)
    rcirc_equiv = rcirc_from_eps_hernquist(eps, M_host, a_host)

    meta.update({
        "vesc_kms": kpcgyr_to_kms(vesc),
        "vcirc_at_r0_kms": kpcgyr_to_kms(vcirc),
        "vtotal_kms": kpcgyr_to_kms(vtot),
        "vtotal_over_vcirc": vtot / vcirc if vcirc > 0 else np.nan,
        "f_energy_vtot_over_vesc": float(f_energy),
        "f_tangential_vt_over_vtot": float(f_tangential),
        "rcirc_equiv_kpc": rcirc_equiv,
        "eps_specific_kms2": eps_int_to_kms2(eps),
        "h_specific_kpc_kms": h_int_to_kpckms(h_abs),
    })

    sol = GeometrySolution(
        y0=y0,
        vx0_kms=kpcgyr_to_kms(vx),
        vy0_kms=kpcgyr_to_kms(vy),
        eps_internal=eps,
        h_internal=h_abs,
        r0=r0,
        vr0_kms=kpcgyr_to_kms(radial_velocity(x0, y0, vx, vy)),
        vt0_kms=kpcgyr_to_kms(h_abs / max(r0, 1e-300)),
    )
    return sol, meta


# Legacy vx-only geometry solver, kept for comparison with the original script.

def geometry_equation_y(y: float, x0: float, eps: float, h: float, M_host: float, a_host: float) -> float:
    if y <= 0:
        return np.inf
    r = math.sqrt(x0 * x0 + y * y)
    vx_abs = h / y
    return 0.5 * vx_abs * vx_abs + phi_hernquist(r, M_host, a_host) - eps


def bisect_root_y(f, lo: float, hi: float, xtol: float = 1.0e-8, maxiter: int = 100) -> Optional[float]:
    flo = f(lo)
    fhi = f(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi):
        return None
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        return None

    a, b = lo, hi
    fa = flo
    for _ in range(maxiter):
        c = 0.5 * (a + b)
        fc = f(c)
        if not np.isfinite(fc):
            return None
        if abs(fc) < 1.0e-12 or abs(b - a) < xtol * max(1.0, abs(c)):
            return c
        if fa * fc <= 0:
            b = c
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def find_vx_only_geometry_solutions(
    eps: float,
    h: float,
    x0: float,
    M_host: float,
    a_host: float,
    y_min: float = 0.1,
    y_max: float = 500.0,
    y_scan_n: int = 1200,
    vx_sign: float = -1.0,
) -> List[GeometrySolution]:
    """Find positive-y solutions for the legacy v0=(vx0,0,0) geometry."""
    if h <= 0 or y_min <= 0 or y_max <= y_min:
        return []

    y_log = np.logspace(math.log10(y_min), math.log10(y_max), max(8, y_scan_n // 2))
    y_lin = np.linspace(y_min, y_max, max(8, y_scan_n // 2))
    ys = np.unique(np.concatenate([y_log, y_lin]))
    ys.sort()

    f = lambda yy: geometry_equation_y(yy, x0, eps, h, M_host, a_host)
    vals = np.array([f(float(yy)) for yy in ys], dtype=float)

    roots: List[float] = []
    for i in range(len(ys) - 1):
        f0, f1 = vals[i], vals[i + 1]
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0:
            roots.append(float(ys[i]))
        elif f0 * f1 < 0:
            root = bisect_root_y(f, float(ys[i]), float(ys[i + 1]))
            if root is not None and np.isfinite(root):
                roots.append(float(root))

    roots_unique: List[float] = []
    for root in roots:
        if not roots_unique or all(abs(root - r) > 1.0e-5 * max(1.0, abs(r)) for r in roots_unique):
            roots_unique.append(root)

    sols: List[GeometrySolution] = []
    for y0 in roots_unique:
        vx_internal = vx_sign * h / y0
        vx0_kms = kpcgyr_to_kms(vx_internal)
        r0 = radius_xy(x0, y0)
        vr0 = x0 * vx0_kms / r0 if r0 > 0 else np.nan
        sols.append(
            GeometrySolution(
                y0=y0,
                vx0_kms=vx0_kms,
                vy0_kms=0.0,
                eps_internal=eps,
                h_internal=h,
                r0=r0,
                vr0_kms=vr0,
                vt0_kms=h_int_to_kpckms(h) / r0,
            )
        )
    return sols


def score_solution(
    sol: GeometrySolution,
    y_preferred: Optional[float],
    vx_preferred_kms: Optional[float],
    weight_y: float = 1.0,
    weight_vx: float = 1.0,
) -> float:
    score = 0.0
    used = False
    if y_preferred is not None and y_preferred > 0 and sol.y0 > 0:
        score += weight_y * abs(math.log(sol.y0 / y_preferred))
        used = True
    if vx_preferred_kms is not None and abs(vx_preferred_kms) > 0 and abs(sol.vx0_kms) > 0:
        score += weight_vx * abs(math.log(abs(sol.vx0_kms) / abs(vx_preferred_kms)))
        used = True
    if not used:
        score = -sol.y0
    return score


def choose_geometry_solution(
    sols: List[GeometrySolution],
    y_preferred: Optional[float] = None,
    vx_preferred_kms: Optional[float] = None,
    root_choice: str = "closest-reference",
) -> Optional[GeometrySolution]:
    if not sols:
        return None
    if root_choice == "smallest-y":
        return min(sols, key=lambda s: s.y0)
    if root_choice == "largest-y":
        return max(sols, key=lambda s: s.y0)
    if root_choice != "closest-reference":
        raise ValueError("root_choice must be 'closest-reference', 'smallest-y', or 'largest-y'.")
    for sol in sols:
        sol.score = score_solution(sol, y_preferred, vx_preferred_kms)
    return min(sols, key=lambda s: s.score)


# =============================================================================
# Notebook configuration and grid driver
# =============================================================================


@dataclass
class OrbitGridConfig:
    """Single object to edit in a Jupyter notebook."""

    # Host model
    M_host_msun: float = 1.0e12
    a_host_kpc: float = 30.0

    # Satellite masses: choose values appropriate to your dwarf models.
    satellite_masses_msun: Mapping[str, float] = field(
        default_factory=lambda: {
            "light": 3.0e8,
            "intermediate": 1.0e9,
            "massive": 3.0e9,
        }
    )

    # Recommended v4 mode: velocity_fraction_grid.
    # Alternatives kept for compatibility: fixed_position_vxy and vx_only_solve_y.
    velocity_mode: str = "velocity_fraction_grid"

    # Initial position. In the recommended mode this is fixed for every grid point.
    x0_kpc: float = 150.0
    y0_kpc: float = 80.0

    # V4 IC-first grid.
    # f_E = v_total / v_escape(r0). Larger values mean higher energy / less bound.
    velocity_energy_fractions: Mapping[str, float] = field(
        default_factory=lambda: {
            "E_low_more_bound": 0.50,
            "E_mid": 0.65,
            "E_high_less_bound": 0.80,
        }
    )

    # f_T = v_t / v_total. Smaller values mean radial; larger values mean high angular momentum.
    tangential_velocity_fractions: Mapping[str, float] = field(
        default_factory=lambda: {
            "L_low_radial": 0.20,
            "L_mid": 0.50,
            "L_high": 0.80,
        }
    )

    # Older energy levels represented by the radius of the circular orbit with
    # the same specific energy. Used only by fixed_position_vxy and vx_only_solve_y.
    energy_rcirc_kpc: Mapping[str, float] = field(
        default_factory=lambda: {
            "E_low_more_bound": 100.0,
            "E_mid": 160.0,
            "E_high_less_bound": 260.0,
        }
    )

    # Older angular momentum levels represented by circularity eta=h/h_circ(eps).
    # Used only by fixed_position_vxy and vx_only_solve_y.
    circularity_eta: Mapping[str, float] = field(
        default_factory=lambda: {
            "radial": 0.15,
            "intermediate_L": 0.45,
            "near_circular": 0.75,
        }
    )

    # Signs of velocity decomposition.
    # radial_sign=-1 means initially infalling.
    # tangential_sign=+1 means positive angular momentum z.
    radial_sign: float = -1.0
    tangential_sign: float = 1.0

    # Legacy vx-only options.
    inward: bool = True
    require_initial_infall: bool = True
    y_preferred_kpc: Optional[float] = 80.0
    vx_preferred_kms: Optional[float] = -220.0
    root_choice: str = "closest-reference"
    y_solve_min_kpc: float = 0.1
    y_solve_max_kpc: float = 700.0
    y_scan_n: int = 1600

    # Optional dynamical friction. This affects integration and event summary;
    # the initial E-L construction remains a specific-orbit construction.
    use_df: bool = False
    lnLambda: float = 3.0
    sigma_mode: str = "local"
    sigma_const_kms: Optional[float] = None

    # Orbit integration and selection.
    apo_max_gyr: float = 5.0
    integrate_tmax_gyr: float = 8.0
    dt_gyr: float = 0.001

    # Output labels.
    orbit_prefix: str = "ORBIT"

    def validate(self) -> None:
        if self.dt_gyr <= 0:
            raise ValueError("dt_gyr must be > 0.")
        if self.apo_max_gyr <= 0:
            raise ValueError("apo_max_gyr must be > 0.")
        if self.integrate_tmax_gyr < self.apo_max_gyr:
            raise ValueError("integrate_tmax_gyr should be >= apo_max_gyr.")
        if self.velocity_mode not in {"velocity_fraction_grid", "fixed_position_vxy", "vx_only_solve_y"}:
            raise ValueError("velocity_mode must be 'velocity_fraction_grid', 'fixed_position_vxy', or 'vx_only_solve_y'.")
        if self.velocity_mode in {"velocity_fraction_grid", "fixed_position_vxy"} and radius_xy(self.x0_kpc, self.y0_kpc) <= 0:
            raise ValueError("The initial radius sqrt(x0^2+y0^2) must be > 0.")
        if self.root_choice not in {"closest-reference", "smallest-y", "largest-y"}:
            raise ValueError("Invalid root_choice.")
        if any(m <= 0 for m in self.satellite_masses_msun.values()):
            raise ValueError("All satellite masses must be > 0.")
        if self.velocity_mode == "velocity_fraction_grid":
            if any((f <= 0 or f >= 1.0) for f in self.velocity_energy_fractions.values()):
                raise ValueError("Use 0 < f_E < 1 for velocity_energy_fractions.")
            if any((f < 0 or f >= 1.0) for f in self.tangential_velocity_fractions.values()):
                raise ValueError("Use 0 <= f_T < 1 for tangential_velocity_fractions.")
        else:
            if any(r <= 0 for r in self.energy_rcirc_kpc.values()):
                raise ValueError("All rcirc values must be > 0.")
            if any((eta <= 0 or eta >= 1.0) for eta in self.circularity_eta.values()):
                raise ValueError("Use 0 < eta < 1 for the circularity grid.")
        if self.sigma_mode not in {"local", "constant"}:
            raise ValueError("sigma_mode must be 'local' or 'constant'.")

def _clean_label(label: str) -> str:
    return str(label).replace(" ", "_").replace("/", "_")


def base_row_velocity_fraction(
    cfg: OrbitGridConfig,
    mass_label: str,
    M_sat: float,
    energy_label: str,
    f_energy: float,
    L_label: str,
    f_tangential: float,
    eps: float,
    h: float,
    rcirc_equiv: float,
    status: str,
) -> Dict[str, object]:
    eps_kms2 = eps_int_to_kms2(eps)
    h_kpckms = h_int_to_kpckms(h)
    orbit_id = f"{_clean_label(mass_label)}__{_clean_label(energy_label)}__{_clean_label(L_label)}"
    return {
        "orbit_id": orbit_id,
        "status": status,
        "mass_label": mass_label,
        "energy_label": energy_label,
        "L_label": L_label,
        "velocity_mode": cfg.velocity_mode,
        "x0_kpc": float(cfg.x0_kpc),
        "M_host_msun": float(cfg.M_host_msun),
        "a_host_kpc": float(cfg.a_host_kpc),
        "M_sat_msun": float(M_sat),
        "lnLambda": float(cfg.lnLambda),
        "use_df": int(cfg.use_df),
        "f_energy_vtot_over_vesc": float(f_energy),
        "f_tangential_vt_over_vtot": float(f_tangential),
        "rcirc_equiv_kpc": float(rcirc_equiv) if np.isfinite(rcirc_equiv) else np.nan,
        "eps_specific_kms2": eps_kms2,
        "E_total_msun_kms2": M_sat * eps_kms2,
        "h_specific_kpc_kms": h_kpckms,
        "L_total_msun_kpc_kms": M_sat * h_kpckms,
        "eta": np.nan,
    }


def base_row(
    cfg: OrbitGridConfig,
    mass_label: str,
    M_sat: float,
    energy_label: str,
    rcirc: float,
    eps: float,
    L_label: str,
    eta: float,
    h: float,
    status: str,
) -> Dict[str, object]:
    eps_kms2 = eps_int_to_kms2(eps)
    h_kpckms = h_int_to_kpckms(h)
    orbit_id = f"{_clean_label(mass_label)}__{_clean_label(energy_label)}__{_clean_label(L_label)}"
    return {
        "orbit_id": orbit_id,
        "status": status,
        "mass_label": mass_label,
        "energy_label": energy_label,
        "L_label": L_label,
        "velocity_mode": cfg.velocity_mode,
        "x0_kpc": float(cfg.x0_kpc),
        "M_host_msun": float(cfg.M_host_msun),
        "a_host_kpc": float(cfg.a_host_kpc),
        "M_sat_msun": float(M_sat),
        "lnLambda": float(cfg.lnLambda),
        "use_df": int(cfg.use_df),
        "rcirc_equiv_kpc": float(rcirc),
        "eps_specific_kms2": eps_kms2,
        "E_total_msun_kms2": M_sat * eps_kms2,
        "h_specific_kpc_kms": h_kpckms,
        "L_total_msun_kpc_kms": M_sat * h_kpckms,
        "eta": float(eta),
    }


def evaluate_velocity_fraction_grid_point(
    cfg: OrbitGridConfig,
    mass_label: str,
    M_sat: float,
    energy_label: str,
    f_energy: float,
    L_label: str,
    f_tangential: float,
) -> Dict[str, object]:
    sol, meta = solve_vxy_from_velocity_fractions(
        x0=cfg.x0_kpc,
        y0=cfg.y0_kpc,
        f_energy=f_energy,
        f_tangential=f_tangential,
        M_host=cfg.M_host_msun,
        a_host=cfg.a_host_kpc,
        radial_sign=cfg.radial_sign,
        tangential_sign=cfg.tangential_sign,
    )

    if sol is None:
        row = base_row_velocity_fraction(
            cfg, mass_label, M_sat, energy_label, f_energy, L_label, f_tangential,
            eps=np.nan, h=np.nan, rcirc_equiv=np.nan, status="no_velocity_fraction_solution"
        )
        row.update(meta)
        row.update({
            "apo1_within_limit": 0,
            "y0_kpc": cfg.y0_kpc,
            "invalid_reason": meta.get("invalid_reason_velocity_fraction", "No velocity solution from velocity fractions."),
        })
        return row

    row = base_row_velocity_fraction(
        cfg, mass_label, M_sat, energy_label, f_energy, L_label, f_tangential,
        eps=sol.eps_internal, h=sol.h_internal,
        rcirc_equiv=meta.get("rcirc_equiv_kpc", np.nan),
        status="pending",
    )
    row.update(meta)
    row.update({
        "y0_kpc": sol.y0,
        "z0_kpc": 0.0,
        "vx0_kms": sol.vx0_kms,
        "vy0_kms": sol.vy0_kms,
        "vz0_kms": 0.0,
        "r0_kpc": sol.r0,
        "vr0_kms": sol.vr0_kms,
        "vt0_kms": sol.vt0_kms,
        "n_geometry_roots": 1,
        "geometry_choice_score": sol.score,
    })

    if cfg.require_initial_infall and not (np.isfinite(sol.vr0_kms) and sol.vr0_kms < 0):
        row.update({
            "status": "not_initially_infalling",
            "apo1_within_limit": 0,
            "invalid_reason": "Initial radial velocity is not negative.",
        })
        return row

    res = integrate_orbit(
        x0=cfg.x0_kpc,
        y0=sol.y0,
        vx0_kms=sol.vx0_kms,
        vy0_kms=sol.vy0_kms,
        M_host=cfg.M_host_msun,
        a_host=cfg.a_host_kpc,
        M_sat=M_sat,
        lnLambda=cfg.lnLambda,
        use_df=cfg.use_df,
        sigma_mode=cfg.sigma_mode,
        sigma_const_kms=cfg.sigma_const_kms,
        tmax=max(cfg.integrate_tmax_gyr, cfg.apo_max_gyr),
        dt=cfg.dt_gyr,
        stop_after_apo1=True,
        store_trajectory=False,
    )
    row.update(summarize_events(res["events"]))

    has_peri = np.isfinite(row.get("t_peri1_gyr", np.nan))
    has_apo = np.isfinite(row.get("t_apo1_after_p1_gyr", np.nan))

    if not has_peri:
        row.update({
            "status": "no_pericenter",
            "apo1_within_limit": 0,
            "invalid_reason": "No first pericenter found within integration time.",
        })
    elif not has_apo:
        row.update({
            "status": "no_apo_after_pericenter",
            "apo1_within_limit": 0,
            "invalid_reason": "No apocenter after first pericenter found within integration time.",
        })
    else:
        t_apo = float(row["t_apo1_after_p1_gyr"])
        within = t_apo <= cfg.apo_max_gyr
        row["apo1_within_limit"] = int(within)
        row["invalid_reason"] = "" if within else "First post-pericenter apocenter occurs after apo_max_gyr."
        row["status"] = "selected" if within else "late_apo1"

    return row


def evaluate_grid_point(
    cfg: OrbitGridConfig,
    mass_label: str,
    M_sat: float,
    energy_label: str,
    rcirc: float,
    eps: float,
    L_label: str,
    eta: float,
) -> Dict[str, object]:
    h = eta * h_circ_hernquist(rcirc, cfg.M_host_msun, cfg.a_host_kpc)
    row = base_row(cfg, mass_label, M_sat, energy_label, rcirc, eps, L_label, eta, h, "pending")

    if cfg.velocity_mode == "fixed_position_vxy":
        sol = solve_vxy_from_energy_angular_momentum(
            x0=cfg.x0_kpc,
            y0=cfg.y0_kpc,
            eps=eps,
            h=h,
            M_host=cfg.M_host_msun,
            a_host=cfg.a_host_kpc,
            radial_sign=cfg.radial_sign,
            tangential_sign=cfg.tangential_sign,
        )
        n_roots = 1 if sol is not None else 0
        if sol is None:
            row.update({
                "status": "no_velocity_solution",
                "apo1_within_limit": 0,
                "y0_kpc": cfg.y0_kpc,
                "invalid_reason": (
                    "No real vx0,vy0 solution at fixed position. Usually eps is too bound "
                    "for the chosen r0, or h is too high for the available kinetic energy."
                ),
            })
            return row
    else:
        sols = find_vx_only_geometry_solutions(
            eps=eps,
            h=h,
            x0=cfg.x0_kpc,
            M_host=cfg.M_host_msun,
            a_host=cfg.a_host_kpc,
            y_min=cfg.y_solve_min_kpc,
            y_max=cfg.y_solve_max_kpc,
            y_scan_n=cfg.y_scan_n,
            vx_sign=-1.0 if cfg.inward else 1.0,
        )
        n_roots = len(sols)
        if not sols:
            row.update({
                "status": "no_geometry_solution",
                "apo1_within_limit": 0,
                "invalid_reason": "No positive-y solution for fixed x0, eps, h, and vy0=0.",
            })
            return row
        sol = choose_geometry_solution(
            sols,
            y_preferred=cfg.y_preferred_kpc,
            vx_preferred_kms=cfg.vx_preferred_kms,
            root_choice=cfg.root_choice,
        )
        if sol is None:
            row.update({
                "status": "no_geometry_solution",
                "apo1_within_limit": 0,
                "invalid_reason": "Geometry solutions found but none selected.",
            })
            return row

    row.update({
        "y0_kpc": sol.y0,
        "z0_kpc": 0.0,
        "vx0_kms": sol.vx0_kms,
        "vy0_kms": sol.vy0_kms,
        "vz0_kms": 0.0,
        "r0_kpc": sol.r0,
        "vr0_kms": sol.vr0_kms,
        "vt0_kms": sol.vt0_kms,
        "n_geometry_roots": n_roots,
        "geometry_choice_score": sol.score,
    })

    if cfg.require_initial_infall and not (np.isfinite(sol.vr0_kms) and sol.vr0_kms < 0):
        row.update({
            "status": "not_initially_infalling",
            "apo1_within_limit": 0,
            "invalid_reason": "Initial radial velocity is not negative.",
        })
        return row

    res = integrate_orbit(
        x0=cfg.x0_kpc,
        y0=sol.y0,
        vx0_kms=sol.vx0_kms,
        vy0_kms=sol.vy0_kms,
        M_host=cfg.M_host_msun,
        a_host=cfg.a_host_kpc,
        M_sat=M_sat,
        lnLambda=cfg.lnLambda,
        use_df=cfg.use_df,
        sigma_mode=cfg.sigma_mode,
        sigma_const_kms=cfg.sigma_const_kms,
        tmax=max(cfg.integrate_tmax_gyr, cfg.apo_max_gyr),
        dt=cfg.dt_gyr,
        stop_after_apo1=True,
        store_trajectory=False,
    )
    row.update(summarize_events(res["events"]))

    has_peri = np.isfinite(row.get("t_peri1_gyr", np.nan))
    has_apo = np.isfinite(row.get("t_apo1_after_p1_gyr", np.nan))

    if not has_peri:
        row.update({
            "status": "no_pericenter",
            "apo1_within_limit": 0,
            "invalid_reason": "No first pericenter found within integration time.",
        })
    elif not has_apo:
        row.update({
            "status": "no_apo_after_pericenter",
            "apo1_within_limit": 0,
            "invalid_reason": "No apocenter after first pericenter found within integration time.",
        })
    else:
        t_apo = float(row["t_apo1_after_p1_gyr"])
        within = t_apo <= cfg.apo_max_gyr
        row["apo1_within_limit"] = int(within)
        row["invalid_reason"] = "" if within else "First post-pericenter apocenter occurs after apo_max_gyr."
        row["status"] = "selected" if within else "late_apo1"

    return row


def run_orbit_grid(cfg: OrbitGridConfig, verbose: bool = True) -> pd.DataFrame:
    """Run the full mass x energy x angular-momentum grid and return a DataFrame.

    In velocity_fraction_grid mode, the grid is mass x f_E x f_T.
    In the older modes, the grid is mass x rcirc_equiv x eta.
    """
    cfg.validate()

    rows: List[Dict[str, object]] = []

    if cfg.velocity_mode == "velocity_fraction_grid":
        total = (
            len(cfg.satellite_masses_msun)
            * len(cfg.velocity_energy_fractions)
            * len(cfg.tangential_velocity_fractions)
        )
        counter = 0
        for mass_label, M_sat in cfg.satellite_masses_msun.items():
            for energy_label, fE in cfg.velocity_energy_fractions.items():
                for L_label, fT in cfg.tangential_velocity_fractions.items():
                    counter += 1
                    if verbose:
                        print(f"[{counter:03d}/{total:03d}] {mass_label} | {energy_label} | {L_label}")
                    row = evaluate_velocity_fraction_grid_point(
                        cfg, mass_label, M_sat, energy_label, fE, L_label, fT
                    )
                    row["grid_index"] = counter
                    rows.append(row)
        return pd.DataFrame(rows)

    total = len(cfg.satellite_masses_msun) * len(cfg.energy_rcirc_kpc) * len(cfg.circularity_eta)
    counter = 0

    for mass_label, M_sat in cfg.satellite_masses_msun.items():
        for energy_label, rcirc in cfg.energy_rcirc_kpc.items():
            eps = eps_circ_hernquist(rcirc, cfg.M_host_msun, cfg.a_host_kpc)
            for L_label, eta in cfg.circularity_eta.items():
                counter += 1
                if verbose:
                    print(f"[{counter:03d}/{total:03d}] {mass_label} | {energy_label} | {L_label}")
                row = evaluate_grid_point(cfg, mass_label, M_sat, energy_label, rcirc, eps, L_label, eta)
                row["grid_index"] = counter
                rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Convenience functions for notebooks
# =============================================================================


def selected_orbits(
    df: pd.DataFrame,
    status: Optional[str] = "selected",
    rperi_range: Optional[Tuple[float, float]] = None,
    rapo_range: Optional[Tuple[float, float]] = None,
    tperi_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Filter usable orbits with optional pericenter/apocenter cuts."""
    out = df.copy()
    if status is not None:
        out = out[out["status"] == status]
    if rperi_range is not None and "rperi1_kpc" in out:
        lo, hi = rperi_range
        out = out[(out["rperi1_kpc"] >= lo) & (out["rperi1_kpc"] <= hi)]
    if rapo_range is not None and "rapo1_after_p1_kpc" in out:
        lo, hi = rapo_range
        out = out[(out["rapo1_after_p1_kpc"] >= lo) & (out["rapo1_after_p1_kpc"] <= hi)]
    if tperi_range is not None and "t_peri1_gyr" in out:
        lo, hi = tperi_range
        out = out[(out["t_peri1_gyr"] >= lo) & (out["t_peri1_gyr"] <= hi)]
    return out.reset_index(drop=True)


def compact_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table with the most important columns."""
    cols = [
        "grid_index", "orbit_id", "status", "velocity_mode",
        "mass_label", "energy_label", "L_label", "M_sat_msun",
        "f_energy_vtot_over_vesc", "f_tangential_vt_over_vtot",
        "vtotal_kms", "vesc_kms", "vcirc_at_r0_kms", "vtotal_over_vcirc",
        "rcirc_equiv_kpc", "eta",
        "x0_kpc", "y0_kpc", "r0_kpc",
        "vx0_kms", "vy0_kms", "vr0_kms", "vt0_kms",
        "eps_specific_kms2", "h_specific_kpc_kms",
        "rperi1_kpc", "t_peri1_gyr",
        "rapo1_after_p1_kpc", "t_apo1_after_p1_gyr",
        "rapo1_over_rperi1", "invalid_reason",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def integrate_from_row(
    row: pd.Series | Mapping[str, object],
    cfg: OrbitGridConfig,
    tmax: Optional[float] = None,
    dt: Optional[float] = None,
    store_trajectory: bool = True,
) -> Dict[str, object]:
    """Re-integrate one orbit from a DataFrame row."""
    if isinstance(row, pd.Series):
        row = row.to_dict()
    required = ["x0_kpc", "y0_kpc", "vx0_kms", "vy0_kms", "M_sat_msun"]
    missing = [k for k in required if k not in row or pd.isna(row[k])]
    if missing:
        raise ValueError(f"Cannot integrate row; missing values: {missing}")

    return integrate_orbit(
        x0=float(row["x0_kpc"]),
        y0=float(row["y0_kpc"]),
        vx0_kms=float(row["vx0_kms"]),
        vy0_kms=float(row["vy0_kms"]),
        M_host=cfg.M_host_msun,
        a_host=cfg.a_host_kpc,
        M_sat=float(row["M_sat_msun"]),
        lnLambda=cfg.lnLambda,
        use_df=cfg.use_df,
        sigma_mode=cfg.sigma_mode,
        sigma_const_kms=cfg.sigma_const_kms,
        tmax=cfg.integrate_tmax_gyr if tmax is None else tmax,
        dt=cfg.dt_gyr if dt is None else dt,
        stop_after_apo1=False,
        store_trajectory=store_trajectory,
    )


def plot_grid_summary(df: pd.DataFrame, annotate: bool = True):
    """Plot quick E-L diagnostic maps for the grid."""
    if plt is None:
        raise ImportError("matplotlib is not available.")

    valid = df[df["status"].isin(["selected", "late_apo1", "no_apo_after_pericenter", "no_pericenter"])].copy()
    if valid.empty:
        raise ValueError("No valid solved-velocity rows to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, col, label in [
        (axes[0], "rperi1_kpc", r"$r_{\rm peri,1}$ [kpc]"),
        (axes[1], "t_apo1_after_p1_gyr", r"$t_{\rm apo,1}$ [Gyr]"),
    ]:
        m = np.isfinite(valid["eps_specific_kms2"]) & np.isfinite(valid["h_specific_kpc_kms"]) & np.isfinite(valid[col])
        sc = ax.scatter(valid.loc[m, "eps_specific_kms2"], valid.loc[m, "h_specific_kpc_kms"], c=valid.loc[m, col], s=70)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(label)
        ax.set_xlabel(r"$\epsilon$ [(km/s)$^2$]")
        ax.set_ylabel(r"$h$ [kpc km/s]")
        if annotate:
            for _, r in valid.loc[m].iterrows():
                ax.annotate(str(r["grid_index"]), (r["eps_specific_kms2"], r["h_specific_kpc_kms"]), fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_orbit(row: pd.Series | Mapping[str, object], cfg: OrbitGridConfig, tmax: Optional[float] = None, dt: Optional[float] = None):
    """Plot x-y trajectory and r(t), v_rad(t) for one row from the DataFrame."""
    if plt is None:
        raise ImportError("matplotlib is not available.")
    res = integrate_from_row(row, cfg, tmax=tmax, dt=dt, store_trajectory=True)
    tr = res["trajectory"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(tr["x_kpc"], tr["y_kpc"])
    axes[0].scatter([0.0], [0.0], marker="x", s=80)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [kpc]")
    axes[0].set_ylabel("y [kpc]")
    axes[0].set_title("Orbit in the orbital plane")

    axes[1].plot(tr["t_gyr"], tr["r_kpc"])
    for ev in res["events"]:
        axes[1].axvline(ev["t_gyr"], linestyle="--", linewidth=1)
        axes[1].annotate(ev["type"], (ev["t_gyr"], ev["r_kpc"]), fontsize=8)
    axes[1].set_xlabel("time [Gyr]")
    axes[1].set_ylabel("r [kpc]")
    axes[1].set_title("Radius")

    axes[2].plot(tr["t_gyr"], tr["vr_kms"])
    axes[2].axhline(0.0, linestyle="--", linewidth=1)
    axes[2].set_xlabel("time [Gyr]")
    axes[2].set_ylabel(r"$v_{\rm rad}$ [km/s]")
    axes[2].set_title("Radial velocity")

    fig.tight_layout()
    return fig, axes, res


def _ordered_labels_from_mapping(mapping: Mapping[str, float], requested: Optional[Sequence[str]] = None) -> List[str]:
    labels = list(mapping.keys())
    if requested is None:
        return labels
    missing = [lab for lab in requested if lab not in labels]
    if missing:
        raise ValueError(f"Requested labels are not in mapping: {missing}")
    return list(requested)


def plot_3x3_radius_vrad_grid(
    df: pd.DataFrame,
    cfg: OrbitGridConfig,
    mass_label: str = "intermediate",
    energy_labels: Optional[Sequence[str]] = None,
    L_labels: Optional[Sequence[str]] = None,
    status_preference: Sequence[str] = ("selected", "late_apo1", "no_apo_after_pericenter", "no_pericenter"),
    tmax: Optional[float] = None,
    dt: Optional[float] = None,
    radius_log: bool = False,
    title: Optional[str] = None,
):
    """Plot a 3x3 grid of r(t) and v_rad(t) in the same panels.

    Rows correspond to energy labels and columns to angular-momentum labels.
    The left y-axis shows radius. The right y-axis shows radial velocity.

    Parameters
    ----------
    df:
        Output from run_orbit_grid.
    cfg:
        The same OrbitGridConfig used to build df.
    mass_label:
        Which satellite mass to display. Use one of cfg.satellite_masses_msun keys.
    energy_labels, L_labels:
        Optional explicit order. In velocity_fraction_grid mode the order comes
        from cfg.velocity_energy_fractions and cfg.tangential_velocity_fractions.
        In the older modes the order comes from cfg.energy_rcirc_kpc and
        cfg.circularity_eta. The function expects exactly three labels in each
        direction for a 3x3 plot.
    status_preference:
        Rows with these statuses can be plotted. The first available row per
        panel is used.
    radius_log:
        Use log scale for radius axis.
    """
    if plt is None:
        raise ImportError("matplotlib is not available.")

    energy_mapping = cfg.velocity_energy_fractions if cfg.velocity_mode == "velocity_fraction_grid" else cfg.energy_rcirc_kpc
    L_mapping = cfg.tangential_velocity_fractions if cfg.velocity_mode == "velocity_fraction_grid" else cfg.circularity_eta
    energy_labels = _ordered_labels_from_mapping(energy_mapping, energy_labels)
    L_labels = _ordered_labels_from_mapping(L_mapping, L_labels)
    if len(energy_labels) != 3 or len(L_labels) != 3:
        raise ValueError("plot_3x3_radius_vrad_grid expects exactly 3 energy labels and 3 L labels.")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)

    plotted_any = False
    for i, e_label in enumerate(energy_labels):
        for j, l_label in enumerate(L_labels):
            ax = axes[i, j]
            ax2 = ax.twinx()

            sub = df[
                (df["mass_label"] == mass_label)
                & (df["energy_label"] == e_label)
                & (df["L_label"] == l_label)
            ].copy()

            if sub.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{e_label}\n{l_label}")
                continue

            # Choose the first row following status_preference.
            row = None
            for st in status_preference:
                cand = sub[sub["status"] == st]
                if not cand.empty:
                    row = cand.iloc[0]
                    break
            if row is None:
                row = sub.iloc[0]

            if pd.isna(row.get("vx0_kms", np.nan)) or pd.isna(row.get("vy0_kms", np.nan)):
                msg = str(row.get("status", "invalid"))
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{e_label}\n{l_label}")
                continue

            res = integrate_from_row(row, cfg, tmax=tmax, dt=dt, store_trajectory=True)
            tr = res["trajectory"]
            ax.plot(tr["t_gyr"], tr["r_kpc"], label="r")
            ax2.plot(tr["t_gyr"], tr["vr_kms"], linestyle="--", label=r"$v_{rad}$")
            ax2.axhline(0.0, linestyle=":", linewidth=1)

            for ev in res["events"]:
                ax.axvline(ev["t_gyr"], linestyle=":", linewidth=1)

            if radius_log:
                ax.set_yscale("log")

            status = row.get("status", "")
            rp = row.get("rperi1_kpc", np.nan)
            ta = row.get("t_apo1_after_p1_gyr", np.nan)
            ax.set_title(f"{e_label}\n{l_label}\n{status}, rp={rp:.1f}, tA={ta:.2f}", fontsize=9)
            plotted_any = True

            if j == 0:
                ax.set_ylabel("r [kpc]")
            if j == 2:
                ax2.set_ylabel(r"$v_{rad}$ [km/s]")
            if i == 2:
                ax.set_xlabel("time [Gyr]")

    if not plotted_any:
        raise ValueError("No valid orbits were available for the requested 3x3 grid.")

    if title is None:
        title = f"Radius and radial velocity grid: mass_label={mass_label}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def describe_velocity_fraction_grid(cfg: OrbitGridConfig) -> pd.DataFrame:
    """Return a small table explaining the v4 grid at the chosen initial radius."""
    if cfg.velocity_mode != "velocity_fraction_grid":
        raise ValueError("describe_velocity_fraction_grid is intended for velocity_fraction_grid mode.")
    r0 = radius_xy(cfg.x0_kpc, cfg.y0_kpc)
    phi0 = phi_hernquist(r0, cfg.M_host_msun, cfg.a_host_kpc)
    vesc = kpcgyr_to_kms(math.sqrt(max(0.0, -2.0 * phi0)))
    vcirc = kpcgyr_to_kms(v_circ_hernquist(r0, cfg.M_host_msun, cfg.a_host_kpc))
    rows = []
    for e_label, fE in cfg.velocity_energy_fractions.items():
        for l_label, fT in cfg.tangential_velocity_fractions.items():
            vtot = fE * vesc
            vt = fT * vtot
            vr = -math.sqrt(max(0.0, vtot * vtot - vt * vt))
            rows.append({
                "energy_label": e_label,
                "L_label": l_label,
                "r0_kpc": r0,
                "f_energy_vtot_over_vesc": fE,
                "f_tangential_vt_over_vtot": fT,
                "vesc_kms": vesc,
                "vcirc_at_r0_kms": vcirc,
                "vtotal_kms": vtot,
                "vr0_radial_component_kms": vr,
                "vt0_tangential_component_kms": vt,
            })
    return pd.DataFrame(rows)




# =============================================================================
# Manual IC-grid helpers
# =============================================================================


def _get_first_present(mapping: Mapping[str, object], names: Sequence[str], default: object = np.nan) -> object:
    """Return the first value whose key appears in mapping."""
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _as_float_or_nan(value: object) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _iter_manual_ic_grid(manual_ics: object) -> List[Dict[str, object]]:
    """Normalize user-provided manual ICs into a list of dictionaries.

    Supported inputs
    ----------------
    1. pandas.DataFrame with columns such as x0_kpc, y0_kpc, vx0_kms, vy0_kms.
    2. list/tuple of dictionaries.
    3. list/tuple of 4-tuples: (x0_kpc, y0_kpc, vx0_kms, vy0_kms).
    4. nested 3x3-like dictionary:

       {
           "row_label_1": {
               "col_label_1": {"x0_kpc": ..., "y0_kpc": ..., "vx0_kms": ..., "vy0_kms": ...},
               "col_label_2": (...),
           },
           "row_label_2": {...},
       }

       In the nested case, the outer key becomes energy_label and the inner key
       becomes L_label by default. The labels are only plotting/grouping labels;
       they do not need to literally represent energy and angular momentum.
    """
    if isinstance(manual_ics, pd.DataFrame):
        return manual_ics.to_dict(orient="records")

    if isinstance(manual_ics, Mapping):
        rows: List[Dict[str, object]] = []
        for row_label, cols in manual_ics.items():
            if not isinstance(cols, Mapping):
                raise ValueError("Nested manual_ics dictionaries must have mapping values at the first level.")
            for col_label, item in cols.items():
                if isinstance(item, Mapping):
                    row = dict(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    row = {
                        "x0_kpc": item[0],
                        "y0_kpc": item[1],
                        "vx0_kms": item[2],
                        "vy0_kms": item[3],
                    }
                    if len(item) >= 5:
                        row["M_sat_msun"] = item[4]
                else:
                    raise ValueError(
                        "Nested manual_ics entries must be dicts or tuples like "
                        "(x0_kpc, y0_kpc, vx0_kms, vy0_kms)."
                    )
                row.setdefault("energy_label", str(row_label))
                row.setdefault("L_label", str(col_label))
                row.setdefault("ic_label", f"{row_label}__{col_label}")
                rows.append(row)
        return rows

    if isinstance(manual_ics, (list, tuple)):
        rows = []
        for i, item in enumerate(manual_ics, start=1):
            if isinstance(item, Mapping):
                row = dict(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                row = {
                    "x0_kpc": item[0],
                    "y0_kpc": item[1],
                    "vx0_kms": item[2],
                    "vy0_kms": item[3],
                }
                if len(item) >= 5:
                    row["M_sat_msun"] = item[4]
            else:
                raise ValueError(
                    "manual_ics list entries must be dictionaries or tuples like "
                    "(x0_kpc, y0_kpc, vx0_kms, vy0_kms)."
                )
            row.setdefault("ic_label", f"manual_{i:02d}")
            rows.append(row)
        return rows

    raise TypeError("manual_ics must be a DataFrame, list/tuple, or nested dictionary.")


def evaluate_manual_ic_row(
    row_input: Mapping[str, object],
    cfg: OrbitGridConfig,
    default_M_sat_msun: Optional[float] = None,
    default_mass_label: str = "manual",
    default_energy_label: str = "manual_row",
    default_L_label: str = "manual_col",
    orbit_prefix: str = "MANUAL",
    grid_index: int = 1,
    integrate: bool = True,
) -> Dict[str, object]:
    """Evaluate one manually supplied IC and return a standard row dictionary.

    The input must contain x0/y0 and vx0/vy0 in kpc and km/s. The function
    computes radial/tangential velocity, specific energy, specific angular
    momentum, circular-equivalent radius, approximate circularity, and optionally
    first pericenter/apocenter diagnostics.
    """
    # Flexible aliases to make notebook use less fragile.
    x0 = _as_float_or_nan(_get_first_present(row_input, ["x0_kpc", "x0", "x_kpc", "x"]))
    y0 = _as_float_or_nan(_get_first_present(row_input, ["y0_kpc", "y0", "y_kpc", "y"]))
    z0 = _as_float_or_nan(_get_first_present(row_input, ["z0_kpc", "z0", "z_kpc", "z"], 0.0))
    vx0 = _as_float_or_nan(_get_first_present(row_input, ["vx0_kms", "vx0", "vx_kms", "vx"]))
    # The user message had '(vx0, vx0, 0)'; here we require/provide vy0 but accept common aliases.
    vy0 = _as_float_or_nan(_get_first_present(row_input, ["vy0_kms", "vy0", "vy_kms", "vy"]))
    vz0 = _as_float_or_nan(_get_first_present(row_input, ["vz0_kms", "vz0", "vz_kms", "vz"], 0.0))

    if default_M_sat_msun is None:
        if len(cfg.satellite_masses_msun) > 0:
            default_M_sat_msun = list(cfg.satellite_masses_msun.values())[0]
        else:
            default_M_sat_msun = 1.0e9

    M_sat = _as_float_or_nan(_get_first_present(row_input, ["M_sat_msun", "M_sat", "msat", "mass_msun"], default_M_sat_msun))
    mass_label = str(_get_first_present(row_input, ["mass_label", "mass_name"], default_mass_label))
    energy_label = str(_get_first_present(row_input, ["energy_label", "row_label", "E_label"], default_energy_label))
    L_label = str(_get_first_present(row_input, ["L_label", "angular_label", "col_label"], default_L_label))
    ic_label = str(_get_first_present(row_input, ["ic_label", "orbit_id", "name"], f"{orbit_prefix}{grid_index:02d}"))

    orbit_id = str(_get_first_present(
        row_input,
        ["orbit_id"],
        f"{_clean_label(mass_label)}__{_clean_label(energy_label)}__{_clean_label(L_label)}"
    ))

    out: Dict[str, object] = {
        "grid_index": int(grid_index),
        "orbit_id": orbit_id,
        "ic_label": ic_label,
        "status": "pending",
        "velocity_mode": "manual_cartesian_ic_grid",
        "mass_label": mass_label,
        "energy_label": energy_label,
        "L_label": L_label,
        "M_host_msun": float(cfg.M_host_msun),
        "a_host_kpc": float(cfg.a_host_kpc),
        "M_sat_msun": float(M_sat) if np.isfinite(M_sat) else np.nan,
        "lnLambda": float(cfg.lnLambda),
        "use_df": int(cfg.use_df),
        "x0_kpc": x0,
        "y0_kpc": y0,
        "z0_kpc": z0,
        "vx0_kms": vx0,
        "vy0_kms": vy0,
        "vz0_kms": vz0,
    }

    missing = [name for name, val in [("x0", x0), ("y0", y0), ("vx0", vx0), ("vy0", vy0), ("M_sat", M_sat)] if not np.isfinite(val)]
    if missing:
        out.update({
            "status": "invalid_manual_ic",
            "apo1_within_limit": 0,
            "invalid_reason": f"Missing or non-finite manual IC values: {missing}",
        })
        return out

    r0 = radius_xy(x0, y0)
    if r0 <= 0:
        out.update({
            "status": "invalid_manual_ic",
            "apo1_within_limit": 0,
            "invalid_reason": "Initial radius sqrt(x0^2+y0^2) must be > 0.",
        })
        return out

    vx_int = kms_to_kpcgyr(vx0)
    vy_int = kms_to_kpcgyr(vy0)
    eps_int = eps_from_state(x0, y0, vx_int, vy_int, cfg.M_host_msun, cfg.a_host_kpc)
    h_int_signed = angular_momentum_z(x0, y0, vx_int, vy_int)
    h_int = abs(h_int_signed)
    vr_int = radial_velocity(x0, y0, vx_int, vy_int)
    vt_int = h_int / r0

    phi0 = phi_hernquist(r0, cfg.M_host_msun, cfg.a_host_kpc)
    vtot_int = math.sqrt(vx_int * vx_int + vy_int * vy_int)
    vesc_int = math.sqrt(max(0.0, -2.0 * phi0))
    vcirc_int = v_circ_hernquist(r0, cfg.M_host_msun, cfg.a_host_kpc)
    rcirc = rcirc_from_eps_hernquist(eps_int, cfg.M_host_msun, cfg.a_host_kpc)
    hcirc = h_circ_hernquist(rcirc, cfg.M_host_msun, cfg.a_host_kpc) if np.isfinite(rcirc) and rcirc > 0 else np.nan
    eta = h_int / hcirc if np.isfinite(hcirc) and hcirc > 0 else np.nan

    eps_kms2 = eps_int_to_kms2(eps_int)
    h_kpckms = h_int_to_kpckms(h_int)
    out.update({
        "r0_kpc": r0,
        "vr0_kms": kpcgyr_to_kms(vr_int),
        "vt0_kms": kpcgyr_to_kms(vt_int),
        "vtotal_kms": kpcgyr_to_kms(vtot_int),
        "vesc_kms": kpcgyr_to_kms(vesc_int),
        "vcirc_at_r0_kms": kpcgyr_to_kms(vcirc_int),
        "vtotal_over_vesc": vtot_int / vesc_int if vesc_int > 0 else np.nan,
        "vtotal_over_vcirc": vtot_int / vcirc_int if vcirc_int > 0 else np.nan,
        "f_tangential_vt_over_vtot": vt_int / vtot_int if vtot_int > 0 else np.nan,
        "eps_specific_kms2": eps_kms2,
        "E_total_msun_kms2": float(M_sat) * eps_kms2,
        "h_specific_kpc_kms": h_kpckms,
        "h_signed_kpc_kms": h_int_to_kpckms(h_int_signed),
        "L_total_msun_kpc_kms": float(M_sat) * h_kpckms,
        "rcirc_equiv_kpc": rcirc,
        "eta": eta,
    })

    if eps_int >= 0:
        out.update({
            "status": "unbound_initial_energy",
            "apo1_within_limit": 0,
            "invalid_reason": "Initial specific energy is >= 0 in the Hernquist potential.",
        })
        return out

    if cfg.require_initial_infall and not (np.isfinite(out["vr0_kms"]) and float(out["vr0_kms"]) < 0):
        out.update({
            "status": "not_initially_infalling",
            "apo1_within_limit": 0,
            "invalid_reason": "Initial radial velocity is not negative.",
        })
        if not integrate:
            return out
        # Keep integrating if requested, but preserve the diagnostic status unless a user changes this criterion.

    if integrate:
        res = integrate_orbit(
            x0=x0,
            y0=y0,
            vx0_kms=vx0,
            vy0_kms=vy0,
            M_host=cfg.M_host_msun,
            a_host=cfg.a_host_kpc,
            M_sat=float(M_sat),
            lnLambda=cfg.lnLambda,
            use_df=cfg.use_df,
            sigma_mode=cfg.sigma_mode,
            sigma_const_kms=cfg.sigma_const_kms,
            tmax=max(cfg.integrate_tmax_gyr, cfg.apo_max_gyr),
            dt=cfg.dt_gyr,
            stop_after_apo1=True,
            store_trajectory=False,
        )
        out.update(summarize_events(res["events"]))

        has_peri = np.isfinite(out.get("t_peri1_gyr", np.nan))
        has_apo = np.isfinite(out.get("t_apo1_after_p1_gyr", np.nan))
        # If the IC was not initially infalling, keep that status; otherwise classify normally.
        if out.get("status") == "not_initially_infalling":
            out["apo1_within_limit"] = 0
        elif not has_peri:
            out.update({
                "status": "no_pericenter",
                "apo1_within_limit": 0,
                "invalid_reason": "No first pericenter found within integration time.",
            })
        elif not has_apo:
            out.update({
                "status": "no_apo_after_pericenter",
                "apo1_within_limit": 0,
                "invalid_reason": "No apocenter after first pericenter found within integration time.",
            })
        else:
            t_apo = float(out["t_apo1_after_p1_gyr"])
            within = t_apo <= cfg.apo_max_gyr
            out["apo1_within_limit"] = int(within)
            out["invalid_reason"] = "" if within else "First post-pericenter apocenter occurs after apo_max_gyr."
            out["status"] = "selected" if within else "late_apo1"
    else:
        out.update({
            "status": "manual_ic_not_integrated" if out.get("status") == "pending" else out.get("status"),
            "apo1_within_limit": 0,
        })

    return out


def build_manual_ic_dataframe(
    manual_ics: object,
    cfg: Optional[OrbitGridConfig] = None,
    default_M_sat_msun: Optional[float] = None,
    default_mass_label: str = "manual",
    default_energy_label: str = "manual_row",
    default_L_label: str = "manual_col",
    orbit_prefix: str = "MANUAL",
    integrate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a standard orbit DataFrame from manually supplied Cartesian ICs.

    Parameters
    ----------
    manual_ics:
        DataFrame, list of dicts/tuples, or nested dictionary. Each IC must
        provide x0_kpc, y0_kpc, vx0_kms, vy0_kms. z0_kpc and vz0_kms default to 0.
    cfg:
        OrbitGridConfig containing the host model and integration settings.
    default_M_sat_msun:
        Satellite mass used when rows do not specify M_sat_msun.
    integrate:
        If True, classify each IC using first pericenter and first post-pericenter
        apocenter diagnostics.
    """
    if cfg is None:
        cfg = OrbitGridConfig()
    cfg.validate()

    rows_in = _iter_manual_ic_grid(manual_ics)
    rows_out: List[Dict[str, object]] = []
    n = len(rows_in)
    for i, row in enumerate(rows_in, start=1):
        if verbose:
            e_label = row.get("energy_label", row.get("row_label", default_energy_label))
            l_label = row.get("L_label", row.get("col_label", default_L_label))
            print(f"[{i:03d}/{n:03d}] manual IC | {e_label} | {l_label}")
        rows_out.append(evaluate_manual_ic_row(
            row,
            cfg=cfg,
            default_M_sat_msun=default_M_sat_msun,
            default_mass_label=default_mass_label,
            default_energy_label=default_energy_label,
            default_L_label=default_L_label,
            orbit_prefix=orbit_prefix,
            grid_index=i,
            integrate=integrate,
        ))
    return pd.DataFrame(rows_out)


def plot_manual_ic_3x3_radius_vrad_grid(
    df: pd.DataFrame,
    cfg: OrbitGridConfig,
    mass_label: Optional[str] = None,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    row_column: str = "energy_label",
    col_column: str = "L_label",
    status_preference: Sequence[str] = ("selected", "late_apo1", "no_apo_after_pericenter", "no_pericenter", "not_initially_infalling"),
    tmax: Optional[float] = None,
    dt: Optional[float] = None,
    radius_log: bool = False,
    title: Optional[str] = None,
):
    """Plot a manual 3x3 IC grid with r(t) and radial velocity in each panel.

    This is the manual-IC analogue of plot_3x3_radius_vrad_grid. The DataFrame
    should usually be the output of build_manual_ic_dataframe, but any DataFrame
    with x0_kpc, y0_kpc, vx0_kms, vy0_kms, M_sat_msun and row/column labels works.

    Rows are selected by row_column; columns are selected by col_column. By
    default these are energy_label and L_label, but for manual grids they are
    just labels and can represent any classification you choose.
    """
    if plt is None:
        raise ImportError("matplotlib is not available.")

    required = ["x0_kpc", "y0_kpc", "vx0_kms", "vy0_kms", "M_sat_msun", row_column, col_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manual IC DataFrame is missing required columns: {missing}")

    work = df.copy()
    if mass_label is not None:
        if "mass_label" not in work.columns:
            raise ValueError("mass_label was requested, but df has no 'mass_label' column.")
        work = work[work["mass_label"] == mass_label].copy()
        if work.empty:
            raise ValueError(f"No rows found for mass_label={mass_label!r}.")

    if row_labels is None:
        row_labels = list(pd.unique(work[row_column]))
    else:
        row_labels = list(row_labels)
    if col_labels is None:
        col_labels = list(pd.unique(work[col_column]))
    else:
        col_labels = list(col_labels)

    if len(row_labels) != 3 or len(col_labels) != 3:
        raise ValueError(
            "plot_manual_ic_3x3_radius_vrad_grid expects exactly 3 row labels and 3 column labels. "
            "Pass row_labels=[...] and col_labels=[...] to control the order."
        )

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    plotted_any = False

    for i, rlab in enumerate(row_labels):
        for j, clab in enumerate(col_labels):
            ax = axes[i, j]
            ax2 = ax.twinx()
            sub = work[(work[row_column] == rlab) & (work[col_column] == clab)].copy()

            if sub.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{rlab}\n{clab}")
                continue

            chosen = None
            if "status" in sub.columns:
                for st in status_preference:
                    cand = sub[sub["status"] == st]
                    if not cand.empty:
                        chosen = cand.iloc[0]
                        break
            if chosen is None:
                chosen = sub.iloc[0]

            if any(pd.isna(chosen.get(c, np.nan)) for c in ["x0_kpc", "y0_kpc", "vx0_kms", "vy0_kms"]):
                msg = str(chosen.get("status", "invalid"))
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{rlab}\n{clab}")
                continue

            res = integrate_from_row(chosen, cfg, tmax=tmax, dt=dt, store_trajectory=True)
            tr = res["trajectory"]
            ax.plot(tr["t_gyr"], tr["r_kpc"], label="r")
            ax2.plot(tr["t_gyr"], tr["vr_kms"], linestyle="--", label=r"$v_{rad}$")
            ax2.axhline(0.0, linestyle=":", linewidth=1)

            for ev in res["events"]:
                ax.axvline(ev["t_gyr"], linestyle=":", linewidth=1)

            if radius_log:
                ax.set_yscale("log")

            status = chosen.get("status", "")
            rp = chosen.get("rperi1_kpc", np.nan)
            ta = chosen.get("t_apo1_after_p1_gyr", np.nan)
            eps = chosen.get("eps_specific_kms2", np.nan)
            h = chosen.get("h_specific_kpc_kms", np.nan)
            ax.set_title(
                f"{rlab}\n{clab}\n{status}, rp={rp:.1f}, tA={ta:.2f}\n"
                f"eps={eps:.0f}, h={h:.0f}",
                fontsize=8,
            )
            plotted_any = True

            if j == 0:
                ax.set_ylabel("r [kpc]")
            if j == 2:
                ax2.set_ylabel(r"$v_{rad}$ [km/s]")
            if i == 2:
                ax.set_xlabel("time [Gyr]")

    if not plotted_any:
        raise ValueError("No valid manual ICs were available for the requested 3x3 grid.")

    if title is None:
        title = "Manual IC grid: radius and radial velocity"
        if mass_label is not None:
            title += f" | mass_label={mass_label}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


# Backward/alternate name: easier to discover next to the original function name.
def plot_3x3_radius_vrad_grid_from_manual_ics(*args, **kwargs):
    """Alias for plot_manual_ic_3x3_radius_vrad_grid."""
    return plot_manual_ic_3x3_radius_vrad_grid(*args, **kwargs)


# =============================================================================
# Output helpers
# =============================================================================


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def scalar_to_str(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isfinite(v):
            return f"{v:.12g}"
        return "nan"
    return str(v)


def write_csv_rows(rows: List[Dict[str, object]], filename: str | Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_grid_tables(df: pd.DataFrame, outdir: str | Path = "ORBIT_GRID_TABLES", prefix: str = "orbit_grid") -> Dict[str, Path]:
    """Save all rows and selected rows as CSV files."""
    outdir = Path(outdir)
    ensure_dir(outdir)
    all_path = outdir / f"{prefix}_all.csv"
    selected_path = outdir / f"{prefix}_selected.csv"
    summary_path = outdir / f"{prefix}_compact_summary.csv"

    df.to_csv(all_path, index=False)
    df[df["status"] == "selected"].to_csv(selected_path, index=False)
    compact_summary(df).to_csv(summary_path, index=False)

    return {"all": all_path, "selected": selected_path, "summary": summary_path}


def row_to_plain_dict(row: pd.Series | Mapping[str, object]) -> Dict[str, object]:
    if isinstance(row, pd.Series):
        out = row.to_dict()
    else:
        out = dict(row)
    clean: Dict[str, object] = {}
    for k, v in out.items():
        if isinstance(v, np.integer):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def write_ic_folder(row: pd.Series | Mapping[str, object], outdir: str | Path, orbit_name: Optional[str] = None) -> Path:
    """Write one folder containing orbit_ic.txt, orbit_ic.csv, and orbit_ic.json."""
    row_dict = row_to_plain_dict(row)
    if orbit_name is None:
        orbit_name = str(row_dict.get("orbit_id", f"ORBIT{int(row_dict.get('grid_index', 0)):02d}"))
    folder = Path(outdir) / orbit_name
    ensure_dir(folder)

    keys_for_ic = [
        "orbit_id", "grid_index", "mass_label", "energy_label", "L_label", "velocity_mode",
        "x0_kpc", "y0_kpc", "z0_kpc", "vx0_kms", "vy0_kms", "vz0_kms",
        "M_sat_msun", "M_host_msun", "a_host_kpc", "lnLambda", "use_df",
        "f_energy_vtot_over_vesc", "f_tangential_vt_over_vtot",
        "vtotal_kms", "vesc_kms", "vcirc_at_r0_kms", "vtotal_over_vcirc",
        "eps_specific_kms2", "E_total_msun_kms2",
        "h_specific_kpc_kms", "L_total_msun_kpc_kms",
        "eta", "rcirc_equiv_kpc", "r0_kpc", "vr0_kms", "vt0_kms",
        "rperi1_kpc", "t_peri1_gyr", "rapo1_after_p1_kpc", "t_apo1_after_p1_gyr",
        "dt_peri1_to_apo1_gyr", "rapo1_over_rperi1",
        "apo1_within_limit", "status", "invalid_reason",
    ]

    txt_path = folder / "orbit_ic.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for key in keys_for_ic:
            if key in row_dict:
                f.write(f"{key} = {scalar_to_str(row_dict[key])}\n")

    pd.DataFrame([row_dict]).to_csv(folder / "orbit_ic.csv", index=False)
    with open(folder / "orbit_ic.json", "w", encoding="utf-8") as f:
        json.dump(row_dict, f, indent=2, allow_nan=True)

    return folder


def write_selected_ic_folders(
    df: pd.DataFrame,
    outdir: str | Path = "IC_ORBITS_GRID",
    status: str = "selected",
    max_count: Optional[int] = None,
) -> List[Path]:
    """Write one IC folder per selected row."""
    selected = df[df["status"] == status].copy()
    if max_count is not None:
        selected = selected.iloc[:max_count]
    ensure_dir(outdir)

    folders: List[Path] = []
    for _, row in selected.iterrows():
        folders.append(write_ic_folder(row, outdir=outdir, orbit_name=row["orbit_id"]))
    return folders


# =============================================================================
# Accepted / best IC display helpers
# =============================================================================


def accepted_ics_table(
    df: pd.DataFrame,
    status: str = "selected",
    sort_by: Sequence[str] = ("mass_label", "energy_label", "L_label"),
    round_digits: Optional[int] = 4,
    include_orbital_summary: bool = True,
) -> pd.DataFrame:
    """Return a clean table of the accepted initial conditions.

    This is the most convenient notebook-level table to inspect before writing
    Gadget4 IC folders. It reports the initial position, initial velocity,
    specific/total orbital energy, and specific/total angular momentum.

    Parameters
    ----------
    df:
        Output from run_orbit_grid.
    status:
        Which rows to display. The default, "selected", corresponds to accepted
        rows whose first post-pericenter apocenter occurs within apo_max_gyr.
    sort_by:
        Columns used to order the table.
    round_digits:
        If not None, numerical columns are rounded for display only.
    include_orbital_summary:
        If True, append first-pericenter and first-apocenter diagnostics.

    Returns
    -------
    pandas.DataFrame
        A display-ready table. The original df is not modified.
    """
    if "status" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'status' column. Did you run run_orbit_grid?")

    out = df[df["status"] == status].copy()
    if out.empty:
        # Return an empty table with useful columns rather than failing.
        return pd.DataFrame(columns=[
            "grid_index", "orbit_id", "status", "mass_label", "energy_label", "L_label",
            "x0_kpc", "y0_kpc", "z0_kpc", "vx0_kms", "vy0_kms", "vz0_kms",
            "eps_specific_kms2", "E_total_msun_kms2",
            "h_specific_kpc_kms", "L_total_msun_kpc_kms",
        ])

    base_cols = [
        "grid_index", "orbit_id", "status",
        "mass_label", "energy_label", "L_label",
        "M_sat_msun",
        "f_energy_vtot_over_vesc", "f_tangential_vt_over_vtot",
        "vtotal_kms", "vesc_kms", "vcirc_at_r0_kms",
        "eta", "rcirc_equiv_kpc",
        "x0_kpc", "y0_kpc", "z0_kpc",
        "vx0_kms", "vy0_kms", "vz0_kms",
        "r0_kpc", "vr0_kms", "vt0_kms",
        "eps_specific_kms2", "E_total_msun_kms2",
        "h_specific_kpc_kms", "L_total_msun_kpc_kms",
    ]
    summary_cols = [
        "rperi1_kpc", "t_peri1_gyr",
        "rapo1_after_p1_kpc", "t_apo1_after_p1_gyr",
        "rapo1_over_rperi1",
    ]
    cols = base_cols + (summary_cols if include_orbital_summary else [])
    cols = [c for c in cols if c in out.columns]

    valid_sort_cols = [c for c in sort_by if c in out.columns]
    if valid_sort_cols:
        out = out.sort_values(valid_sort_cols)

    out = out[cols].reset_index(drop=True)
    if round_digits is not None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].round(round_digits)
    return out


def best_ics_table(
    df: pd.DataFrame,
    status: str = "selected",
    top_n: Optional[int] = None,
    target_rperi_kpc: Optional[float] = None,
    target_tperi_gyr: Optional[float] = None,
    target_rapo_kpc: Optional[float] = None,
    weights: Mapping[str, float] = None,
    round_digits: Optional[int] = 4,
) -> pd.DataFrame:
    """Rank accepted ICs by optional target orbital properties.

    If no target is provided, this simply returns accepted_ics_table. If one or
    more targets are provided, the function adds a dimensionless 'score' column
    and sorts from best to worst. Lower score is better.

    Examples
    --------
    # Just list all accepted ICs:
    best_ics_table(df)

    # Prefer accepted ICs with first pericenter around 40 kpc:
    best_ics_table(df, target_rperi_kpc=40)

    # Prefer first pericenter around 40 kpc and first apocenter around 180 kpc:
    best_ics_table(df, target_rperi_kpc=40, target_rapo_kpc=180)
    """
    if weights is None:
        weights = {"rperi": 1.0, "tperi": 1.0, "rapo": 1.0}

    out = df[df["status"] == status].copy()
    if out.empty:
        return accepted_ics_table(df, status=status, round_digits=round_digits)

    targets = []
    if target_rperi_kpc is not None:
        targets.append(("rperi1_kpc", float(target_rperi_kpc), float(weights.get("rperi", 1.0))))
    if target_tperi_gyr is not None:
        targets.append(("t_peri1_gyr", float(target_tperi_gyr), float(weights.get("tperi", 1.0))))
    if target_rapo_kpc is not None:
        targets.append(("rapo1_after_p1_kpc", float(target_rapo_kpc), float(weights.get("rapo", 1.0))))

    if targets:
        score = np.zeros(len(out), dtype=float)
        for col, target, weight in targets:
            if col not in out.columns:
                raise ValueError(f"Cannot use target for '{col}' because this column is missing.")
            denom = max(abs(target), 1.0e-12)
            score += weight * ((out[col].astype(float).to_numpy() - target) / denom) ** 2
        out["score"] = np.sqrt(score)
        out = out.sort_values("score")
    else:
        out = out.sort_values([c for c in ("mass_label", "energy_label", "L_label") if c in out.columns])

    if top_n is not None:
        out = out.iloc[:int(top_n)]

    table = accepted_ics_table(
        out,
        status=status,
        sort_by=("score",) if "score" in out.columns else ("mass_label", "energy_label", "L_label"),
        round_digits=round_digits,
        include_orbital_summary=True,
    )
    if "score" in out.columns and "score" not in table.columns:
        # Insert score after status while preserving the accepted_ics_table layout.
        scores = out.reset_index(drop=True)["score"].round(round_digits) if round_digits is not None else out.reset_index(drop=True)["score"]
        table.insert(3, "score", scores.to_numpy())
    return table


def save_accepted_ics_table(
    df: pd.DataFrame,
    filename: str | Path = "accepted_initial_conditions.csv",
    status: str = "selected",
    round_digits: Optional[int] = None,
) -> Path:
    """Save accepted ICs to a CSV table and return the output path."""
    path = Path(filename)
    ensure_dir(path.parent if str(path.parent) != "." else ".")
    table = accepted_ics_table(df, status=status, round_digits=round_digits)
    table.to_csv(path, index=False)
    return path



G_KPC_KMS2_MSUN = 4.30091e-6  # kpc (km/s)^2 / Msun


def phi_hernquist_kms2(r, M_host_msun=1.0e12, a_host_kpc=30.0):
    return -G_KPC_KMS2_MSUN * M_host_msun / (r + a_host_kpc)


def radial_velocity_squared_kms2(
    r,
    eps_specific_kms2,
    h_specific_kpc_kms,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    phi = phi_hernquist_kms2(r, M_host_msun, a_host_kpc)
    return 2.0 * (eps_specific_kms2 - phi) - (h_specific_kpc_kms / r) ** 2


def find_turning_points_from_Eh(
    eps_specific_kms2,
    h_specific_kpc_kms,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    rmin_kpc=1.0e-3,
    rmax_kpc=5000.0,
    nscan=20000,
):
    """
    Find rperi and rapo for a bound orbit in a Hernquist potential.
    """
    rgrid = np.logspace(np.log10(rmin_kpc), np.log10(rmax_kpc), nscan)

    vals = np.array([
        radial_velocity_squared_kms2(
            r,
            eps_specific_kms2,
            h_specific_kpc_kms,
            M_host_msun,
            a_host_kpc,
        )
        for r in rgrid
    ])

    roots = []

    for i in range(len(rgrid) - 1):
        f0 = vals[i]
        f1 = vals[i + 1]

        if not np.isfinite(f0) or not np.isfinite(f1):
            continue

        if f0 == 0:
            roots.append(rgrid[i])

        if f0 * f1 < 0:
            root = brentq(
                lambda rr: radial_velocity_squared_kms2(
                    rr,
                    eps_specific_kms2,
                    h_specific_kpc_kms,
                    M_host_msun,
                    a_host_kpc,
                ),
                rgrid[i],
                rgrid[i + 1],
            )
            roots.append(root)

    roots = np.array(sorted(roots))

    if len(roots) < 2:
        raise ValueError(
            f"No bound orbit found for eps={eps_specific_kms2} and h={h_specific_kpc_kms}."
        )

    return roots[0], roots[-1]


def ic_from_exact_Eh(
    eps_specific_kms2,
    h_specific_kpc_kms,
    M_sat_msun,
    mass_label="massive",
    energy_label=None,
    L_label=None,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    theta_ref_xy=(150.0, 80.0),
    start_fraction_from_peri=0.85,
    inward=True,
    prograde=True,
):
    """
    Build one IC from exact epsilon and h.

    start_fraction_from_peri = 0.0 gives r0 = rperi
    start_fraction_from_peri = 1.0 gives r0 = rapo

    A value around 0.80--0.90 starts the satellite near apocentre,
    but still with non-zero inward radial velocity.
    """

    rperi, rapo = find_turning_points_from_Eh(
        eps_specific_kms2,
        h_specific_kpc_kms,
        M_host_msun,
        a_host_kpc,
    )

    f = start_fraction_from_peri
    r0 = rperi + f * (rapo - rperi)

    phi0 = phi_hernquist_kms2(r0, M_host_msun, a_host_kpc)

    vt0 = h_specific_kpc_kms / r0
    if not prograde:
        vt0 *= -1.0

    vr2 = 2.0 * (eps_specific_kms2 - phi0) - vt0**2

    if vr2 < 0:
        raise ValueError(
            f"No real velocity at r0={r0:.3f} kpc. "
            f"Computed vr2={vr2:.6e}."
        )

    vr0 = np.sqrt(vr2)
    if inward:
        vr0 *= -1.0

    x_ref, y_ref = theta_ref_xy
    theta = np.arctan2(y_ref, x_ref)

    er = np.array([np.cos(theta), np.sin(theta)])
    et = np.array([-np.sin(theta), np.cos(theta)])

    x0, y0 = r0 * er
    vx0, vy0 = vr0 * er + vt0 * et

    return {
        "mass_label": mass_label,
        "energy_label": energy_label,
        "L_label": L_label,

        "M_sat_msun": M_sat_msun,
        "M_host_msun": M_host_msun,
        "a_host_kpc": a_host_kpc,

        "x0_kpc": x0,
        "y0_kpc": y0,
        "z0_kpc": 0.0,

        "vx0_kms": vx0,
        "vy0_kms": vy0,
        "vz0_kms": 0.0,

        "r0_kpc": r0,
        "theta0_deg": np.degrees(theta),

        "vr0_kms": vr0,
        "vt0_kms": vt0,
        "vtotal_kms": np.sqrt(vr0**2 + vt0**2),

        "eps_specific_kms2": eps_specific_kms2,
        "h_specific_kpc_kms": h_specific_kpc_kms,

        "E_total_msun_kms2": M_sat_msun * eps_specific_kms2,
        "L_total_msun_kpc_kms": M_sat_msun * h_specific_kpc_kms,

        "rperi_target_kpc": rperi,
        "rapo_target_kpc": rapo,
        "start_fraction_from_peri": start_fraction_from_peri,
    }


def build_exact_Eh_grid(
    eps_grid,
    h_grid,
    M_sat_msun,
    mass_label="massive",
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    theta_ref_xy=(150.0, 80.0),
    start_fraction_from_peri=0.85,
):
    rows = []

    for energy_label, eps in eps_grid.items():
        for L_label, h in h_grid.items():
            row = ic_from_exact_Eh(
                eps_specific_kms2=eps,
                h_specific_kpc_kms=h,
                M_sat_msun=M_sat_msun,
                mass_label=mass_label,
                energy_label=energy_label,
                L_label=L_label,
                M_host_msun=M_host_msun,
                a_host_kpc=a_host_kpc,
                theta_ref_xy=theta_ref_xy,
                start_fraction_from_peri=start_fraction_from_peri,
            )
            rows.append(row)

    return pd.DataFrame(rows)

def trajetory_angle(r, v):
    """
    Calcula o ângulo de trajetória de voo (gamma) em graus.
    r: vetor posição [x, y, z]
    v: vetor velocidade [vx, vy, vz]
    """
    # Converte para arrays do numpy
    r_vec = np.array(r)
    v_vec = np.array(v)
    
    # Calcula o produto escalar
    produto_escalar = np.dot(r_vec, v_vec)
    
    # Calcula as magnitudes (normas) dos vetores
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)
    
    # Calcula o seno de gamma
    sin_gamma = produto_escalar / (r_norm * v_norm)
    
    # Garante que o valor fique entre -1 e 1 (evita erros numéricos)
    sin_gamma = np.clip(sin_gamma, -1.0, 1.0)
    
    # Calcula o arco seno em radianos e converte para graus
    gamma_rad = np.arcsin(sin_gamma)
    gamma_deg = np.degrees(gamma_rad)
    
    return gamma_deg


def ic_from_exact_Eh_fixed_position(
    eps_specific_kms2,
    h_specific_kpc_kms,
    x0_kpc,
    y0_kpc,
    M_sat_msun,
    mass_label="massive",
    energy_label=None,
    L_label=None,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    inward=True,
    prograde=True,
):
    """
    Build one IC from exact epsilon and h, but fixing x0 and y0.

    This is only possible if the chosen fixed radius lies inside the allowed
    radial range of the orbit, i.e. if vr2 >= 0.
    """

    r0 = np.sqrt(x0_kpc**2 + y0_kpc**2)

    rperi, rapo = find_turning_points_from_Eh(
        eps_specific_kms2,
        h_specific_kpc_kms,
        M_host_msun=M_host_msun,
        a_host_kpc=a_host_kpc,
    )

    phi0 = phi_hernquist_kms2(r0, M_host_msun, a_host_kpc)

    vt0 = h_specific_kpc_kms / r0
    if not prograde:
        vt0 *= -1.0

    vr2 = 2.0 * (eps_specific_kms2 - phi0) - vt0**2

    if vr2 < 0:
        return {
            "mass_label": mass_label,
            "energy_label": energy_label,
            "L_label": L_label,
            "status": "no_fixed_position_solution",
            "invalid_reason": (
                f"No real velocity at fixed r0={r0:.3f} kpc. "
                f"Allowed range is rperi={rperi:.3f} to rapo={rapo:.3f} kpc. "
                f"Computed vr2={vr2:.6e}."
            ),
            "M_sat_msun": M_sat_msun,
            "M_host_msun": M_host_msun,
            "a_host_kpc": a_host_kpc,
            "x0_kpc": x0_kpc,
            "y0_kpc": y0_kpc,
            "z0_kpc": 0.0,
            "r0_kpc": r0,
            "eps_specific_kms2": eps_specific_kms2,
            "h_specific_kpc_kms": h_specific_kpc_kms,
            "rperi_target_kpc": rperi,
            "rapo_target_kpc": rapo,
        }

    vr0 = np.sqrt(vr2)
    if inward:
        vr0 *= -1.0

    er = np.array([x0_kpc / r0, y0_kpc / r0])
    et = np.array([-y0_kpc / r0, x0_kpc / r0])

    vx0, vy0 = vr0 * er + vt0 * et

    start_fraction_actual = (r0 - rperi) / (rapo - rperi)

    return {
        "mass_label": mass_label,
        "energy_label": energy_label,
        "L_label": L_label,
        "status": "selected",

        "M_sat_msun": M_sat_msun,
        "M_host_msun": M_host_msun,
        "a_host_kpc": a_host_kpc,

        "x0_kpc": x0_kpc,
        "y0_kpc": y0_kpc,
        "z0_kpc": 0.0,

        "vx0_kms": vx0,
        "vy0_kms": vy0,
        "vz0_kms": 0.0,
        
        "trajetory_angle": trajetory_angle(np.array([x0_kpc, y0_kpc, 0]), np.array([vx0, vy0, 0])),

        "r0_kpc": r0,
        "theta0_deg": np.degrees(np.arctan2(y0_kpc, x0_kpc)),

        "vr0_kms": vr0,
        "vt0_kms": vt0,
        "vtotal_kms": np.sqrt(vr0**2 + vt0**2),

        "eps_specific_kms2": eps_specific_kms2,
        "h_specific_kpc_kms": h_specific_kpc_kms,

        "E_total_msun_kms2": M_sat_msun * eps_specific_kms2,
        "L_total_msun_kpc_kms": M_sat_msun * h_specific_kpc_kms,

        "rperi_target_kpc": rperi,
        "rapo_target_kpc": rapo,
        "start_fraction_actual": start_fraction_actual,
    }


def build_exact_Eh_grid_fixed_position(
    eps_grid,
    h_grid,
    x0_kpc,
    y0_kpc,
    M_sat_msun,
    mass_label="massive",
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    inward=True,
    prograde=True,
):
    rows = []

    for energy_label, eps in eps_grid.items():
        for L_label, h in h_grid.items():
            row = ic_from_exact_Eh_fixed_position(
                eps_specific_kms2=eps,
                h_specific_kpc_kms=h,
                x0_kpc=x0_kpc,
                y0_kpc=y0_kpc,
                M_sat_msun=M_sat_msun,
                mass_label=mass_label,
                energy_label=energy_label,
                L_label=L_label,
                M_host_msun=M_host_msun,
                a_host_kpc=a_host_kpc,
                inward=inward,
                prograde=prograde,
            )
            rows.append(row)

    return pd.DataFrame(rows)

# =============================================================================
# V6 add-on: fixed-position E-h grids with physically meaningful angular labels
# =============================================================================
# These definitions intentionally appear after the older manual helpers.  When the
# module is imported, Python keeps the last definition of functions with the same
# name, so build_exact_Eh_grid_fixed_position and ic_from_exact_Eh_fixed_position
# below replace the simpler v5 versions while preserving backward compatibility.


def v_circ_hernquist_kms(r_kpc, M_host_msun=1.0e12, a_host_kpc=30.0):
    """Circular speed in km/s for the Hernquist potential."""
    r_kpc = float(r_kpc)
    if r_kpc <= 0:
        return np.nan
    return np.sqrt(G_KPC_KMS2_MSUN * M_host_msun * r_kpc / (r_kpc + a_host_kpc) ** 2)


def eps_circ_hernquist_kms2(r_kpc, M_host_msun=1.0e12, a_host_kpc=30.0):
    """Specific energy of the circular orbit at radius r, in (km/s)^2."""
    vc = v_circ_hernquist_kms(r_kpc, M_host_msun, a_host_kpc)
    return 0.5 * vc * vc + phi_hernquist_kms2(r_kpc, M_host_msun, a_host_kpc)


def h_circ_hernquist_kpckms(r_kpc, M_host_msun=1.0e12, a_host_kpc=30.0):
    """Specific angular momentum of the circular orbit at r, in kpc km/s."""
    return float(r_kpc) * v_circ_hernquist_kms(r_kpc, M_host_msun, a_host_kpc)


def rcirc_from_eps_hernquist_kms2(
    eps_specific_kms2,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    r_min_kpc=1.0e-5,
    r_max_kpc=1.0e7,
):
    """Radius of the circular orbit with the same specific energy.

    Returns np.nan if eps is not bound or outside the circular-orbit range of
    the adopted Hernquist model.
    """
    eps = float(eps_specific_kms2)
    if not np.isfinite(eps) or eps >= 0:
        return np.nan

    def f(rr):
        return eps_circ_hernquist_kms2(rr, M_host_msun, a_host_kpc) - eps

    lo = float(r_min_kpc)
    hi = float(r_max_kpc)
    flo = f(lo)
    fhi = f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan
    return brentq(f, lo, hi, maxiter=300)


def h_circ_from_eps_hernquist_kpckms(eps_specific_kms2, M_host_msun=1.0e12, a_host_kpc=30.0):
    """Circular angular momentum h_circ(eps), in kpc km/s."""
    rc = rcirc_from_eps_hernquist_kms2(eps_specific_kms2, M_host_msun, a_host_kpc)
    if not np.isfinite(rc):
        return np.nan
    return h_circ_hernquist_kpckms(rc, M_host_msun, a_host_kpc)


def fixed_position_speed_from_eps_kms(
    eps_specific_kms2,
    r0_kpc,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    """Total speed required by eps at fixed r0, in km/s."""
    phi0 = phi_hernquist_kms2(r0_kpc, M_host_msun, a_host_kpc)
    v2 = 2.0 * (float(eps_specific_kms2) - phi0)
    if not np.isfinite(v2) or v2 < 0:
        return np.nan
    return np.sqrt(max(0.0, v2))


def h_max_at_fixed_position_kpckms(
    eps_specific_kms2,
    r0_kpc,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    """Maximum |h| possible at fixed r0 and eps.

    This is obtained for vr=0, i.e. a locally tangential velocity at the chosen
    starting point. It is not necessarily the angular momentum of a circular
    orbit unless eps == eps_circ(r0).
    """
    vtot = fixed_position_speed_from_eps_kms(eps_specific_kms2, r0_kpc, M_host_msun, a_host_kpc)
    if not np.isfinite(vtot):
        return np.nan
    return float(r0_kpc) * vtot


def flight_path_angle_deg(r, v):
    """Signed flight-path angle gamma in degrees.

    gamma = asin(v_r / |v|).  Therefore:
      * gamma = -90 deg: purely radial inward
      * gamma =   0 deg: locally tangential
      * gamma = +90 deg: purely radial outward

    This is the same mathematical convention used by the older misspelled
    trajetory_angle helper, but with a clearer name.
    """
    r_vec = np.asarray(r, dtype=float)
    v_vec = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)
    if r_norm <= 0 or v_norm <= 0:
        return np.nan
    sin_gamma = np.dot(r_vec, v_vec) / (r_norm * v_norm)
    sin_gamma = np.clip(sin_gamma, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_gamma)))


def angle_between_r_and_v_deg(r, v):
    """Unsigned angle between the position and velocity vectors, in degrees.

    Pure inward radial motion gives 180 deg, pure outward radial motion gives
    0 deg, and a locally tangential/circular-like velocity gives 90 deg.
    """
    r_vec = np.asarray(r, dtype=float)
    v_vec = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)
    if r_norm <= 0 or v_norm <= 0:
        return np.nan
    cos_alpha = np.dot(r_vec, v_vec) / (r_norm * v_norm)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_alpha)))


# Backward-compatible alias for notebooks that already use the typo.
def trajetory_angle(r, v):  # noqa: D401 - keep the old public name
    """Backward-compatible alias of flight_path_angle_deg."""
    return flight_path_angle_deg(r, v)


def _signed_vt_from_h(h_specific_kpc_kms, r0_kpc, prograde=True):
    vt_abs = abs(float(h_specific_kpc_kms)) / float(r0_kpc)
    return vt_abs if prograde else -vt_abs


def _target_h_from_angular_mode(
    angular_grid_mode,
    target_value,
    eps_specific_kms2,
    r0_kpc,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    """Convert a human-readable angular target into h at fixed r0 and eps.

    Supported angular_grid_mode values
    ----------------------------------
    'h' or 'absolute_h':
        target_value is already h in kpc km/s.
    'local_tangential_fraction' or 'f_t':
        target_value = |v_t| / |v| at the initial point. This is usually the
        most transparent choice if the initial position is fixed.
    'flight_angle' or 'gamma':
        target_value is the desired signed inward flight-path angle in degrees.
        For inward orbits use negative values, e.g. -80, -45, -10.
    'circularity_eta' or 'eta':
        target_value = h / h_circ(eps). This is a true orbital circularity, but
        it may not be feasible at the chosen fixed position.
    """
    mode = str(angular_grid_mode).lower()
    vtot = fixed_position_speed_from_eps_kms(eps_specific_kms2, r0_kpc, M_host_msun, a_host_kpc)
    if not np.isfinite(vtot):
        return np.nan, {"invalid_reason_angular_mode": "eps gives no real speed at the fixed r0."}

    if mode in {"h", "absolute_h", "h_specific", "h_specific_kpc_kms"}:
        h = abs(float(target_value))
        meta = {"h_target_input_kpc_kms": h}
    elif mode in {"local_tangential_fraction", "tangential_fraction", "f_t", "ft", "vt_over_v"}:
        f_t = float(target_value)
        if not (0.0 <= f_t <= 1.0):
            return np.nan, {"invalid_reason_angular_mode": "local tangential fraction must satisfy 0 <= f_t <= 1."}
        h = float(r0_kpc) * vtot * f_t
        gamma_abs = np.degrees(np.arccos(np.clip(f_t, 0.0, 1.0)))
        meta = {
            "target_local_tangential_fraction": f_t,
            "target_flight_path_angle_deg": -float(gamma_abs),
        }
    elif mode in {"flight_angle", "gamma", "flight_path_angle"}:
        gamma = float(target_value)
        # gamma is asin(vr/v).  The tangential fraction is cos(gamma).
        if abs(gamma) > 90.0:
            return np.nan, {"invalid_reason_angular_mode": "flight angle must be between -90 and +90 deg."}
        f_t = float(np.cos(np.radians(gamma)))
        h = float(r0_kpc) * vtot * f_t
        meta = {
            "target_flight_path_angle_deg": gamma,
            "target_local_tangential_fraction": f_t,
        }
    elif mode in {"circularity_eta", "eta", "h_over_hcirc_e"}:
        eta = float(target_value)
        h_circ_eps = h_circ_from_eps_hernquist_kpckms(eps_specific_kms2, M_host_msun, a_host_kpc)
        if not np.isfinite(h_circ_eps):
            return np.nan, {"invalid_reason_angular_mode": "could not compute h_circ(eps)."}
        h = eta * h_circ_eps
        meta = {
            "target_eta": eta,
            "h_circ_eps_kpc_kms": h_circ_eps,
        }
    else:
        raise ValueError(
            "angular_grid_mode must be one of: 'h', 'local_tangential_fraction', "
            "'flight_angle', or 'circularity_eta'."
        )

    hmax = h_max_at_fixed_position_kpckms(eps_specific_kms2, r0_kpc, M_host_msun, a_host_kpc)
    meta.update({
        "angular_grid_mode": angular_grid_mode,
        "h_max_fixed_r0_kpc_kms": hmax,
        "h_over_hmax_fixed_r0": h / hmax if np.isfinite(hmax) and hmax > 0 else np.nan,
    })
    return h, meta


def _value_from_simple_or_nested_grid(grid, energy_label, L_label):
    """Read either grid[L_label] or grid[energy_label][L_label]."""
    if isinstance(grid, Mapping) and energy_label in grid and isinstance(grid[energy_label], Mapping):
        return grid[energy_label][L_label]
    return grid[L_label]


def _L_labels_from_simple_or_nested_grid(grid, energy_label):
    if isinstance(grid, Mapping) and energy_label in grid and isinstance(grid[energy_label], Mapping):
        return list(grid[energy_label].keys())
    return list(grid.keys())


def fixed_position_Eh_diagnostic_table(
    eps_grid,
    x0_kpc,
    y0_kpc,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    """Diagnostic table for deciding which h values are feasible at fixed r0."""
    r0 = float(np.hypot(x0_kpc, y0_kpc))
    rows = []
    for energy_label, eps in eps_grid.items():
        vtot = fixed_position_speed_from_eps_kms(eps, r0, M_host_msun, a_host_kpc)
        hmax = h_max_at_fixed_position_kpckms(eps, r0, M_host_msun, a_host_kpc)
        rcirc = rcirc_from_eps_hernquist_kms2(eps, M_host_msun, a_host_kpc)
        hcirc_eps = h_circ_from_eps_hernquist_kpckms(eps, M_host_msun, a_host_kpc)
        rows.append({
            "energy_label": energy_label,
            "eps_specific_kms2": eps,
            "x0_kpc": x0_kpc,
            "y0_kpc": y0_kpc,
            "r0_kpc": r0,
            "phi0_kms2": phi_hernquist_kms2(r0, M_host_msun, a_host_kpc),
            "vtotal_from_E_at_r0_kms": vtot,
            "h_max_fixed_r0_kpc_kms": hmax,
            "rcirc_equiv_kpc": rcirc,
            "h_circ_eps_kpc_kms": hcirc_eps,
            "hmax_over_hcirc_eps": hmax / hcirc_eps if np.isfinite(hmax) and np.isfinite(hcirc_eps) and hcirc_eps > 0 else np.nan,
            "eps_circ_at_r0_kms2": eps_circ_hernquist_kms2(r0, M_host_msun, a_host_kpc),
            "h_circ_at_r0_kpc_kms": h_circ_hernquist_kpckms(r0, M_host_msun, a_host_kpc),
        })
    return pd.DataFrame(rows)


def make_h_grid_fixed_position_from_tangential_fractions(
    eps_grid,
    tangential_fraction_grid,
    x0_kpc,
    y0_kpc,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    """Return a nested h-grid h_grid[energy_label][L_label].

    This is useful when you want the labels 'radial', 'mid', and 'high-L' to
    mean the same local velocity geometry at the fixed starting point for every
    energy.  The exact h values are allowed to vary with energy.
    """
    r0 = float(np.hypot(x0_kpc, y0_kpc))
    out = {}
    for e_label, eps in eps_grid.items():
        out[e_label] = {}
        vtot = fixed_position_speed_from_eps_kms(eps, r0, M_host_msun, a_host_kpc)
        for l_label, f_t in tangential_fraction_grid.items():
            out[e_label][l_label] = r0 * vtot * float(f_t)
    return out


def make_h_grid_fixed_position_from_flight_angles(
    eps_grid,
    flight_angle_grid_deg,
    x0_kpc,
    y0_kpc,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
):
    """Return nested h-grid using target flight-path angles at fixed r0.

    Example: {'radial': -80, 'intermediate': -45, 'near_tangential': -10}.
    """
    r0 = float(np.hypot(x0_kpc, y0_kpc))
    out = {}
    for e_label, eps in eps_grid.items():
        out[e_label] = {}
        vtot = fixed_position_speed_from_eps_kms(eps, r0, M_host_msun, a_host_kpc)
        for l_label, gamma in flight_angle_grid_deg.items():
            f_t = np.cos(np.radians(float(gamma)))
            out[e_label][l_label] = r0 * vtot * f_t
    return out


def ic_from_exact_Eh_fixed_position(
    eps_specific_kms2,
    h_specific_kpc_kms,
    x0_kpc,
    y0_kpc,
    M_sat_msun,
    mass_label="massive",
    energy_label=None,
    L_label=None,
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    inward=True,
    prograde=True,
    angular_grid_mode="h",
    angular_target_value=None,
    extra_metadata=None,
    tolerance_vr2=-1.0e-9,
):
    """Build one IC from exact eps and h while fixing x0 and y0.

    The velocity is decomposed as

        v = v_r e_r + v_t e_t,

    at the fixed position.  A row is only physically possible if

        v_r^2 = 2 [eps - Phi(r0)] - (h/r0)^2 >= 0.

    Important interpretation
    ------------------------
    At fixed r0, the same absolute h value does *not* represent the same orbit
    shape for different energies, because |v| changes with eps.  The columns
    local_tangential_fraction and flight_path_angle_deg should therefore be used
    to check whether the labels radial/intermediate/high-L mean what you want.
    """
    r0 = float(np.hypot(x0_kpc, y0_kpc))
    phi0 = phi_hernquist_kms2(r0, M_host_msun, a_host_kpc)
    vtot2 = 2.0 * (float(eps_specific_kms2) - phi0)
    vtot = np.sqrt(max(0.0, vtot2)) if np.isfinite(vtot2) and vtot2 >= 0 else np.nan
    h_abs = abs(float(h_specific_kpc_kms)) if np.isfinite(h_specific_kpc_kms) else np.nan
    vt0 = _signed_vt_from_h(h_abs, r0, prograde=prograde) if np.isfinite(h_abs) and r0 > 0 else np.nan
    vr2 = vtot2 - vt0**2 if np.isfinite(vtot2) and np.isfinite(vt0) else np.nan

    rcirc_equiv = rcirc_from_eps_hernquist_kms2(eps_specific_kms2, M_host_msun, a_host_kpc)
    h_circ_eps = h_circ_from_eps_hernquist_kpckms(eps_specific_kms2, M_host_msun, a_host_kpc)
    hmax = h_max_at_fixed_position_kpckms(eps_specific_kms2, r0, M_host_msun, a_host_kpc)

    base = {
        "mass_label": mass_label,
        "energy_label": energy_label,
        "L_label": L_label,
        "M_sat_msun": M_sat_msun,
        "M_host_msun": M_host_msun,
        "a_host_kpc": a_host_kpc,
        "x0_kpc": x0_kpc,
        "y0_kpc": y0_kpc,
        "z0_kpc": 0.0,
        "r0_kpc": r0,
        "theta0_deg": float(np.degrees(np.arctan2(y0_kpc, x0_kpc))),
        "eps_specific_kms2": float(eps_specific_kms2),
        "h_specific_kpc_kms": h_abs,
        "E_total_msun_kms2": M_sat_msun * float(eps_specific_kms2),
        "L_total_msun_kpc_kms": M_sat_msun * h_abs if np.isfinite(h_abs) else np.nan,
        "phi0_kms2": phi0,
        "vtotal_from_E_at_r0_kms": vtot,
        "h_max_fixed_r0_kpc_kms": hmax,
        "h_over_hmax_fixed_r0": h_abs / hmax if np.isfinite(h_abs) and np.isfinite(hmax) and hmax > 0 else np.nan,
        "rcirc_equiv_kpc": rcirc_equiv,
        "h_circ_eps_kpc_kms": h_circ_eps,
        "eta": h_abs / h_circ_eps if np.isfinite(h_abs) and np.isfinite(h_circ_eps) and h_circ_eps > 0 else np.nan,
        "eps_circ_at_r0_kms2": eps_circ_hernquist_kms2(r0, M_host_msun, a_host_kpc),
        "h_circ_at_r0_kpc_kms": h_circ_hernquist_kpckms(r0, M_host_msun, a_host_kpc),
        "angular_grid_mode": angular_grid_mode,
        "angular_target_value": angular_target_value,
    }
    if extra_metadata:
        base.update(extra_metadata)

    try:
        rperi, rapo = find_turning_points_from_Eh(
            eps_specific_kms2,
            h_abs,
            M_host_msun=M_host_msun,
            a_host_kpc=a_host_kpc,
        )
        base.update({
            "rperi_target_kpc": rperi,
            "rapo_target_kpc": rapo,
            "start_fraction_actual": (r0 - rperi) / (rapo - rperi) if rapo > rperi else np.nan,
        })
    except Exception as exc:
        base.update({
            "rperi_target_kpc": np.nan,
            "rapo_target_kpc": np.nan,
            "start_fraction_actual": np.nan,
            "turning_point_warning": str(exc),
        })

    if not np.isfinite(vtot2) or vtot2 < 0:
        base.update({
            "status": "no_fixed_position_energy_solution",
            "invalid_reason": (
                f"No real speed at fixed r0={r0:.3f} kpc for eps={eps_specific_kms2:.6g}. "
                f"At this r0, eps must be larger than Phi(r0)={phi0:.6g}."
            ),
        })
        return base

    if not np.isfinite(vr2) or vr2 < tolerance_vr2:
        base.update({
            "status": "no_fixed_position_solution",
            "vx0_kms": np.nan,
            "vy0_kms": np.nan,
            "vz0_kms": 0.0,
            "vr0_kms": np.nan,
            "vt0_kms": vt0,
            "vtotal_kms": vtot,
            "local_tangential_fraction": abs(vt0) / vtot if np.isfinite(vt0) and np.isfinite(vtot) and vtot > 0 else np.nan,
            "flight_path_angle_deg": np.nan,
            "angle_between_r_and_v_deg": np.nan,
            "invalid_reason": (
                f"No real velocity at fixed r0={r0:.3f} kpc: vr^2={vr2:.6e}. "
                f"For this energy, require |h| <= h_max_fixed_r0={hmax:.3f} kpc km/s."
            ),
        })
        return base

    vr0 = np.sqrt(max(0.0, vr2))
    if inward:
        vr0 *= -1.0

    er = np.array([x0_kpc / r0, y0_kpc / r0], dtype=float)
    et = np.array([-y0_kpc / r0, x0_kpc / r0], dtype=float)
    vx0, vy0 = vr0 * er + vt0 * et

    gamma = flight_path_angle_deg(np.array([x0_kpc, y0_kpc, 0.0]), np.array([vx0, vy0, 0.0]))
    alpha = angle_between_r_and_v_deg(np.array([x0_kpc, y0_kpc, 0.0]), np.array([vx0, vy0, 0.0]))

    # Direct reconstruction checks in the same output units.
    eps_check = 0.5 * (vx0 * vx0 + vy0 * vy0) + phi0
    h_check = abs(x0_kpc * vy0 - y0_kpc * vx0)

    base.update({
        "status": "selected",
        "vx0_kms": float(vx0),
        "vy0_kms": float(vy0),
        "vz0_kms": 0.0,
        "vr0_kms": float(vr0),
        "vt0_kms": float(vt0),
        "vtotal_kms": float(np.sqrt(vr0**2 + vt0**2)),
        "local_tangential_fraction": abs(vt0) / np.sqrt(vr0**2 + vt0**2),
        "local_radial_fraction_signed": vr0 / np.sqrt(vr0**2 + vt0**2),
        "flight_path_angle_deg": gamma,
        "trajetory_angle": gamma,  # backward-compatible misspelled column
        "angle_between_r_and_v_deg": alpha,
        "eps_reconstructed_kms2": eps_check,
        "h_reconstructed_kpc_kms": h_check,
        "eps_error_kms2": eps_check - float(eps_specific_kms2),
        "h_error_kpc_kms": h_check - h_abs,
        "invalid_reason": "",
    })
    return base


def build_exact_Eh_grid_fixed_position(
    eps_grid,
    h_grid=None,
    x0_kpc=150.0,
    y0_kpc=80.0,
    M_sat_msun=1.0e9,
    mass_label="massive",
    M_host_msun=1.0e12,
    a_host_kpc=30.0,
    inward=True,
    prograde=True,
    angular_grid_mode="h",
    tangential_fraction_grid=None,
    flight_angle_grid_deg=None,
    circularity_eta_grid=None,
    add_grid_index=True,
):
    """Build a fixed-position grid from exact energy and angular momentum.

    This function is backward-compatible with the older call

        build_exact_Eh_grid_fixed_position(eps_grid, h_grid, x0, y0, ...)

    but it also supports safer angular grids where h is derived separately for
    each energy so that the labels keep a consistent meaning at the fixed
    starting position.

    Recommended modes
    -----------------
    1. angular_grid_mode='local_tangential_fraction'
       Pass tangential_fraction_grid={'L_radial': 0.20, 'L_mid': 0.55,
       'L_tangential': 0.90}. This makes radial/mid/high-L comparable across
       energies because |v_t|/|v| is fixed by label.

    2. angular_grid_mode='flight_angle'
       Pass flight_angle_grid_deg={'radial': -80, 'mid': -45,
       'near_tangential': -10}. This directly controls the reported
       flight_path_angle_deg.

    3. angular_grid_mode='circularity_eta'
       Pass circularity_eta_grid={'eta_low': 0.2, 'eta_mid': 0.5,
       'eta_high': 0.8}. This uses h=eta*h_circ(eps), but some rows can be
       impossible at the chosen fixed position if h > h_max_fixed_r0.

    4. angular_grid_mode='h'
       Pass h_grid={'L_low': 4000, ...} or a nested h_grid[energy][L].
       This is the original behavior.  It is exact, but labels like 'radial'
       are only meaningful after checking local_tangential_fraction or
       flight_path_angle_deg.
    """
    mode = str(angular_grid_mode).lower()

    if mode in {"local_tangential_fraction", "tangential_fraction", "f_t", "ft", "vt_over_v"}:
        active_grid = tangential_fraction_grid if tangential_fraction_grid is not None else h_grid
        if active_grid is None:
            active_grid = {"L_radial": 0.20, "L_mid": 0.55, "L_high_tangential": 0.85}
    elif mode in {"flight_angle", "gamma", "flight_path_angle"}:
        active_grid = flight_angle_grid_deg if flight_angle_grid_deg is not None else h_grid
        if active_grid is None:
            active_grid = {"L_radial": -80.0, "L_mid": -45.0, "L_high_tangential": -15.0}
    elif mode in {"circularity_eta", "eta", "h_over_hcirc_e"}:
        active_grid = circularity_eta_grid if circularity_eta_grid is not None else h_grid
        if active_grid is None:
            active_grid = {"eta_low": 0.20, "eta_mid": 0.50, "eta_high": 0.80}
    elif mode in {"h", "absolute_h", "h_specific", "h_specific_kpc_kms"}:
        active_grid = h_grid
        if active_grid is None:
            raise ValueError("h_grid must be provided when angular_grid_mode='h'.")
    else:
        raise ValueError(
            "angular_grid_mode must be 'h', 'local_tangential_fraction', "
            "'flight_angle', or 'circularity_eta'."
        )

    rows = []
    r0 = float(np.hypot(x0_kpc, y0_kpc))
    counter = 0
    for energy_label, eps in eps_grid.items():
        for L_label in _L_labels_from_simple_or_nested_grid(active_grid, energy_label):
            target_value = _value_from_simple_or_nested_grid(active_grid, energy_label, L_label)
            h, meta = _target_h_from_angular_mode(
                mode,
                target_value,
                eps,
                r0,
                M_host_msun=M_host_msun,
                a_host_kpc=a_host_kpc,
            )
            counter += 1
            row = ic_from_exact_Eh_fixed_position(
                eps_specific_kms2=eps,
                h_specific_kpc_kms=h,
                x0_kpc=x0_kpc,
                y0_kpc=y0_kpc,
                M_sat_msun=M_sat_msun,
                mass_label=mass_label,
                energy_label=energy_label,
                L_label=L_label,
                M_host_msun=M_host_msun,
                a_host_kpc=a_host_kpc,
                inward=inward,
                prograde=prograde,
                angular_grid_mode=mode,
                angular_target_value=target_value,
                extra_metadata=meta,
            )
            row["orbit_id"] = f"{_clean_label(mass_label)}__{_clean_label(energy_label)}__{_clean_label(L_label)}"
            if add_grid_index:
                row["grid_index"] = counter
            rows.append(row)

    return pd.DataFrame(rows)


def accepted_fixed_position_ics_table(df, round_digits=3, status="selected"):
    """Compact display table for the improved fixed-position E-h builder."""
    out = df.copy()
    if status is not None and "status" in out.columns:
        out = out[out["status"] == status].copy()
    cols = [
        "grid_index", "status", "energy_label", "L_label", "angular_grid_mode",
        "angular_target_value", "target_local_tangential_fraction", "target_flight_path_angle_deg",
        "x0_kpc", "y0_kpc", "r0_kpc",
        "vx0_kms", "vy0_kms", "vr0_kms", "vt0_kms", "vtotal_kms",
        "local_tangential_fraction", "flight_path_angle_deg", "angle_between_r_and_v_deg",
        "eps_specific_kms2", "h_specific_kpc_kms", "eta", "h_over_hmax_fixed_r0",
        "rperi_target_kpc", "rapo_target_kpc", "start_fraction_actual", "invalid_reason",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].reset_index(drop=True)
    if round_digits is not None and not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].round(round_digits)
    return out
