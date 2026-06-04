#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:35:59 2026

@author: abhner
"""
from orbit_satellite_analysis_v02 import (
    AnalysisConfig, HostHaloConfig, analyze_all,
    inspect_snapshot_structure, compare_labels
)


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#%%
cfg = AnalysisConfig(
    root="./../SIMULATIONS/ORBIT/HigherRes",
    labels=["E_mid_L_radial", "E_mid_L_mid", "E_mid_L_high"],
    output_dir="orbit_analysis_outputs",

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

    initial_satellite_gas_radius_kpc=None,
    default_initial_gas_radius_kpc=30.0,
    dm_selection_mode="all",

    fixed_aperture_radii_kpc=(1.0, 3.0),

    compute_star_formation=True,
    sfr_unit_to_msun_per_yr=1.0,
    star_forming_sfr_threshold_msun_per_yr=0.0,

    young_star_age_myr=100.0,
    stellar_birth_time_mode="code_time",

    pericentre_reference="first",

    include_all_stars_as_satellite=True,

    make_maps=True,
    make_gifs=True,
    verbose=True,
)

#%%
results = analyze_all(cfg)
compare_labels(results, cfg.output_dir)

#%%

import numpy as np
import pandas as pd

def summarize_orbits(results):
    rows = []

    for label, df in results.items():

        R = np.asarray(df["R_host_kpc"], dtype=float)
        t = np.asarray(df["time_gyr"], dtype=float)

        i_peri = np.nanargmin(R)

        r0 = df[["x_sat_kpc", "y_sat_kpc", "z_sat_kpc"]].iloc[0].values.astype(float)
        v0 = df[["vx_sat_kms", "vy_sat_kms", "vz_sat_kms"]].iloc[0].values.astype(float)

        R0 = np.linalg.norm(r0)
        V0 = np.linalg.norm(v0)

        if R0 > 0:
            rhat0 = r0 / R0
            v_rad0 = np.dot(v0, rhat0)
            v_tan0 = np.sqrt(max(V0**2 - v_rad0**2, 0.0))
        else:
            v_rad0 = np.nan
            v_tan0 = np.nan

        Lvec0 = np.cross(r0, v0)
        L0 = np.linalg.norm(Lvec0)

        rows.append({
            "label": label,
            "R0_kpc": R0,
            "V0_kms": V0,
            "Vrad0_kms": v_rad0,
            "Vtan0_kms": v_tan0,
            "Vtan0_over_V0": v_tan0 / V0 if V0 > 0 else np.nan,
            "L0_kpc_kms": L0,
            "Rperi_kpc": R[i_peri],
            "tperi_gyr": t[i_peri],
        })

    return pd.DataFrame(rows).sort_values("Rperi_kpc")

orbit_summary = summarize_orbits(results)
print(orbit_summary)

#%%
from pathlib import Path

from plot_functions_gadget import (
    load_results_from_output,
    plot_quantity,
    plot_main_comparison_panel,
    plot_mass_budget,
    plot_retained_fraction,
    plot_star_formation_panel,
    plot_ram_pressure_panel,
    plot_standard_gadget_suite,
)

#%%

xmin = -1.5
xmax = 3

#%%
plot_dir = Path(cfg.output_dir) / "gadget_plots"

plot_standard_gadget_suite(
    results,
    outdir=plot_dir,
    phase_xlim=(xmin, xmax),
)

#%%

plot_main_comparison_panel(
    results,
    outdir=plot_dir,
    filename="phase_main_comparison_panel",
    xmode="phase",
    smooth=True,
)

#%%
plot_mass_budget(
    results,
    outdir=plot_dir,
    aperture="rhalf",
    filename="phase_mass_budget_inside_rt",
    xmode="phase",
    smooth=True,
    xlim=(xmin, xmax),

)

#%%
plot_mass_budget(
    results,
    outdir=plot_dir,
    aperture="1kpc",
    filename="phase_mass_budget_inside_rt",
    xmode="phase",
    smooth=True,
    xlim=(xmin, xmax),

)

#%%

def check_projected_vs_3d_pericentre(results):
    rows = []

    for label, df in results.items():
        x = np.asarray(df["x_sat_kpc"], dtype=float)
        y = np.asarray(df["y_sat_kpc"], dtype=float)
        z = np.asarray(df["z_sat_kpc"], dtype=float)
        t = np.asarray(df["time_gyr"], dtype=float)
        R3d = np.asarray(df["R_host_kpc"], dtype=float)

        Rxy = np.sqrt(x**2 + y**2)

        i3d = np.nanargmin(R3d)
        ixy = np.nanargmin(Rxy)

        rows.append({
            "label": label,

            "Rperi_3D_kpc": R3d[i3d],
            "tperi_3D_gyr": t[i3d],
            "x_at_3Dperi": x[i3d],
            "y_at_3Dperi": y[i3d],
            "z_at_3Dperi": z[i3d],
            "Rxy_at_3Dperi_kpc": Rxy[i3d],

            "Rperi_projected_xy_kpc": Rxy[ixy],
            "tperi_projected_xy_gyr": t[ixy],
            "x_at_xyperi": x[ixy],
            "y_at_xyperi": y[ixy],
            "z_at_xyperi": z[ixy],
            "R3D_at_xyperi_kpc": R3d[ixy],
        })

    return pd.DataFrame(rows).sort_values("Rperi_3D_kpc")

check_projected_vs_3d_pericentre(results)

#%%

for label, df in results.items():
    print("\n", label)

    R = np.asarray(df["R_host_kpc"], dtype=float)
    t = np.asarray(df["time_gyr"], dtype=float)

    for i in range(1, len(R) - 1):
        if R[i] < R[i - 1] and R[i] < R[i + 1]:
            print(
                "pericentre candidate:",
                "i =", i,
                "t =", t[i],
                "R =", R[i],
            )
            
#%%

for label, df in results.items():
    print("\n", label)

    print("initial:")
    print(df[["time_gyr", "x_sat_kpc", "y_sat_kpc", "z_sat_kpc",
              "vx_sat_kms", "vy_sat_kms", "vz_sat_kms",
              "R_host_kpc", "V_rad_kms"inner_stars_dm, "V_tan_kms"]].iloc[0])

    print("z range:")
    print("min z =", np.nanmin(df["z_sat_kpc"]))
    print("max z =", np.nanmax(df["z_sat_kpc"]))
    print("max |z| =", np.nanmax(np.abs(df["z_sat_kpc"])))

    print("vz range:")
    print("min vz =", np.nanmin(df["vz_sat_kms"]))
    print("max vz =", np.nanmax(df["vz_sat_kms"]))
    
#%%

import numpy as np
import pandas as pd

G = 4.30091e-6  # kpc (km/s)^2 Msun^-1

def f_nfw(x):
    return np.log1p(x) - x / (1.0 + x)

def M_nfw(R, M200=1e12, R200=210.0, c=10.0):
    x = c * R / R200
    return M200 * f_nfw(x) / f_nfw(c)

def phi_nfw(R, M200=1e12, R200=210.0, c=10.0):
    # Potential with Phi(infinity)=0 for NFW halo normalized by M200.
    rs = R200 / c
    A = M200 / f_nfw(c)
    return -G * A * np.log(1.0 + R / rs) / R

rows = []

for label, df in results.items():
    r0 = df[["x_sat_kpc", "y_sat_kpc", "z_sat_kpc"]].iloc[0].values.astype(float)
    v0 = df[["vx_sat_kms", "vy_sat_kms", "vz_sat_kms"]].iloc[0].values.astype(float)

    R0 = np.linalg.norm(r0)
    V0 = np.linalg.norm(v0)
    L0 = np.linalg.norm(np.cross(r0, v0))
    E0 = 0.5 * V0**2 + phi_nfw(R0)

    rows.append({
        "label": label,
        "R0_kpc": R0,
        "V0_kms": V0,
        "Vrad0_kms": df["V_rad_kms"].iloc[0],
        "Vtan0_kms": df["V_tan_kms"].iloc[0],
        "L0_kpc_kms": L0,
        "E0_km2_s2": E0,
    })

pd.DataFrame(rows).sort_values("L0_kpc_kms")


#%%

# %%
# COMPARE CENTER DEFINITIONS

all_results = {}

for center_mode, velocity_mode in [
    ("stars_com", "stars_com"),
    ("stellar_core", "inner_stars"),
]:
    cfg = AnalysisConfig(
        root="./../SIMULATIONS/ORBIT/HigherRes",
        labels=[
            "E_mid_L_radial",
            "E_mid_L_mid",
            "E_mid_L_high",
        ],
        output_dir=f"orbit_analysis_outputs_{center_mode}",

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

        center_mode=center_mode,
        velocity_mode=velocity_mode,

        make_maps=False,
        make_gifs=False,
        dm_selection_mode="all",
        verbose=True,
    )

    results = analyze_all(cfg)
    all_results[center_mode] = results