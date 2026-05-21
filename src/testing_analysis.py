#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:35:59 2026

@author: abhner
"""
from orbit_satellite_analysis import (
    AnalysisConfig, HostHaloConfig, analyze_all,
    inspect_snapshot_structure, compare_labels
)


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#%%
cfg = AnalysisConfig(
    root="./../SIMULATIONS/ORBIT/HigherRes",
    labels=["E_mid_L_radial", "E_mid_L_mid"], 
    output_dir="orbit_analysis_outputs",

    length_unit_to_kpc=1.0,
    velocity_unit_to_kms=1.0,
    mass_unit_to_msun=1.0e10,  # mude para 1.0 se as massas já estiverem em Msun
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

    make_maps=True,
    make_gifs=True,
    verbose=True,
)

#%%

results = analyze_all(cfg)
compare_labels(results, cfg.output_dir)

#%%


comparison_dir = Path(cfg.output_dir) / "comparison_plots"
comparison_dir.mkdir(parents=True, exist_ok=True)


def log10_safe(values):
    """
    Calcula log10 evitando problemas com zero ou valores negativos.
    Valores <= 0 viram NaN.
    """
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, np.log10(values), np.nan)


def get_time_axis(df):
    """
    Usa tempo em Gyr se estiver disponível; caso contrário usa número do snapshot.
    """
    if "time_gyr" in df.columns and np.all(np.isfinite(df["time_gyr"])):
        return df["time_gyr"].values, "Time [Gyr]"
    else:
        return df["snapshot_number"].values, "Snapshot"


def plot_compare_log_quantity(results, column, ylabel, filename, title=None):
    """
    Compara uma quantidade em log10 entre diferentes simulações.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for label, df in results.items():
        if column not in df.columns:
            print(f"Coluna {column} não encontrada em {label}")
            continue

        x, xlabel = get_time_axis(df)
        y = log10_safe(df[column])

        ax.plot(x, y, marker="o", lw=1.8, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(comparison_dir / filename, dpi=200)
    plt.show()
    
#%%

plot_compare_log_quantity(
    results,
    column="Mstar_inside_rt_msun",
    ylabel=r"$\log_{10}(M_\star(<r_t) / M_\odot)$",
    filename="compare_log10_Mstar_inside_rt.png",
    title="Stellar mass inside tidal radius"
)

plot_compare_log_quantity(
    results,
    column="Mgas_inside_rt_msun",
    ylabel=r"$\log_{10}(M_{\rm gas}(<r_t) / M_\odot)$",
    filename="compare_log10_Mgas_inside_rt.png",
    title="Satellite gas mass inside tidal radius"
)

plot_compare_log_quantity(
    results,
    column="Mdm_inside_rt_msun",
    ylabel=r"$\log_{10}(M_{\rm DM}(<r_t) / M_\odot)$",
    filename="compare_log10_Mdm_inside_rt.png",
    title="Dark matter mass inside tidal radius"
)

plot_compare_log_quantity(
    results,
    column="Msat_inside_rt_msun",
    ylabel=r"$\log_{10}(M_{\rm sat}(<r_t) / M_\odot)$",
    filename="compare_log10_Msat_inside_rt.png",
    title="Total satellite mass inside tidal radius"
)

#%%

plot_compare_log_quantity(
    results,
    column="R_host_kpc",
    ylabel=r"$\log_{10}(R_{\rm host} / {\rm kpc})$",
    filename="compare_log10_Rhost.png",
    title="Orbital radius"
)

plot_compare_log_quantity(
    results,
    column="r_tidal_kpc",
    ylabel=r"$\log_{10}(r_t / {\rm kpc})$",
    filename="compare_log10_rt.png",
    title="Tidal radius"
)

plot_compare_log_quantity(
    results,
    column="rhalf_star_kpc",
    ylabel=r"$\log_{10}(r_{1/2,\star} / {\rm kpc})$",
    filename="compare_log10_rhalf_star.png",
    title="Stellar half-mass radius"
)

plot_compare_log_quantity(
    results,
    column="r90_star_kpc",
    ylabel=r"$\log_{10}(r_{90,\star} / {\rm kpc})$",
    filename="compare_log10_r90_star.png",
    title="Stellar 90 per cent mass radius"
)

#%%

def plot_main_comparison_panel(results, filename="main_comparison_panel.png"):
    quantities = [
        (
            "R_host_kpc",
            r"$\log_{10}(R_{\rm host}/{\rm kpc})$",
            "Orbital radius"
        ),
        (
            "r_tidal_kpc",
            r"$\log_{10}(r_t/{\rm kpc})$",
            "Tidal radius"
        ),
        (
            "rhalf_star_kpc",
            r"$\log_{10}(r_{1/2,\star}/{\rm kpc})$",
            "Stellar half-mass radius"
        ),
        (
            "Msat_inside_rt_msun",
            r"$\log_{10}(M_{\rm sat}(<r_t)/M_\odot)$",
            "Satellite mass inside tidal radius"
        ),
        (
            "Mstar_inside_rt_msun",
            r"$\log_{10}(M_\star(<r_t)/M_\odot)$",
            "Stellar mass inside tidal radius"
        ),
        (
            "Mgas_inside_rt_msun",
            r"$\log_{10}(M_{\rm gas}(<r_t)/M_\odot)$",
            "Gas mass inside tidal radius"
        ),
        (
            "Mdm_inside_rt_msun",
            r"$\log_{10}(M_{\rm DM}(<r_t)/M_\odot)$",
            "DM mass inside tidal radius"
        ),
        (
            "Mstar_inside_rhalf_msun",
            r"$\log_{10}(M_\star(<r_{1/2,\star})/M_\odot)$",
            "Stellar mass inside stellar half-mass radius"
        ),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=False)
    axes = axes.ravel()

    for ax, (column, ylabel, title) in zip(axes, quantities):

        for label, df in results.items():
            if column not in df.columns:
                continue

            x, xlabel = get_time_axis(df)
            y = log10_safe(df[column])

            ax.plot(x, y, marker="o", lw=1.6, label=label)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(comparison_dir / filename, dpi=220)
    plt.show()


plot_main_comparison_panel(results)

#%%

def plot_retained_fraction(results, filename="compare_retained_fractions.png"):

    fractions = [
        ("Mstar_inside_rt_msun", "Mstar_tracked_msun", "stars"),
        ("Mgas_inside_rt_msun", "Mgas_tracked_msun", "gas"),
        ("Mdm_inside_rt_msun", "Mdm_tracked_msun", "DM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, (inside_col, total_col, name) in zip(axes, fractions):

        for label, df in results.items():
            x, xlabel = get_time_axis(df)

            total = np.asarray(df[total_col], dtype=float)
            inside = np.asarray(df[inside_col], dtype=float)

            frac = np.where(total > 0, inside / total, np.nan)

            ax.plot(x, frac, marker="o", lw=1.6, label=label)

        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$M(<r_t)/M_{\rm tracked}$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    axes[0].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(comparison_dir / filename, dpi=220)
    plt.show()


plot_retained_fraction(results)

#%%

plot_compare_log_quantity(
    results,
    "Mgas_inside_rt_msun",
    r"$\log_{10}(M_{\rm gas}(<r_t) / M_\odot)$",
    "compare_log10_Mgas_inside_rt.png",
    "Gas stripping comparison"
)

plot_compare_log_quantity(
    results,
    "Mdm_inside_rt_msun",
    r"$\log_{10}(M_{\rm DM}(<r_t) / M_\odot)$",
    "compare_log10_Mdm_inside_rt.png",
    "Dark matter stripping comparison"
)

plot_compare_log_quantity(
    results,
    "rhalf_star_kpc",
    r"$\log_{10}(r_{1/2,\star} / {\rm kpc})$",
    "compare_log10_rhalf_star.png",
    "Size evolution comparison"
)