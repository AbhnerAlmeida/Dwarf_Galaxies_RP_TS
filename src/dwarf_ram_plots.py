#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dwarf_ram_plots.py

Plotting module for the organized Gadget4 dwarf ram-pressure workflow.

This file is intentionally separated from dwarf_ram_analysis.py:
  - dwarf_ram_analysis.py reads snapshots/tables and computes diagnostics.
  - dwarf_ram_plots.py only makes figures from prepared RunData objects,
    evolution tables, profiles, and yt snapshots.

Notebook-friendly usage
-----------------------
import dwarf_ram_analysis as dra
import dwarf_ram_plots as drp

run = dra.prepare_run("runA/dwarf_ram_pressure_evolution_final.txt", label="A")
drp.plot_summary_evolution(run, outname="summary_A.png")
drp.plot_pericenter_diagnostics(run, outname="peri_A.png")

runs = dra.prepare_runs(["runA/table.txt", "runB/table.txt"], labels=["A", "B"])
drp.make_comparison_plot_set(runs, outdir="compare_plots")

Command-line examples
---------------------
python dwarf_ram_plots.py summary runA/dwarf_ram_pressure_evolution_final.txt --label A
python dwarf_ram_plots.py compare runA/table.txt runB/table.txt --labels A B
python dwarf_ram_plots.py maps /path/to/output --width 80 --axis z --center-mode stars-com
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import dwarf_ram_analysis as dra


# =============================================================================
# Generic plotting helpers
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(fig: plt.Figure, outbase: str, dpi: int = 250, formats: Sequence[str] = ("png", "pdf")) -> List[str]:
    fig.tight_layout()
    written: List[str] = []
    base = str(outbase)
    suffix = Path(base).suffix.lower()
    if suffix in [".png", ".pdf", ".jpg", ".jpeg", ".svg"]:
        plt.show()
        fig.savefig(base, dpi=dpi, bbox_inches="tight")
        written.append(base)
    else:
        for fmt in formats:
            fname = base + "." + fmt
            plt.show()
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            written.append(fname)
    plt.close(fig)
    return written


def _col(run: dra.RunData, name: str, default: Optional[np.ndarray] = None) -> np.ndarray:
    if name in run.cols:
        return np.asarray(run.cols[name], dtype=float)
    if default is not None:
        return default
    return np.full_like(run.t, np.nan, dtype=float)


def _good_xy(x: Sequence[float], y: Sequence[float], positive_y: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if positive_y:
        good &= y > 0
    return good


def safe_plot(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    ylabel: str,
    logy: bool = False,
    tperi: Optional[float] = None,
    label: Optional[str] = None,
    ls: str = "-",
    marker: Optional[str] = "o",
    lw: float = 1.5,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = _good_xy(x, y, positive_y=logy)
    if np.any(good):
        ax.plot(x[good], y[good], marker=marker, lw=lw, ls=ls, label=label)
        if logy:
            ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    if tperi is not None and np.isfinite(tperi):
        ax.axvline(tperi, ls="--", lw=1.2, color="0.4")
    if label is not None:
        ax.legend(frameon=False, fontsize=9)


def add_tau0(ax: plt.Axes) -> None:
    ax.axvline(0.0, ls="--", lw=1.1, color="0.35")


def sanitize_label(label: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(label))
    return out.strip("_") or "run"


# =============================================================================
# Single-run table plots
# =============================================================================

def plot_summary_evolution(run: dra.RunData, outname: str = "summary_ram_pressure_final", use_time: bool = True) -> List[str]:
    """Main diagnostic figure equivalent to the old summary_ram_pressure_final plot."""
    x = run.t if use_time else run.tau
    xlabel = "Time [Gyr]" if use_time else r"$\tau=t-t_{\rm peri}$ [Gyr]"
    tline = run.t_peri if use_time else 0.0

    fig, axes = plt.subplots(6, 2, figsize=(12, 18), sharex=True)
    axes = axes.ravel()

    safe_plot(axes[0], x, _col(run, "dist_host_kpc"), r"$d_{\rm host}\ [{\rm kpc}]$", tperi=tline)

    pram_main_name = "pram_upstream_dyn_cm2" if "pram_upstream_dyn_cm2" in run.cols else "pram_dyn_cm2"
    safe_plot(axes[1], x, _col(run, pram_main_name), r"$P_{\rm ram}\ [{\rm dyn\,cm^{-2}}]$", logy=True, tperi=tline, label="upstream/preferred")
    if "pram_shell_dyn_cm2" in run.cols:
        safe_plot(axes[1], x, _col(run, "pram_shell_dyn_cm2"), r"$P_{\rm ram}\ [{\rm dyn\,cm^{-2}}]$", logy=True, tperi=tline, label="shell", ls=":")

    safe_plot(axes[2], x, _col(run, "tidal_strength_msun_kpc3"), r"$M_{\rm host}(<d)/d^3$", logy=True, tperi=tline)
    safe_plot(axes[3], x, _col(run, "tidal_to_self_star"), r"$a_{\rm tide}/a_{\rm self}$", logy=True, tperi=tline, label="stars")
    if "tidal_to_self_gas" in run.cols:
        safe_plot(axes[3], x, _col(run, "tidal_to_self_gas"), r"$a_{\rm tide}/a_{\rm self}$", logy=True, tperi=tline, label="SF gas", ls=":")

    safe_plot(axes[4], x, _col(run, "sfr_msun_per_yr"), r"${\rm SFR}\ [{\rm M_\odot\,yr^{-1}}]$", logy=True, tperi=tline)
    safe_plot(axes[5], x, _col(run, "f_sfr_central"), "Central / leading SF fractions", tperi=tline, label="central SF")
    if "sfr_leading_frac" in run.cols:
        safe_plot(axes[5], x, _col(run, "sfr_leading_frac"), "Central / leading SF fractions", tperi=tline, label="leading SF", ls=":")

    safe_plot(axes[6], x, _col(run, "rhalf_star_kpc"), "Sizes [kpc]", tperi=tline, label="all stars")
    if "rhalf_old_kpc" in run.cols:
        safe_plot(axes[6], x, _col(run, "rhalf_old_kpc"), "Sizes [kpc]", tperi=tline, label="old stars", ls=":")
    if "rhalf_young_kpc" in run.cols:
        safe_plot(axes[6], x, _col(run, "rhalf_young_kpc"), "Sizes [kpc]", tperi=tline, label="young stars", ls="--")
    if "rhalf_gas_sf_kpc" in run.cols:
        safe_plot(axes[6], x, _col(run, "rhalf_gas_sf_kpc"), "Sizes [kpc]", tperi=tline, label="SF gas", ls="-.")

    safe_plot(axes[7], x, _col(run, "tidal_radius_kpc"), r"$r_{\rm tidal}$ and sizes [kpc]", tperi=tline, label=r"$r_{\rm tidal}$")
    if "rhalf_star_kpc" in run.cols:
        safe_plot(axes[7], x, _col(run, "rhalf_star_kpc"), r"$r_{\rm tidal}$ and sizes [kpc]", tperi=tline, label=r"$R_{1/2,\star}$", ls=":")
    if "rhalf_gas_sf_kpc" in run.cols:
        safe_plot(axes[7], x, _col(run, "rhalf_gas_sf_kpc"), r"$r_{\rm tidal}$ and sizes [kpc]", tperi=tline, label=r"$R_{1/2,{\rm gas,SF}}$", ls="--")

    safe_plot(axes[8], x, _col(run, "rhalf_star_over_rtidal"), r"$R/r_{\rm tidal}$", logy=True, tperi=tline, label="stars")
    if "rhalf_young_over_rtidal" in run.cols:
        safe_plot(axes[8], x, _col(run, "rhalf_young_over_rtidal"), r"$R/r_{\rm tidal}$", logy=True, tperi=tline, label="young", ls="--")
    if "rhalf_gas_sf_over_rtidal" in run.cols:
        safe_plot(axes[8], x, _col(run, "rhalf_gas_sf_over_rtidal"), r"$R/r_{\rm tidal}$", logy=True, tperi=tline, label="SF gas", ls=":")

    safe_plot(axes[9], x, _col(run, "rho_leading_over_trailing"), "Leading-side gas diagnostics", tperi=tline, label=r"$\rho_{\rm lead}/\rho_{\rm trail}$")
    if "mgas_leading_frac" in run.cols:
        safe_plot(axes[9], x, _col(run, "mgas_leading_frac"), "Leading-side gas diagnostics", tperi=tline, label="gas lead frac", ls=":")
    if "mgas_sf_leading_frac" in run.cols:
        safe_plot(axes[9], x, _col(run, "mgas_sf_leading_frac"), "Leading-side gas diagnostics", tperi=tline, label="SF gas lead frac", ls="--")

    safe_plot(axes[10], x, _col(run, "mdwarf_proxy_msun"), r"Mass proxies [$\rm M_\odot$]", logy=True, tperi=tline, label="dwarf proxy")
    if "mhost_enclosed_msun" in run.cols:
        safe_plot(axes[10], x, _col(run, "mhost_enclosed_msun"), r"Mass proxies [$\rm M_\odot$]", logy=True, tperi=tline, label=r"$M_{\rm host}(<d)$", ls=":")
    if "mdm_ap_msun" in run.cols:
        safe_plot(axes[10], x, _col(run, "mdm_ap_msun"), r"Mass proxies [$\rm M_\odot$]", logy=True, tperi=tline, label=r"$M_{\rm DM,ap}$", ls="--")

    safe_plot(axes[11], x, _col(run, "cos_wind_host_radial"), r"$\cos(\hat v_{\rm wind},\hat r_{\rm host})$", tperi=tline)
    if "gas_com_along_upstream_kpc" in run.cols:
        safe_plot(axes[11], x, _col(run, "gas_com_along_upstream_kpc"), "Wind/host alignment and gas offset", tperi=tline, label="gas COM along wind", ls=":")

    axes[-2].set_xlabel(xlabel)
    axes[-1].set_xlabel(xlabel)
    return savefig(fig, outname, dpi=220)


def plot_pericenter_diagnostics(
    run: dra.RunData,
    outname: str = "pericenter_diagnostics",
    smooth: int = 5,
    pre_window: Tuple[float, float] = (-0.6, -0.1),
) -> List[str]:
    """Compact 2x2 plot from the old pos_process.py, aligned to first pericenter."""
    tau = run.tau
    d = _col(run, "dist_host_kpc")
    pram_col = "pram_upstream_dyn_cm2" if "pram_upstream_dyn_cm2" in run.cols else "pram_dyn_cm2"
    pram_s = dra.running_nanmedian(_col(run, pram_col), smooth)

    def norm_col(name: str) -> np.ndarray:
        return dra.running_nanmedian(dra.normalize_to_window(tau, _col(run, name), pre_window[0], pre_window[1]), smooth)

    sfr_n = norm_col("sfr_msun_per_yr")
    fcen_n = norm_col("f_sfr_central")
    mgas_n = norm_col("mgas_msun")
    mgas_sf_n = norm_col("mgas_sf_msun")
    rgas_n = norm_col("rhalf_gas_sf_kpc")
    ryoung_n = norm_col("rhalf_young_kpc") if "rhalf_young_kpc" in run.cols else None

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    a = ax[0, 0]
    a.plot(tau, d, lw=2, label=r"$d_{\rm host}$")
    a.set_ylabel(r"$d_{\rm host}\ [{\rm kpc}]$")
    add_tau0(a)
    a2 = a.twinx()
    good = _good_xy(tau, pram_s, positive_y=True)
    a2.plot(tau[good], pram_s[good], lw=2, ls=":", label=r"$P_{\rm ram}$")
    a2.set_yscale("log")
    a2.set_ylabel(r"$P_{\rm ram}\ [{\rm dyn\,cm^{-2}}]$")

    a = ax[0, 1]
    good = _good_xy(tau, sfr_n, positive_y=True)
    a.plot(tau[good], sfr_n[good], lw=2, label=r"${\rm SFR/SFR_{pre}}$")
    good = _good_xy(tau, fcen_n, positive_y=True)
    a.plot(tau[good], fcen_n[good], lw=2, ls="--", label=r"$f_{\rm central}/f_{\rm central,pre}$")
    add_tau0(a)
    a.set_ylabel("Normalized SF diagnostics")
    a.legend(frameon=False)

    a = ax[1, 0]
    good = _good_xy(tau, mgas_n, positive_y=True)
    a.plot(tau[good], mgas_n[good], lw=2, label=r"$M_{\rm gas}/M_{\rm gas,pre}$")
    good = _good_xy(tau, mgas_sf_n, positive_y=True)
    a.plot(tau[good], mgas_sf_n[good], lw=2, ls="--", label=r"$M_{\rm gas,SF}/M_{\rm gas,SF,pre}$")
    add_tau0(a)
    a.set_xlabel(r"$\tau=t-t_{\rm peri}$ [Gyr]")
    a.set_ylabel("Normalized gas masses")
    a.legend(frameon=False)

    a = ax[1, 1]
    good = _good_xy(tau, rgas_n, positive_y=True)
    a.plot(tau[good], rgas_n[good], lw=2, label=r"$R_{\rm gas,SF}/R_{\rm pre}$")
    if ryoung_n is not None:
        good = _good_xy(tau, ryoung_n, positive_y=True)
        a.plot(tau[good], ryoung_n[good], lw=2, ls="--", label=r"$R_{\rm young}/R_{\rm pre}$")
    add_tau0(a)
    a.set_xlabel(r"$\tau=t-t_{\rm peri}$ [Gyr]")
    a.set_ylabel("Normalized sizes")
    a.legend(frameon=False)
    return savefig(fig, outname, dpi=250)


# =============================================================================
# Multi-run comparison plots
# =============================================================================

def plot_quantity_over_runs(
    runs: Sequence[dra.RunData],
    colname: str,
    ylabel: str,
    outbase: str,
    xlim: Optional[Tuple[float, float]] = None,
    logy: bool = False,
    use_smooth: bool = False,
) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for run in runs:
        name = colname + "__smooth" if use_smooth and colname + "__smooth" in run.cols else colname
        if name not in run.cols:
            continue
        y = np.asarray(run.cols[name], dtype=float)
        good = _good_xy(run.tau, y, positive_y=logy)
        if not np.any(good):
            continue
        ax.plot(run.tau[good], y[good], lw=2, label=run.label)
    add_tau0(ax)
    ax.set_xlabel(r"$\tau=t-t_{\rm peri}$ [Gyr]")
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    return savefig(fig, outbase)


def plot_compare_forcing(runs: Sequence[dra.RunData], outdir: str, xlim: Optional[Tuple[float, float]] = None) -> List[str]:
    ensure_dir(outdir)
    outputs: List[str] = []
    fig, axs = plt.subplots(3, 1, figsize=(7.8, 8.6), sharex=True)

    for run in runs:
        d = _col(run, "dist_host_kpc")
        good = _good_xy(run.tau, d)
        if np.any(good):
            axs[0].plot(run.tau[good], d[good], lw=2, label=run.label)

        pram_name = "pram_upstream_dyn_cm2" if "pram_upstream_dyn_cm2" in run.cols else "pram_dyn_cm2"
        p = _col(run, pram_name)
        good = _good_xy(run.tau, p, positive_y=True)
        if np.any(good):
            axs[1].plot(run.tau[good], p[good], lw=2, label=run.label)

        tid = _col(run, "tidal_to_self_star")
        good = _good_xy(run.tau, tid, positive_y=True)
        if np.any(good):
            axs[2].plot(run.tau[good], tid[good], lw=2, label=run.label)

    for a in axs:
        add_tau0(a)
        if xlim is not None:
            a.set_xlim(*xlim)
        a.legend(frameon=False, fontsize=9)

    axs[0].set_ylabel(r"$d_{\rm host}$ [kpc]")
    axs[1].set_ylabel(r"$P_{\rm ram}$ [dyn cm$^{-2}$]")
    axs[1].set_yscale("log")
    axs[2].set_ylabel(r"$a_{\rm tide}/a_{\rm self,\star}$")
    axs[2].set_yscale("log")
    axs[2].set_xlabel(r"$\tau=t-t_{\rm peri}$ [Gyr]")
    outputs.extend(savefig(fig, os.path.join(outdir, "compare_forcing")))
    return outputs


def plot_compare_structure(runs: Sequence[dra.RunData], outdir: str, xlim: Optional[Tuple[float, float]] = None) -> List[str]:
    ensure_dir(outdir)
    outputs: List[str] = []
    outputs += plot_quantity_over_runs(runs, "rhalf_star_kpc__norm__smooth", r"$R_{1/2,\star}/R_{\rm pre}$", os.path.join(outdir, "compare_rhalf_star_norm"), xlim=xlim)
    outputs += plot_quantity_over_runs(runs, "rhalf_young_kpc__norm__smooth", r"$R_{1/2,{\rm young}}/R_{\rm pre}$", os.path.join(outdir, "compare_rhalf_young_norm"), xlim=xlim)
    outputs += plot_quantity_over_runs(runs, "rhalf_old_kpc__norm__smooth", r"$R_{1/2,{\rm old}}/R_{\rm pre}$", os.path.join(outdir, "compare_rhalf_old_norm"), xlim=xlim)
    return outputs


def plot_compare_gas_sf(runs: Sequence[dra.RunData], outdir: str, xlim: Optional[Tuple[float, float]] = None) -> List[str]:
    ensure_dir(outdir)
    outputs: List[str] = []
    outputs += plot_quantity_over_runs(runs, "mgas_msun__norm__smooth", r"$M_{\rm gas}/M_{\rm gas,pre}$", os.path.join(outdir, "compare_mgas_norm"), xlim=xlim, logy=True)
    outputs += plot_quantity_over_runs(runs, "mgas_sf_msun__norm__smooth", r"$M_{\rm gas,SF}/M_{\rm gas,SF,pre}$", os.path.join(outdir, "compare_mgas_sf_norm"), xlim=xlim, logy=True)
    outputs += plot_quantity_over_runs(runs, "sfr_msun_per_yr__norm__smooth", r"${\rm SFR}/{\rm SFR}_{\rm pre}$", os.path.join(outdir, "compare_sfr_norm"), xlim=xlim, logy=True)
    outputs += plot_quantity_over_runs(runs, "f_sfr_central__norm__smooth", r"$f_{\rm central}/f_{\rm central,pre}$", os.path.join(outdir, "compare_fcentral_norm"), xlim=xlim)
    return outputs


def plot_compare_asymmetry(runs: Sequence[dra.RunData], outdir: str, xlim: Optional[Tuple[float, float]] = None) -> List[str]:
    ensure_dir(outdir)
    outputs: List[str] = []
    outputs += plot_quantity_over_runs(runs, "rho_leading_over_trailing__smooth", r"$\rho_{\rm lead}/\rho_{\rm trail}$", os.path.join(outdir, "compare_rho_leading_trailing"), xlim=xlim)
    outputs += plot_quantity_over_runs(runs, "sfr_leading_frac__smooth", r"$f_{\rm SFR,lead}$", os.path.join(outdir, "compare_sfr_leading_frac"), xlim=xlim)
    outputs += plot_quantity_over_runs(runs, "gas_com_along_upstream_kpc__smooth", r"Gas COM offset along wind [kpc]", os.path.join(outdir, "compare_gas_com_offset"), xlim=xlim)
    return outputs


def plot_summary_pram_vs_tidal(runs: Sequence[dra.RunData], outdir: str) -> List[str]:
    ensure_dir(outdir)
    x = np.array([run.metrics.get("pram_peak", np.nan) for run in runs], dtype=float)
    y = np.array([run.metrics.get("tidal_peak", np.nan) for run in runs], dtype=float)
    mask = dra.finite_mask(x, y) & (x > 0) & (y > 0)
    if not np.any(mask):
        return []
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(x[mask], y[mask], s=70)
    for i, run in enumerate(runs):
        if mask[i]:
            ax.annotate(run.label, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Peak $P_{\rm ram}$ [dyn cm$^{-2}$]")
    ax.set_ylabel(r"Peak $a_{\rm tide}/a_{\rm self,\star}$")
    return savefig(fig, os.path.join(outdir, "summary_pram_vs_tidal"))


def plot_summary_young_vs_old(runs: Sequence[dra.RunData], outdir: str) -> List[str]:
    ensure_dir(outdir)
    x = np.array([run.metrics.get("old_norm_min", np.nan) for run in runs], dtype=float)
    y = np.array([run.metrics.get("young_norm_min", np.nan) for run in runs], dtype=float)
    mask = dra.finite_mask(x, y)
    if not np.any(mask):
        return []
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x[mask], y[mask], s=70)
    xmin = float(np.nanmin(np.r_[x[mask], y[mask]]))
    xmax = float(np.nanmax(np.r_[x[mask], y[mask]]))
    ax.plot([xmin, xmax], [xmin, xmax], lw=1.2, ls="--")
    for i, run in enumerate(runs):
        if mask[i]:
            ax.annotate(run.label, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"min $R_{1/2,{\rm old}}/R_{\rm pre}$")
    ax.set_ylabel(r"min $R_{1/2,{\rm young}}/R_{\rm pre}$")
    return savefig(fig, os.path.join(outdir, "summary_young_vs_old_compaction"))


def make_comparison_plot_set(
    runs: Sequence[dra.RunData],
    outdir: str = "compare_runs",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[str]:
    ensure_dir(outdir)
    outputs: List[str] = []
    outputs += plot_compare_forcing(runs, outdir, xlim=xlim)
    outputs += plot_compare_structure(runs, outdir, xlim=xlim)
    outputs += plot_compare_gas_sf(runs, outdir, xlim=xlim)
    outputs += plot_compare_asymmetry(runs, outdir, xlim=xlim)
    outputs += plot_summary_pram_vs_tidal(runs, outdir)
    outputs += plot_summary_young_vs_old(runs, outdir)
    return outputs


# =============================================================================
# Section-5 / compaction-channel plots
# =============================================================================

def plot_compaction_rates(runs: Sequence[dra.RunData], outdir: str) -> List[str]:
    ensure_dir(outdir)
    labels = [run.label for run in runs]
    x = np.arange(len(runs), dtype=float)
    y1 = np.array([run.metrics.get("compaction_rate_entry_to_gasloss_gyr-1", np.nan) for run in runs], dtype=float)
    y2 = np.array([run.metrics.get("compaction_rate_nogas_gyr-1", np.nan) for run in runs], dtype=float)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(runs) + 3.0), 4.8))
    width = 0.36
    ax.bar(x - width / 2.0, y1, width=width, label="entry → gas loss/final")
    ax.bar(x + width / 2.0, y2, width=width, label="after gas loss")
    ax.axhline(0.0, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta R_{1/2,\star}/(R_{1/2,\star}\Delta t)$ [Gyr$^{-1}$]")
    ax.legend(frameon=False)
    return savefig(fig, os.path.join(outdir, "fig9_like_compaction_rates"))


def plot_gas_compression_proxy(runs: Sequence[dra.RunData], outdir: str) -> List[str]:
    ensure_dir(outdir)
    labels = [run.label for run in runs]
    x = np.arange(len(runs), dtype=float)

    metrics = [
        ("gas_total_change_entry_to_peri", r"$\Delta M_{\rm gas}$ entry→peri"),
        ("gas_sf_change_entry_to_peri", r"$\Delta M_{\rm gas,SF}$ entry→peri"),
        ("rhalf_gas_sf_change_entry_to_peri", r"$\Delta R_{\rm gas,SF}$ entry→peri"),
        ("f_sfr_central_change_entry_to_peri", r"$\Delta f_{\rm central}$ entry→peri"),
    ]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(runs) + 3.0), 5.0))
    width = 0.18
    for j, (key, label) in enumerate(metrics):
        vals = np.array([run.metrics.get(key, np.nan) for run in runs], dtype=float)
        ax.bar(x + (j - 1.5) * width, vals, width=width, label=label)
    ax.axhline(0.0, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Fractional change")
    ax.legend(frameon=False, fontsize=9)
    return savefig(fig, os.path.join(outdir, "gas_compression_proxy_comparison"))


def plot_stellar_compaction_proxy(runs: Sequence[dra.RunData], outdir: str) -> List[str]:
    ensure_dir(outdir)
    labels = [run.label for run in runs]
    x = np.arange(len(runs), dtype=float)
    metrics = [
        ("rhalf_star_change_entry_to_final", r"$\Delta R_{1/2,\star}$ entry→final"),
        ("mstar_change_entry_to_final", r"$\Delta M_\star$ entry→final"),
        ("delta_rhalf_over_rentry", r"$\Delta R_{1/2,\star}/R_{\rm entry}$"),
    ]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(runs) + 3.0), 5.0))
    width = 0.23
    for j, (key, label) in enumerate(metrics):
        vals = np.array([run.metrics.get(key, np.nan) for run in runs], dtype=float)
        ax.bar(x + (j - 1.0) * width, vals, width=width, label=label)
    ax.axhline(0.0, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Fractional change")
    ax.legend(frameon=False, fontsize=9)
    return savefig(fig, os.path.join(outdir, "stellar_compaction_proxy_comparison"))


def plot_mass_budget_rows(rows: Sequence[Dict[str, Any]], outdir: str) -> List[str]:
    ensure_dir(outdir)
    if not rows:
        return []
    outputs: List[str] = []
    labels = [str(r["label"]) for r in rows]
    x = np.arange(len(rows), dtype=float)

    def vals(key: str) -> np.ndarray:
        return np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(rows) + 3.0), 5.0))
    width = 0.23
    ax.bar(x - width, vals("dgas_inner_entry_to_comp_over_mgas_entry"), width=width, label="inner gas")
    ax.bar(x, vals("dgas_outer_entry_to_comp_over_mgas_entry"), width=width, label="outer gas")
    ax.bar(x + width, vals("dgas_outer_entry_to_peri_over_mgas_entry"), width=width, label="outer gas entry→peri")
    ax.axhline(0.0, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta M_{\rm gas}/M_{\rm gas,entry}$")
    ax.legend(frameon=False)
    outputs += savefig(fig, os.path.join(outdir, "gas_inner_outer_change"))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(rows) + 3.0), 5.0))
    width = 0.25
    ax.bar(x - width / 2.0, vals("dstar_inner_entry_to_comp_over_mstar_entry"), width=width, label="inner stars")
    ax.bar(x + width / 2.0, vals("dstar_outer_entry_to_comp_over_mstar_entry"), width=width, label="outer stars")
    ax.axhline(0.0, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta M_\star/M_{\star,\rm entry}$")
    ax.legend(frameon=False)
    outputs += savefig(fig, os.path.join(outdir, "stars_inner_outer_change"))
    return outputs


def plot_stellar_profile_pair(profile_pair: Dict[str, Any], outdir: str) -> List[str]:
    ensure_dir(outdir)
    label = str(profile_pair["label"])
    entry = profile_pair["entry_profile"]
    final = profile_pair["final_profile"]

    fig, axs = plt.subplots(2, 1, figsize=(6.8, 8.2), sharex=True)

    ax = axs[0]
    ax.plot(entry["r_cum_kpc"], entry["cum_mass_msun"], lw=2.0, label="entry")
    ax.plot(final["r_cum_kpc"], final["cum_mass_msun"], lw=2.0, label="final")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$M_\star(<r)\ [{\rm M_\odot}]$")
    ax.legend(framealpha=0.95)
    ax.set_title(label)

    ax = axs[1]
    good1 = _good_xy(entry["r_rho_kpc"], entry["rho_r2_msun_kpc-1"], positive_y=True)
    good2 = _good_xy(final["r_rho_kpc"], final["rho_r2_msun_kpc-1"], positive_y=True)
    ax.plot(entry["r_rho_kpc"][good1], entry["rho_r2_msun_kpc-1"][good1], lw=2.0, label="entry")
    ax.plot(final["r_rho_kpc"][good2], final["rho_r2_msun_kpc-1"][good2], lw=2.0, label="final")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r\ [{\rm kpc}]$")
    ax.set_ylabel(r"$\rho_\star(r)\,r^2\ [{\rm M_\odot\,kpc^{-1}}]$")

    return savefig(fig, os.path.join(outdir, "fig10_like_profiles_{0}".format(sanitize_label(label))))


# =============================================================================
# yt snapshot maps
# =============================================================================

def choose_existing_field(ds: Any, candidates: Sequence[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    return dra.choose_existing_field(ds, candidates)


def get_stellar_com(ds: Any) -> Any:
    star = dra.get_star_data(ds)
    if star is None:
        return ds.domain_center
    try:
        pos = star["pos"]
        mass = star["mass"]
        mtot = mass.sum()
        if float(mtot) <= 0:
            return ds.domain_center
        return (pos * mass[:, None]).sum(axis=0) / mtot
    except Exception:
        return ds.domain_center


def get_sfr_field(ds: Any) -> Optional[Tuple[str, str]]:
    return choose_existing_field(ds, [("PartType0", "StarFormationRate"), ("gas", "star_formation_rate"), ("gas", "sfr")])


def compute_total_sfr(ds: Any) -> float:
    sfr_field = get_sfr_field(ds)
    if sfr_field is None:
        return np.nan
    ad = ds.all_data()
    try:
        return float(ad[sfr_field].sum().to("Msun/yr"))
    except Exception:
        try:
            return float(ad[sfr_field].sum())
        except Exception:
            return np.nan


def annotate_common(plot_obj: Any, total_sfr: Optional[float] = None) -> None:
    try:
        plot_obj.annotate_timestamp(corner="upper_left", redshift=True, draw_inset_box=True)
    except Exception:
        pass
    try:
        plot_obj.annotate_scale(corner="upper_right")
    except Exception:
        pass
    if total_sfr is not None and np.isfinite(total_sfr):
        try:
            plot_obj.annotate_text((0.03, 0.92), "Total SFR = {0:.3e} Msun/yr".format(total_sfr), coord_system="axis", text_args={"color": "white"})
        except Exception:
            pass


def make_gas_density_plot(ds: Any, outname: str, axis: str = "z", center: Any = None, width_kpc: float = 50.0, zlim: Optional[Sequence[float]] = None, total_sfr: Optional[float] = None) -> Optional[str]:
    yt = dra.try_import_yt()
    gas_field = choose_existing_field(ds, [("gas", "density"), ("PartType0", "Density"), ("PartType0", "density")])
    if gas_field is None:
        return None
    p = yt.ProjectionPlot(ds, axis, gas_field, center=center, width=(width_kpc, "kpc"))
    try:
        p.set_unit(gas_field, "Msun/kpc**2")
    except Exception:
        pass
    if zlim is not None:
        p.set_zlim(gas_field, zlim[0], zlim[1])
    try:
        p.set_cmap(gas_field, "inferno")
    except Exception:
        pass
    annotate_common(p, total_sfr=total_sfr)
    p.save(outname)
    return outname


def make_star_particle_plot(ds: Any, outname: str, axis: str = "z", center: Any = None, width_kpc: float = 50.0, mass_lim: Optional[Sequence[float]] = None, total_sfr: Optional[float] = None) -> Optional[str]:
    yt = dra.try_import_yt()
    mass_field = choose_existing_field(ds, [("PartType4", "Masses"), ("PartType4", "particle_mass"), ("stars", "particle_mass")])
    if mass_field is None:
        return None

    if axis == "z":
        x_field, y_field = ("PartType4", "particle_position_x"), ("PartType4", "particle_position_y")
    elif axis == "y":
        x_field, y_field = ("PartType4", "particle_position_x"), ("PartType4", "particle_position_z")
    elif axis == "x":
        x_field, y_field = ("PartType4", "particle_position_y"), ("PartType4", "particle_position_z")
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    fields = set(ds.field_list) | set(ds.derived_field_list)
    if x_field not in fields or y_field not in fields:
        return None

    p = yt.ParticlePlot(ds, x_field, y_field, mass_field, center=center, width=(width_kpc, "kpc"), depth=(width_kpc, "kpc"))
    try:
        p.set_unit(mass_field, "Msun")
    except Exception:
        pass
    if mass_lim is not None:
        p.set_zlim(mass_field, mass_lim[0], mass_lim[1])
    try:
        p.set_cmap(mass_field, "bone")
    except Exception:
        pass
    annotate_common(p, total_sfr=total_sfr)
    p.save(outname)
    return outname


def make_sfr_map(ds: Any, outname: str, axis: str = "z", center: Any = None, width_kpc: float = 50.0, zlim: Optional[Sequence[float]] = None, total_sfr: Optional[float] = None) -> Optional[str]:
    yt = dra.try_import_yt()
    sfr_field = get_sfr_field(ds)
    if sfr_field is None:
        return None

    if axis == "z":
        x_field, y_field = ("PartType0", "particle_position_x"), ("PartType0", "particle_position_y")
    elif axis == "y":
        x_field, y_field = ("PartType0", "particle_position_x"), ("PartType0", "particle_position_z")
    elif axis == "x":
        x_field, y_field = ("PartType0", "particle_position_y"), ("PartType0", "particle_position_z")
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    fields = set(ds.field_list) | set(ds.derived_field_list)
    if x_field not in fields or y_field not in fields:
        return None

    p = yt.ParticlePlot(ds, x_field, y_field, sfr_field, center=center, width=(width_kpc, "kpc"), depth=(width_kpc, "kpc"))
    try:
        p.set_unit(sfr_field, "Msun/yr")
    except Exception:
        pass
    if zlim is not None:
        p.set_zlim(sfr_field, zlim[0], zlim[1])
    try:
        p.set_cmap(sfr_field, "magma")
    except Exception:
        pass
    annotate_common(p, total_sfr=total_sfr)
    p.save(outname)
    return outname


def make_sfr_history_plot(times_gyr: Sequence[float], sfr_totals: Sequence[float], outname: str) -> List[str]:
    t = np.asarray(times_gyr, dtype=float)
    sfr = np.asarray(sfr_totals, dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    good = _good_xy(t, sfr, positive_y=True)
    if np.any(good):
        ax.plot(t[good], sfr[good], marker="o", lw=1.8)
        ax.set_yscale("log")
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel(r"Total SFR [$M_\odot\,{\rm yr}^{-1}$]")
    return savefig(fig, outname)


def make_snapshot_map_series(
    snapdir: str,
    outdir: Optional[str] = None,
    axis: str = "z",
    width_kpc: float = 50.0,
    center_mode: str = "domain",
    center: Optional[Sequence[float]] = None,
    gas_lim: Optional[Sequence[float]] = None,
    star_lim: Optional[Sequence[float]] = None,
    sfr_lim: Optional[Sequence[float]] = None,
    no_sfr_annotation: bool = False,
    max_snaps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    yt = dra.try_import_yt()
    snapdir = os.path.abspath(snapdir)
    outdir = outdir or os.path.join(snapdir, "plots_yt")

    gas_dir = os.path.join(outdir, "gas_density")
    star_dir = os.path.join(outdir, "stellar_density")
    sfr_dir = os.path.join(outdir, "sfr_maps")
    for d in [outdir, gas_dir, star_dir, sfr_dir]:
        ensure_dir(d)

    snapfiles = dra.find_snapshots(snapdir)
    if max_snaps is not None:
        snapfiles = snapfiles[:max_snaps]
    if len(snapfiles) == 0:
        raise FileNotFoundError("No snapshot_XXX.hdf5 files found in {0}".format(snapdir))

    times: List[float] = []
    redshifts: List[float] = []
    sfrs: List[float] = []
    outputs: List[str] = []

    for i, snap in enumerate(snapfiles):
        base = os.path.basename(snap)
        snapnum = "{0:03d}".format(dra.snapshot_number(snap)) if dra.snapshot_number(snap) is not None else "unknown"
        if verbose:
            print("[{0:04d}/{1:04d}] {2}".format(i + 1, len(snapfiles), base))
        try:
            ds = yt.load(snap)
        except Exception as exc:
            print("[WARN] Could not load {0}: {1}".format(base, exc))
            times.append(np.nan)
            redshifts.append(np.nan)
            sfrs.append(np.nan)
            continue

        if center is not None:
            ctr = center
        elif center_mode in ("stars_com", "stars-com", "stellar_com"):
            ctr = get_stellar_com(ds)
        else:
            ctr = ds.domain_center

        total_sfr = compute_total_sfr(ds)
        times.append(dra.get_time_gyr(ds))
        redshifts.append(dra.get_redshift(ds))
        sfrs.append(total_sfr)
        annot_sfr = None if no_sfr_annotation else total_sfr

        tasks = [
            (make_gas_density_plot, os.path.join(gas_dir, "gas_density_{0}.png".format(snapnum)), gas_lim),
            (make_star_particle_plot, os.path.join(star_dir, "stellar_density_{0}.png".format(snapnum)), star_lim),
            (make_sfr_map, os.path.join(sfr_dir, "sfr_map_{0}.png".format(snapnum)), sfr_lim),
        ]

        for func, outname, lim in tasks:
            try:
                result = func(ds, outname, axis=axis, center=ctr, width_kpc=width_kpc, zlim=lim, total_sfr=annot_sfr) if func is not make_star_particle_plot else func(ds, outname, axis=axis, center=ctr, width_kpc=width_kpc, mass_lim=lim, total_sfr=annot_sfr)
                if result is not None:
                    outputs.append(result)
            except Exception as exc:
                print("[WARN] Plot failed for {0}: {1}".format(base, exc))
                if verbose:
                    traceback.print_exc()

    outputs += make_sfr_history_plot(times, sfrs, os.path.join(outdir, "sfr_history"))
    return {"times_gyr": np.asarray(times), "redshifts": np.asarray(redshifts), "sfr_msun_per_yr": np.asarray(sfrs), "outputs": outputs}


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plotting utilities for controlled Gadget4 dwarf ram-pressure runs.")
    sub = p.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summary", help="Make summary and pericenter plots for one table.")
    p_sum.add_argument("table")
    p_sum.add_argument("--label", default=None)
    p_sum.add_argument("--outdir", default="single_run_plots")
    p_sum.add_argument("--smooth", type=int, default=5)
    p_sum.add_argument("--use-tau", action="store_true")

    p_cmp = sub.add_parser("compare", help="Make comparison plot set from multiple evolution tables.")
    p_cmp.add_argument("tables", nargs="+")
    p_cmp.add_argument("--labels", nargs="*", default=None)
    p_cmp.add_argument("--outdir", default="compare_runs")
    p_cmp.add_argument("--smooth", type=int, default=5)
    p_cmp.add_argument("--xlim", nargs=2, type=float, default=None)
    p_cmp.add_argument("--pre-window", nargs=2, type=float, default=(-0.5, -0.1))
    p_cmp.add_argument("--force-window", nargs=2, type=float, default=(-0.3, 0.3))
    p_cmp.add_argument("--response-window", nargs=2, type=float, default=(-0.1, 1.0))

    p_sec = sub.add_parser("section5", help="Make Section-5-inspired plots and optional mass-budget plots.")
    p_sec.add_argument("tables", nargs="+")
    p_sec.add_argument("--labels", nargs="*", default=None)
    p_sec.add_argument("--snapshot-dirs", nargs="*", default=None)
    p_sec.add_argument("--outdir", default="section5_plots")
    p_sec.add_argument("--entry-radius-kpc", type=float, default=250.0)
    p_sec.add_argument("--gas-loss-fraction", type=float, default=0.05)
    p_sec.add_argument("--gas-loss-mode", choices=["total", "sf"], default="total")
    p_sec.add_argument("--min-gas-floor-abs", type=float, default=1e5)
    p_sec.add_argument("--split-mode", choices=["final_rhalf", "fixed"], default="final_rhalf")
    p_sec.add_argument("--split-radius-kpc", type=float, default=None)

    p_maps = sub.add_parser("maps", help="Make yt gas/star/SFR maps and SFR history.")
    p_maps.add_argument("path")
    p_maps.add_argument("--outdir", default=None)
    p_maps.add_argument("--axis", choices=["x", "y", "z"], default="z")
    p_maps.add_argument("--width", type=float, default=50.0)
    p_maps.add_argument("--center-mode", choices=["domain", "stars-com", "stars_com"], default="domain")
    p_maps.add_argument("--center", nargs=3, type=float, default=None)
    p_maps.add_argument("--gas-lim", nargs=2, type=float, default=(1e-5, 5e1))
    p_maps.add_argument("--star-lim", nargs=2, type=float, default=(1e-5, 5e1))
    p_maps.add_argument("--sfr-lim", nargs=2, type=float, default=(1e15, 3e18))
    p_maps.add_argument("--no-sfr-annotation", action="store_true")
    p_maps.add_argument("--max-snaps", type=int, default=None)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "summary":
        ensure_dir(args.outdir)
        run = dra.prepare_run(args.table, label=args.label, smooth=args.smooth)
        base = sanitize_label(run.label)
        out1 = plot_summary_evolution(run, os.path.join(args.outdir, "summary_{0}".format(base)), use_time=not args.use_tau)
        out2 = plot_pericenter_diagnostics(run, os.path.join(args.outdir, "pericenter_diagnostics_{0}".format(base)), smooth=args.smooth)
        print("[INFO] Wrote {0} files".format(len(out1) + len(out2)))

    elif args.command == "compare":
        ensure_dir(args.outdir)
        labels = args.labels if args.labels not in (None, []) else None
        runs = dra.prepare_runs(
            args.tables,
            labels=labels,
            smooth=args.smooth,
            pre_window=tuple(args.pre_window),
            force_window=tuple(args.force_window),
            response_window=tuple(args.response_window),
        )
        xlim = tuple(args.xlim) if args.xlim is not None else None
        outputs = make_comparison_plot_set(runs, outdir=args.outdir, xlim=xlim)
        dra.write_csv(dra.metrics_rows(runs), os.path.join(args.outdir, "summary_metrics.csv"))
        print("[INFO] Wrote {0} figure files plus summary_metrics.csv".format(len(outputs)))

    elif args.command == "section5":
        ensure_dir(args.outdir)
        labels = args.labels if args.labels not in (None, []) else None
        snapdirs = args.snapshot_dirs if args.snapshot_dirs not in (None, []) else None
        runs = dra.prepare_runs(
            args.tables,
            labels=labels,
            snapshot_dirs=snapdirs,
            entry_radius_kpc=args.entry_radius_kpc,
            gas_loss_fraction=args.gas_loss_fraction,
            gas_loss_mode=args.gas_loss_mode,
            min_gas_floor_abs=args.min_gas_floor_abs,
        )
        outputs: List[str] = []
        outputs += plot_compaction_rates(runs, args.outdir)
        outputs += plot_gas_compression_proxy(runs, args.outdir)
        outputs += plot_stellar_compaction_proxy(runs, args.outdir)
        dra.write_csv(dra.metrics_rows(runs), os.path.join(args.outdir, "run_epoch_summary.csv"))

        if snapdirs is not None:
            rows = []
            for run in runs:
                row = dra.mass_budget_for_run(run, split_mode=args.split_mode, split_radius_kpc=args.split_radius_kpc)
                if row is not None:
                    rows.append(row)
                try:
                    pair = dra.profile_pair_for_run(run)
                    if pair is not None:
                        outputs += plot_stellar_profile_pair(pair, args.outdir)
                except Exception as exc:
                    print("[WARN] Could not make profile for {0}: {1}".format(run.label, exc))
            outputs += plot_mass_budget_rows(rows, args.outdir)
            dra.write_csv(rows, os.path.join(args.outdir, "epoch_mass_budget.csv"))

        print("[INFO] Wrote {0} figure files".format(len(outputs)))

    elif args.command == "maps":
        result = make_snapshot_map_series(
            args.path,
            outdir=args.outdir,
            axis=args.axis,
            width_kpc=args.width,
            center_mode=args.center_mode,
            center=args.center,
            gas_lim=args.gas_lim,
            star_lim=args.star_lim,
            sfr_lim=args.sfr_lim,
            no_sfr_annotation=args.no_sfr_annotation,
            max_snaps=args.max_snaps,
        )
        print("[INFO] Wrote {0} outputs".format(len(result["outputs"])))


if __name__ == "__main__":
    main()
