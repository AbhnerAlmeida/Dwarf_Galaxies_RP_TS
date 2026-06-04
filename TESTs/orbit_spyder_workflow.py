#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_spyder_workflow.py

Workflow passo-a-passo para usar:
    1. orbit_analysis_tools.py
    2. orbit_science_plots.py

Este arquivo foi escrito para ser usado como um "notebook em formato .py" no
Spyder.  Ele usa células `# %%`, então você pode rodar seção por seção com
Run current cell.

Resumo da reorganização dos scripts originais
---------------------------------------------

1. orbit_analysis_full_spyder.py
   Papel original:
       Pipeline mais completo. Lê snapshots HDF5 do Gadget/Gadget-4, cria um
       catálogo inicial de IDs do satélite, rastreia o centro, mede órbita,
       raio de maré, massas, tamanhos, SFR, sSFR, SFE, tempo de depleção,
       pressão de ram, proxies de maré, pericentros e apocentros.
   Decisão:
       Foi usado como base principal de `orbit_analysis_tools.py`, porque já
       contém quase tudo que é necessário para a análise física.

2. orbit_center_tracking_spyder.py
   Papel original:
       Script focado em testar e diagnosticar o rastreamento do centro. Ele é
       muito útil para comparar métodos como all-stars COM, stellar_core e
       inner_ids_shrinking.
   Decisão:
       A lógica essencial de centro robusto já está presente no pipeline
       completo. Por isso, o novo workflow recomenda usar
       center_mode="inner_ids_shrinking" e velocity_mode="inner_ids".
       O script antigo fica conceitualmente como ferramenta de validação, mas
       não precisa ser o pipeline principal.

3. orbit_comparison_tools_spyder.py
   Papel original:
       Primeira versão das rotinas de comparação entre simulações. Adiciona
       proxies simples de campo de maré e plota quantidades em função do tempo.
   Decisão:
       Foi considerado uma versão útil, mas mais antiga. Suas ideias foram
       incorporadas ao módulo consolidado de plotagem.

4. orbit_comparison_tools_spyder_v2.py
   Papel original:
       Versão mais completa das ferramentas de comparação. Adiciona eixo
       t - t_first_pericentre, quantidades derivadas de SFR/sSFR/SFE/tdep,
       tabelas de pericentro/apocentro e mais painéis.
   Decisão:
       Serviu como uma das bases para `orbit_science_plots.py`.

5. orbit_plots_full_spyder.py
   Papel original:
       Companheiro de plotagem do pipeline completo. Pode usar `results` e
       `cfg` em memória ou recarregar CSVs salvos. É o mais compatível com a
       saída de `orbit_analysis_full_spyder.py`.
   Decisão:
       Foi usado como base direta de `orbit_science_plots.py`.

Estrutura recomendada agora
---------------------------
    orbit_analysis_tools.py   -> somente análise, leitura HDF5 e tabelas
    orbit_science_plots.py    -> somente pós-processamento e figuras
    orbit_spyder_workflow.py  -> guia executável passo-a-passo

A vantagem é separar:
    - física/medidas dos snapshots;
    - visualização científica;
    - escolhas específicas do projeto.
"""

# %%
# =============================================================================
# 0. Imports
# =============================================================================

from pathlib import Path
import pandas as pd

from orbit_analysis_tools import (
    HostHaloConfig,
    OrbitAnalysisConfig,
    discover_labels,
    find_snapshots_for_label,
    inspect_snapshot_structure,
    analyze_all,
    load_results_from_output,
    compute_orbital_extrema,
)

from orbit_science_plots import *


# %%
# =============================================================================
# 1. Escolhas gerais do workflow
# =============================================================================

# Execute uma etapa por vez.  Para usar no Spyder, altere True/False abaixo.
RUN_STRUCTURE_INSPECTION = False
RUN_ANALYSIS_FROM_HDF5 = True
LOAD_EXISTING_RESULTS = False
RUN_PLOTS = True

# Se True, gera figuras com eixo temporal normal e também com t - t_pericentro.
GENERATE_BOTH_TIME_AXES = True

# Opções aceitas se GENERATE_BOTH_TIME_AXES = False:
#     "time"
#     "snapshot"
#     "time_since_first_pericentre"
#     "snapshot_since_first_pericentre"
X_AXIS_MODE = "time_since_first_pericentre"

ANNOTATE_EXTREMA = True


# %%
# =============================================================================
# 2. Configuração física e dos arquivos
# =============================================================================

cfg = OrbitAnalysisConfig(
    # Estrutura esperada:
    #     ROOT / LABEL / output / snapshot_*.hdf5
    root="./../SIMULATIONS/ORBIT/HigherRes",

    # Coloque None para descobrir automaticamente todos os labels em root.
    labels=[
        "E_mid_L_radial",
        "E_mid_L_mid",
        "E_mid_L_high",
    ],

    output_dir="orbit_full_analysis_outputs",
    snapshot_glob="snapshot_*.hdf5",

    # Tipos de partículas no Gadget/Gadget-4.
    gas_ptype=0,
    dm_ptype=1,
    star_ptype=4,

    # Conversões de unidades.
    # Ajuste estes fatores se seus snapshots não estiverem em kpc, km/s,
    # 1e10 Msun e unidades de tempo compatíveis com Gadget.
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
        truncate_mass_at_r200=False,
    ),

    # Recomendação para seus testes de órbita:
    # - define IDs estelares centrais no primeiro snapshot;
    # - segue esses IDs ao longo do tempo;
    # - evita que caudas de maré puxem o centro.
    initial_center_mode="stars_com",
    center_mode="inner_ids_shrinking",
    velocity_mode="inner_ids",

    # Seleção inicial de estrelas centrais.
    inner_radius_factor_rhalf=1.0,
    inner_min_radius_kpc=0.5,
    inner_max_radius_kpc=5.0,
    inner_min_particles=50,

    # Gás inicial do satélite.
    initial_satellite_gas_radius_kpc=None,
    default_initial_gas_radius_kpc=30.0,
    gas_radius_factor_rhalf=8.0,

    # DM inicial.
    # Use "all" se PartType1 contém apenas o DM do satélite.
    # Use "radius" se PartType1 contém DM do satélite + halo hospedeiro.
    dm_selection_mode="all",
    initial_satellite_dm_radius_kpc=None,
    default_initial_dm_radius_kpc=80.0,
    dm_radius_factor_rhalf=20.0,

    # Rastreamento do centro.
    shrink_initial_radius_kpc=None,
    shrink_factor=0.75,
    shrink_min_particles=100,
    shrink_min_radius_kpc=0.2,
    center_search_radius_kpc=40.0,

    # Associação estelar / stripping.
    member_mode="tidal_kinematic",
    member_tidal_factor=1.0,
    stripped_tidal_factor=1.5,
    v_escape_factor=1.25,
    stripped_consecutive_snapshots=2,

    # Raio de maré.
    tidal_max_iterations=40,
    tidal_tolerance=1.0e-3,
    tidal_initial_fraction_of_R=0.15,
    tidal_max_fraction_of_R=0.5,
    tidal_min_radius_kpc=0.05,

    # Pressão de ram.
    compute_ram_pressure=True,
    ram_pressure_density_radius_kpc=10.0,
    ram_pressure_max_density_radius_kpc=30.0,
    ram_pressure_min_cgm_particles=16,
    ram_pressure_velocity_mode="local_cgm",  # "local_cgm" ou "host_frame"
    ram_pressure_use_cgm_only=True,

    event_window_snapshots=1,
    save_snapshot_json=False,
    verbose=True,
)


# %%
# =============================================================================
# 3. Inspecionar estrutura HDF5 antes de rodar a análise completa
# =============================================================================

if RUN_STRUCTURE_INSPECTION:
    labels_to_check = cfg.labels if cfg.labels is not None else discover_labels(cfg.root)
    first_label = labels_to_check[0]
    first_snapshot = find_snapshots_for_label(cfg, first_label)[0]

    print("\nInspecionando primeiro snapshot:")
    print(first_snapshot)
    inspect_snapshot_structure(first_snapshot)


# %%
# =============================================================================
# 4. Rodar análise a partir dos snapshots HDF5
# =============================================================================

if RUN_ANALYSIS_FROM_HDF5:
    results = analyze_all(cfg)

    # Tabelas úteis no Variable Explorer do Spyder.
    combined_df = pd.concat(
        [df.assign(label=label) for label, df in results.items()],
        ignore_index=True,
    )

    extrema_all_list = []
    for label, df in results.items():
        ev = compute_orbital_extrema(df)
        if len(ev):
            extrema_all_list.append(ev.assign(label=label))

    extrema_all_df = (
        pd.concat(extrema_all_list, ignore_index=True)
        if extrema_all_list else pd.DataFrame()
    )

    print("\nObjetos criados:")
    print("  results[label]")
    print("  combined_df")
    print("  extrema_all_df")


# %%
# =============================================================================
# 5. Carregar resultados já salvos, sem reler os HDF5
# =============================================================================

if LOAD_EXISTING_RESULTS:
    results = load_results_from_output(cfg.output_dir)

    combined_df = pd.concat(
        [df.assign(label=label) for label, df in results.items()],
        ignore_index=True,
    )

    print("\nResultados carregados de:")
    print(Path(cfg.output_dir).resolve())
    print("Labels:", list(results.keys()))


# %%
# =============================================================================
# 6. Pós-processamento leve antes de plotar
# =============================================================================

# Esta etapa adiciona colunas derivadas úteis, mas não altera os CSVs originais.
# Ex.: time_since_first_pericentre_gyr, rhalf_over_rt, SFR_tracked_use_msun_yr.
prepared_results = prepare_results_for_comparison(results, cfg=cfg)

prepared_combined_df = pd.concat(
    [df.assign(label=label) for label, df in prepared_results.items()],
    ignore_index=True,
)

print("\nColunas disponíveis para plotagem:")
print(sorted(prepared_combined_df.columns))


# %%
# =============================================================================
# 7. Configurar aparência das figuras
# =============================================================================

configure_matplotlib_for_paper(
    base_fontsize=12,
    use_latex=False,  # mude para True apenas se o LaTeX funcionar no seu sistema
)


# %%
# =============================================================================
# 8. Gerar figuras científicas comparando todos os labels
# =============================================================================

if RUN_PLOTS:
    if GENERATE_BOTH_TIME_AXES:
        comparison_outputs = run_standard_and_pericentre_normalized_plots(
            results,
            cfg=cfg,
            annotate_extrema=ANNOTATE_EXTREMA,
        )
    else:
        comparison_outputs = run_all_comparison_plots(
            results,
            cfg=cfg,
            output_subdir=f"comparison_plots_{X_AXIS_MODE}",
            annotate_extrema=ANNOTATE_EXTREMA,
            x_axis_mode=X_AXIS_MODE,
        )

    print("\nArquivos gerados:")
    for key, value in comparison_outputs.items():
        print(f"  {key}: {value}")


#%%
labels = ["E_mid_L_radial",
            "E_mid_L_mid",
            "E_mid_L_high"]

for label in labels:
    map_outputs = plot_map_suite(
        results,
        cfg,
        label=label,
        snapshot_index=0,
        fields=("gas_density", "gas_sfr"),
        width_kpc=80.0,
        axis="z",
        center_on="satellite",
        annotate_orbit=True,
        annotate_tidal_radius=True,
        annotate_rhalf=True,
    )

# %%
# =============================================================================
# 9. Exemplos de inspeção rápida no Spyder
# =============================================================================

# Exemplos úteis para rodar manualmente depois que `results` existir:
#
# 1. Ver as primeiras linhas de uma simulação:
#       results["E_mid_L_radial"].head()
#
# 2. Ver as colunas disponíveis:
#       results["E_mid_L_radial"].columns
#
# 3. Comparar pericentros/apocentros:
#       extrema_all_df
#
# 4. Fazer um gráfico rápido no console:
#       df = results["E_mid_L_radial"]
#       df.plot(x="time_gyr", y=["R_host_kpc", "r_tidal_kpc"])
#
# 5. Salvar uma tabela reduzida:
#       cols = ["time_gyr", "R_host_kpc", "V_rad_kms", "r_tidal_kpc",
#               "Mstar_member_msun", "P_ram_dyne_cm2"]
#       results["E_mid_L_radial"][cols].to_csv("quick_check.csv", index=False)
