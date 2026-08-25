"""Pipeline stage runners for SIR_ANALYSIS."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from vshake_pipeline.config import PipelineConfig, ResolvedConfig
from vshake_pipeline.script_paths import (
    BUILD_MASTER_XYZ,
    PLOT_GEOM_SCATTER,
    SIR_COMPUTE,
    SIR_VIZ,
)

log = logging.getLogger("vshake_pipeline")


def _run_cmd(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    log.info("CMD: %s", " ".join(cmd))
    if cwd:
        log.info("  cwd=%s", cwd)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def write_resolved_config(resolved: ResolvedConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.resolved.yaml"
    with out.open("w") as f:
        yaml.safe_dump(resolved.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)


def step_master_xyz(
    resolved: ResolvedConfig,
    run_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    cfg = resolved.pipeline
    out_root = run_dir / "01_master_xyz"
    mx = cfg.master_xyz

    for iso in cfg.isotopes:
        iso_dir = out_root / iso
        marker = iso_dir / "master.xyz"
        if marker.is_file() and not force:
            log.info("Skip master_xyz %s (exists)", iso)
            continue
        iso_dir.mkdir(parents=True, exist_ok=True)
        inputs = [str(p) for p in resolved.discovery.by_isotope[iso]]
        cmd = [
            sys.executable,
            str(BUILD_MASTER_XYZ),
            "--input-files",
            *inputs,
            "--output-xyz",
            str(iso_dir / "master.xyz"),
            "--output-plot",
            str(iso_dir / "coupling_vs_energy.png"),
        ]
        if mx.x_lim is not None:
            cmd.extend(["--x-lim", str(mx.x_lim[0]), str(mx.x_lim[1])])
        if mx.y_lim is not None:
            cmd.extend(["--y-lim", str(mx.y_lim[0]), str(mx.y_lim[1])])
        if not mx.log_minor_ticks:
            cmd.append("--no-log-minor-ticks")
        if mx.yscale != "log":
            cmd.extend(["--yscale", mx.yscale])
        _run_cmd(cmd, dry_run=dry_run)

    hd_dir = out_root / "HD"
    hd_plot = hd_dir / "coupling_vs_energy.png"
    if set(cfg.isotopes) >= {"H", "D"}:
        if hd_plot.is_file() and not force:
            log.info("Skip master_xyz HD combined scatter (exists)")
        else:
            hd_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(BUILD_MASTER_XYZ),
                "--hd-combined-scatter",
                "--input-files-h",
                *[str(p) for p in resolved.discovery.by_isotope["H"]],
                "--input-files-d",
                *[str(p) for p in resolved.discovery.by_isotope["D"]],
                "--output-plot",
                str(hd_plot),
            ]
            if mx.x_lim is not None:
                cmd.extend(["--x-lim", str(mx.x_lim[0]), str(mx.x_lim[1])])
            if mx.y_lim is not None:
                cmd.extend(["--y-lim", str(mx.y_lim[0]), str(mx.y_lim[1])])
            if not mx.log_minor_ticks:
                cmd.append("--no-log-minor-ticks")
            if mx.yscale != "log":
                cmd.extend(["--yscale", mx.yscale])
            _run_cmd(cmd, dry_run=dry_run)

    return out_root


def step_geom(
    resolved: ResolvedConfig,
    run_dir: Path,
    geom_config_path: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    cfg = resolved.pipeline
    master_root = run_dir / "01_master_xyz"
    out_root = run_dir / "02_geom"

    if not cfg.geom.observables or not cfg.geom.plots:
        log.warning(
            "geom section empty in run config; using geom_config_path defaults only"
        )

    for iso in cfg.isotopes:
        iso_out = out_root / iso / "plots"
        marker = iso_out / ".geom_done"
        master = master_root / iso / "master.xyz"
        if marker.is_file() and not force:
            log.info("Skip geom %s (done marker exists)", iso)
            continue
        iso_out.mkdir(parents=True, exist_ok=True)
        if force and marker.is_file() and not dry_run:
            marker.unlink()
        cmd = [
            sys.executable,
            str(PLOT_GEOM_SCATTER),
            "--xyz",
            str(master),
            "--output-dir",
            str(iso_out),
            "--config",
            str(geom_config_path),
            "--isotope",
            iso,
        ]
        _run_cmd(cmd, dry_run=dry_run)
        if not dry_run:
            marker.touch()

    isotopes = set(cfg.isotopes)
    if isotopes >= {"H", "D"}:
        combo_dir = out_root / "combo_plots"
        combo_marker = combo_dir / ".geom_combo_done"
        master_h = master_root / "H" / "master.xyz"
        master_d = master_root / "D" / "master.xyz"
        if (not combo_marker.is_file() or force) and master_h.is_file() and master_d.is_file():
            combo_dir.mkdir(parents=True, exist_ok=True)
            if force and combo_marker.is_file() and not dry_run:
                combo_marker.unlink()
            cmd = [
                sys.executable,
                str(PLOT_GEOM_SCATTER),
                "--hd-combo-hists",
                "--xyz-h",
                str(master_h),
                "--xyz-d",
                str(master_d),
                "--output-dir",
                str(combo_dir),
                "--config",
                str(geom_config_path),
            ]
            _run_cmd(cmd, dry_run=dry_run)
            if not dry_run:
                combo_marker.touch()
        elif not master_h.is_file() or not master_d.is_file():
            log.warning("Skip geom combo_plots (missing H or D master.xyz)")

    return out_root


def step_sir(
    resolved: ResolvedConfig,
    run_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    """SIR eigen-decomposition of V on the seam manifold.

    One ``sir_compute.py`` invocation covers every isotope so it can also emit
    the grouped H/D eigenvalue bar chart under ``combo_bar_plots/``.
    """
    cfg = resolved.pipeline
    master_root = run_dir / "01_master_xyz"
    out_root = run_dir / "03_sir"
    sr = cfg.sir

    markers = [out_root / iso / "sir_results.npz" for iso in cfg.isotopes]
    if all(m.is_file() for m in markers) and not force:
        log.info("Skip sir (NPZ exists for all isotopes)")
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SIR_COMPUTE),
        "--output-dir",
        str(out_root),
        "--run-name",
        cfg.run_name,
    ]
    for iso in cfg.isotopes:
        master = master_root / iso / "master.xyz"
        if not master.is_file() and not dry_run:
            raise FileNotFoundError(f"Missing master XYZ: {master}")
        cmd.extend(["--dataset", iso, str(master)])
    cmd.extend(
        [
            "--nslices",
            str(sr.nslices),
            "--var",
            str(sr.var),
            "--n-eigenvalues-plot",
            str(sr.n_eigenvalues_plot),
            "--n-eigenvectors-viz",
            str(sr.n_eigenvectors_viz),
        ]
    )
    if force:
        cmd.append("--force")
    _run_cmd(cmd, dry_run=dry_run)

    return out_root


def step_sir_viz(
    resolved: ResolvedConfig,
    run_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    """PyMOL rendering of the SIR eigenvectors, g_perp, and grad V.

    Static arrow overlays carry the nuclear-density isomesh from the per-isotope
    cube file; swing GIFs stay skeleton-only.
    """
    cfg = resolved.pipeline
    sir_root = run_dir / "03_sir"
    out_root = run_dir / "04_sir_viz"
    sv = cfg.sir_viz
    sr = cfg.sir

    for iso in cfg.isotopes:
        npz = sir_root / iso / "sir_results.npz"
        if not npz.is_file() and not dry_run:
            raise FileNotFoundError(f"Missing SIR NPZ: {npz}")

        qchem = Path(resolved.qchem_paths[iso])
        if not qchem.is_file() and not dry_run:
            raise FileNotFoundError(f"Q-Chem MECP output not found: {qchem}")

        cube = Path(resolved.density_cube_paths[iso])
        if not cube.is_file() and not dry_run:
            raise FileNotFoundError(f"Nuclear-density cube not found: {cube}")

        iso_dir = out_root / iso
        iso_dir.mkdir(parents=True, exist_ok=True)
        viz = sv.viz_for_isotope(iso)
        cmd = [
            sys.executable,
            str(SIR_VIZ),
            "--npz",
            str(npz),
            "--qchem",
            str(qchem),
            "--output-dir",
            str(iso_dir),
            "--n-vectors",
            str(sr.n_eigenvectors_viz),
            "--n-skip-rows",
            str(cfg.n_skip_rows),
            "--arrow-scale",
            str(sv.arrow_scale),
            "--arrow-color",
            sv.arrow_color[iso],
            "--gperp-color",
            sv.gperp_color,
            "--gradv-color",
            sv.gradv_color,
            "--arrow-mol-transparency",
            str(sv.arrow_mol_transparency),
            "--density-cube",
            str(cube),
            "--density-isovalue",
            str(sv.density.isovalue),
            "--density-color",
            ",".join(str(c) for c in sv.density.color),
            "--amplitude",
            str(viz["amplitude"]),
            "--n-frames",
            str(viz["n_frames"]),
            "--zoom-center",
            str(viz["zoom_center"]),
            "--turn-x",
            str(viz["turn_x"]),
            "--turn-y",
            str(viz["turn_y"]),
            "--turn-z",
            str(viz["turn_z"]),
            "--png-width",
            str(viz["png_width"]),
            "--png-height",
            str(viz["png_height"]),
            "--png-dpi",
            str(viz["png_dpi"]),
            "--ray",
            str(viz["ray"]),
            "--gif-delay",
            str(viz["gif_delay"]),
            "--gif-loop",
            str(viz["gif_loop"]),
            "--gif-extra-args",
            viz["gif_extra_args"],
            "--viz-overrides",
            json.dumps(sv.per_vector),
        ]
        if sv.cos_barchart_xlim is not None:
            cmd.extend(
                ["--barchart-xlim", str(sv.cos_barchart_xlim[0]), str(sv.cos_barchart_xlim[1])]
            )
        if sv.cos_barchart_ylim is not None:
            cmd.extend(
                ["--barchart-ylim", str(sv.cos_barchart_ylim[0]), str(sv.cos_barchart_ylim[1])]
            )
        if force:
            cmd.append("--force")
        _run_cmd(cmd, dry_run=dry_run)

    return out_root


def write_geom_sidecar(cfg: PipelineConfig, path: Path) -> None:
    geom = cfg.geom
    data = {
        "observables": geom.observables,
        "plots": geom.plots,
        "write_table": geom.write_table,
        "table_path": geom.table_path,
        "plot_1d_hist_splines": geom.plot_1d_hist_splines,
    }
    if geom.plot_defaults:
        data["plot_defaults"] = geom.plot_defaults
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
