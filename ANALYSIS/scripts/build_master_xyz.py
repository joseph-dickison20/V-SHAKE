#!/usr/bin/env python3
"""Concatenate seam geometries, Kabsch-align to MECP, and plot coupling vs energy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from hd_plot_colors import (
    COLOR_D_MECP,
    COLOR_D_SCATTER,
    COLOR_H_MECP,
    COLOR_H_SCATTER,
)

HARTREE_TO_KCAL = 627.509
MECP_COORD_TOL = 1e-4

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "Nimbus Sans",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

EXCLUDED_ELEMENTS = frozenset({"Hd", "Ha"})

# Set to (xmin, xmax) to bypass auto x limits, e.g. (0, 60)
#X_LIM_OVERRIDE: tuple[float, float] | None = None
X_LIM_OVERRIDE: tuple[float, float] | None = (0, 60)


def parse_comment_line(comment: str) -> dict[str, float | int]:
    """Parse key=value fields from an XYZ comment line."""
    out: dict[str, float | int] = {}
    for token in comment.split():
        if "=" not in token:
            continue
        key, val = token.split("=", 1)
        if key == "step":
            out["step"] = int(val)
        elif key == "time_fs":
            out["time_fs"] = float(val)
        elif key == "sigma":
            out["sigma"] = float(val)
        elif key == "delta_cm^-1":
            out["delta_cm_inv"] = float(val)
        elif key == "E0":
            out["E0"] = float(val)
    required = ("step", "time_fs", "sigma", "delta_cm_inv", "E0")
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"Comment line missing fields {missing}: {comment!r}")
    return out


def parse_seam_xyz(filepath: str | Path) -> list[dict]:
    """Read all frames from a seam_geometries.xyz file."""
    path = Path(filepath)
    frames: list[dict] = []
    with path.open() as f:
        while True:
            natom_line = f.readline()
            if not natom_line:
                break
            natom = int(natom_line.strip())
            comment = f.readline().rstrip("\n")
            if not comment:
                raise ValueError(f"Unexpected EOF after atom count in {path}")

            meta = parse_comment_line(comment)
            real_atoms: list[tuple[str, float, float, float]] = []
            for _ in range(natom):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF reading atoms in {path}")
                parts = line.split()
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if element not in EXCLUDED_ELEMENTS:
                    real_atoms.append((element, x, y, z))

            coords = np.array([[a[1], a[2], a[3]] for a in real_atoms], dtype=float)
            frames.append(
                {
                    "atoms": real_atoms,
                    "coords": coords,
                    "comment": comment,
                    "step": meta["step"],
                    "time_fs": meta["time_fs"],
                    "sigma": meta["sigma"],
                    "delta_cm_inv": meta["delta_cm_inv"],
                    "E0": meta["E0"],
                    "source_file": str(path),
                }
            )
    return frames


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align coordinates P to reference Q (both shape (n, 3)). Returns aligned P."""
    centroid_p = P.mean(axis=0)
    centroid_q = Q.mean(axis=0)
    P_centered = P - centroid_p
    Q_centered = Q - centroid_q
    H = P_centered.T @ Q_centered
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    return P_centered @ R.T + centroid_q


def write_xyz_frame(
    handle,
    atoms: list[tuple[str, float, float, float]],
    comment: str,
) -> None:
    handle.write(f"{len(atoms)}\n")
    handle.write(f"{comment}\n")
    for element, x, y, z in atoms:
        handle.write(f"{element:<2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def atoms_from_coords(
    template: list[tuple[str, float, float, float]],
    coords: np.ndarray,
) -> list[tuple[str, float, float, float]]:
    return [
        (template[i][0], coords[i, 0], coords[i, 1], coords[i, 2])
        for i in range(len(template))
    ]





@dataclass
class IsotopeBuildResult:
    output_frames: list[dict]
    E0_mecp: float
    delta_mecp: float
    n_real: int
    frames_per_file: list[int]
    input_paths: list[Path]


def build_aligned_frames(input_paths: list[Path]) -> IsotopeBuildResult:
    """Parse trajectories, validate MECP, Kabsch-align samples to reference MECP."""
    frames_per_file: list[int] = []
    all_parsed: list[list[dict]] = []
    for p in input_paths:
        frames = parse_seam_xyz(p)
        frames_per_file.append(len(frames))
        all_parsed.append(frames)

    if not all_parsed or not all_parsed[0]:
        raise SystemExit("No frames found in first input file.")

    mecp_frame = all_parsed[0][0]
    if mecp_frame["step"] != 0:
        print(
            f"Warning: first frame of {input_paths[0]} has step={mecp_frame['step']}, expected 0."
        )

    mecp_coords = mecp_frame["coords"].copy()
    mecp_atoms = mecp_frame["atoms"]
    E0_mecp = float(mecp_frame["E0"])
    delta_mecp = float(mecp_frame["delta_cm_inv"])
    n_real = mecp_coords.shape[0]

    for i, (path, frames) in enumerate(zip(input_paths, all_parsed)):
        if not frames:
            raise SystemExit(f"No frames in {path}")
        ref = frames[0]
        if ref["coords"].shape != mecp_coords.shape:
            raise SystemExit(
                f"Atom count mismatch in {path}: "
                f"{ref['coords'].shape[0]} vs MECP {n_real}"
            )
        max_dev = np.max(np.abs(ref["coords"] - mecp_coords))
        if i > 0 and max_dev >= MECP_COORD_TOL:
            print(
                f"Warning: MECP in {path} differs from reference by "
                f"{max_dev:.6e} Å (tolerance {MECP_COORD_TOL})."
            )

    output_frames: list[dict] = [
        {
            "atoms": mecp_atoms,
            "comment": mecp_frame["comment"],
            "E0": E0_mecp,
            "delta_cm_inv": delta_mecp,
        }
    ]

    for frames in all_parsed:
        for frame in frames[1:]:
            if frame["coords"].shape[0] != n_real:
                raise SystemExit(
                    f"Real atom count mismatch in {frame['source_file']} "
                    f"step={frame['step']}"
                )
            aligned = kabsch_align(frame["coords"], mecp_coords)
            output_frames.append(
                {
                    "atoms": atoms_from_coords(frame["atoms"], aligned),
                    "comment": frame["comment"],
                    "E0": frame["E0"],
                    "delta_cm_inv": frame["delta_cm_inv"],
                }
            )

    return IsotopeBuildResult(
        output_frames=output_frames,
        E0_mecp=E0_mecp,
        delta_mecp=delta_mecp,
        n_real=n_real,
        frames_per_file=frames_per_file,
        input_paths=input_paths,
    )


def scatter_series(result: IsotopeBuildResult) -> tuple[np.ndarray, np.ndarray, float]:
    """ΔE and V for non-MECP frames relative to this isotope's MECP."""
    delta_e = np.array(
        [
            (fr["E0"] - result.E0_mecp) * HARTREE_TO_KCAL
            for fr in result.output_frames[1:]
        ],
        dtype=float,
    )
    v_coupling = np.array(
        [fr["delta_cm_inv"] / 2.0 for fr in result.output_frames[1:]],
        dtype=float,
    )
    v_mecp = result.delta_mecp / 2.0
    return delta_e, v_coupling, v_mecp


def write_master_xyz(path: Path, output_frames: list[dict]) -> None:
    with path.open("w") as fh:
        for fr in output_frames:
            write_xyz_frame(fh, fr["atoms"], fr["comment"])


def _apply_scatter_axis_limits(
    ax: plt.Axes,
    delta_e_arrays: list[np.ndarray],
    v_arrays: list[np.ndarray],
    v_mecps: list[float],
    log_minor_ticks: bool,
    x_lim_override: tuple[float, float] | None = None,
    y_lim_override: tuple[float, float] | None = None,
    *,
    yscale: str = "log",
) -> None:
    ax.set_xlabel(r"$\Delta E_{\mathrm{seam}}$ [kcal mol$^{-1}$]")
    ax.set_ylabel(r"$V$ [cm$^{-1}$]")

    x_all = np.concatenate([np.array([0.0])] + list(delta_e_arrays))
    x_max = float(np.max(x_all))
    effective_x_lim = (
        x_lim_override
        if x_lim_override is not None
        else (X_LIM_OVERRIDE if X_LIM_OVERRIDE is not None else (0.0, np.ceil(x_max)))
    )
    ax.set_xlim(*effective_x_lim)

    v_all = np.concatenate(list(v_arrays) + [np.asarray(v_mecps, dtype=float)])
    y_is_log = yscale == "log"

    if y_is_log:
        ax.set_yscale("log")
        log_y_floor = 0.01
        if y_lim_override is not None:
            y_lo = (
                float(y_lim_override[0])
                if float(y_lim_override[0]) > 0.0
                else log_y_floor
            )
            y_hi = float(y_lim_override[1])
            if y_hi <= y_lo:
                y_hi = y_lo * 10.0
            ax.set_ylim(y_lo, y_hi)
        else:
            pos = v_all[v_all > 0]
            y_lo = float(max(np.min(pos), log_y_floor)) if pos.size else log_y_floor
            y_hi = float(np.max(v_all)) if v_all.size else y_lo * 10.0
            if y_hi <= y_lo:
                y_hi = y_lo * 10.0
            else:
                y_hi = y_hi * 1.08
            ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12)
        )
        if log_minor_ticks:
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=48)
            )
    else:
        ax.set_yscale("linear")
        if y_lim_override is not None:
            y_lo = float(y_lim_override[0])
            y_hi = float(y_lim_override[1])
            if y_hi <= y_lo:
                y_hi = y_lo + 1.0
            ax.set_ylim(y_lo, y_hi)
        else:
            y_lo = float(np.min(v_all)) if v_all.size else 0.0
            y_hi = float(np.max(v_all)) if v_all.size else 1.0
            if y_hi <= y_lo:
                y_hi = y_lo + 1.0
            else:
                pad = 0.08 * (y_hi - y_lo)
                y_lo = y_lo - pad
                y_hi = y_hi + pad
            ax.set_ylim(y_lo, y_hi)


def print_isotope_summary(result: IsotopeBuildResult, *, label: str, xyz_path: Path | None) -> None:
    e0_rel_all = [
        (fr["E0"] - result.E0_mecp) * HARTREE_TO_KCAL for fr in result.output_frames
    ]
    v_all = [fr["delta_cm_inv"] / 2.0 for fr in result.output_frames]

    if label:
        print(f"--- {label} ---")
    if xyz_path is not None:
        print(f"Wrote master XYZ: {xyz_path} ({len(result.output_frames)} frames)")
    print(f"Input files processed: {len(result.input_paths)}")
    for path, n in zip(result.input_paths, result.frames_per_file):
        print(f"  {path}: {n} frames")
    print(f"Real atoms per frame: {result.n_real}")
    print(f"MECP E0 (Hartree): {result.E0_mecp:.11f}")
    print(f"MECP tunneling splitting (cm^-1): {result.delta_mecp:.6f}")
    print(
        f"ΔE relative to MECP (kcal/mol): "
        f"{min(e0_rel_all):.4f} to {max(e0_rel_all):.4f}"
    )
    print(f"Vibronic coupling V (cm^-1): {min(v_all):.6f} to {max(v_all):.6f}")


def make_scatter_plot(
    delta_e_kcal: np.ndarray,
    v_coupling: np.ndarray,
    v_mecp: float,
    output_path: Path,
    *,
    log_minor_ticks: bool = True,
    isotope: str = "H",
    x_lim_override: tuple[float, float] | None = None,
    y_lim_override: tuple[float, float] | None = None,
    yscale: str = "log",
) -> None:
    scatter_color = COLOR_D_SCATTER if isotope == "D" else COLOR_H_SCATTER
    mecp_color = COLOR_D_MECP if isotope == "D" else COLOR_H_MECP
    fig, ax = plt.subplots(figsize=(3.33, 3.33))
    ax.scatter(
        delta_e_kcal,
        v_coupling,
        s=8,
        alpha=0.6,
        edgecolors="none",
        color=scatter_color,
    )
    ax.axhline(v_mecp, color=mecp_color, ls="--", lw=1.0, zorder=5)
    _apply_scatter_axis_limits(
        ax, [delta_e_kcal], [v_coupling], [v_mecp], log_minor_ticks, x_lim_override,
        y_lim_override,
        yscale=yscale,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_scatter_plot_hd(
    delta_e_h: np.ndarray,
    v_h: np.ndarray,
    v_mecp_h: float,
    delta_e_d: np.ndarray,
    v_d: np.ndarray,
    v_mecp_d: float,
    output_path: Path,
    *,
    log_minor_ticks: bool = True,
    x_lim_override: tuple[float, float] | None = None,
    y_lim_override: tuple[float, float] | None = None,
    yscale: str = "log",
) -> None:
    fig, ax = plt.subplots(figsize=(3.33, 3.33))
    ax.scatter(
        delta_e_h,
        v_h,
        s=8,
        alpha=0.6,
        edgecolors="none",
        color=COLOR_H_SCATTER,
        label="H",
    )
    ax.scatter(
        delta_e_d,
        v_d,
        s=8,
        alpha=0.6,
        edgecolors="none",
        color=COLOR_D_SCATTER,
        label="D",
    )
    ax.axhline(v_mecp_h, color=COLOR_H_MECP, ls="--", lw=1.0, zorder=5)
    ax.axhline(v_mecp_d, color=COLOR_D_MECP, ls="--", lw=1.0, zorder=5)
    _apply_scatter_axis_limits(
        ax,
        [delta_e_h, delta_e_d],
        [v_h, v_d],
        [v_mecp_h, v_mecp_d],
        log_minor_ticks,
        x_lim_override,
        y_lim_override,
        yscale=yscale,
    )
    ax.legend(loc="best", frameon=True, markerscale=2)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_single_isotope(
    input_paths: list[Path],
    output_xyz: Path,
    output_plot: Path,
    log_minor_ticks: bool,
    x_lim_override: tuple[float, float] | None = None,
    y_lim_override: tuple[float, float] | None = None,
    *,
    yscale: str = "log",
) -> None:
    result = build_aligned_frames(input_paths)
    write_master_xyz(output_xyz, result.output_frames)
    delta_e, v_coupling, v_mecp = scatter_series(result)
    isotope = "D" if "/D/" in str(input_paths[0]).replace("\\", "/") else "H"
    make_scatter_plot(
        delta_e,
        v_coupling,
        v_mecp,
        output_plot,
        log_minor_ticks=log_minor_ticks,
        isotope=isotope,
        x_lim_override=x_lim_override,
        y_lim_override=y_lim_override,
        yscale=yscale,
    )
    print_isotope_summary(result, label="", xyz_path=output_xyz)
    print(f"Wrote plot: {output_plot}")


def run_hd_combined_scatter(
    input_paths_h: list[Path],
    input_paths_d: list[Path],
    output_plot: Path,
    log_minor_ticks: bool,
    x_lim_override: tuple[float, float] | None = None,
    y_lim_override: tuple[float, float] | None = None,
    *,
    yscale: str = "log",
) -> None:
    result_h = build_aligned_frames(input_paths_h)
    result_d = build_aligned_frames(input_paths_d)
    delta_e_h, v_h, v_mecp_h = scatter_series(result_h)
    delta_e_d, v_d, v_mecp_d = scatter_series(result_d)
    make_scatter_plot_hd(
        delta_e_h,
        v_h,
        v_mecp_h,
        delta_e_d,
        v_d,
        v_mecp_d,
        output_plot,
        log_minor_ticks=log_minor_ticks,
        x_lim_override=x_lim_override,
        y_lim_override=y_lim_override,
        yscale=yscale,
    )
    print("Combined H/D scatter plot (no master XYZ written).")
    print(f"Wrote plot: {output_plot}")
    print_isotope_summary(result_h, label="H", xyz_path=None)
    print_isotope_summary(result_d, label="D", xyz_path=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Kabsch-aligned master XYZ and coupling vs energy plot."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help="Paths to seam_geometries.xyz (same system/isotope; single-isotope mode)",
    )
    parser.add_argument(
        "--input-files-h",
        nargs="+",
        default=None,
        help="H isotope trajectories (with --hd-combined-scatter)",
    )
    parser.add_argument(
        "--input-files-d",
        nargs="+",
        default=None,
        help="D isotope trajectories (with --hd-combined-scatter)",
    )
    parser.add_argument(
        "--hd-combined-scatter",
        action="store_true",
        help="Overlay H and D on one scatter plot only (no master XYZ).",
    )
    parser.add_argument(
        "--output-xyz",
        default=None,
        help="Output master XYZ path (single-isotope mode)",
    )
    parser.add_argument("--output-plot", required=True, help="Output PNG path")
    parser.add_argument(
        "--log-minor-ticks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Minor y-axis ticks between log decades (default: on). "
        "Use --no-log-minor-ticks to disable. Ignored when --yscale linear.",
    )
    parser.add_argument(
        "--yscale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale for coupling scatter (default: log).",
    )
    parser.add_argument(
        "--x-lim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Fixed x-axis limits for coupling scatter (kcal/mol); "
        "overrides X_LIM_OVERRIDE in script",
    )
    parser.add_argument(
        "--y-lim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Fixed y-axis limits for coupling scatter (cm^-1); "
        "default: auto from data",
    )
    args = parser.parse_args()

    x_lim_override: tuple[float, float] | None = None
    if args.x_lim is not None:
        x_lim_override = (float(args.x_lim[0]), float(args.x_lim[1]))
    y_lim_override: tuple[float, float] | None = None
    if args.y_lim is not None:
        y_lim_override = (float(args.y_lim[0]), float(args.y_lim[1]))

    if args.hd_combined_scatter:
        if not args.input_files_h or not args.input_files_d:
            parser.error(
                "--hd-combined-scatter requires --input-files-h and --input-files-d"
            )
        if args.input_files or args.output_xyz:
            parser.error(
                "Use --input-files and --output-xyz only in single-isotope mode "
                "(omit them with --hd-combined-scatter)"
            )
        run_hd_combined_scatter(
            [Path(p) for p in args.input_files_h],
            [Path(p) for p in args.input_files_d],
            Path(args.output_plot),
            args.log_minor_ticks,
            x_lim_override=x_lim_override,
            y_lim_override=y_lim_override,
            yscale=args.yscale,
        )
        return

    if not args.input_files or not args.output_xyz:
        parser.error(
            "Single-isotope mode requires --input-files and --output-xyz "
            "(or use --hd-combined-scatter)"
        )
    run_single_isotope(
        [Path(p) for p in args.input_files],
        Path(args.output_xyz),
        Path(args.output_plot),
        args.log_minor_ticks,
        x_lim_override=x_lim_override,
        y_lim_override=y_lim_override,
        yscale=args.yscale,
    )


if __name__ == "__main__":
    main()
