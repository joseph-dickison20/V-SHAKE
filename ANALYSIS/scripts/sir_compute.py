#!/usr/bin/env python3
"""Sliced inverse regression of the vibronic coupling on the seam manifold.

Reads a MECP-aligned master XYZ (frame 0 = MECP), response V = delta_cm^-1 / 2,
PCA-whitens Cartesians, runs SIR, and writes the eigenvalue spectrum plus the
eigenvectors (as Cartesian displacement directions) for step ``sir_viz``.

Always produces three families per isotope under the output directory:

* ``all/`` — SIR on every frame (current unfiltered behavior)
* ``filtered/`` — drop frames with ``V < noise_floor`` (MECP always kept)
* ``bootstrap_pXX/`` — classical bootstrap on the filtered set at each YAML
  fraction, with bar + Tukey box/whisker eigenvalue plots and η1 |cos|/angle
  vs the filtered direction

One invocation handles every isotope via repeated ``--dataset ISO XYZ``. Grouped
H/D charts go under ``combo_bar_plots/``; with ``--bootstrap-combo-spectrum``
(default) a single filtered+bootstrap mega chart is written instead of separate
per-fraction bootstrap combo plots.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Nimbus Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)

BOHR_PER_ANGSTROM = 1.8897259886

# Minimum frames for PCA whitening (need n >= 3 so n-1 >= 2 components).
_MIN_FRAMES_PCA = 3

try:
    from hd_plot_colors import series_color
except ImportError:
    series_color = lambda iso: "#d62728" if iso == "H" else "#1f77b4"  # noqa: E731


# ---------------------------------------------------------------------------
# XYZ I/O
# ---------------------------------------------------------------------------

def parse_comment_line(comment: str) -> dict:
    out = {}
    for tok in comment.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    if "delta_cm^-1" not in out:
        raise ValueError(f"XYZ comment missing delta_cm^-1: {comment!r}")
    return {
        "delta_cm_inv": float(out["delta_cm^-1"]),
        "E0": float(out["E0"]) if "E0" in out else None,
        "step": int(float(out["step"])) if "step" in out else None,
    }


def read_master_xyz(path: Path):
    frames, symbols, y, meta = [], None, [], []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].split()[0])
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        info = parse_comment_line(comment)
        block = lines[i + 2 : i + 2 + n_atoms]
        if len(block) < n_atoms:
            raise ValueError(f"Truncated frame at line {i + 1} in {path}")
        syms = [ln.split()[0] for ln in block]
        xyz = np.array([[float(x) for x in ln.split()[1:4]] for ln in block])
        if symbols is None:
            symbols = syms
        elif syms != symbols:
            raise ValueError(f"Atom-order mismatch at frame starting line {i + 1}")
        frames.append(xyz)
        y.append(info["delta_cm_inv"] / 2.0)
        meta.append(info)
        i += 2 + n_atoms
    if not frames:
        raise ValueError(f"No frames in {path}")
    return symbols, np.asarray(frames), np.asarray(y, dtype=float), meta


# ---------------------------------------------------------------------------
# Alignment / PCA / SIR
# ---------------------------------------------------------------------------

def kabsch_align(mobile, ref):
    mobile_c = mobile - mobile.mean(axis=0)
    ref_c = ref - ref.mean(axis=0)
    H = mobile_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (R @ mobile_c.T).T + ref.mean(axis=0)


def align_to_mecp(coords):
    ref = coords[0]
    return np.array([kabsch_align(f, ref) for f in coords])


def pca_whiten(X, var_cutoff=0.95):
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (X.shape[0] - 1)
    frac = np.cumsum(eigvals) / eigvals.sum()
    k = max(2, int(np.searchsorted(frac, var_cutoff) + 1))
    k = min(k, X.shape[0] - 1, len(eigvals))
    V = Vt[:k].T
    scores = Xc @ V
    Z = scores / np.sqrt(eigvals[:k])
    return Z, V, eigvals[:k], k


def sir(Z, y, n_slices=10):
    n = len(y)
    order = np.argsort(y)
    Z_sorted = Z[order]
    edges = np.linspace(0, n, n_slices + 1).astype(int)
    M = np.zeros((Z.shape[1], Z.shape[1]))
    for s in range(n_slices):
        sl = Z_sorted[edges[s] : edges[s + 1]]
        if len(sl) == 0:
            continue
        weight = len(sl) / n
        mean_h = sl.mean(axis=0)
        M += weight * np.outer(mean_h, mean_h)
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


# ---------------------------------------------------------------------------
# Filtering / bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_dirname(fraction: float) -> str:
    return f"bootstrap_p{int(round(100 * fraction)):02d}"


def filtered_indices(y: np.ndarray, noise_floor: float) -> np.ndarray:
    """Indices with V >= noise_floor; always include frame 0 (MECP) first."""
    keep = np.flatnonzero(y >= noise_floor)
    if 0 not in keep:
        keep = np.concatenate(([0], keep))
    else:
        keep = np.concatenate(([0], keep[keep != 0]))
    return keep.astype(int)


def eta1_abs_cos_angle(eta_ref: np.ndarray, eta_boot: np.ndarray) -> tuple[float, float]:
    """Sign-invariant |cos| and angle (deg) between two unit Cartesian directions."""
    a = np.asarray(eta_ref, dtype=float).ravel()
    b = np.asarray(eta_boot, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0, 90.0
    cos_abs = float(np.clip(abs(np.dot(a / na, b / nb)), 0.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_abs)))
    return cos_abs, angle_deg


def _pct_label(fraction: float) -> str:
    return f"{int(round(100 * fraction))}%"


def require_enough_frames(n: int, nslices: int, *, where: str) -> None:
    if n < _MIN_FRAMES_PCA:
        raise ValueError(
            f"{where}: need at least {_MIN_FRAMES_PCA} frames for PCA whitening; "
            f"got {n}"
        )
    if n < nslices:
        raise ValueError(
            f"{where}: need at least nslices={nslices} frames for SIR slicing; "
            f"got {n}"
        )


def compute_sir_core(coords: np.ndarray, y: np.ndarray, *, nslices: int, var: float):
    """Align, PCA-whiten, SIR, un-whiten. MECP must be at index 0."""
    n_frames, n_atoms, _ = coords.shape
    aligned = align_to_mecp(coords)
    ref_geom = aligned[0].copy()
    X = aligned.reshape(n_frames, 3 * n_atoms)
    X = X - X.mean(axis=0)
    Z, V, pca_eigvals, k = pca_whiten(X, var_cutoff=var)
    sir_vals, sir_vecs = sir(Z, y, n_slices=nslices)
    betas = V @ (sir_vecs / np.sqrt(pca_eigvals)[:, None])
    betas /= np.linalg.norm(betas, axis=0, keepdims=True)
    return {
        "ref_geom": ref_geom,
        "sir_vals": sir_vals,
        "betas": betas,
        "pca_eigvals": pca_eigvals,
        "k": k,
        "n_frames": n_frames,
        "n_atoms": n_atoms,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def write_molden_modes(path, symbols, ref_coords, vectors, eigenvalues):
    with open(path, "w") as f:
        f.write("[Molden Format]\n[FREQ]\n")
        for ev in eigenvalues:
            f.write(f"{ev:10.4f}\n")
        f.write("[FR-COORD]\n")
        for sym, (x, y, z) in zip(symbols, ref_coords):
            f.write(
                f"{sym:2s} {x * BOHR_PER_ANGSTROM:12.6f} "
                f"{y * BOHR_PER_ANGSTROM:12.6f} {z * BOHR_PER_ANGSTROM:12.6f}\n"
            )
        f.write("[FR-NORM-COORD]\n")
        for i, vec in enumerate(vectors, start=1):
            f.write(f"vibration {i}\n")
            for dx, dy, dz in vec:
                f.write(f"{dx:12.6f} {dy:12.6f} {dz:12.6f}\n")


# ---------------------------------------------------------------------------
# Eigenvalue bar charts
# ---------------------------------------------------------------------------

def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _spectrum_x_extent(eigenvalues: np.ndarray, n_plot: int) -> int:
    return min(int(n_plot), len(eigenvalues))


def _spectrum_ylim(max_value: float) -> tuple[float, float]:
    """Top of the y-axis on a clean step above the tallest bar.

    SIR eigenvalues are not normalized the way diffusion-map ones are, so the
    hardcoded (0, 1) of the DC spectrum plot would clip or dwarf them.
    """
    if not np.isfinite(max_value) or max_value <= 0.0:
        return 0.0, 1.0
    step = 10.0 ** np.floor(np.log10(max_value)) / 2.0
    return 0.0, float(step * np.ceil(max_value / step - 1e-12))


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    out_path: Path,
    *,
    color: str,
    n_plot: int,
) -> None:
    """Bar chart of the first ``n_plot`` SIR eigenvalues."""
    n = _spectrum_x_extent(eigenvalues, n_plot)
    if n == 0:
        return
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(3.33, 3.33))
    ax.bar(x, eigenvalues[:n], color=color, width=0.7)
    ax.set_xlabel(r"Eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(*_spectrum_ylim(float(np.max(eigenvalues[:n]))))
    tick_step = 2 if n > 12 else 1
    ax.set_xticks(x[::tick_step])
    _save_fig(fig, out_path)


def _combo_fig_width(n: int) -> float:
    return max(4.0, 0.45 * n + 1.5)


def _combo_group_geometry(n_labels: int, n: int) -> tuple[np.ndarray, float, np.ndarray]:
    group_centers = np.arange(1, n + 1, dtype=float)
    bar_width = 0.8 / n_labels
    offsets = (np.arange(n_labels) - (n_labels - 1) / 2.0) * bar_width
    return group_centers, bar_width, offsets


def plot_combined_grouped_eigenvalue_bars(
    spectra_by_label: dict[str, np.ndarray],
    out_path: Path,
    n_plot: int,
) -> None:
    """Grouped bar chart: per eigenvalue index, bars for each isotope touch."""
    labels = list(spectra_by_label.keys())
    n_labels = len(labels)
    if n_labels == 0:
        return
    n = min(int(n_plot), min(len(s) for s in spectra_by_label.values()))
    if n == 0:
        return

    group_centers, bar_width, offsets = _combo_group_geometry(n_labels, n)
    leg_fs = plt.rcParams["font.size"]

    fig, ax = plt.subplots(figsize=(_combo_fig_width(n), 3.33))
    y_max = 0.0
    for li, label in enumerate(labels):
        eig = spectra_by_label[label]
        y_max = max(y_max, float(np.max(eig[:n])))
        ax.bar(
            group_centers + offsets[li],
            eig[:n],
            width=bar_width * 0.95,
            color=series_color(label),
            label=label,
        )

    ax.set_xlabel(r"Eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(*_spectrum_ylim(y_max))
    tick_step = 2 if n > 12 else 1
    ax.set_xticks(group_centers[::tick_step])
    ax.legend(fontsize=leg_fs, loc="best")
    _save_fig(fig, out_path)


def _tukey_box_stats(values: np.ndarray) -> dict[str, float]:
    """Tukey five-number summary with whiskers at 1.5×IQR."""
    q1 = float(np.percentile(values, 25))
    median = float(np.percentile(values, 50))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    inside_lo = values[values >= lo_fence]
    inside_hi = values[values <= hi_fence]
    low = float(np.min(inside_lo)) if len(inside_lo) else float(np.min(values))
    high = float(np.max(inside_hi)) if len(inside_hi) else float(np.max(values))
    return {"q1": q1, "median": median, "q3": q3, "whislo": low, "whishi": high}


def _draw_box_whisker(ax, x: float, stats: dict[str, float], *, width: float) -> None:
    """Overlay a Tukey box-whisker centered at ``x`` (mean bar drawn separately)."""
    half = 0.35 * width
    q1, med, q3 = stats["q1"], stats["median"], stats["q3"]
    lo, hi = stats["whislo"], stats["whishi"]
    ax.plot([x, x], [lo, q1], color="k", linewidth=0.8, zorder=3)
    ax.plot([x, x], [q3, hi], color="k", linewidth=0.8, zorder=3)
    ax.plot([x - half * 0.6, x + half * 0.6], [lo, lo], color="k", linewidth=0.8, zorder=3)
    ax.plot([x - half * 0.6, x + half * 0.6], [hi, hi], color="k", linewidth=0.8, zorder=3)
    ax.add_patch(
        plt.Rectangle(
            (x - half, q1),
            2 * half,
            max(q3 - q1, 0.0),
            fill=False,
            edgecolor="k",
            linewidth=0.8,
            zorder=3,
        )
    )
    ax.plot([x - half, x + half], [med, med], color="k", linewidth=1.0, zorder=4)


def plot_bootstrap_eigenvalue_spectrum(
    sir_vals_boot: np.ndarray,
    out_path: Path,
    *,
    color: str,
    n_plot: int,
) -> None:
    """Mean bars over bootstrap replicates with Tukey box/whiskers overlaid."""
    n_reps, n_eig = sir_vals_boot.shape
    n = min(int(n_plot), n_eig)
    if n == 0 or n_reps == 0:
        return
    means = sir_vals_boot[:, :n].mean(axis=0)
    x = np.arange(1, n + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(3.33, 3.33))
    ax.bar(x, means, color=color, width=0.7, zorder=2)
    y_max = float(np.max(means))
    for i in range(n):
        stats = _tukey_box_stats(sir_vals_boot[:, i])
        _draw_box_whisker(ax, x[i], stats, width=0.7)
        y_max = max(y_max, stats["whishi"])
    ax.set_xlabel(r"Eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(*_spectrum_ylim(y_max))
    tick_step = 2 if n > 12 else 1
    ax.set_xticks(x[::tick_step])
    _save_fig(fig, out_path)


def plot_combined_bootstrap_eigenvalue_bars(
    boot_by_label: dict[str, np.ndarray],
    out_path: Path,
    n_plot: int,
) -> None:
    """Grouped mean bars + whiskers for H/D bootstrap spectra."""
    labels = list(boot_by_label.keys())
    n_labels = len(labels)
    if n_labels == 0:
        return
    n = min(int(n_plot), min(arr.shape[1] for arr in boot_by_label.values()))
    if n == 0:
        return

    group_centers, bar_width, offsets = _combo_group_geometry(n_labels, n)
    leg_fs = plt.rcParams["font.size"]
    fig, ax = plt.subplots(figsize=(_combo_fig_width(n), 3.33))
    y_max = 0.0
    for li, label in enumerate(labels):
        arr = boot_by_label[label]
        means = arr[:, :n].mean(axis=0)
        y_max = max(y_max, float(np.max(means)))
        xpos = group_centers + offsets[li]
        ax.bar(
            xpos,
            means,
            width=bar_width * 0.95,
            color=series_color(label),
            label=label,
            zorder=2,
        )
        for i in range(n):
            stats = _tukey_box_stats(arr[:, i])
            _draw_box_whisker(ax, xpos[i], stats, width=bar_width * 0.95)
            y_max = max(y_max, stats["whishi"])

    ax.set_xlabel(r"Eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(*_spectrum_ylim(y_max))
    tick_step = 2 if n > 12 else 1
    ax.set_xticks(group_centers[::tick_step])
    ax.legend(fontsize=leg_fs, loc="best")
    _save_fig(fig, out_path)


def plot_combined_filtered_bootstrap_spectrum(
    filtered_by_label: dict[str, np.ndarray],
    boot_by_fraction: dict[float, dict[str, np.ndarray]],
    out_path: Path,
    n_plot: int,
    *,
    isotope_order: list[str] | None = None,
) -> None:
    """One combo chart: filtered 100% then bootstrap fractions (desc), H then D.

    At each eigenvalue index, bars are fraction-major with isotopes nested:
    e.g. H 100%, D 100%, H 85%, D 85%, H 75%, D 75%. Bootstrap bars carry
    Tukey whiskers; the 100% filtered bars do not.
    """
    if isotope_order is None:
        isotope_order = sorted(filtered_by_label.keys())
    isotopes = [iso for iso in isotope_order if iso in filtered_by_label]
    if len(isotopes) < 2:
        return

    fractions_desc = sorted(boot_by_fraction.keys(), reverse=True)
    # Fraction-major: 100% filtered (H, D), then each bootstrap fraction (H, D).
    series: list[tuple[float, str, bool]] = []
    for iso in isotopes:
        if iso in filtered_by_label:
            series.append((1.0, iso, False))
    for frac in fractions_desc:
        by_iso = boot_by_fraction.get(frac, {})
        for iso in isotopes:
            if iso in by_iso:
                series.append((frac, iso, True))

    n_series = len(series)
    if n_series == 0:
        return

    lengths = [len(filtered_by_label[iso]) for iso in isotopes]
    for frac in fractions_desc:
        for iso, arr in boot_by_fraction[frac].items():
            lengths.append(arr.shape[1])
    n = min(int(n_plot), min(lengths))
    if n == 0:
        return

    group_centers, bar_width, offsets = _combo_group_geometry(n_series, n)
    fig_w = max(6.0, 0.55 * n * (n_series / 2.0) + 2.0)
    leg_fs = max(8.0, plt.rcParams["font.size"] - 2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.33))
    y_max = 0.0

    # Distinct hatch per bootstrap fraction so 75% vs 85% are visually separable.
    _BOOT_HATCHES = ("//", "xx", "\\\\", "..", "++", "OO")
    hatch_by_frac: dict[float, str] = {
        frac: _BOOT_HATCHES[i % len(_BOOT_HATCHES)]
        for i, frac in enumerate(fractions_desc)
    }

    for si, (frac, iso, is_boot) in enumerate(series):
        label = f"{iso} {_pct_label(frac)}"
        color = series_color(iso)
        hatch = hatch_by_frac[frac] if is_boot else None
        xpos = group_centers + offsets[si]
        if is_boot:
            arr = boot_by_fraction[frac][iso]
            heights = arr[:, :n].mean(axis=0)
            y_max = max(y_max, float(np.max(heights)))
            ax.bar(
                xpos,
                heights,
                width=bar_width * 0.95,
                color=color,
                hatch=hatch,
                edgecolor="k",
                linewidth=0.4,
                label=label,
                zorder=2,
            )
            for i in range(n):
                stats = _tukey_box_stats(arr[:, i])
                _draw_box_whisker(ax, xpos[i], stats, width=bar_width * 0.95)
                y_max = max(y_max, stats["whishi"])
        else:
            eig = filtered_by_label[iso]
            heights = eig[:n]
            y_max = max(y_max, float(np.max(heights)))
            ax.bar(
                xpos,
                heights,
                width=bar_width * 0.95,
                color=color,
                edgecolor="k",
                linewidth=0.4,
                label=label,
                zorder=2,
            )

    ax.set_xlabel(r"Eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(*_spectrum_ylim(y_max))
    tick_step = 2 if n > 12 else 1
    ax.set_xticks(group_centers[::tick_step])
    ax.legend(fontsize=leg_fs, loc="best", ncol=2)
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Per-isotope SIR (full write for all / filtered)
# ---------------------------------------------------------------------------

def run_sir_on_frames(
    coords: np.ndarray,
    y: np.ndarray,
    symbols,
    outdir: Path,
    *,
    isotope: str,
    run_name: str,
    xyz_path: Path,
    nslices: int,
    var: float,
    n_eigenvalues_plot: int,
    n_eigenvectors_viz: int,
    variant_label: str,
    noise_floor: float | None = None,
    n_total_frames: int | None = None,
) -> np.ndarray:
    """Run SIR on a pre-selected (coords, y) with MECP at index 0; write outputs."""
    outdir.mkdir(parents=True, exist_ok=True)
    require_enough_frames(len(y), nslices, where=f"{run_name}/{isotope}/{variant_label}")

    print(f"=== SIR {run_name} / {isotope} / {variant_label} ===")
    print(
        f"  {len(y)} frames, {coords.shape[1]} atoms; "
        f"V in [{y.min():.3f}, {y.max():.3f}] cm^-1"
    )

    core = compute_sir_core(coords, y, nslices=nslices, var=var)
    sir_vals = core["sir_vals"]
    betas = core["betas"]
    pca_eigvals = core["pca_eigvals"]
    k = core["k"]
    n_frames = core["n_frames"]
    n_atoms = core["n_atoms"]
    ref_geom = core["ref_geom"]

    print(f"  PCA components retained: {k} (var={var})")

    n_viz = min(max(1, int(n_eigenvectors_viz)), betas.shape[1])
    print(f"  {betas.shape[1]} SIR directions available; first {n_viz} flagged for viz")

    plot_eigenvalue_spectrum(
        sir_vals,
        outdir / "sir_eigenvalue_spectrum.png",
        color=series_color(isotope),
        n_plot=n_eigenvalues_plot,
    )

    total = float(sir_vals.sum()) if sir_vals.sum() > 0 else 1.0
    top_vectors = []
    report_lines = [
        f"SIR vibronic-coupling analysis: {run_name} / {isotope} / {variant_label}",
        f"XYZ: {xyz_path}",
        f"Frames used: {n_frames}  Atoms: {n_atoms}",
    ]
    if n_total_frames is not None:
        report_lines.append(f"Frames in master XYZ: {n_total_frames}")
    if noise_floor is not None:
        report_lines.append(f"Noise floor: {noise_floor:.6f} cm^-1")
    report_lines.extend(
        [
            f"Response: V = delta_cm^-1 / 2  "
            f"[min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}] cm^-1",
            f"MECP V: {y[0]:.6f} cm^-1",
            f"PCA components retained: {k} (var cutoff {var})",
            f"SIR slices: {nslices}",
            f"SIR directions available: {betas.shape[1]}",
            f"Directions visualized in step sir_viz: {n_viz}",
            f"Eigenvalues on the bar chart: "
            f"{_spectrum_x_extent(sir_vals, n_eigenvalues_plot)}",
            "",
            "Eigenvalue spectrum:",
            f"{'i':>4s}  {'lambda_i':>14s}  {'% of spectrum':>14s}",
        ]
    )
    for i in range(_spectrum_x_extent(sir_vals, n_eigenvalues_plot)):
        report_lines.append(
            f"{i + 1:4d}  {sir_vals[i]:14.6f}  {100.0 * sir_vals[i] / total:14.2f}"
        )
    report_lines.append("")
    report_lines.append(f"Top {n_viz} coupling-driving directions:")
    report_lines.append("")
    for r in range(n_viz):
        vec = betas[:, r].reshape(n_atoms, 3)
        top_vectors.append(vec)
        mags = np.linalg.norm(vec, axis=1)
        top_atoms = np.argsort(mags)[::-1][:5]
        report_lines.append(
            f"Direction {r + 1}: SIR eigenvalue = {sir_vals[r]:.6f}  "
            f"({100.0 * sir_vals[r] / total:.1f}% of spectrum)"
        )
        report_lines.append("  Dominant atomic contributions (index, symbol, |disp|):")
        for a in top_atoms:
            report_lines.append(f"    {a:4d}  {symbols[a]:2s}  {mags[a]:.4f}")
        report_lines.append("")
    (outdir / "sir_report.txt").write_text("\n".join(report_lines) + "\n")

    np.savez_compressed(
        outdir / "sir_results.npz",
        symbols=np.array(symbols),
        ref_coords=ref_geom,
        y=y,
        sir_vals=sir_vals,
        betas=betas,
        pca_eigvals=pca_eigvals,
        n_slices=nslices,
        var_cutoff=var,
        n_eigenvalues_plot=int(n_eigenvalues_plot),
        n_eigenvectors_viz=int(n_viz),
        isotope=isotope,
        run_name=run_name,
        variant=variant_label,
        noise_floor=float(noise_floor) if noise_floor is not None else np.nan,
    )

    write_molden_modes(
        outdir / "sir_modes.molden",
        symbols,
        ref_geom,
        top_vectors,
        sir_vals[:n_viz],
    )
    print(f"  Wrote {outdir}")
    return sir_vals


def run_bootstrap_fraction(
    coords_filt: np.ndarray,
    y_filt: np.ndarray,
    symbols,
    outdir: Path,
    *,
    isotope: str,
    run_name: str,
    xyz_path: Path,
    fraction: float,
    n_replicates: int,
    rng: np.random.Generator,
    nslices: int,
    var: float,
    n_eigenvalues_plot: int,
    noise_floor: float,
    eta1_filtered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical bootstrap on the filtered set; write NPZ + whisker plot + report.

    Returns ``(mean_sir_vals, eta1_cos_abs, eta1_angle_deg)``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    n_filtered = len(y_filt)
    m = int(math.floor(fraction * n_filtered))
    require_enough_frames(
        m, nslices, where=f"{run_name}/{isotope}/{outdir.name} (sample size m)"
    )
    if m < 2:
        raise ValueError(
            f"{run_name}/{isotope}/{outdir.name}: sample size m={m} too small "
            f"(fraction={fraction}, n_filtered={n_filtered})"
        )

    # Pool of non-MECP filtered frames (indices into coords_filt / y_filt).
    pool = np.arange(1, n_filtered)
    if len(pool) == 0:
        raise ValueError(
            f"{run_name}/{isotope}/{outdir.name}: filtered set has only the MECP; "
            "cannot bootstrap"
        )

    print(
        f"=== SIR bootstrap {run_name} / {isotope} / {outdir.name} "
        f"(f={fraction}, N={n_replicates}, m={m}) ==="
    )

    boot_vals: list[np.ndarray] = []
    cos_abs_list: list[float] = []
    angle_list: list[float] = []
    for rep in range(n_replicates):
        # Keep MECP at index 0; draw m-1 indices with replacement from the pool.
        drawn = rng.choice(pool, size=m - 1, replace=True)
        idx = np.concatenate(([0], drawn))
        core = compute_sir_core(
            coords_filt[idx], y_filt[idx], nslices=nslices, var=var
        )
        boot_vals.append(core["sir_vals"])
        eta1_boot = core["betas"][:, 0]
        cos_abs, angle_deg = eta1_abs_cos_angle(eta1_filtered, eta1_boot)
        cos_abs_list.append(cos_abs)
        angle_list.append(angle_deg)
        print(
            f"  replicate {rep + 1}/{n_replicates}: top λ = {core['sir_vals'][0]:.6f}; "
            f"|cos(η1)| = {cos_abs:.6f}; angle = {angle_deg:.3f} deg"
        )

    # Pad/truncate to a common eigenvalue length (shortest replicate).
    n_eig = min(len(v) for v in boot_vals)
    sir_vals_boot = np.stack([v[:n_eig] for v in boot_vals], axis=0)
    eta1_cos_abs = np.asarray(cos_abs_list, dtype=float)
    eta1_angle_deg = np.asarray(angle_list, dtype=float)

    plot_bootstrap_eigenvalue_spectrum(
        sir_vals_boot,
        outdir / "sir_eigenvalue_spectrum.png",
        color=series_color(isotope),
        n_plot=n_eigenvalues_plot,
    )

    means = sir_vals_boot.mean(axis=0)
    stds = sir_vals_boot.std(axis=0, ddof=1) if n_replicates > 1 else np.zeros_like(means)
    n_plot = _spectrum_x_extent(means, n_eigenvalues_plot)
    cos_mean = float(eta1_cos_abs.mean())
    cos_std = float(eta1_cos_abs.std(ddof=1)) if n_replicates > 1 else 0.0
    ang_mean = float(eta1_angle_deg.mean())
    ang_std = float(eta1_angle_deg.std(ddof=1)) if n_replicates > 1 else 0.0
    report_lines = [
        f"SIR bootstrap: {run_name} / {isotope} / {outdir.name}",
        f"XYZ: {xyz_path}",
        f"Noise floor: {noise_floor:.6f} cm^-1",
        f"Filtered frames: {n_filtered}",
        f"Fraction: {fraction}",
        f"Sample size m = floor(f * n_filtered): {m}",
        f"Replicates N: {n_replicates}",
        f"Sampling: with replacement; MECP always at index 0",
        f"PCA var cutoff: {var}",
        f"SIR slices: {nslices}",
        "",
        "η1 vs filtered 100% (sign-invariant |cos| and angle):",
        f"{'rep':>4s}  {'|cos|':>14s}  {'angle_deg':>14s}",
    ]
    for rep in range(n_replicates):
        report_lines.append(
            f"{rep + 1:4d}  {eta1_cos_abs[rep]:14.6f}  {eta1_angle_deg[rep]:14.3f}"
        )
    report_lines.extend(
        [
            f"mean ± std |cos|: {cos_mean:.6f} ± {cos_std:.6f}",
            f"mean ± std angle_deg: {ang_mean:.3f} ± {ang_std:.3f}",
            "",
            "Mean eigenvalue spectrum over replicates "
            "(bar = mean; box/whisker = Tukey Q1/median/Q3, whiskers to 1.5×IQR):",
            f"{'i':>4s}  {'mean_lambda':>14s}  {'std':>14s}  "
            f"{'Q1':>14s}  {'median':>14s}  {'Q3':>14s}",
        ]
    )
    for i in range(n_plot):
        stats = _tukey_box_stats(sir_vals_boot[:, i])
        report_lines.append(
            f"{i + 1:4d}  {means[i]:14.6f}  {stds[i]:14.6f}  "
            f"{stats['q1']:14.6f}  {stats['median']:14.6f}  {stats['q3']:14.6f}"
        )
    (outdir / "sir_report.txt").write_text("\n".join(report_lines) + "\n")

    np.savez_compressed(
        outdir / "sir_bootstrap.npz",
        sir_vals_boot=sir_vals_boot,
        eta1_cos_abs=eta1_cos_abs,
        eta1_angle_deg=eta1_angle_deg,
        fraction=float(fraction),
        n_replicates=int(n_replicates),
        sample_size=int(m),
        noise_floor=float(noise_floor),
        n_filtered=int(n_filtered),
        n_slices=int(nslices),
        var_cutoff=float(var),
        n_eigenvalues_plot=int(n_eigenvalues_plot),
        isotope=isotope,
        run_name=run_name,
    )
    print(
        f"  η1 vs filtered: |cos| = {cos_mean:.6f} ± {cos_std:.6f}; "
        f"angle = {ang_mean:.3f} ± {ang_std:.3f} deg"
    )
    print(f"  Wrote {outdir}")
    return means, eta1_cos_abs, eta1_angle_deg


def run_isotope_variants(
    xyz_path: Path,
    outdir: Path,
    *,
    isotope: str,
    run_name: str,
    nslices: int,
    var: float,
    n_eigenvalues_plot: int,
    n_eigenvectors_viz: int,
    noise_floor: float,
    bootstrap_fractions: list[float],
    bootstrap_n: int,
    bootstrap_seed: int,
    isotope_index: int,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute all / filtered / bootstrap variants for one isotope.

    Returns ``(spectra, boot_stacks)`` where ``spectra`` maps variant name to
    a 1-d eigenvalue vector (mean over replicates for bootstrap) and
    ``boot_stacks`` maps bootstrap variant names to ``(N, n_eig)`` arrays.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    symbols, coords_all, y_all, _meta = read_master_xyz(xyz_path)
    n_total = len(y_all)
    print(
        f"=== SIR {run_name} / {isotope}: {n_total} frames in master XYZ; "
        f"V in [{y_all.min():.3f}, {y_all.max():.3f}] cm^-1 ==="
    )

    spectra: dict[str, np.ndarray] = {}
    boot_stacks: dict[str, np.ndarray] = {}

    # --- all ---
    all_dir = outdir / "all"
    all_marker = all_dir / "sir_results.npz"
    if all_marker.is_file() and not force:
        print(f"Skip SIR {isotope}/all (exists: {all_marker})")
        spectra["all"] = np.load(all_marker, allow_pickle=True)["sir_vals"]
    else:
        spectra["all"] = run_sir_on_frames(
            coords_all,
            y_all,
            symbols,
            all_dir,
            isotope=isotope,
            run_name=run_name,
            xyz_path=xyz_path,
            nslices=nslices,
            var=var,
            n_eigenvalues_plot=n_eigenvalues_plot,
            n_eigenvectors_viz=n_eigenvectors_viz,
            variant_label="all",
            noise_floor=None,
            n_total_frames=n_total,
        )

    # --- filtered ---
    idx_filt = filtered_indices(y_all, noise_floor)
    coords_filt = coords_all[idx_filt]
    y_filt = y_all[idx_filt]
    n_dropped = n_total - len(idx_filt)
    print(
        f"  Filtered: kept {len(idx_filt)} / {n_total} frames "
        f"(dropped {n_dropped} with V < {noise_floor} cm^-1; MECP forced in)"
    )
    require_enough_frames(
        len(y_filt), nslices, where=f"{run_name}/{isotope}/filtered"
    )

    filt_dir = outdir / "filtered"
    filt_marker = filt_dir / "sir_results.npz"
    if filt_marker.is_file() and not force:
        print(f"Skip SIR {isotope}/filtered (exists: {filt_marker})")
        filt_data = np.load(filt_marker, allow_pickle=True)
        spectra["filtered"] = filt_data["sir_vals"]
        eta1_filtered = np.asarray(filt_data["betas"][:, 0], dtype=float)
    else:
        spectra["filtered"] = run_sir_on_frames(
            coords_filt,
            y_filt,
            symbols,
            filt_dir,
            isotope=isotope,
            run_name=run_name,
            xyz_path=xyz_path,
            nslices=nslices,
            var=var,
            n_eigenvalues_plot=n_eigenvalues_plot,
            n_eigenvectors_viz=n_eigenvectors_viz,
            variant_label="filtered",
            noise_floor=noise_floor,
            n_total_frames=n_total,
        )
        eta1_filtered = np.asarray(
            np.load(filt_marker, allow_pickle=True)["betas"][:, 0], dtype=float
        )

    # --- bootstrap ---
    similarity_lines = [
        f"η1 similarity vs filtered 100%: {run_name} / {isotope}",
        f"Sign-invariant |cos| = |η1_filt · η1_boot|; angle = arccos(|cos|) in degrees.",
        "",
    ]
    for fi, fraction in enumerate(bootstrap_fractions):
        vname = bootstrap_dirname(fraction)
        boot_dir = outdir / vname
        boot_marker = boot_dir / "sir_bootstrap.npz"
        if boot_marker.is_file() and not force:
            print(f"Skip SIR {isotope}/{vname} (exists: {boot_marker})")
            data = np.load(boot_marker, allow_pickle=True)
            stack = data["sir_vals_boot"]
            boot_stacks[vname] = stack
            spectra[vname] = stack.mean(axis=0)
            if "eta1_cos_abs" in data.files:
                cos_arr = np.asarray(data["eta1_cos_abs"], dtype=float)
                ang_arr = np.asarray(data["eta1_angle_deg"], dtype=float)
            else:
                cos_arr = np.array([])
                ang_arr = np.array([])
        else:
            # Seed offset by isotope and fraction for reproducibility across runs.
            seed = int(bootstrap_seed) + 1009 * (isotope_index + 1) + 17 * (fi + 1)
            rng = np.random.default_rng(seed)
            means, cos_arr, ang_arr = run_bootstrap_fraction(
                coords_filt,
                y_filt,
                symbols,
                boot_dir,
                isotope=isotope,
                run_name=run_name,
                xyz_path=xyz_path,
                fraction=fraction,
                n_replicates=bootstrap_n,
                rng=rng,
                nslices=nslices,
                var=var,
                n_eigenvalues_plot=n_eigenvalues_plot,
                noise_floor=noise_floor,
                eta1_filtered=eta1_filtered,
            )
            stack = np.load(boot_marker, allow_pickle=True)["sir_vals_boot"]
            boot_stacks[vname] = stack
            spectra[vname] = means

        similarity_lines.append(f"=== {vname} (fraction={fraction}) ===")
        if len(cos_arr) == 0:
            similarity_lines.append(
                "  (no eta1_cos_abs in cached NPZ; re-run with --force to compute)"
            )
        else:
            cos_std = float(cos_arr.std(ddof=1)) if len(cos_arr) > 1 else 0.0
            ang_std = float(ang_arr.std(ddof=1)) if len(ang_arr) > 1 else 0.0
            similarity_lines.append(
                f"{'rep':>4s}  {'|cos|':>14s}  {'angle_deg':>14s}"
            )
            for rep in range(len(cos_arr)):
                similarity_lines.append(
                    f"{rep + 1:4d}  {cos_arr[rep]:14.6f}  {ang_arr[rep]:14.3f}"
                )
            similarity_lines.append(
                f"mean ± std |cos|: {float(cos_arr.mean()):.6f} ± {cos_std:.6f}"
            )
            similarity_lines.append(
                f"mean ± std angle_deg: {float(ang_arr.mean()):.3f} ± {ang_std:.3f}"
            )
        similarity_lines.append("")

    (outdir / "eta1_vs_filtered.txt").write_text("\n".join(similarity_lines) + "\n")
    print(f"  Wrote {outdir / 'eta1_vs_filtered.txt'}")

    return spectra, boot_stacks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("ISOTOPE", "XYZ"),
        required=True,
        help="Isotope label and its master XYZ; repeat once per isotope",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--nslices", type=int, default=10)
    ap.add_argument("--var", type=float, default=0.95)
    ap.add_argument("--n-eigenvalues-plot", type=int, default=3)
    ap.add_argument("--n-eigenvectors-viz", type=int, default=3)
    ap.add_argument(
        "--noise-floor",
        type=float,
        default=1.0,
        help="cm^-1; frames with V < this are dropped for filtered/bootstrap",
    )
    ap.add_argument(
        "--bootstrap-fractions",
        type=float,
        nargs="+",
        default=[0.75, 0.85],
        help="Fractions of the filtered set for classical bootstrap",
    )
    ap.add_argument(
        "--bootstrap-n",
        type=int,
        default=100,
        help="Number of bootstrap replicates per fraction",
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Base RNG seed (offset per isotope/fraction)",
    )
    ap.add_argument(
        "--bootstrap-combo-spectrum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="One combo plot (filtered 100%% + all bootstrap fractions × H/D) "
        "instead of separate per-fraction combo bootstrap charts",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.noise_floor < 0.0:
        ap.error(f"--noise-floor must be >= 0; got {args.noise_floor!r}")
    if args.bootstrap_n < 1:
        ap.error(f"--bootstrap-n must be >= 1; got {args.bootstrap_n!r}")
    for f in args.bootstrap_fractions:
        if not (0.0 < f <= 1.0):
            ap.error(f"--bootstrap-fractions entries must be in (0, 1]; got {f!r}")

    labels_seen: set[str] = set()
    datasets = []
    for label, xyz in args.dataset:
        if label in labels_seen:
            ap.error(f"Duplicate dataset label: {label!r}")
        labels_seen.add(label)
        xyz_path = Path(xyz)
        if not xyz_path.is_file():
            ap.error(f"Master XYZ not found for {label!r}: {xyz_path}")
        datasets.append((label, xyz_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-variant spectra across isotopes for combo plots.
    combo_plain: dict[str, dict[str, np.ndarray]] = {
        "all": {},
        "filtered": {},
    }
    combo_boot: dict[str, dict[str, np.ndarray]] = {
        bootstrap_dirname(f): {} for f in args.bootstrap_fractions
    }

    for iso_i, (label, xyz_path) in enumerate(datasets):
        iso_outdir = args.output_dir / label
        spectra, boot_stacks = run_isotope_variants(
            xyz_path,
            iso_outdir,
            isotope=label,
            run_name=args.run_name,
            nslices=args.nslices,
            var=args.var,
            n_eigenvalues_plot=args.n_eigenvalues_plot,
            n_eigenvectors_viz=args.n_eigenvectors_viz,
            noise_floor=args.noise_floor,
            bootstrap_fractions=list(args.bootstrap_fractions),
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
            isotope_index=iso_i,
            force=args.force,
        )
        for vname in ("all", "filtered"):
            if vname in spectra:
                combo_plain[vname][label] = spectra[vname]
        for vname, stack in boot_stacks.items():
            combo_boot.setdefault(vname, {})[label] = stack

    if len(datasets) > 1:
        combo_root = args.output_dir / "combo_bar_plots"
        isotope_order = [label for label, _ in datasets]
        for vname, by_label in combo_plain.items():
            if len(by_label) < 2:
                continue
            out_dir = combo_root / vname
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_combined_grouped_eigenvalue_bars(
                by_label,
                out_dir / "sir_eigenvalue_spectrum.png",
                n_plot=args.n_eigenvalues_plot,
            )
            print(f"Combined bar plot written to {out_dir}")

        if args.bootstrap_combo_spectrum:
            # Map fraction -> {iso: sir_vals_boot}
            boot_by_fraction: dict[float, dict[str, np.ndarray]] = {}
            for frac in args.bootstrap_fractions:
                vname = bootstrap_dirname(frac)
                by_iso = combo_boot.get(vname, {})
                if by_iso:
                    boot_by_fraction[float(frac)] = by_iso
            if len(combo_plain.get("filtered", {})) >= 2 and boot_by_fraction:
                out_dir = combo_root / "bootstrap_combo"
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_combined_filtered_bootstrap_spectrum(
                    combo_plain["filtered"],
                    boot_by_fraction,
                    out_dir / "sir_eigenvalue_spectrum.png",
                    n_plot=args.n_eigenvalues_plot,
                    isotope_order=isotope_order,
                )
                print(f"Combined filtered+bootstrap spectrum written to {out_dir}")
        else:
            for vname, by_label in combo_boot.items():
                if len(by_label) < 2:
                    continue
                out_dir = combo_root / vname
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_combined_bootstrap_eigenvalue_bars(
                    by_label,
                    out_dir / "sir_eigenvalue_spectrum.png",
                    n_plot=args.n_eigenvalues_plot,
                )
                print(f"Combined bootstrap bar+whisker plot written to {out_dir}")

    print(f"All outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
