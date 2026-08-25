#!/usr/bin/env python3
"""Sliced inverse regression of the vibronic coupling on the seam manifold.

Reads a MECP-aligned master XYZ (frame 0 = MECP), response V = delta_cm^-1 / 2,
PCA-whitens Cartesians, runs SIR, and writes the eigenvalue spectrum plus the
eigenvectors (as Cartesian displacement directions) for step ``sir_viz``.

One invocation handles every isotope via repeated ``--dataset ISO XYZ``; a
grouped H/D eigenvalue bar chart is written to ``combo_bar_plots/`` when more
than one dataset is given.
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Per-isotope SIR
# ---------------------------------------------------------------------------

def run_sir(
    xyz_path: Path,
    outdir: Path,
    *,
    isotope: str,
    run_name: str,
    nslices: int,
    var: float,
    n_eigenvalues_plot: int,
    n_eigenvectors_viz: int,
) -> np.ndarray:
    """Run SIR for one isotope; return the eigenvalue spectrum."""
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== SIR {run_name} / {isotope} ===")
    symbols, coords, y, _meta = read_master_xyz(xyz_path)
    n_frames, n_atoms, _ = coords.shape
    print(
        f"  {n_frames} frames, {n_atoms} atoms; "
        f"V in [{y.min():.3f}, {y.max():.3f}] cm^-1"
    )

    aligned = align_to_mecp(coords)
    ref_geom = aligned[0].copy()
    X = aligned.reshape(n_frames, 3 * n_atoms)
    X = X - X.mean(axis=0)

    Z, V, pca_eigvals, k = pca_whiten(X, var_cutoff=var)
    print(f"  PCA components retained: {k} (var={var})")

    sir_vals, sir_vecs = sir(Z, y, n_slices=nslices)
    betas = V @ (sir_vecs / np.sqrt(pca_eigvals)[:, None])
    betas /= np.linalg.norm(betas, axis=0, keepdims=True)

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
        f"SIR vibronic-coupling analysis: {run_name} / {isotope}",
        f"XYZ: {xyz_path}",
        f"Frames: {n_frames}  Atoms: {n_atoms}",
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
    ap.add_argument("--n-eigenvalues-plot", type=int, default=10)
    ap.add_argument("--n-eigenvectors-viz", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

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

    spectra_by_label: dict[str, np.ndarray] = {}
    for label, xyz_path in datasets:
        outdir = args.output_dir / label
        marker = outdir / "sir_results.npz"
        if marker.is_file() and not args.force:
            print(f"Skip SIR {label} (exists: {marker}); reusing eigenvalues")
            spectra_by_label[label] = np.load(marker, allow_pickle=True)["sir_vals"]
            continue
        spectra_by_label[label] = run_sir(
            xyz_path,
            outdir,
            isotope=label,
            run_name=args.run_name,
            nslices=args.nslices,
            var=args.var,
            n_eigenvalues_plot=args.n_eigenvalues_plot,
            n_eigenvectors_viz=args.n_eigenvectors_viz,
        )

    if len(spectra_by_label) > 1:
        combo_dir = args.output_dir / "combo_bar_plots"
        combo_dir.mkdir(parents=True, exist_ok=True)
        plot_combined_grouped_eigenvalue_bars(
            spectra_by_label,
            combo_dir / "sir_eigenvalue_spectrum.png",
            n_plot=args.n_eigenvalues_plot,
        )
        print(f"Combined bar plot written to {combo_dir}")

    print(f"All outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
