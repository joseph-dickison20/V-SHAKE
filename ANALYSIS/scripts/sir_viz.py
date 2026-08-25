#!/usr/bin/env python3
"""PyMOL visualization of SIR eigenvectors, g_perp, and grad V at the MECP.

Loads ``sir_results.npz`` from step ``sir`` and the NEO-MSDFT Q-Chem MECP
output, then renders for each isotope:

* ``sir{k}``      — the first ``n_vectors`` SIR eigenvectors
* ``gperp``       — the tunneling coordinate, g_perp = grad(sigma01)
* ``gradV``       — grad V = (grad E1 - grad E0) / 2
* ``gperp_gradV`` — both of the above overlaid on one structure

Static arrow overlays are drawn on the full MECP structure with the transferred
nucleus' density rendered as an isomesh from a Gaussian cube file. Swing GIFs
show the heavy-atom skeleton only (no density).

Three families of orthogonality checks are reported, each as a per-atom
decomposition bar chart plus a text table: cos(g_perp, grad V) in
``gperp_gradV/``, and cos(g_perp, sir{k}) and cos(grad V, sir{k}) in each
``sir{k}/``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PYMOL_SET_SETTINGS = [
    ("ray_opaque_background", 1),
    ("ambient", 0.2),
    ("ray_shadows", 0),
    ("spec_reflect", 0.6),
    ("spec_power", 600),
    ("spec_count", 3),
    ("shininess", 70),
    ("reflect", 0.5),
    ("ray_trace_mode", 1),
    ("ray_texture", 2),
    ("antialias", 3),
    ("fog", 0),
]

_ARROW_SHAFT_RADIUS = 0.03
_ARROW_HEAD_RADIUS = 0.08
_ARROW_HEAD_FRAC = 0.30
_ARROW_MIN_LEN = 1e-6

_VIZ_FAMILIES = ("sir", "gperp", "gradv")


# ---------------------------------------------------------------------------
# XYZ I/O
# ---------------------------------------------------------------------------

def write_xyz_frame(f, elements, coords, comment=""):
    n_atoms = len(elements)
    f.write(f"{n_atoms}\n")
    f.write(f"{comment}\n")
    for i in range(n_atoms):
        f.write(
            f"{elements[i]:4s} {coords[i, 0]:16.8f} "
            f"{coords[i, 1]:16.8f} {coords[i, 2]:16.8f}\n"
        )


def write_xyz_trajectory(path, elements, frames, comments=None):
    comments = comments or ["" for _ in frames]
    with open(path, "w") as f:
        for coords, comment in zip(frames, comments):
            write_xyz_frame(f, elements, coords, comment=comment)


def build_swing_frames(R0, v, amplitude, n_frames):
    phis = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    frames, comments = [], []
    for j, phi in enumerate(phis):
        frames.append(R0 + amplitude * np.sin(phi) * v)
        comments.append(f"frame={j} phi={phi:.6f} disp={amplitude * np.sin(phi):+.6f}")
    return frames, comments


# ---------------------------------------------------------------------------
# Q-Chem parsing (shared convention with DC_ANALYSIS/scripts/jacobian_viz.py)
# ---------------------------------------------------------------------------

def parse_qchem_vector(filepath, header, n_skip_rows=2):
    """Parse an (n_total x 3) block following ``header``, in Hartree/Bohr.

    The first ``n_skip_rows`` rows are the quantum-nuclear basis centers and are
    dropped so the result aligns with the classical atom ordering.
    """
    with open(filepath) as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if header in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Header '{header}' not found in {filepath}")

    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            break
        if len(vals) != 3:
            break
        rows.append(vals)

    if len(rows) <= n_skip_rows:
        raise ValueError(
            f"Parsed {len(rows)} rows for '{header}', need > {n_skip_rows} "
            f"to skip basis centers."
        )
    return np.array(rows)[n_skip_rows:]


def parse_reference_geometry(filepath):
    """Elements and coordinates (Angstrom) from the Standard Nuclear Orientation."""
    with open(filepath) as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "Standard Nuclear Orientation" in line:
            start = i
            break
    if start is None:
        raise ValueError(f"'Standard Nuclear Orientation' block not found in {filepath}")

    i = start + 1
    while i < len(lines) and set(lines[i].strip()) != {"-"}:
        i += 1
    i += 1

    elements, coords = [], []
    while i < len(lines):
        stripped = lines[i].strip()
        if set(stripped) == {"-"} or not stripped:
            break
        parts = stripped.split()
        if len(parts) < 5:
            break
        try:
            xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
        except ValueError:
            break
        elements.append(parts[1])
        coords.append(xyz)
        i += 1

    if not elements:
        raise ValueError(
            f"Parsed no atoms from the Standard Nuclear Orientation block in {filepath}"
        )
    return elements, np.array(coords, dtype=float)


# ---------------------------------------------------------------------------
# Cosine analysis
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a_flat = np.asarray(a).ravel()
    b_flat = np.asarray(b).ravel()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def per_atom_dot_products(a, b):
    """Per-atom c_i = (a_i . b_i) / (|a| |b|), so that sum_i c_i = cos(theta)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return np.zeros(a.shape[0])
    return np.sum(a * b, axis=1) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Per-atom cos(theta) bar chart
# ---------------------------------------------------------------------------

_Y_SCALE = 1e-2
_BAR_PLOT_ELEMENT_ORDER = ("O", "C", "N", "H")


def _symmetric_y_axis(ax, values):
    """Equal +/- y limits; <=5 ticks (endpoints and 0); labels in units of 10^-2."""
    vals = np.asarray(values, dtype=float)
    raw = float(np.max(np.abs(vals))) if vals.size else 0.0
    step = 0.01
    if raw == 0.0:
        ymax = step
    else:
        ymax = step * np.ceil(raw / step - 1e-12)
        while ymax / step > 8:
            step *= 2
            ymax = step * np.ceil(raw / step - 1e-12)

    s_max = ymax / _Y_SCALE
    if s_max <= 3.0:
        ticks_scaled = np.array([-s_max, 0.0, s_max])
    else:
        half = s_max / 2.0
        ticks_scaled = np.array([-s_max, -half, 0.0, half, s_max])

    ax.set_ylim(-ymax, ymax)
    ax.set_yticks(ticks_scaled * _Y_SCALE)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks_scaled])
    ax.margins(y=0)
    ax.set_autoscaley_on(False)


def _bar_plot_element_rank(elem: str) -> int:
    try:
        return _BAR_PLOT_ELEMENT_ORDER.index(elem)
    except ValueError:
        return len(_BAR_PLOT_ELEMENT_ORDER)


def _bar_plot_atom_order(elements: list[str]) -> np.ndarray:
    """Indices sorted by O, C, N, H, then other elements; stable within each type."""
    return np.array(
        sorted(range(len(elements)), key=lambda i: (_bar_plot_element_rank(elements[i]), i)),
        dtype=int,
    )


def _bar_plot_atom_labels(elements: list[str], order: np.ndarray) -> list[str]:
    """Labels like O1, O2, C1, ... with numbering restarted per element."""
    counts: dict[str, int] = {}
    labels: list[str] = []
    for i in order:
        elem = elements[i]
        counts[elem] = counts.get(elem, 0) + 1
        labels.append(f"{elem}{counts[elem]}")
    return labels


def make_cos_figure(
    elements,
    per_atom_contrib,
    cos_theta,
    output_path,
    *,
    xlim=None,
    ylim=None,
):
    """Per-atom dot-product decomposition bar chart for cos(a, b)."""
    sans = ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans", "sans-serif"]
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 3.33), constrained_layout=True)

    order = _bar_plot_atom_order(elements)
    contrib = np.asarray(per_atom_contrib, dtype=float)[order]
    n_atoms = len(elements)
    x_pos = np.arange(n_atoms)

    colors = ["#CC0000" if c < 0 else "#0055CC" for c in contrib]
    ax.bar(x_pos, contrib, color=colors, edgecolor="none", width=0.7, alpha=0.85)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.margins(y=0)
        ax.set_autoscaley_on(False)
    else:
        _symmetric_y_axis(ax, contrib)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(_bar_plot_atom_labels(elements, order), rotation=45, ha="right")
    ax.set_ylabel(r"Per-atom contrib. to $\cos\theta$ [$10^{-2}$ arb.]")
    ax.axhline(0, color="black", linewidth=0.5)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        ax.set_xlim(-0.5, n_atoms - 0.5)

    theta_deg = np.degrees(np.arccos(np.clip(abs(cos_theta), 0, 1)))
    ax.text(
        0.98,
        0.95,
        f"$\\cos\\theta = {cos_theta:+.4f}$\n$\\theta = {theta_deg:.1f}°$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path, bbox_inches="tight")
    print(f"  Figure saved to {output_path}")
    plt.close(fig)


def write_cos_txt(path, label_a, label_b, elements, contrib, cos_theta, a, b):
    """Plain-text cos value plus the per-atom contribution table."""
    theta_deg = np.degrees(np.arccos(np.clip(abs(cos_theta), 0, 1)))
    col_a = f"|{label_a}_i|"
    col_b = f"|{label_b}_i|"
    with open(path, "w") as f:
        f.write(f"# cos({label_a}, {label_b}) decomposition\n")
        f.write(f"cos_theta   = {cos_theta:+.8f}\n")
        f.write(f"|cos_theta| = {abs(cos_theta):.8f}\n")
        f.write(f"theta_deg   = {theta_deg:.4f}\n")
        f.write("# SIR eigenvector signs are arbitrary; theta uses |cos_theta|.\n")
        f.write("\n")
        f.write(f"{'Atom':<6s} {'Contribution':>14s} {col_a:>14s} {col_b:>14s}\n")
        f.write(f"{'-' * 52}\n")
        for i, elem in enumerate(elements):
            f.write(
                f"{elem}{i + 1:<4d} {contrib[i]:+14.8f} "
                f"{np.linalg.norm(a[i]):14.6e} {np.linalg.norm(b[i]):14.6e}\n"
            )
        f.write(f"{'Sum':<6s} {contrib.sum():+14.8f}\n")
    print(f"  Wrote cosine summary to {path}")


def report_overlap(out_dir, stem, label_a, label_b, a, b, elements, *, xlim, ylim):
    """Cosine similarity between two Cartesian vectors: chart + text table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cos_theta = cosine_similarity(a, b)
    contrib = per_atom_dot_products(a, b)
    theta_deg = np.degrees(np.arccos(np.clip(abs(cos_theta), 0, 1)))
    print(
        f"  cos({label_a}, {label_b}) = {cos_theta:+.6f}  (theta = {theta_deg:.1f} deg)"
    )
    make_cos_figure(
        elements, contrib, cos_theta, out_dir / f"{stem}_barchart.png",
        xlim=xlim, ylim=ylim,
    )
    write_cos_txt(
        out_dir / f"{stem}.txt", label_a, label_b, elements, contrib, cos_theta, a, b
    )
    return cos_theta


# ---------------------------------------------------------------------------
# PyMOL script generation
# ---------------------------------------------------------------------------

def _pymol_preamble(body, xyz_path, *, mol_transparency=None):
    body.append("cmd.reinitialize()")
    body.append("cmd.set('auto_zoom', 0)")
    body.append("cmd.bg_color('white')")
    for name, val in _PYMOL_SET_SETTINGS:
        body.append(f"cmd.set({name!r}, {val!r})")
    body.append(f"cmd.load({str(xyz_path)!r}, 'mol')")
    body.append("cmd.hide('lines', 'mol')")
    body.append("cmd.show('sticks', 'mol')")
    body.append("cmd.set_bond('stick_radius', 0.08, 'mol')")
    body.append("cmd.show('spheres', 'mol')")
    body.append("cmd.set('sphere_scale', 0.2, 'mol')")
    body.append("cmd.color('gray60', '(mol and elem C)')")
    body.append("cmd.set('stick_color', 'gray90', 'mol')")
    if mol_transparency is not None:
        t = float(mol_transparency)
        # Global and object-scoped: ray tracing misses object-only sets on some builds.
        body.append(f"cmd.set('sphere_transparency', {t!r})")
        body.append(f"cmd.set('stick_transparency', {t!r})")
        body.append(f"cmd.set('sphere_transparency', {t!r}, 'mol')")
        body.append(f"cmd.set('stick_transparency', {t!r}, 'mol')")


def _pymol_camera(body, *, turn_x, turn_y, turn_z, zoom_center, center_selection):
    body.append("cmd.set('state', 1)")
    body.append("cmd.reset()")
    body.append(f"cmd.turn('x', {float(turn_x)!r})")
    body.append(f"cmd.turn('y', {float(turn_y)!r})")
    body.append(f"cmd.turn('z', {float(turn_z)!r})")
    body.append(f"cmd.center({center_selection!r})")
    body.append(f"cmd.zoom('center', {float(zoom_center)!r})")
    body.append("view = cmd.get_view()")


def _pymol_density(body, cube_path, isovalue, color_rgb):
    """Nuclear-density isomesh from a Gaussian cube; hides the cube's own atoms."""
    r, g, b = (int(c) for c in color_rgb)
    body.append(f"cmd.load({str(cube_path)!r}, 'dens')")
    body.append("cmd.hide('everything', 'dens')")
    body.append(f"cmd.isomesh('density', 'dens', {float(isovalue)!r})")
    body.append(f"cmd.set_color('denscol', [{r}, {g}, {b}])")
    body.append("cmd.color('denscol', 'density')")


def _pymol_png(body, png_path, *, png_width, png_height, png_dpi, ray):
    body.append(f"W, H, DPI = {int(png_width)}, {int(png_height)}, {int(png_dpi)}")
    body.append(f"png = {str(png_path)!r}")
    if int(ray):
        body.append("cmd.png(png, width=W, height=H, dpi=DPI, ray=1)")
    else:
        body.append("cmd.draw(W, H, antialias=1)")
        body.append("cmd.png(png, dpi=DPI)")


def _write_pml(pml_path, body):
    with open(pml_path, "w") as f:
        f.write("\n".join(["python"] + body + ["python end"]) + "\n")


def write_swing_pml(pml_path, xyz_path, frames_dir, *,
                    turn_x, turn_y, turn_z, zoom_center,
                    png_width, png_height, png_dpi, ray):
    """Swing-GIF frames: heavy-atom skeleton only, no density, opaque."""
    body = ["from pymol import cmd", "import os"]
    _pymol_preamble(body, xyz_path)
    body.append("n_states = cmd.count_states('mol')")
    _pymol_camera(
        body,
        turn_x=turn_x,
        turn_y=turn_y,
        turn_z=turn_z,
        zoom_center=zoom_center,
        center_selection="mol",
    )
    body.append(f"frames_dir = {str(frames_dir)!r}")
    body.append(f"W, H, DPI = {int(png_width)}, {int(png_height)}, {int(png_dpi)}")
    body.append("for s in range(1, n_states + 1):")
    body.append("    cmd.set('state', s)")
    body.append("    cmd.set_view(view)")
    body.append("    png = os.path.join(frames_dir, 'frame_%04d.png' % (s - 1))")
    if int(ray):
        body.append("    cmd.png(png, width=W, height=H, dpi=DPI, ray=1)")
    else:
        body.append("    cmd.draw(W, H, antialias=1)")
        body.append("    cmd.png(png, dpi=DPI)")
    body.append("cmd.quit()")
    _write_pml(pml_path, body)


def write_arrows_pml(pml_path, xyz_path, png_path, overlays, *,
                     turn_x, turn_y, turn_z, zoom_center,
                     png_width, png_height, png_dpi, ray,
                     mol_transparency, density=None):
    """Static overlay PNG: MECP structure, nuclear density, one arrow set per overlay.

    ``overlays`` is a list of ``(bases, tips, color)`` triples, so a single figure
    can carry several vectors in distinguishable colors. ``color`` accepts a hex
    spec (``0xRRGGBB`` / ``#RRGGBB``) or a PyMOL named color. ``density``, when
    given, is ``(cube_path, isovalue, (r, g, b))``.

    The camera is centered on ``mol`` alone so the framing stays registered with
    the corresponding GIF, which has no density map loaded.
    """
    t = float(mol_transparency)
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"mol_transparency must be in [0, 1]; got {mol_transparency!r}")

    body = ["from pymol import cmd", "from pymol.cgo import CYLINDER, CONE", "import os"]
    _pymol_preamble(body, xyz_path, mol_transparency=t)
    if density is not None:
        _pymol_density(body, density[0], density[1], density[2])
    _pymol_camera(
        body,
        turn_x=turn_x,
        turn_y=turn_y,
        turn_z=turn_z,
        zoom_center=zoom_center,
        center_selection="mol",
    )

    body.append(f"SHAFT_R = {_ARROW_SHAFT_RADIUS!r}")
    body.append(f"HEAD_R = {_ARROW_HEAD_RADIUS!r}")
    body.append(f"HEAD_FRAC = {_ARROW_HEAD_FRAC!r}")
    body.append(f"MIN_LEN = {_ARROW_MIN_LEN!r}")
    body.append("def _rgb(col_spec):")
    body.append("    if col_spec.startswith('0x') or col_spec.startswith('#'):")
    body.append("        _h = col_spec[2:] if col_spec.startswith('0x') else col_spec[1:]")
    body.append("        return tuple(int(_h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))")
    body.append("    return cmd.get_color_tuple(cmd.get_color_index(col_spec))")

    for idx, (bases, tips, color) in enumerate(overlays):
        body.append(f"bases = {bases!r}")
        body.append(f"tips = {tips!r}")
        body.append(f"r, g, b = _rgb({str(color)!r})")
        body.append("arrow = []")
        body.append("for base, tip in zip(bases, tips):")
        body.append("    dx, dy, dz = tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]")
        body.append("    length = (dx * dx + dy * dy + dz * dz) ** 0.5")
        body.append("    if length < MIN_LEN:")
        body.append("        continue")
        body.append("    sx = base[0] + (1.0 - HEAD_FRAC) * dx")
        body.append("    sy = base[1] + (1.0 - HEAD_FRAC) * dy")
        body.append("    sz = base[2] + (1.0 - HEAD_FRAC) * dz")
        body.append(
            "    arrow += [CYLINDER, base[0], base[1], base[2], sx, sy, sz, "
            "SHAFT_R, r, g, b, r, g, b]"
        )
        body.append(
            "    arrow += [CONE, sx, sy, sz, tip[0], tip[1], tip[2], HEAD_R, "
            "0.0, r, g, b, r, g, b, 1.0, 1.0]"
        )
        body.append(f"cmd.load_cgo(arrow, 'arrows{idx}')")

    body.append("cmd.set_view(view)")
    _pymol_png(
        body, png_path, png_width=png_width, png_height=png_height, png_dpi=png_dpi, ray=ray
    )
    body.append("cmd.quit()")
    _write_pml(pml_path, body)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _imagemagick_exe() -> str:
    return shutil.which("convert") or shutil.which("magick") or "convert"


def _pymol_exe() -> str:
    return shutil.which("pymol") or "pymol"


def _assemble_gif(frame_pngs, gif_path, cfg) -> None:
    cmd = [_imagemagick_exe(), "-delay", str(cfg.gif_delay), "-loop", str(cfg.gif_loop)]
    extra = cfg.gif_extra_args or ""
    if extra.strip():
        cmd.extend(shlex.split(extra))
    cmd.extend(str(p) for p in frame_pngs)
    cmd.append(str(gif_path))
    print(f"  Assembling GIF ({len(frame_pngs)} frames) -> {gif_path}")
    subprocess.run(cmd, check=True)


def render_vector_gif(name, R0, v, elements, out_dir, cfg, *, force):
    """Swing GIF of one vector: skeleton only, deliberately without density."""
    out_dir = Path(out_dir)
    gif_path = out_dir / f"{name}.gif"
    if gif_path.is_file() and not force:
        print(f"  Skip {name} GIF (exists: {gif_path})")
        return gif_path

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if frames_dir.is_dir():
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = out_dir / f"{name}.xyz"
    frames, comments = build_swing_frames(R0, v, cfg.amplitude, cfg.n_frames)
    write_xyz_trajectory(xyz_path, elements, frames, comments)
    print(f"  Wrote {len(frames)}-frame swing trajectory to {xyz_path}")

    pml_path = out_dir / f"{name}.pml"
    write_swing_pml(
        pml_path,
        xyz_path,
        frames_dir,
        turn_x=cfg.turn_x,
        turn_y=cfg.turn_y,
        turn_z=cfg.turn_z,
        zoom_center=cfg.zoom_center,
        png_width=cfg.png_width,
        png_height=cfg.png_height,
        png_dpi=cfg.png_dpi,
        ray=cfg.ray,
    )

    print(f"  Rendering {cfg.n_frames} frames with PyMOL ...")
    subprocess.run([_pymol_exe(), "-qc", str(pml_path)], check=True)

    frame_pngs = sorted(frames_dir.glob("frame_*.png"))
    if not frame_pngs:
        raise RuntimeError(f"PyMOL produced no PNG frames for {name} ({frames_dir})")
    _assemble_gif(frame_pngs, gif_path, cfg)
    return gif_path


def render_vector_arrows(name, R0, vectors, elements, out_dir, cfg, *,
                         arrow_scale, colors, density, force):
    """Static arrow overlay on the MECP structure, with the nuclear-density mesh.

    ``vectors`` and ``colors`` are parallel sequences, so one figure can show
    several vectors at once (used for the combined g_perp / grad V overlay).
    """
    out_dir = Path(out_dir)
    png_path = out_dir / f"{name}_arrows.png"
    if png_path.is_file() and not force:
        print(f"  Skip {name} arrow overlay (exists: {png_path})")
        return png_path

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_xyz = out_dir / f"{name}_arrows.xyz"
    write_xyz_trajectory(ref_xyz, elements, [R0], ["arrows reference (MECP geometry R0)"])

    n = len(elements)
    overlays = []
    for v, color in zip(vectors, colors):
        bases = [[float(R0[i, 0]), float(R0[i, 1]), float(R0[i, 2])] for i in range(n)]
        tips = [
            [
                float(R0[i, 0] + arrow_scale * v[i, 0]),
                float(R0[i, 1] + arrow_scale * v[i, 1]),
                float(R0[i, 2] + arrow_scale * v[i, 2]),
            ]
            for i in range(n)
        ]
        overlays.append((bases, tips, color))

    pml_path = out_dir / f"{name}_arrows.pml"
    write_arrows_pml(
        pml_path,
        ref_xyz,
        png_path,
        overlays,
        turn_x=cfg.turn_x,
        turn_y=cfg.turn_y,
        turn_z=cfg.turn_z,
        zoom_center=cfg.zoom_center,
        png_width=cfg.png_width,
        png_height=cfg.png_height,
        png_dpi=cfg.png_dpi,
        ray=cfg.ray,
        mol_transparency=cfg.arrow_mol_transparency,
        density=density,
    )

    print(f"  Rendering static arrow overlay with PyMOL -> {png_path}")
    subprocess.run([_pymol_exe(), "-qc", str(pml_path)], check=True)
    if not png_path.is_file():
        raise RuntimeError(f"PyMOL produced no arrow-overlay PNG for {name}")
    return png_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [p for p in str(text).replace(",", " ").split() if p]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"density color must be three 0-255 integers, got {text!r}"
        )
    vals = []
    for p in parts:
        v = int(float(p))
        if not (0 <= v <= 255):
            raise argparse.ArgumentTypeError(f"density color component out of range: {p!r}")
        vals.append(v)
    return (vals[0], vals[1], vals[2])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--qchem", type=Path, required=True,
                    help="NEO-MSDFT Q-Chem MECP output (msdft_0.out)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-vectors", type=int, default=0,
                    help="SIR eigenvectors to visualize; 0 = use the value in the NPZ")
    ap.add_argument("--n-skip-rows", type=int, default=2,
                    help="Leading rows to skip in Q-Chem vector blocks")
    # arrows
    ap.add_argument("--arrow-scale", type=float, default=3.0)
    ap.add_argument("--arrow-color", default="#d62728",
                    help="SIR eigenvector arrow color for this isotope")
    ap.add_argument("--gperp-color", default="#2ca02c")
    ap.add_argument("--gradv-color", default="#ff7f0e")
    ap.add_argument("--arrow-mol-transparency", type=float, default=0.2)
    # density
    ap.add_argument("--density-cube", type=Path, default=None,
                    help="Gaussian cube of the transferred nucleus' density")
    ap.add_argument("--density-isovalue", type=float, default=0.04)
    ap.add_argument("--density-color", type=_parse_rgb, default=(128, 229, 255))
    # swing / render
    ap.add_argument("--amplitude", type=float, default=0.5)
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--zoom-center", type=float, default=3.0)
    ap.add_argument("--turn-x", type=float, default=0.0)
    ap.add_argument("--turn-y", type=float, default=0.0)
    ap.add_argument("--turn-z", type=float, default=0.0)
    ap.add_argument("--png-width", type=int, default=2000)
    ap.add_argument("--png-height", type=int, default=1400)
    ap.add_argument("--png-dpi", type=int, default=600)
    ap.add_argument("--ray", type=int, default=1)
    ap.add_argument("--gif-delay", type=int, default=5)
    ap.add_argument("--gif-loop", type=int, default=0)
    ap.add_argument("--gif-extra-args", default="")
    ap.add_argument("--viz-overrides", default="{}",
                    help="JSON: per-vector-family overrides keyed sir/gperp/gradv")
    # bar charts
    ap.add_argument("--barchart-xlim", nargs=2, type=float,
                    metavar=("XMIN", "XMAX"), default=None)
    ap.add_argument("--barchart-ylim", nargs=2, type=float,
                    metavar=("YMIN", "YMAX"), default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not (0.0 <= args.arrow_mol_transparency <= 1.0):
        ap.error(
            f"--arrow-mol-transparency must be in [0, 1] (got {args.arrow_mol_transparency!r})"
        )

    # --- Per-vector-family visualization parameters ---
    _viz_fields = ("amplitude", "n_frames", "zoom_center", "turn_x", "turn_y", "turn_z",
                   "png_width", "png_height", "png_dpi", "ray", "gif_delay", "gif_loop",
                   "gif_extra_args")
    base_viz = {f: getattr(args, f) for f in _viz_fields}
    base_viz["arrow_mol_transparency"] = args.arrow_mol_transparency
    try:
        viz_overrides = json.loads(args.viz_overrides or "{}")
    except json.JSONDecodeError as exc:
        ap.error(f"--viz-overrides is not valid JSON: {exc}")
    for key in viz_overrides:
        if key not in _VIZ_FAMILIES:
            ap.error(
                f"--viz-overrides key {key!r} unknown; expected one of {_VIZ_FAMILIES}"
            )

    def params_for(family):
        p = dict(base_viz)
        p.update(viz_overrides.get(family, {}))
        return SimpleNamespace(**p)

    # --- SIR results ---
    data = np.load(args.npz, allow_pickle=True)
    symbols = [str(s) for s in data["symbols"]]
    ref = np.asarray(data["ref_coords"], dtype=float)
    betas = np.asarray(data["betas"], dtype=float)
    n_atoms = len(symbols)
    n_avail = betas.shape[1]
    n_viz = int(args.n_vectors) if args.n_vectors > 0 else int(data["n_eigenvectors_viz"])
    if n_viz > n_avail:
        print(
            f"WARNING: requested {n_viz} SIR eigenvectors but only {n_avail} stored; "
            f"clipping to {n_avail}."
        )
        n_viz = n_avail
    n_viz = max(1, n_viz)

    # --- Q-Chem vectors ---
    print(f"Parsing {args.qchem} ...")
    g_perp = parse_qchem_vector(
        args.qchem, "Gradient of sigma01 (gradH00 - gradH11):", args.n_skip_rows
    )
    grad_E0 = parse_qchem_vector(args.qchem, "Gradient of E0:", args.n_skip_rows)
    grad_E1 = parse_qchem_vector(args.qchem, "Gradient of E1:", args.n_skip_rows)
    if not (g_perp.shape == grad_E0.shape == grad_E1.shape):
        print(
            f"ERROR: gradient blocks disagree — sigma01 {g_perp.shape}, "
            f"E0 {grad_E0.shape}, E1 {grad_E1.shape}.",
            file=sys.stderr,
        )
        sys.exit(1)
    # V = Delta/2, so grad V = (grad E1 - grad E0) / 2.
    grad_V = 0.5 * (grad_E1 - grad_E0)
    print(f"  |g_perp| = {np.linalg.norm(g_perp):.6e} Hartree/Bohr")
    print(f"  |grad V| = {np.linalg.norm(grad_V):.6e} Hartree/Bohr")

    geom_elements, _geom_coords = parse_reference_geometry(args.qchem)
    n_g = g_perp.shape[0]
    n_drop = len(geom_elements) - n_g
    if n_drop < 0:
        print(
            f"ERROR: reference geometry has {len(geom_elements)} atoms but the "
            f"gradient blocks have {n_g} rows after skipping {args.n_skip_rows}.",
            file=sys.stderr,
        )
        sys.exit(1)
    elements = geom_elements[n_drop:]
    if n_drop:
        print(f"  Dropped {n_drop} leading quantum-nucleus row(s) from the geometry.")

    if not (n_g == len(elements) == n_atoms):
        print(
            f"ERROR: atom-count mismatch — gradients={n_g}, Q-Chem geometry="
            f"{len(elements)}, SIR NPZ={n_atoms}. All three must share one ordering.",
            file=sys.stderr,
        )
        sys.exit(1)
    if elements != symbols:
        print(
            f"ERROR: element ordering differs between the Q-Chem geometry "
            f"({elements}) and the SIR NPZ ({symbols}).",
            file=sys.stderr,
        )
        sys.exit(1)

    density = None
    if args.density_cube is not None:
        if not Path(args.density_cube).is_file():
            print(f"ERROR: density cube not found: {args.density_cube}", file=sys.stderr)
            sys.exit(1)
        density = (args.density_cube, args.density_isovalue, args.density_color)
        print(f"  Density cube: {args.density_cube} (isovalue {args.density_isovalue})")

    xlim = tuple(args.barchart_xlim) if args.barchart_xlim is not None else None
    ylim = tuple(args.barchart_ylim) if args.barchart_ylim is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    g_perp_unit = g_perp / np.linalg.norm(g_perp)
    grad_V_unit = grad_V / np.linalg.norm(grad_V)

    # --- g_perp (tunneling coordinate) ---
    print("\n=== g_perp (tunneling coordinate) ===")
    gperp_dir = args.output_dir / "gperp"
    gperp_cfg = params_for("gperp")
    render_vector_gif("gperp", ref, g_perp_unit, symbols, gperp_dir, gperp_cfg,
                      force=args.force)
    render_vector_arrows("gperp", ref, [g_perp_unit], symbols, gperp_dir, gperp_cfg,
                         arrow_scale=args.arrow_scale, colors=[args.gperp_color],
                         density=density, force=args.force)

    # --- grad V ---
    print("\n=== grad V (vibronic-coupling gradient) ===")
    gradv_dir = args.output_dir / "gradV"
    gradv_cfg = params_for("gradv")
    render_vector_gif("gradV", ref, grad_V_unit, symbols, gradv_dir, gradv_cfg,
                      force=args.force)
    render_vector_arrows("gradV", ref, [grad_V_unit], symbols, gradv_dir, gradv_cfg,
                         arrow_scale=args.arrow_scale, colors=[args.gradv_color],
                         density=density, force=args.force)

    # --- combined g_perp + grad V overlay, and their mutual overlap ---
    print("\n=== g_perp + grad V combined overlay ===")
    combo_dir = args.output_dir / "gperp_gradV"
    render_vector_arrows(
        "gperp_gradV",
        ref,
        [g_perp_unit, grad_V_unit],
        symbols,
        combo_dir,
        gperp_cfg,
        arrow_scale=args.arrow_scale,
        colors=[args.gperp_color, args.gradv_color],
        density=density,
        force=args.force,
    )
    report_overlap(
        combo_dir, "gperp_vs_gradV", "g_perp", "grad_V", g_perp, grad_V, symbols,
        xlim=xlim, ylim=ylim,
    )

    # --- SIR eigenvectors, each checked against both reference vectors ---
    sir_cfg = params_for("sir")
    print(f"\nRendering SIR eigenvectors 1..{n_viz} -> {args.output_dir}")
    for r in range(n_viz):
        name = f"sir{r + 1}"
        vec = betas[:, r].reshape(n_atoms, 3)
        vec_dir = args.output_dir / name
        print(f"\n=== {name} ===")
        render_vector_gif(name, ref, vec, symbols, vec_dir, sir_cfg, force=args.force)
        render_vector_arrows(name, ref, [vec], symbols, vec_dir, sir_cfg,
                             arrow_scale=args.arrow_scale, colors=[args.arrow_color],
                             density=density, force=args.force)

        report_overlap(
            vec_dir, f"{name}_vs_gperp", "g_perp", name, g_perp, vec, symbols,
            xlim=xlim, ylim=ylim,
        )
        report_overlap(
            vec_dir, f"{name}_vs_gradV", "grad_V", name, grad_V, vec, symbols,
            xlim=xlim, ylim=ylim,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
