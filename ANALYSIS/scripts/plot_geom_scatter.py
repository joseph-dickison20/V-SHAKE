#!/usr/bin/env python3
"""
plot_geom_scatter.py
====================
Read a seam-sampling master XYZ (from SCATTER_PLOTS/MASTER_XYZ/build_master_xyz.py),
compute configured geometric observables per frame, and make colored scatter
plots: 2D colored scatter (x_id, y_id, color) or 1D scatter (x_id, y_field).

A per-variable distribution (normalized histogram) with an optional renormalized
cubic-spline density overlay is produced for each 1D plot (``distribution_*.png``;
this is density estimation of a histogram, not a regression of an observable, and
is retained). Bivariate Gaussian kernel density estimates give 2D density contour
maps for each 2D geometric plot (``density2d_*.png``).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np
from ase import Atoms
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde

from hd_plot_colors import (
    COLOR_D,
    COLOR_D_MECP,
    COLOR_H,
    COLOR_H_MECP,
    mecp_color as _mecp_line_color,
    series_color as _isotope_series_color,
)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Nimbus Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        # Mathtext constructs superscripts from regular glyphs (no Unicode
        # SUPERSCRIPT MINUS needed). dejavusans keeps $...$ near the body face
        # without Computer Modern or cursive-italic fallbacks.
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 300,
        "savefig.dpi": 600,
    }
)

HARTREE_TO_KCAL = 627.509
EXCLUDED_ELEMENTS = frozenset({"Hd", "Ha"})
FIGSIZE = (4.44, 3.33)
SCATTER_POINT_SIZE = 14.0
GRID_N = 500  # evaluation grid resolution for joint-PDF contours
PLOT_1D_HIST_SPLINES = True
HIST_ALPHA = 0.55
JOINT_PDF_CONTOUR_LEVELS = 12


def _infer_isotope_from_path(path: Path) -> str | None:
    """Return 'H' or 'D' if a path segment is exactly that isotope label."""
    for part in Path(path).parts:
        if part in ("H", "D"):
            return part
    return None

# ═════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

XYZ_FILE = "master.xyz"
OUTPUT_DIR = "plots"
WRITE_TABLE = False
TABLE_PATH = "geom_scatter_table.txt"

# bond (2 indices), angle (3), dihedral (4) — 1-based atom indices
OBSERVABLES = [
    {
        "id": "ROO",
        "type": "bond",
        "indices": [1, 2],
        "label": r"$R_{\mathrm{OO}}$",
    },
    {
        "id": "methyl_tor",
        "type": "dihedral",
        "indices": [2, 4, 8, 9],
        "label": "Methyl Torsion",
        "period_deg": 120,
    },
]

# 2D: x_id, y_id, color (RdBu_r colorbar), label = colorbar title
# 1D: x_id, y_field (y axis), label = y-axis quantity name
PLOTS = [
    {
        "x_id": "ROO",
        "y_id": "methyl_tor",
        "color": "vibronic_coupling",
        "label": r"$V$",
        "output": "coupling_vs_geom.png",
        "xlim": [2.3, 2.8],
        "ylim": [0.0, 120.0],
        "clim": [0, 1500],
    },
        {
        "x_id": "ROO",
        "y_id": "methyl_tor",
        "color": "delta_E_kcal",
        "label": r"$\Delta E_{\mathrm{seam}}$",
        "output": "deltaE_vs_geom.png",
        "xlim": [2.3, 2.8],
        "ylim": [0.0, 120.0],
        "clim": [0, 40],
    },
    {
        "x_id": "methyl_tor",
        "y_field": "vibronic_coupling",
        "label": r"$V$",
        "output": "V_vs_methyl_tor.png",
    },
    {
        "x_id": "ROO",
        "y_field": "vibronic_coupling",
        "label": r"$V$",
        "output": "V_vs_ROO.png",
    },
    {
        "x_id": "methyl_tor",
        "y_field": "delta_E_kcal",
        "label": r"$\Delta E_{\mathrm{seam}}$",
        "output": "deltaE_vs_methyl_tor.png",
    },
    {
        "x_id": "ROO",
        "y_field": "delta_E_kcal",
        "label": r"$\Delta E_{\mathrm{seam}}$",
        "output": "deltaE_vs_ROO.png",
    },
]

GEOM_UNITS = {"bond": r"Å", "angle": r"deg.", "dihedral": r"deg."}
# Inverse units via mathtext ($^{-1}$): Nimbus/Arial lack Unicode superscript
# glyphs (U+207B), which caused missing-glyph warnings with cm⁻¹-style labels.
COLOR_FIELD_UNITS = {
    "vibronic_coupling": r"cm$^{-1}$",
    "delta_E_kcal": r"kcal mol$^{-1}$",
}
Y_FIELD_UNITS = COLOR_FIELD_UNITS
VALID_SCALAR_FIELDS = frozenset(COLOR_FIELD_UNITS)
# Fields shown on a logarithmic scale (visual dynamic range only; this is a
# display transform, not a regression): 2D colorbars and 1D y-axes.
LOG_SCALE_FIELDS = frozenset({"vibronic_coupling"})
LOG_Y_FLOOR = 0.01
_VALID_YSCALES = frozenset({"log", "linear"})


def _y_is_log(pl: dict) -> bool:
    """Whether a 1D plot uses a log y-axis.

    For ``y_field: vibronic_coupling``, an explicit ``yscale: log|linear``
    (from plot_defaults or the plot entry) overrides the field default.
    Other y-fields keep the ``LOG_SCALE_FIELDS`` rule so a shared
    ``plot_defaults.yscale`` does not flip ΔE plots to log.
    """
    yscale = pl.get("yscale")
    if pl.get("y_field") == "vibronic_coupling" and yscale is not None:
        key = str(yscale).lower()
        if key not in _VALID_YSCALES:
            raise ValueError(
                f"yscale must be 'log' or 'linear', got {yscale!r}"
            )
        return key == "log"
    return pl["y_field"] in LOG_SCALE_FIELDS


def _inverse_geom_unit(obs_type: str) -> str:
    return rf"{GEOM_UNITS[obs_type]}$^{{-1}}$"


def _ylabel_prob_den_1d(obs: dict) -> str:
    return f"Prob. Density [{_inverse_geom_unit(obs['type'])}]"


def _colorbar_joint_prob_den(obs_x: dict, obs_y: dict) -> str:
    unit_x = GEOM_UNITS[obs_x["type"]]
    unit_y = GEOM_UNITS[obs_y["type"]]
    if unit_x == unit_y:
        return rf"Prob. Density [{unit_x}$^{{-2}}$]"
    inv_x = _inverse_geom_unit(obs_x["type"])
    inv_y = _inverse_geom_unit(obs_y["type"])
    return f"Prob. Density [{inv_x}·{inv_y}]"


def _is_hinge_vs_hinge(x_id: str, y_id: str) -> bool:
    """True only for hinge–hinge 2D maps (special-case y = x overlay)."""
    return str(x_id).startswith("hinge") and str(y_id).startswith("hinge")


# ═════════════════════════════════════════════════════════════════════
#  XYZ PARSING
# ═════════════════════════════════════════════════════════════════════


def parse_comment_line(comment: str) -> dict[str, float | int]:
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
    required = ("step", "delta_cm_inv", "E0")
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"Comment line missing {missing}: {comment!r}")
    return out


def parse_master_xyz(path: str | Path) -> list[dict]:
    path = Path(path)
    frames: list[dict] = []
    with path.open() as f:
        while True:
            natom_line = f.readline()
            if not natom_line:
                break
            natom = int(natom_line.strip())
            comment = f.readline().rstrip("\n")
            if not comment:
                raise ValueError(f"Unexpected EOF in {path}")

            meta = parse_comment_line(comment)
            elements: list[str] = []
            coords: list[list[float]] = []
            for _ in range(natom):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF reading atoms in {path}")
                parts = line.split()
                element = parts[0]
                if element in EXCLUDED_ELEMENTS:
                    continue
                elements.append(element)
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            frames.append(
                {
                    "elements": elements,
                    "coords": np.array(coords, dtype=float),
                    "comment": comment,
                    "step": int(meta["step"]),
                    "delta_cm_inv": float(meta["delta_cm_inv"]),
                    "E0": float(meta["E0"]),
                }
            )
    return frames


# ═════════════════════════════════════════════════════════════════════
#  GEOMETRIC OBSERVABLES
# ═════════════════════════════════════════════════════════════════════


def normalize_dihedral(deg: float) -> float:
    """Fold torsions above 180° to a smaller magnitude reference."""
    if deg > 180.0:
        return abs(deg - 360.0)
    return deg


def fold_angle_period(deg: float, period: float) -> float:
    """Map angles above ``period`` into ``(0, period]`` by subtracting ``period``.

    Used for methyl torsion, where unique conformations span only 0–120°.
    """
    period = float(period)
    if period <= 0.0:
        raise ValueError(f"period_deg must be positive, got {period}")
    x = float(deg)
    while x > period:
        x -= period
    return x


def compute_geom(elements: list[str], coords: np.ndarray, obs: dict) -> float:
    atoms = Atoms(symbols=elements, positions=coords)
    idx = [i - 1 for i in obs["indices"]]
    t = obs["type"]
    if t == "bond":
        return atoms.get_distance(idx[0], idx[1])
    if t == "angle":
        return atoms.get_angle(idx[0], idx[1], idx[2])
    if t == "dihedral":
        val = normalize_dihedral(
            atoms.get_dihedral(idx[0], idx[1], idx[2], idx[3])
        )
        period = obs.get("period_deg")
        if period is not None:
            val = fold_angle_period(val, float(period))
        return val
    raise ValueError(f"Unknown type {t!r}")


def geom_label(obs: dict) -> str:
    return f'{obs["label"]} [{GEOM_UNITS[obs["type"]]}]'


# ═════════════════════════════════════════════════════════════════════
#  DATASET CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════


def build_dataset(frames: list[dict], observables: list[dict]) -> dict:
    e0_mecp = frames[0]["E0"]
    n = len(frames)
    geom: dict[str, np.ndarray] = {}
    for obs in observables:
        arr = np.zeros(n)
        for fi, fr in enumerate(frames):
            arr[fi] = compute_geom(fr["elements"], fr["coords"], obs)
        geom[obs["id"]] = arr

    vibronic_coupling = np.array([fr["delta_cm_inv"] / 2.0 for fr in frames])
    E0 = np.array([fr["E0"] for fr in frames])
    delta_E_kcal = (E0 - e0_mecp) * HARTREE_TO_KCAL
    step = np.array([fr["step"] for fr in frames], dtype=int)

    return {
        "geom": geom,
        "vibronic_coupling": vibronic_coupling,
        "E0": E0,
        "delta_E_kcal": delta_E_kcal,
        "step": step,
        "e0_mecp": e0_mecp,
    }


def scalar_values(data: dict, field: str) -> np.ndarray:
    if field not in VALID_SCALAR_FIELDS:
        raise ValueError(f"field must be one of {sorted(VALID_SCALAR_FIELDS)}")
    return data[field].astype(float)


def is_2d_color_plot(pl: dict) -> bool:
    return "y_id" in pl


def is_1d_plot(pl: dict) -> bool:
    return "y_field" in pl and "y_id" not in pl


def scalar_label(label: str, field: str) -> str:
    return f'{label} [{Y_FIELD_UNITS[field]}]'


# ═════════════════════════════════════════════════════════════════════
#  CONFIGURATION VALIDATION
# ═════════════════════════════════════════════════════════════════════


def validate_config(observables: list[dict], plots: list[dict]) -> dict[str, dict]:
    obs_by_id: dict[str, dict] = {}
    for obs in observables:
        oid = obs["id"]
        if oid in obs_by_id:
            raise ValueError(f"Duplicate observable id {oid!r}")
        n = {"bond": 2, "angle": 3, "dihedral": 4}[obs["type"]]
        if len(obs["indices"]) != n:
            raise ValueError(f'Observable "{oid}": expected {n} indices')
        obs_by_id[oid] = obs

    for pl in plots:
        if pl["x_id"] not in obs_by_id:
            raise ValueError(f'Plot references unknown x_id {pl["x_id"]!r}')
        if "output" not in pl:
            raise ValueError("Each plot needs an output filename")
        if is_2d_color_plot(pl) and is_1d_plot(pl):
            raise ValueError("Plot cannot have both y_id and y_field")
        if is_2d_color_plot(pl):
            if pl["y_id"] not in obs_by_id:
                raise ValueError(f'Plot references unknown y_id {pl["y_id"]!r}')
            if pl.get("color") not in VALID_SCALAR_FIELDS:
                raise ValueError(f'Unknown color field {pl.get("color")!r}')
        elif is_1d_plot(pl):
            if pl["y_field"] not in VALID_SCALAR_FIELDS:
                raise ValueError(f'Unknown y_field {pl["y_field"]!r}')
        else:
            raise ValueError(
                'Each plot needs either (y_id + color) for 2D or y_field for 1D'
            )
    return obs_by_id


# ═════════════════════════════════════════════════════════════════════
#  TABLE OUTPUT
# ═════════════════════════════════════════════════════════════════════


def write_table(
    path: str,
    frames: list[dict],
    data: dict,
    observables: list[dict],
) -> None:
    n = len(frames)
    with open(path, "w") as fh:
        fh.write(f"# Geom scatter table\n# Source: {XYZ_FILE}\n# {date.today()}\n#\n")
        header = f"{'step':>8} {'V[cm-1]':>14} {'E0[Ha]':>18} {'dE[kcal/mol]':>14}"
        for obs in observables:
            header += f" {obs['id']:>14}"
        fh.write(f"# {header}\n")
        for fi in range(n):
            line = (
                f"{data['step'][fi]:8d}"
                f" {data['vibronic_coupling'][fi]:14.6f}"
                f" {data['E0'][fi]:18.11f}"
                f" {data['delta_E_kcal'][fi]:14.6f}"
            )
            for obs in observables:
                line += f" {data['geom'][obs['id']][fi]:14.6f}"
            fh.write(f"  {line}\n")
    print(f"Wrote {path}")


# ═════════════════════════════════════════════════════════════════════
#  AXIS / COLORBAR UTILITIES
# ═════════════════════════════════════════════════════════════════════


def _limits_on_ticks(
    vmin: float, vmax: float, nbins: int = 5
) -> tuple[float, float, np.ndarray]:
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if np.isclose(vmin, vmax, rtol=0.0, atol=1e-12):
        pad = max(abs(vmin) * 0.05, 1e-6)
        vmin, vmax = vmin - pad, vmax + pad
    loc = mticker.MaxNLocator(nbins=nbins, min_n_ticks=3)
    ticks = loc.tick_values(vmin, vmax)
    return float(ticks[0]), float(ticks[-1]), ticks


def _apply_axis_limits_on_ticks(
    ax: plt.Axes, axis: str, vmin: float, vmax: float, nbins: int = 5
) -> None:
    lo, hi, ticks = _limits_on_ticks(vmin, vmax, nbins=nbins)
    if axis == "x":
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    elif axis == "y":
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    else:
        raise ValueError(axis)


def _five_ticks_rounded(lo: float, hi: float) -> np.ndarray:
    ticks = np.round(np.linspace(lo, hi, 5), 2)
    ticks[0] = np.round(lo, 2)
    ticks[-1] = np.round(hi, 2)
    if len(np.unique(ticks)) != 5:
        raise ValueError(
            f"Cannot place 5 distinct ticks (2 decimal places) on [{lo}, {hi}]"
        )
    return ticks


def _fixed_tick_format(ticks: np.ndarray, *, integer_ticks: bool) -> str:
    """Choose label precision from the placed ticks.

    Angles/dihedrals stay integer. Otherwise use one decimal when every tick's
    hundredths digit is 0 (e.g. 2.40 → ``2.4``), and two decimals when any tick
    needs a non-zero hundredths place (e.g. 2.35).
    """
    if integer_ticks:
        return "%.0f"
    hundredths = np.round(np.asarray(ticks, dtype=float) * 100.0)
    if np.all(np.isclose(hundredths % 10.0, 0.0)):
        return "%.1f"
    return "%.2f"


def _fixed_lims_use_integer_ticks(obs: dict, lim: tuple[float, float] | None) -> bool:
    if lim is None:
        return False
    return obs["type"] in ("angle", "dihedral")


_LEGACY_AXIS_LIMIT = object()


def plot_axis_limit(pl: dict, key: str) -> tuple[float, float] | None | object:
    """
    Resolve axis limits from a plot YAML entry.

    - Key absent → legacy auto ticks from data (_LEGACY_AXIS_LIMIT)
    - ``null`` → matplotlib autoscale (return None)
    - ``[lo, hi]`` → fixed limits
    """
    if key not in pl:
        return _LEGACY_AXIS_LIMIT
    val = pl[key]
    if val is None:
        return None
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError(f"plot entry {key!r} must be null or [min, max]")
    return (float(val[0]), float(val[1]))


def _apply_one_axis_limit(
    ax: plt.Axes,
    axis: str,
    spec: tuple[float, float] | None | object,
    data_min: float,
    data_max: float,
    *,
    integer_ticks: bool = False,
    nbins: int = 5,
) -> None:
    if spec is _LEGACY_AXIS_LIMIT:
        _apply_axis_limits_on_ticks(ax, axis, data_min, data_max, nbins=nbins)
    elif spec is None:
        return
    else:
        _apply_fixed_axis_limits(
            ax,
            axis,
            float(spec[0]),
            float(spec[1]),
            integer_ticks=integer_ticks,
        )


def _positive_y_floor(y_values: np.ndarray) -> float:
    pos = y_values[np.asarray(y_values) > 0]
    if pos.size:
        return float(max(np.min(pos), LOG_Y_FLOOR))
    return LOG_Y_FLOOR


def _apply_log_y_axis(
    ax: plt.Axes,
    ylim: tuple[float, float] | None,
    y_values: np.ndarray,
) -> None:
    """Logarithmic y-axis for vibronic-coupling (and other LOG_SCALE_FIELDS)."""
    ax.set_yscale("log")
    if ylim is not None:
        y_lo = float(ylim[0]) if float(ylim[0]) > 0.0 else LOG_Y_FLOOR
        y_hi = float(ylim[1])
        if y_hi <= y_lo:
            y_hi = y_lo * 10.0
        ax.set_ylim(y_lo, y_hi)
    else:
        y_lo = _positive_y_floor(y_values)
        y_hi = float(np.max(y_values)) if y_values.size else y_lo * 10.0
        if y_hi <= y_lo:
            y_hi = y_lo * 10.0
        ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12)
    )
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=48)
    )


def color_limits_from_plot_entry(
    pl: dict,
) -> tuple[float, float] | None:
    """Fixed scatter color limits, or None to derive from visible data.

    ``clim`` now sets the scatter color-scale limits (the NW heatmaps it once
    governed are gone).
    """
    if "clim" not in pl or pl["clim"] is None:
        return None
    clim = pl["clim"]
    if not isinstance(clim, (list, tuple)) or len(clim) != 2:
        raise ValueError("plot entry clim must be null or [min, max]")
    return (float(clim[0]), float(clim[1]))


def _fixed_lim_tuple(
    spec: tuple[float, float] | None | object,
) -> tuple[float, float] | None:
    if isinstance(spec, tuple):
        return spec
    return None


def _apply_fixed_axis_limits(
    ax: plt.Axes,
    axis: str,
    lo: float,
    hi: float,
    *,
    integer_ticks: bool = False,
) -> None:
    ticks = _five_ticks_rounded(lo, hi)
    formatter = mticker.FormatStrFormatter(
        _fixed_tick_format(ticks, integer_ticks=integer_ticks)
    )
    if axis == "x":
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError(axis)


def _apply_plot_xy_limits(
    ax: plt.Axes,
    xlim: tuple[float, float] | None | object,
    ylim: tuple[float, float] | None | object,
    data_xmin: float,
    data_xmax: float,
    data_ymin: float,
    data_ymax: float,
    *,
    x_fixed_integer_ticks: bool = False,
    y_fixed_integer_ticks: bool = False,
) -> None:
    _apply_one_axis_limit(
        ax,
        "x",
        xlim,
        data_xmin,
        data_xmax,
        integer_ticks=x_fixed_integer_ticks,
    )
    _apply_one_axis_limit(
        ax,
        "y",
        ylim,
        data_ymin,
        data_ymax,
        integer_ticks=y_fixed_integer_ticks,
    )


# ═════════════════════════════════════════════════════════════════════
#  HISTOGRAM / JOINT-PDF DENSITY ESTIMATION (retained; not a regression)
# ═════════════════════════════════════════════════════════════════════


def _marginal_pdf_path(output_dir: Path, x_id: str) -> Path:
    """Single-isotope 1D marginal probability-density plot path."""
    return output_dir / f"distribution_{x_id}.png"


def _hd_distribution_compare_path(output_dir: Path, x_id: str) -> Path:
    """Paired H/D distribution-comparison plot path.

    This overlays the H and D distributions of one observable on a single axis;
    it is a paired comparison, not a marginal density, hence its own name.
    """
    return output_dir / f"hd_distribution_compare_{x_id}.png"


def _hd_scatter_path(output_dir: Path, plot_output: str) -> Path:
    """Paired H/D 1D scatter path derived from the per-isotope plot filename."""
    name = Path(plot_output).name
    return output_dir / f"hd_{name}"


def _joint_pdf_path_for_geom(output_dir: Path, x_id: str, y_id: str) -> Path:
    return output_dir / f"density2d_{x_id}_vs_{y_id}.png"


def _histogram_bin_edges(samples: np.ndarray) -> np.ndarray:
    """Freedman–Diaconis bin edges, with a modest cap on bin count."""
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n < 2:
        lo, hi = float(np.min(samples)), float(np.max(samples))
        pad = max(abs(lo) * 0.05, 1e-6)
        return np.linspace(lo - pad, hi + pad, 3)
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = q75 - q25
    if iqr <= 0.0:
        iqr = float(np.ptp(samples))
    if iqr <= 0.0:
        iqr = 1.0
    bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if bin_width <= 0.0:
        bin_width = max(float(np.ptp(samples)) / 10.0, 1e-6)
    lo, hi = float(np.min(samples)), float(np.max(samples))
    span = hi - lo
    if span <= 0.0:
        pad = max(abs(lo) * 0.05, 1e-6)
        return np.linspace(lo - pad, hi + pad, 3)
    nbins = int(np.ceil(span / bin_width))
    nbins = max(8, min(nbins, 60))
    return np.linspace(lo, hi, nbins + 1)


def _spline_pdf_from_histogram(
    bin_edges: np.ndarray,
    density: np.ndarray,
    x_curve: np.ndarray,
    *,
    x_bound_lo: float | None = None,
    x_bound_hi: float | None = None,
) -> np.ndarray:
    """Smooth histogram density with a spline and renormalize to unit integral."""
    density = np.asarray(density, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    if len(density) < 2:
        return np.zeros_like(x_curve, dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    y_fit = density.copy()
    x_fit = centers.copy()
    if x_bound_lo is not None and x_bound_hi is not None and x_bound_lo < x_bound_hi:
        x_fit = np.concatenate(([float(x_bound_lo)], x_fit, [float(x_bound_hi)]))
        y_fit = np.concatenate(([0.0], y_fit, [0.0]))

    var_d = float(np.var(y_fit))
    smooth = max(1e-12, 0.02 * len(y_fit) * var_d)
    k = min(3, len(x_fit) - 1)
    spline = UnivariateSpline(x_fit, y_fit, k=k, s=smooth)
    pdf = np.maximum(spline(x_curve), 0.0)
    if x_bound_lo is not None and x_bound_hi is not None:
        pdf[(x_curve < x_bound_lo) | (x_curve > x_bound_hi)] = 0.0
        on_lo = np.isclose(x_curve, x_bound_lo, rtol=0.0, atol=1e-9)
        on_hi = np.isclose(x_curve, x_bound_hi, rtol=0.0, atol=1e-9)
        pdf[on_lo | on_hi] = 0.0
    integral = float(np.trapezoid(pdf, x_curve))
    if integral > 0.0:
        pdf /= integral
    return pdf


def _pdf_integral_on_range(
    x_grid: np.ndarray, pdf: np.ndarray, lo: float, hi: float
) -> float:
    x_grid = np.asarray(x_grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    if hi <= lo:
        return 0.0
    mask = (x_grid >= lo) & (x_grid <= hi)
    if not np.any(mask):
        return 0.0
    x_seg = x_grid[mask]
    p_seg = pdf[mask]
    if len(x_seg) < 2:
        return 0.0
    return float(np.trapezoid(p_seg, x_seg))


def _joint_pdf_on_grid(
    x_data: np.ndarray, y_data: np.ndarray, X_grid: np.ndarray, Y_grid: np.ndarray
) -> np.ndarray:
    kde = gaussian_kde(np.vstack([x_data, y_data]))
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    pdf = kde(positions).reshape(X_grid.shape)
    return pdf


def _grid_ranges(
    x_data: np.ndarray,
    y_data: np.ndarray,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if xlim is not None:
        x_lo, x_hi = float(xlim[0]), float(xlim[1])
    else:
        x_lo, x_hi = float(np.min(x_data)), float(np.max(x_data))
    if ylim is not None:
        y_lo, y_hi = float(ylim[0]), float(ylim[1])
    else:
        y_lo, y_hi = float(np.min(y_data)), float(np.max(y_data))
    x_1d = np.linspace(x_lo, x_hi, GRID_N)
    y_1d = np.linspace(y_lo, y_hi, GRID_N)
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d)
    return X_grid, Y_grid


# ═════════════════════════════════════════════════════════════════════
#  PLOT RENDERING
# ═════════════════════════════════════════════════════════════════════


def _plot_mecp_point(
    ax: plt.Axes, x_mecp: float, y_mecp: float, *, color: str
) -> None:
    """Filled circle marking the MECP geometry on a 2D axes.

    ``color`` matches the isotope series (H red / D blue).
    """
    ax.plot(
        x_mecp,
        y_mecp,
        "o",
        color=color,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=5,
    )


def _add_mecp_vline(
    ax: plt.Axes,
    x_mecp: float,
    *,
    color: str,
    label: str | None = None,
) -> None:
    """Vertical dashed line at the MECP value of the x-axis observable."""
    kwargs: dict = {
        "color": color,
        "linestyle": "--",
        "linewidth": 1.2,
        "zorder": 5,
    }
    if label is not None:
        kwargs["label"] = label
    ax.axvline(x_mecp, **kwargs)


def _add_y_equals_x_line(ax: plt.Axes) -> None:
    """Dashed black y = x reference (hinge–hinge 2D plots only)."""
    ax.axline(
        (0.0, 0.0),
        slope=1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )


def _add_mecp_figure_legend(fig: plt.Figure, *, color: str) -> None:
    """MECP key in figure margin; excluded from tight_layout so axes size stays fixed."""
    handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    leg = fig.legend(
        handles=[handle],
        labels=["MECP"],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        fontsize=12,
        frameon=True,
        framealpha=1.0,
        borderpad=0.35,
        handletextpad=0.5,
    )
    if hasattr(leg, "set_in_layout"):
        leg.set_in_layout(False)


def make_scatter_plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    x_label: str,
    y_label: str,
    c_label: str,
    output_path: Path,
    clim: tuple[float, float] | None,
    color_is_log: bool,
    x_mecp: float,
    y_mecp: float,
    mecp_color: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    x_fixed_integer_ticks: bool = False,
    y_fixed_integer_ticks: bool = False,
    draw_y_equals_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if color_is_log:
        pos = c[c > 0]
        if clim is not None and clim[0] > 0.0:
            vmin = float(clim[0])
        else:
            vmin = float(np.min(pos)) if pos.size else 1e-3
        vmax = float(clim[1]) if clim is not None else float(np.max(c))
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
        sc = ax.scatter(
            x,
            y,
            c=c,
            s=SCATTER_POINT_SIZE,
            cmap=plt.colormaps["RdBu_r"],
            norm=norm,
            edgecolors="0.15",
            linewidths=0.2,
        )
        cbar = fig.colorbar(sc, ax=ax, label=c_label)
        cbar.ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12)
        )
        cbar.ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=48)
        )
    else:
        if clim is not None:
            c_lo, c_hi, c_ticks = _limits_on_ticks(clim[0], clim[1])
        else:
            c_lo, c_hi, c_ticks = _limits_on_ticks(
                float(np.min(c)), float(np.max(c))
            )
        norm = Normalize(vmin=c_lo, vmax=c_hi)
        sc = ax.scatter(
            x,
            y,
            c=c,
            s=SCATTER_POINT_SIZE,
            cmap=plt.colormaps["RdBu_r"],
            norm=norm,
            edgecolors="0.15",
            linewidths=0.2,
        )
        cbar = fig.colorbar(sc, ax=ax, label=c_label)
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(c_ticks))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_plot_xy_limits(
        ax,
        xlim,
        ylim,
        float(np.min(x)),
        float(np.max(x)),
        float(np.min(y)),
        float(np.max(y)),
        x_fixed_integer_ticks=x_fixed_integer_ticks,
        y_fixed_integer_ticks=y_fixed_integer_ticks,
    )
    if draw_y_equals_x:
        _add_y_equals_x_line(ax)
    _plot_mecp_point(ax, x_mecp, y_mecp, color=mecp_color)
    fig.tight_layout()
    _add_mecp_figure_legend(fig, color=mecp_color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def make_scatter_plot_1d(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    output_path: Path,
    x_mecp: float,
    mecp_color: str,
    scatter_color: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    x_fixed_integer_ticks: bool = False,
    y_is_log: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        x,
        y,
        s=SCATTER_POINT_SIZE,
        color=scatter_color,
        alpha=0.65,
        edgecolors="0.15",
        linewidths=0.2,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_one_axis_limit(
        ax,
        "x",
        xlim,
        float(np.min(x)),
        float(np.max(x)),
        integer_ticks=x_fixed_integer_ticks,
    )
    if y_is_log:
        _apply_log_y_axis(ax, ylim, y)
    else:
        _apply_one_axis_limit(ax, "y", ylim, float(np.min(y)), float(np.max(y)))
    _add_mecp_vline(ax, x_mecp, color=mecp_color)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def make_scatter_plot_1d_hd(
    x_h: np.ndarray,
    y_h: np.ndarray,
    x_d: np.ndarray,
    y_d: np.ndarray,
    x_label: str,
    y_label: str,
    output_path: Path,
    x_mecp_h: float,
    x_mecp_d: float,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    x_fixed_integer_ticks: bool = False,
    y_is_log: bool = False,
) -> None:
    """Paired H/D 1D scatter overlay (step-1 combo styling)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        x_h,
        y_h,
        s=SCATTER_POINT_SIZE,
        color=COLOR_H,
        alpha=0.65,
        edgecolors="0.15",
        linewidths=0.2,
        label="H",
    )
    ax.scatter(
        x_d,
        y_d,
        s=SCATTER_POINT_SIZE,
        color=COLOR_D,
        alpha=0.65,
        edgecolors="0.15",
        linewidths=0.2,
        label="D",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_one_axis_limit(
        ax,
        "x",
        xlim,
        float(min(np.min(x_h), np.min(x_d))),
        float(max(np.max(x_h), np.max(x_d))),
        integer_ticks=x_fixed_integer_ticks,
    )
    if y_is_log:
        _apply_log_y_axis(ax, ylim, np.concatenate([y_h, y_d]))
    else:
        _apply_one_axis_limit(
            ax,
            "y",
            ylim,
            float(min(np.min(y_h), np.min(y_d))),
            float(max(np.max(y_h), np.max(y_d))),
        )
    _add_mecp_vline(ax, x_mecp_h, color=COLOR_H_MECP)
    _add_mecp_vline(ax, x_mecp_d, color=COLOR_D_MECP)
    ax.legend(loc="best", fontsize=9, frameon=True, markerscale=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def make_hist_pdf_1d(
    samples: np.ndarray,
    x_label: str,
    output_path: Path,
    y_pdf_label: str,
    x_mecp: float,
    mecp_color: str,
    bar_color: str,
    xlim: tuple[float, float] | None | object = _LEGACY_AXIS_LIMIT,
    ylim: tuple[float, float] | None | object = _LEGACY_AXIS_LIMIT,
    *,
    plot_splines: bool = True,
    x_fixed_integer_ticks: bool = False,
) -> None:
    samples = np.asarray(samples, dtype=float)
    data_xmin = float(np.min(samples))
    data_xmax = float(np.max(samples))

    x_fixed = _fixed_lim_tuple(xlim)
    if x_fixed is not None:
        x_lo, x_hi = float(x_fixed[0]), float(x_fixed[1])
    else:
        x_lo, x_hi = data_xmin, data_xmax

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bin_edges = _histogram_bin_edges(samples)
    density, edges = np.histogram(samples, bins=bin_edges, density=True)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(
        centers,
        density,
        width=widths,
        align="center",
        facecolor=bar_color,
        alpha=HIST_ALPHA,
        edgecolor="0.15",
        linewidth=0.65,
    )

    pdf_curve: np.ndarray | None = None
    if plot_splines:
        pad = 0.05 * max(x_hi - x_lo, 1e-6)
        x_curve = np.unique(
            np.concatenate(
                [
                    np.linspace(x_lo - pad, x_hi + pad, 500),
                    np.array([x_lo, x_hi], dtype=float),
                ]
            )
        )
        pdf_curve = _spline_pdf_from_histogram(
            edges, density, x_curve, x_bound_lo=x_lo, x_bound_hi=x_hi
        )
        ax.plot(
            x_curve,
            pdf_curve,
            color="black",
            linewidth=0.85,
            zorder=4,
        )

        hist_integral = float(np.sum(density * widths))
        pdf_integral_full = float(np.trapezoid(pdf_curve, x_curve))
        pdf_integral_axis = _pdf_integral_on_range(x_curve, pdf_curve, x_lo, x_hi)
        print(
            f"  {output_path.name}: histogram ∫p dx = {hist_integral:.6f} "
            f"(expect 1); spline PDF ∫_axis dx = {pdf_integral_axis:.6f}, "
            f"∫_spline_grid dx = {pdf_integral_full:.6f} (expect 1)"
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_pdf_label)
    _apply_one_axis_limit(
        ax,
        "x",
        xlim,
        data_xmin,
        data_xmax,
        integer_ticks=x_fixed_integer_ticks,
    )
    y_max = float(np.max(density) if len(density) else 0.0)
    if pdf_curve is not None:
        y_max = max(y_max, float(np.max(pdf_curve)))
    y_max = max(y_max, 1e-12)
    _apply_one_axis_limit(ax, "y", ylim, 0.0, y_max * 1.08)
    _add_mecp_vline(ax, x_mecp, color=mecp_color)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def _x_curve_for_marginal_pdf(x_lo: float, x_hi: float) -> np.ndarray:
    pad = 0.05 * max(x_hi - x_lo, 1e-6)
    return np.unique(
        np.concatenate(
            [
                np.linspace(x_lo - pad, x_hi + pad, 500),
                np.array([x_lo, x_hi], dtype=float),
            ]
        )
    )


def make_hist_pdf_1d_hd(
    samples_h: np.ndarray,
    samples_d: np.ndarray,
    x_label: str,
    output_path: Path,
    y_pdf_label: str,
    x_mecp_h: float,
    x_mecp_d: float,
    xlim: tuple[float, float] | None | object = _LEGACY_AXIS_LIMIT,
    ylim: tuple[float, float] | None | object = _LEGACY_AXIS_LIMIT,
    *,
    plot_splines: bool = True,
    x_fixed_integer_ticks: bool = False,
) -> None:
    """Paired H/D distribution comparison: overlay H (red) and D (blue)
    histograms of one observable on a single axis, with optional spline PDFs.

    MECP observable values are marked with dashed vertical lines
    (red for H, blue for D), matching step-1 combo V vs E coloring.
    """
    samples_h = np.asarray(samples_h, dtype=float)
    samples_d = np.asarray(samples_d, dtype=float)
    data_xmin = float(min(np.min(samples_h), np.min(samples_d)))
    data_xmax = float(max(np.max(samples_h), np.max(samples_d)))

    x_fixed = _fixed_lim_tuple(xlim)
    if x_fixed is not None:
        x_lo, x_hi = float(x_fixed[0]), float(x_fixed[1])
    else:
        x_lo, x_hi = data_xmin, data_xmax

    bin_edges = _histogram_bin_edges(np.concatenate([samples_h, samples_d]))
    x_curve = _x_curve_for_marginal_pdf(x_lo, x_hi) if plot_splines else None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    y_max = 0.0

    for label, samples, color in (
        ("H", samples_h, COLOR_H),
        ("D", samples_d, COLOR_D),
    ):
        density, edges = np.histogram(samples, bins=bin_edges, density=True)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(
            centers,
            density,
            width=widths,
            align="center",
            facecolor=color,
            alpha=HIST_ALPHA,
            edgecolor="0.15",
            linewidth=0.65,
            label=label,
        )
        y_max = max(y_max, float(np.max(density) if len(density) else 0.0))
        if plot_splines and x_curve is not None:
            pdf_curve = _spline_pdf_from_histogram(
                edges, density, x_curve, x_bound_lo=x_lo, x_bound_hi=x_hi
            )
            ax.plot(
                x_curve,
                pdf_curve,
                color=color,
                linewidth=0.85,
                zorder=4,
            )
            y_max = max(y_max, float(np.max(pdf_curve)))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_pdf_label)
    _apply_one_axis_limit(
        ax,
        "x",
        xlim,
        data_xmin,
        data_xmax,
        integer_ticks=x_fixed_integer_ticks,
    )
    _apply_one_axis_limit(ax, "y", ylim, 0.0, max(y_max * 1.08, 1e-12))
    _add_mecp_vline(ax, x_mecp_h, color=COLOR_H_MECP)
    _add_mecp_vline(ax, x_mecp_d, color=COLOR_D_MECP)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def run_hd_combo_hist_plots(
    xyz_h: Path,
    xyz_d: Path,
    output_dir: Path,
    observables: list[dict],
    plots: list[dict],
    *,
    plot_splines: bool = True,
) -> None:
    obs_by_id = validate_config(observables, plots)
    frames_h = parse_master_xyz(xyz_h)
    frames_d = parse_master_xyz(xyz_d)
    print(f"Parsed {len(frames_h)} H frames from {xyz_h}")
    print(f"Parsed {len(frames_d)} D frames from {xyz_d}")
    data_h = build_dataset(frames_h, observables)
    data_d = build_dataset(frames_d, observables)

    out_root = Path(output_dir)
    written_marginals: set[str] = set()
    for pl in plots:
        if not is_1d_plot(pl):
            continue
        x_id = pl["x_id"]
        x_h = data_h["geom"][x_id]
        x_d = data_d["geom"][x_id]
        x_lab = geom_label(obs_by_id[x_id])
        xlim = plot_axis_limit(pl, "xlim")
        ylim = plot_axis_limit(pl, "ylim")
        x_fixed_int = _fixed_lims_use_integer_ticks(
            obs_by_id[x_id], _fixed_lim_tuple(xlim)
        )

        y_h = scalar_values(data_h, pl["y_field"])
        y_d = scalar_values(data_d, pl["y_field"])
        y_lab = scalar_label(pl["label"], pl["y_field"])
        scatter_out = _hd_scatter_path(out_root, pl["output"])
        print(f'  H/D 1D scatter {pl["x_id"]} vs {pl["y_field"]}')
        make_scatter_plot_1d_hd(
            x_h,
            y_h,
            x_d,
            y_d,
            x_lab,
            y_lab,
            scatter_out,
            float(x_h[0]),
            float(x_d[0]),
            xlim=xlim,
            ylim=ylim,
            x_fixed_integer_ticks=x_fixed_int,
            y_is_log=_y_is_log(pl),
        )

        if x_id in written_marginals:
            continue
        written_marginals.add(x_id)
        y_pdf_lab = _ylabel_prob_den_1d(obs_by_id[x_id])
        hist_out = _hd_distribution_compare_path(out_root, x_id)
        print(f'  H/D distribution comparison {x_id}')
        make_hist_pdf_1d_hd(
            x_h,
            x_d,
            x_lab,
            hist_out,
            y_pdf_lab,
            float(x_h[0]),
            float(x_d[0]),
            xlim=xlim,
            ylim=None,
            plot_splines=plot_splines,
            x_fixed_integer_ticks=x_fixed_int,
        )


def make_joint_pdf_contour_2d(
    x: np.ndarray,
    y: np.ndarray,
    x_mecp: float,
    y_mecp: float,
    mecp_color: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    cbar_label: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    data_xmin: float,
    data_xmax: float,
    data_ymin: float,
    data_ymax: float,
    *,
    x_fixed_integer_ticks: bool = False,
    y_fixed_integer_ticks: bool = False,
    draw_y_equals_x: bool = False,
) -> None:
    X_grid, Y_grid = _grid_ranges(x, y, xlim, ylim)
    pdf = _joint_pdf_on_grid(x, y, X_grid, Y_grid)
    pdf_max = float(np.max(pdf))
    if pdf_max <= 0.0 or not np.isfinite(pdf_max):
        print(f"  Skipping 2D density (degenerate density): {output_path}")
        return

    _, c_hi, c_ticks = _limits_on_ticks(0.0, pdf_max)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    norm_c = Normalize(vmin=0.0, vmax=c_hi)
    levels = np.linspace(0.0, pdf_max, JOINT_PDF_CONTOUR_LEVELS + 1)[1:]
    cf = ax.contourf(
        X_grid,
        Y_grid,
        pdf,
        levels=levels,
        cmap=plt.colormaps["RdBu_r"],
        norm=norm_c,
    )
    ax.contour(
        X_grid,
        Y_grid,
        pdf,
        levels=levels,
        colors="black",
        linewidths=0.35,
        alpha=0.45,
    )
    _plot_mecp_point(ax, x_mecp, y_mecp, color=mecp_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_plot_xy_limits(
        ax,
        xlim,
        ylim,
        data_xmin,
        data_xmax,
        data_ymin,
        data_ymax,
        x_fixed_integer_ticks=x_fixed_integer_ticks,
        y_fixed_integer_ticks=y_fixed_integer_ticks,
    )
    if draw_y_equals_x:
        _add_y_equals_x_line(ax)
    cbar = fig.colorbar(cf, ax=ax, label=cbar_label)
    cf.set_clim(0.0, c_hi)
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(c_ticks))
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    _add_mecp_figure_legend(fig, color=mecp_color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geometric scatter plots (raw scatter + density) for master XYZ."
    )
    parser.add_argument(
        "--xyz",
        default=None,
        help=f"Master XYZ path (default: {XYZ_FILE} in cwd)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Plot output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help="YAML with observables, plots, write_table, table_path",
    )
    parser.add_argument(
        "--hd-combo-hists",
        action="store_true",
        help="Paired H/D combo plots for each 1D geom plot: overlay scatters "
        "(hd_<output>) and distribution comparisons (requires --xyz-h, --xyz-d)",
    )
    parser.add_argument(
        "--xyz-h",
        default=None,
        type=Path,
        help="Master XYZ for H (with --hd-combo-hists)",
    )
    parser.add_argument(
        "--xyz-d",
        default=None,
        type=Path,
        help="Master XYZ for D (with --hd-combo-hists)",
    )
    parser.add_argument(
        "--isotope",
        default=None,
        choices=("H", "D"),
        help="Isotope label for MECP line color in single-isotope plots "
        "(red for H, blue for D). Inferred from --xyz/--output-dir if omitted.",
    )
    args = parser.parse_args()

    xyz_file = args.xyz or XYZ_FILE
    output_dir = args.output_dir or OUTPUT_DIR
    do_write_table = WRITE_TABLE
    table_path = TABLE_PATH
    observables = OBSERVABLES
    plots = PLOTS
    plot_1d_hist_splines = PLOT_1D_HIST_SPLINES

    if args.config is not None:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML required for --config") from exc
        with args.config.open() as f:
            geom_cfg = yaml.safe_load(f)
        if not isinstance(geom_cfg, dict):
            raise ValueError(f"Geom config must be a mapping: {args.config}")
        observables = geom_cfg.get("observables") or observables
        plot_defaults = geom_cfg.get("plot_defaults") or {}
        raw_plots = geom_cfg.get("plots") or plots
        plots = [{**plot_defaults, **pl} for pl in raw_plots]
        do_write_table = bool(geom_cfg.get("write_table", do_write_table))
        table_path = str(geom_cfg.get("table_path", table_path))
        if "plot_1d_hist_splines" in geom_cfg:
            plot_1d_hist_splines = bool(geom_cfg["plot_1d_hist_splines"])

    if args.hd_combo_hists:
        if args.xyz_h is None or args.xyz_d is None:
            raise SystemExit("--hd-combo-hists requires --xyz-h and --xyz-d")
        xyz_h = Path(args.xyz_h)
        xyz_d = Path(args.xyz_d)
        if not xyz_h.is_file():
            raise FileNotFoundError(f"{xyz_h} not found")
        if not xyz_d.is_file():
            raise FileNotFoundError(f"{xyz_d} not found")
        print("H/D combo 1D scatter and distribution plots")
        run_hd_combo_hist_plots(
            xyz_h,
            xyz_d,
            Path(output_dir),
            observables,
            plots,
            plot_splines=plot_1d_hist_splines,
        )
        print("Done.")
        return

    isotope = args.isotope
    if isotope is None:
        isotope = _infer_isotope_from_path(Path(xyz_file)) or _infer_isotope_from_path(
            Path(output_dir)
        )
    if isotope is None:
        raise SystemExit(
            "Could not determine isotope for MECP line color; pass --isotope H|D"
        )
    mecp_color = _mecp_line_color(isotope)
    isotope_color = _isotope_series_color(isotope)

    obs_by_id = validate_config(observables, plots)
    xyz_path = Path(xyz_file)
    if not xyz_path.is_file():
        raise FileNotFoundError(f"{xyz_file} not found in {Path.cwd()}")

    frames = parse_master_xyz(xyz_path)
    print(f"Parsed {len(frames)} frames from {xyz_file}")
    data = build_dataset(frames, observables)
    print(f"MECP E0 = {data['e0_mecp']:.11f} Hartree (frame step=0)")
    print(f"Isotope = {isotope} (MECP line {mecp_color}, series {isotope_color})")

    if do_write_table:
        write_table(table_path, frames, data, observables)

    out_root = Path(output_dir)
    written_marginals: set[str] = set()
    written_joints: set[tuple[str, str]] = set()
    for pl in plots:
        x = data["geom"][pl["x_id"]]
        x_lab = geom_label(obs_by_id[pl["x_id"]])
        out = out_root / pl["output"]
        xlim = plot_axis_limit(pl, "xlim")
        ylim = plot_axis_limit(pl, "ylim")
        x_fixed_int = _fixed_lims_use_integer_ticks(
            obs_by_id[pl["x_id"]], _fixed_lim_tuple(xlim)
        )

        x_mecp = float(x[0])

        if is_1d_plot(pl):
            y = scalar_values(data, pl["y_field"])
            y_lab = scalar_label(pl["label"], pl["y_field"])
            print(f'Plot 1D {pl["x_id"]} vs {pl["y_field"]}')
            make_scatter_plot_1d(
                x,
                y,
                x_lab,
                y_lab,
                out,
                x_mecp,
                mecp_color,
                isotope_color,
                xlim=xlim,
                ylim=ylim,
                x_fixed_integer_ticks=x_fixed_int,
                y_is_log=_y_is_log(pl),
            )
            x_id = pl["x_id"]
            if x_id not in written_marginals:
                written_marginals.add(x_id)
                print(f'  Distribution (1D density) {x_id}')
                make_hist_pdf_1d(
                    x,
                    x_lab,
                    _marginal_pdf_path(out_root, x_id),
                    _ylabel_prob_den_1d(obs_by_id[x_id]),
                    x_mecp,
                    mecp_color,
                    isotope_color,
                    xlim=xlim,
                    ylim=None,
                    plot_splines=plot_1d_hist_splines,
                    x_fixed_integer_ticks=x_fixed_int,
                )
            continue

        y_fixed_int = _fixed_lims_use_integer_ticks(
            obs_by_id[pl["y_id"]], _fixed_lim_tuple(ylim)
        )

        y = data["geom"][pl["y_id"]]
        c = scalar_values(data, pl["color"])
        y_lab = geom_label(obs_by_id[pl["y_id"]])
        c_lab = scalar_label(pl["label"], pl["color"])
        color_is_log = pl["color"] in LOG_SCALE_FIELDS
        y_mecp = float(y[0])

        clim_fixed = color_limits_from_plot_entry(pl)
        x_mask = _fixed_lim_tuple(xlim)
        y_mask = _fixed_lim_tuple(ylim)
        if clim_fixed is not None:
            clim = clim_fixed
        else:
            mask = np.ones(len(c), dtype=bool)
            if x_mask is not None:
                mask &= (x >= x_mask[0]) & (x <= x_mask[1])
            if y_mask is not None:
                mask &= (y >= y_mask[0]) & (y <= y_mask[1])
            c_visible = c[mask]
            if len(c_visible) == 0:
                c_visible = c
            clim = (float(np.min(c_visible)), float(np.max(c_visible)))

        print(f'Plot 2D {pl["x_id"]} vs {pl["y_id"]} color={pl["color"]}'
              f'{" [log color]" if color_is_log else ""}')
        hinge_diag = _is_hinge_vs_hinge(pl["x_id"], pl["y_id"])
        make_scatter_plot_2d(
            x,
            y,
            c,
            x_lab,
            y_lab,
            c_lab,
            out,
            clim,
            color_is_log,
            x_mecp,
            y_mecp,
            mecp_color,
            xlim=xlim,
            ylim=ylim,
            x_fixed_integer_ticks=x_fixed_int,
            y_fixed_integer_ticks=y_fixed_int,
            draw_y_equals_x=hinge_diag,
        )

        data_xmin = float(np.min(x))
        data_xmax = float(np.max(x))
        data_ymin = float(np.min(y))
        data_ymax = float(np.max(y))

        joint_key = (pl["x_id"], pl["y_id"])
        if joint_key not in written_joints:
            written_joints.add(joint_key)
            print(f'  2D density {pl["x_id"]} vs {pl["y_id"]}')
            make_joint_pdf_contour_2d(
                x,
                y,
                x_mecp,
                y_mecp,
                mecp_color,
                x_lab,
                y_lab,
                _joint_pdf_path_for_geom(out_root, pl["x_id"], pl["y_id"]),
                _colorbar_joint_prob_den(
                    obs_by_id[pl["x_id"]], obs_by_id[pl["y_id"]]
                ),
                xlim,
                ylim,
                data_xmin,
                data_xmax,
                data_ymin,
                data_ymax,
                x_fixed_integer_ticks=x_fixed_int,
                y_fixed_integer_ticks=y_fixed_int,
                draw_y_equals_x=hinge_diag,
            )

    print("Done.")


if __name__ == "__main__":
    main()
