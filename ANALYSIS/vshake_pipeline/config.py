"""Load and validate SIR analysis pipeline YAML.

Steps: ``master_xyz`` → ``geom`` → ``sir`` → ``sir_viz``.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vshake_pipeline.discover import (
    DiscoveryResult,
    density_cube_path,
    discover_seam_xyz,
    qchem_mecp_path,
)
from vshake_pipeline.limits import parse_optional_lim_pair_from_section

# Everything is located relative to this file, so the analysis directory can be
# renamed or moved without editing any code. PROJECT_ROOT is its parent, i.e.
# the directory that also holds TRAJS/.
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ANALYSIS_ROOT.parent
SCRIPTS_ROOT = ANALYSIS_ROOT / "scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from hd_plot_colors import COLOR_D, COLOR_H  # noqa: E402

ALL_STEPS = ("master_xyz", "geom", "sir", "sir_viz")

_README = f"{ANALYSIS_ROOT.name}/README.md"

# g_perp and grad V arrows must stay distinguishable from the H/D eigenvector
# colors, so they get their own green/orange pair.
COLOR_GPERP = "#2ca02c"
COLOR_GRADV = "#ff7f0e"

DEFAULT_DENSITY_ROOT = PROJECT_ROOT / "TRAJS" / "xDENPLT" / "OUTPUT"

_VECTOR_FAMILIES = ("sir", "gperp", "gradv")

_VIZ_FIELDS: dict[str, Any] = {
    "amplitude": float,
    "n_frames": int,
    "zoom_center": float,
    "turn_x": float,
    "turn_y": float,
    "turn_z": float,
    "png_width": int,
    "png_height": int,
    "png_dpi": int,
    "ray": int,
    "gif_delay": int,
    "gif_loop": int,
    "gif_extra_args": str,
}


@dataclass
class MasterXyzConfig:
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    log_minor_ticks: bool = True
    yscale: str = "log"


@dataclass
class SirConfig:
    nslices: int = 10
    var: float = 0.95
    n_eigenvalues_plot: int = 10
    n_eigenvectors_viz: int = 3


@dataclass
class DensityConfig:
    """Nuclear-density cube overlaid on the static arrow figures."""

    root: str = str(DEFAULT_DENSITY_ROOT)
    isovalue: float = 0.04
    color: tuple[int, int, int] = (128, 229, 255)
    paths: dict[str, str] = field(default_factory=dict)

    def cube_for(self, run_name: str, iso: str) -> Path:
        override = self.paths.get(iso)
        if override:
            return resolve_input_path(override)
        return density_cube_path(resolve_input_path(self.root), run_name, iso)


@dataclass
class SirVizConfig:
    arrow_scale: float = 1.0
    arrow_color: dict[str, str] = field(
        default_factory=lambda: {"H": COLOR_H, "D": COLOR_D}
    )
    gperp_color: str = COLOR_GPERP
    gradv_color: str = COLOR_GRADV
    arrow_mol_transparency: float = 0.5
    amplitude: float = 0.5
    n_frames: int = 24
    zoom_center: float = 3.0
    turn_x: float = 0.0
    turn_y: float = 0.0
    turn_z: float = 0.0
    png_width: int = 2000
    png_height: int = 1400
    png_dpi: int = 300
    ray: int = 1
    gif_delay: int = 5
    gif_loop: int = 0
    gif_extra_args: str = ""
    density: DensityConfig = field(default_factory=DensityConfig)
    cos_barchart_xlim: tuple[float, float] | None = None
    cos_barchart_ylim: tuple[float, float] | None = None
    per_isotope: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_vector: dict[str, dict[str, Any]] = field(default_factory=dict)

    def base_viz(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in _VIZ_FIELDS}

    def viz_for_isotope(self, iso: str) -> dict[str, Any]:
        d = self.base_viz()
        d.update(self.per_isotope.get(iso, {}))
        return d


@dataclass
class QchemMecpConfig:
    temp: str = "HIGH_TEMP"
    batch: str = "BATCH1"


@dataclass
class GeomConfig:
    observables: list[dict[str, Any]] = field(default_factory=list)
    plots: list[dict[str, Any]] = field(default_factory=list)
    plot_defaults: dict[str, Any] = field(default_factory=dict)
    write_table: bool = False
    table_path: str = "geom_scatter_table.txt"
    plot_1d_hist_splines: bool = True


@dataclass
class PipelineConfig:
    run_name: str
    trajectory_root: Path
    temps: list[str]
    isotopes: list[str]
    batches: list[int] | None
    master_xyz: MasterXyzConfig
    sir: SirConfig
    sir_viz: SirVizConfig
    qchem_mecp: QchemMecpConfig
    geom: GeomConfig
    n_skip_rows: int = 2

    @property
    def run_dir(self) -> Path:
        return ANALYSIS_ROOT / "runs" / self.run_name


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key)
    return dict(val) if isinstance(val, dict) else {}


def resolve_input_path(raw: str | Path) -> Path:
    """Absolute paths are used as-is; relative ones hang off PROJECT_ROOT.

    Keeps configs portable: ``../TRAJS/CBO`` and ``TRAJS/CBO`` both work no
    matter what the analysis directory is called or where it is invoked from.
    """
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _arrow_mol_transparency(raw: Any) -> float:
    val = float(raw)
    if not (0.0 <= val <= 1.0):
        raise ValueError(
            "sir_viz.arrow_mol_transparency must be in [0, 1]; "
            f"got {val!r}"
        )
    return val


def _arrow_color(raw: Any) -> dict[str, str]:
    defaults = {"H": COLOR_H, "D": COLOR_D}
    if raw is None:
        return dict(defaults)
    if isinstance(raw, str):
        return {"H": raw, "D": raw}
    if isinstance(raw, dict):
        out = dict(defaults)
        for k, v in raw.items():
            key = str(k)
            if key not in ("H", "D"):
                raise ValueError(
                    f"sir_viz.arrow_color key {key!r} not allowed; use H and/or D."
                )
            out[key] = str(v)
        return out
    raise ValueError(
        "sir_viz.arrow_color must be a string or {H, D} mapping; "
        f"got {type(raw).__name__}."
    )


def _coerce_viz_overrides(section: dict[str, Any], where: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in section.items():
        if k not in _VIZ_FIELDS:
            raise ValueError(
                f"Unknown sir_viz.{where} key {k!r}. Overridable fields: "
                f"{', '.join(sorted(_VIZ_FIELDS))}."
            )
        out[k] = _VIZ_FIELDS[k](v)
    return out


def _parse_per_isotope(sv: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_isotope: dict[str, dict[str, Any]] = {}
    for iso_k, sec in (sv.get("per_isotope") or {}).items():
        per_isotope[str(iso_k)] = _coerce_viz_overrides(
            dict(sec or {}), f"per_isotope.{iso_k}"
        )
    return per_isotope


def _parse_per_vector(sv: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-vector-family swing/render overrides keyed sir / gperp / gradv."""
    per_vector: dict[str, dict[str, Any]] = {}
    for vec_k, sec in (sv.get("per_vector") or {}).items():
        key = str(vec_k)
        if key not in _VECTOR_FAMILIES:
            raise ValueError(
                f"Unknown sir_viz.per_vector key {key!r}; expected one of "
                f"{', '.join(_VECTOR_FAMILIES)}."
            )
        per_vector[key] = _coerce_viz_overrides(dict(sec or {}), f"per_vector.{key}")
    return per_vector


def _parse_density(sv: dict[str, Any]) -> DensityConfig:
    raw = sv.get("density")
    if raw is None:
        return DensityConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"sir_viz.density must be a mapping; got {type(raw).__name__}."
        )
    allowed = {"root", "isovalue", "color", "paths"}
    unknown = set(map(str, raw)) - allowed
    if unknown:
        raise ValueError(
            f"Unknown sir_viz.density key(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )

    color_raw = raw.get("color", (128, 229, 255))
    color = tuple(int(c) for c in color_raw)
    if len(color) != 3 or any(not (0 <= c <= 255) for c in color):
        raise ValueError(
            f"sir_viz.density.color must be three 0-255 integers; got {color_raw!r}"
        )

    paths: dict[str, str] = {}
    for k, v in (raw.get("paths") or {}).items():
        key = str(k)
        if key not in ("H", "D"):
            raise ValueError(
                f"sir_viz.density.paths key {key!r} not allowed; use H and/or D."
            )
        paths[key] = str(v)

    return DensityConfig(
        root=str(raw.get("root", DEFAULT_DENSITY_ROOT)),
        isovalue=float(raw.get("isovalue", 0.04)),
        color=color,
        paths=paths,
    )


def _reject_retired_keys(raw: dict[str, Any]) -> None:
    for dead_section in ("dmap", "jacobian_viz", "displace", "bin_select", "tun_coord"):
        if isinstance(raw.get(dead_section), dict):
            raise ValueError(
                f"Config contains '{dead_section}:' which is not used by the SIR "
                f"pipeline. Steps are master_xyz → geom → sir → sir_viz. "
                f"See {_README}."
            )
    if "xyz_root" in raw:
        raise ValueError(
            "xyz_root is retired; use trajectory_root (TRAJS) and step master_xyz. "
            f"See {_README}."
        )
    sir_raw = raw.get("sir")
    if isinstance(sir_raw, dict):
        dead_sir = {"alpha", "n_beyond", "max_pairs"} & set(map(str, sir_raw))
        if dead_sir:
            raise ValueError(
                f"Config contains retired sir key(s): {', '.join(sorted(dead_sir))}. "
                "The Li chi-squared dimension test and the atom-pair importance "
                "analysis were removed; use sir.n_eigenvalues_plot and "
                f"sir.n_eigenvectors_viz instead. See {_README}."
            )


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path).expanduser()
    if not path.is_absolute() and not path.exists():
        # Bare "CBO.yaml" should find the config shipped next to the pipeline,
        # whatever the analysis directory is called and wherever it is invoked.
        candidate = ANALYSIS_ROOT / path
        if candidate.is_file():
            path = candidate
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    _reject_retired_keys(raw)

    run_name = raw.get("run_name")
    traj_root = raw.get("trajectory_root")
    if not run_name or not traj_root:
        raise ValueError("Config requires run_name and trajectory_root")

    mx = _section(raw, "master_xyz")
    x_lim_tuple = parse_optional_lim_pair_from_section(
        mx, "x_lim", field_name="master_xyz.x_lim"
    )
    y_lim_tuple = parse_optional_lim_pair_from_section(
        mx, "y_lim", field_name="master_xyz.y_lim"
    )
    yscale_raw = mx.get("yscale", "log")
    yscale = str(yscale_raw).lower()
    if yscale not in ("log", "linear"):
        raise ValueError(
            f"master_xyz.yscale must be 'log' or 'linear', got {yscale_raw!r}"
        )

    sr = _section(raw, "sir")
    sv = _section(raw, "sir_viz")
    qc = _section(raw, "qchem_mecp")
    geom = _section(raw, "geom")

    n_eig_plot = int(sr.get("n_eigenvalues_plot", 10))
    if n_eig_plot < 1:
        raise ValueError(f"sir.n_eigenvalues_plot must be >= 1; got {n_eig_plot!r}")
    n_vec_viz = int(sr.get("n_eigenvectors_viz", 3))
    if n_vec_viz < 1:
        raise ValueError(f"sir.n_eigenvectors_viz must be >= 1; got {n_vec_viz!r}")

    cos_bar = _section(sv, "cos_barchart")
    cos_xlim = parse_optional_lim_pair_from_section(
        cos_bar, "xlim", field_name="sir_viz.cos_barchart.xlim"
    )
    cos_ylim = parse_optional_lim_pair_from_section(
        cos_bar, "ylim", field_name="sir_viz.cos_barchart.ylim"
    )

    return PipelineConfig(
        run_name=str(run_name),
        trajectory_root=resolve_input_path(traj_root),
        temps=list(raw.get("temps") or ["HIGH_TEMP", "LOW_TEMP"]),
        isotopes=list(raw.get("isotopes") or ["H", "D"]),
        batches=raw.get("batches"),
        master_xyz=MasterXyzConfig(
            x_lim=x_lim_tuple,
            y_lim=y_lim_tuple,
            log_minor_ticks=bool(mx.get("log_minor_ticks", True)),
            yscale=yscale,
        ),
        sir=SirConfig(
            nslices=int(sr.get("nslices", 10)),
            var=float(sr.get("var", 0.95)),
            n_eigenvalues_plot=n_eig_plot,
            n_eigenvectors_viz=n_vec_viz,
        ),
        sir_viz=SirVizConfig(
            arrow_scale=float(sv.get("arrow_scale", 1.0)),
            arrow_color=_arrow_color(sv.get("arrow_color")),
            gperp_color=str(sv.get("gperp_color", COLOR_GPERP)),
            gradv_color=str(sv.get("gradv_color", COLOR_GRADV)),
            arrow_mol_transparency=_arrow_mol_transparency(
                sv.get("arrow_mol_transparency", 0.5)
            ),
            amplitude=float(sv.get("amplitude", 0.5)),
            n_frames=int(sv.get("n_frames", 24)),
            zoom_center=float(sv.get("zoom_center", 3.0)),
            turn_x=float(sv.get("turn_x", 0.0)),
            turn_y=float(sv.get("turn_y", 0.0)),
            turn_z=float(sv.get("turn_z", 0.0)),
            png_width=int(sv.get("png_width", 2000)),
            png_height=int(sv.get("png_height", 1400)),
            png_dpi=int(sv.get("png_dpi", 300)),
            ray=int(sv.get("ray", 1)),
            gif_delay=int(sv.get("gif_delay", 5)),
            gif_loop=int(sv.get("gif_loop", 0)),
            gif_extra_args=str(sv.get("gif_extra_args", "")),
            density=_parse_density(sv),
            cos_barchart_xlim=cos_xlim,
            cos_barchart_ylim=cos_ylim,
            per_isotope=_parse_per_isotope(sv),
            per_vector=_parse_per_vector(sv),
        ),
        qchem_mecp=QchemMecpConfig(
            temp=str(qc.get("temp", "HIGH_TEMP")),
            batch=str(qc.get("batch", "BATCH1")),
        ),
        n_skip_rows=int(raw.get("n_skip_rows", 2)),
        geom=GeomConfig(
            observables=list(geom.get("observables") or []),
            plots=list(geom.get("plots") or []),
            plot_defaults=dict(geom.get("plot_defaults") or {}),
            write_table=bool(geom.get("write_table", False)),
            table_path=str(geom.get("table_path", "geom_scatter_table.txt")),
            plot_1d_hist_splines=bool(geom.get("plot_1d_hist_splines", True)),
        ),
    )


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


@dataclass
class ResolvedConfig:
    pipeline: PipelineConfig
    discovery: DiscoveryResult
    seam_paths: dict[str, list[str]]
    qchem_paths: dict[str, str]
    density_cube_paths: dict[str, str]

    def to_yaml_dict(self) -> dict[str, Any]:
        p = self.pipeline
        return {
            "run_name": p.run_name,
            "trajectory_root": str(self.discovery.trajectory_root),
            "temps": list(p.temps),
            "isotopes": list(p.isotopes),
            "batches": list(p.batches) if p.batches is not None else None,
            "seam_geometries": self.seam_paths,
            "qchem_mecp_paths": self.qchem_paths,
            "density_cube_paths": self.density_cube_paths,
            "master_xyz": _plain(p.master_xyz),
            "sir": _plain(p.sir),
            "sir_viz": _plain(p.sir_viz),
            "qchem_mecp": _plain(p.qchem_mecp),
            "n_skip_rows": p.n_skip_rows,
            "geom": _plain(p.geom),
        }


def resolve_config(cfg: PipelineConfig) -> ResolvedConfig:
    discovery = discover_seam_xyz(
        cfg.trajectory_root,
        temps=cfg.temps,
        isotopes=cfg.isotopes,
        batches=cfg.batches,
    )
    seam_paths = {
        iso: [str(p) for p in paths] for iso, paths in discovery.by_isotope.items()
    }
    qchem_paths: dict[str, str] = {}
    density_cube_paths: dict[str, str] = {}
    for iso in cfg.isotopes:
        qchem_paths[iso] = str(
            qchem_mecp_path(
                discovery.trajectory_root,
                temp=cfg.qchem_mecp.temp,
                batch=cfg.qchem_mecp.batch,
                isotope=iso,
            )
        )
        density_cube_paths[iso] = str(cfg.sir_viz.density.cube_for(cfg.run_name, iso))
    return ResolvedConfig(
        pipeline=cfg,
        discovery=discovery,
        seam_paths=seam_paths,
        qchem_paths=qchem_paths,
        density_cube_paths=density_cube_paths,
    )
