"""Discover seam_geometries.xyz under a SECOND_BATCH-style trajectory root."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

VALID_TEMPS = frozenset({"HIGH_TEMP", "LOW_TEMP"})
VALID_ISOTOPES = frozenset({"H", "D"})
SEAM_NAME = "seam_geometries.xyz"
BATCH_RE = re.compile(r"^BATCH(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveryResult:
    """Ordered seam geometry paths per isotope."""

    by_isotope: dict[str, list[Path]]
    trajectory_root: Path


def _batch_number(batch_dir_name: str) -> int | None:
    m = BATCH_RE.match(batch_dir_name)
    return int(m.group(1)) if m else None


def discover_seam_xyz(
    trajectory_root: str | Path,
    *,
    temps: list[str] | None = None,
    isotopes: list[str] | None = None,
    batches: list[int] | None = None,
) -> DiscoveryResult:
    """
    Find {TEMP}/BATCH{N}/{H|D}/seam_geometries.xyz under trajectory_root.

    Paths are sorted by temp (HIGH_TEMP before LOW_TEMP), batch number, then path.
    """
    root = Path(trajectory_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trajectory_root is not a directory: {root}")

    want_temps = list(temps) if temps is not None else sorted(VALID_TEMPS)
    want_isotopes = list(isotopes) if isotopes is not None else ["H", "D"]
    want_batches = set(batches) if batches is not None else None

    for t in want_temps:
        if t not in VALID_TEMPS:
            raise ValueError(f"Invalid temp {t!r}; expected one of {sorted(VALID_TEMPS)}")
    for iso in want_isotopes:
        if iso not in VALID_ISOTOPES:
            raise ValueError(
                f"Invalid isotope {iso!r}; expected one of {sorted(VALID_ISOTOPES)}"
            )

    collected: dict[str, list[tuple[int, int, Path]]] = {
        iso: [] for iso in want_isotopes
    }

    for temp in want_temps:
        temp_dir = root / temp
        if not temp_dir.is_dir():
            print(
                f"Warning: missing temp directory {temp_dir} (skipping)",
                file=sys.stderr,
            )
            continue
        for batch_dir in sorted(temp_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            bn = _batch_number(batch_dir.name)
            if bn is None:
                continue
            if want_batches is not None and bn not in want_batches:
                continue
            for iso in want_isotopes:
                seam = batch_dir / iso / SEAM_NAME
                if seam.is_file():
                    temp_ord = 0 if temp == "HIGH_TEMP" else 1
                    collected[iso].append((temp_ord, bn, seam))

    by_isotope: dict[str, list[Path]] = {}
    for iso in want_isotopes:
        entries = collected[iso]
        entries.sort(key=lambda x: (x[0], x[1], str(x[2])))
        paths = [e[2] for e in entries]
        if not paths:
            raise FileNotFoundError(
                f"No {SEAM_NAME} files found for isotope {iso!r} under {root}. "
                f"Expected layout: {{HIGH_TEMP,LOW_TEMP}}/BATCH{{N}}/{{H,D}}/{SEAM_NAME}"
            )
        by_isotope[iso] = paths

    return DiscoveryResult(by_isotope=by_isotope, trajectory_root=root)


def qchem_mecp_path(
    trajectory_root: Path,
    *,
    temp: str,
    batch: str,
    isotope: str,
) -> Path:
    """Reference-batch NEO-MSDFT Q-Chem MECP output for sir_viz."""
    if temp not in VALID_TEMPS:
        raise ValueError(f"Invalid qchem_mecp.temp: {temp!r}")
    if isotope not in VALID_ISOTOPES:
        raise ValueError(f"Invalid isotope: {isotope!r}")
    batch_name = batch if batch.upper().startswith("BATCH") else f"BATCH{batch}"
    path = (
        trajectory_root
        / temp
        / batch_name
        / isotope
        / "output"
        / "msdft"
        / "msdft_0.out"
    )
    return path


def density_cube_path(root: str | Path, run_name: str, isotope: str) -> Path:
    """Nuclear-density cube for one system/isotope: ``{root}/{RUN}_{ISO}.cube``."""
    if isotope not in VALID_ISOTOPES:
        raise ValueError(f"Invalid isotope: {isotope!r}")
    return Path(root) / f"{run_name}_{isotope}.cube"
