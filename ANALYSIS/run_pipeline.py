#!/usr/bin/env python3
"""Run the SIR vibronic-coupling analysis pipeline from a YAML config.

Steps (in order): ``master_xyz`` → ``geom`` → ``sir`` → ``sir_viz``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from vshake_pipeline.config import ALL_STEPS, load_config, resolve_config
from vshake_pipeline.stages import (
    step_geom,
    step_master_xyz,
    step_sir,
    step_sir_viz,
    write_geom_sidecar,
    write_resolved_config,
)

STAGE_FUNCS = {
    "master_xyz": step_master_xyz,
    "geom": step_geom,
    "sir": step_sir,
    "sir_viz": step_sir_viz,
}


def _setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    root = logging.getLogger("vshake_pipeline")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def main() -> None:
    parser = argparse.ArgumentParser(description="SIR vibronic-coupling analysis pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML config")
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated steps (default: all). Choices: {','.join(ALL_STEPS)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolved = resolve_config(cfg)
    run_dir = cfg.run_dir
    _setup_logging(run_dir)
    log = logging.getLogger("vshake_pipeline")

    write_resolved_config(resolved, run_dir)
    geom_sidecar = run_dir / "geom_config.yaml"
    write_geom_sidecar(cfg, geom_sidecar)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in steps:
        if s not in STAGE_FUNCS:
            raise SystemExit(f"Unknown step {s!r}; choices: {', '.join(ALL_STEPS)}")

    log.info("Run %s  steps=%s  force=%s", cfg.run_name, steps, args.force)
    for s in steps:
        log.info("=== step %s ===", s)
        fn = STAGE_FUNCS[s]
        if s == "geom":
            fn(resolved, run_dir, geom_sidecar, dry_run=args.dry_run, force=args.force)
        else:
            fn(resolved, run_dir, dry_run=args.dry_run, force=args.force)

    log.info("Done.")


if __name__ == "__main__":
    main()
