#!/usr/bin/env python3
"""Verbose per-step parser for seam-sampling trajectory NPZ archives.

Usage:
    python inspect_trajectory.py
    python inspect_trajectory.py --path trajectory.npz
    python inspect_trajectory.py --start-frame 0 --end-frame 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO

import numpy as np


EXPECTED_KEYS = [
    "step",
    "time / fs",
    "R",
    "V / bohr/(a.u. time)",
    "E_D / E_h",
    "E_A / E_h",
    "sigma",
    "E_0 / E_h",
    "E_1 / E_h",
    "H_DA / E_h",
    "S_DA",
    "tunneling_splitting / cm-1",
    "grad_sigma / E_h/bohr",
    "grad_E_0_all_rows / E_h/bohr",
    "grad_E_1_all_rows / E_h/bohr",
    "grad_H01_all_rows / E_h/bohr",
    "grad_S01_all_rows / 1/bohr",
    "grad_sigma_all_rows / E_h/bohr",
    "nacv_d10_all_rows / 1/bohr",
    "msdft_row_labels",
    "g",
    "k",
    "kinetic_energy / E_h",
    "total_energy / E_h",
    "temperature / K",
    "n_shake_iterations",
    "proton_center_donor / bohr",
    "proton_center_acceptor / bohr",
    "diabatic_contributions_pct",
    "mb_resample_recovery_used",
    "mb_resample_attempt",
]


def _fmt_float(value: float) -> str:
    return f"{value:.8e}"


def _is_per_frame_array(arr: np.ndarray, n_frames: int) -> bool:
    return arr.ndim >= 1 and arr.shape[0] == n_frames


def _value_to_string(value: np.ndarray | float | int | bool) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, separator=", ")
    if isinstance(value, (np.floating, float)):
        return _fmt_float(float(value))
    return str(value)


def _emit(handle: TextIO, message: str = "") -> None:
    handle.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print all trajectory.npz content, frame-by-frame."
    )
    parser.add_argument(
        "--path",
        default="trajectory.npz",
        help="Path to NPZ trajectory archive (default: trajectory.npz).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to print (default: 0).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame index to print (inclusive). Default: last available frame.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Floating-point precision for printed arrays (default: 8).",
    )
    parser.add_argument(
        "--linewidth",
        type=int,
        default=200,
        help="Max characters per line when printing arrays (default: 200).",
    )
    parser.add_argument(
        "--out",
        default="npz_analysis.out",
        help="Output report path (default: npz_analysis.out). Use '-' for stdout.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory archive not found: {path}")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if args.end_frame is not None and args.end_frame < 0:
        raise ValueError("--end-frame must be >= 0")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("--end-frame must be >= --start-frame")
    if args.precision < 1:
        raise ValueError("--precision must be >= 1")
    if args.linewidth < 20:
        raise ValueError("--linewidth should be >= 20")

    np.set_printoptions(precision=args.precision, suppress=False, linewidth=args.linewidth)

    out_handle: TextIO
    close_out = False
    if args.out == "-":
        import sys

        out_handle = sys.stdout
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("w", encoding="utf-8")
        close_out = True

    try:
        with np.load(path) as data:
            keys = list(data.files)
            _emit(out_handle, f"Trajectory file: {path}")
            _emit(out_handle, f"Keys ({len(keys)}): {', '.join(keys)}")

            missing = [k for k in EXPECTED_KEYS if k not in keys]
            extra = [k for k in keys if k not in EXPECTED_KEYS]
            _emit(
                out_handle,
                f"Expected keys present: {len(EXPECTED_KEYS) - len(missing)}/{len(EXPECTED_KEYS)}",
            )
            if missing:
                _emit(out_handle, f"Missing expected keys: {', '.join(missing)}")
            if extra:
                _emit(out_handle, f"Extra keys: {', '.join(extra)}")

            if "step" not in data:
                raise KeyError("trajectory archive does not contain required key 'step'")
            n_frames = int(data["step"].shape[0])
            _emit(out_handle, f"Frames available: {n_frames}")
            if n_frames == 0:
                _emit(out_handle, "No frames in trajectory.")
                return

            end_frame = args.end_frame if args.end_frame is not None else (n_frames - 1)
            if end_frame >= n_frames:
                raise IndexError(
                    f"--end-frame={end_frame} is out of bounds for n_frames={n_frames}"
                )

            _emit(out_handle, f"Frames requested: {args.start_frame} -> {end_frame} (inclusive)")

            _emit(out_handle, "")
            _emit(out_handle, "=== Array inventory ===")
            for name in keys:
                arr = data[name]
                per_frame = _is_per_frame_array(arr, n_frames)
                _emit(
                    out_handle,
                    f"- {name}: shape={arr.shape}, dtype={arr.dtype}, "
                    f"{'per-frame' if per_frame else 'global'}"
                )

            _emit(out_handle, "")
            _emit(out_handle, "=== Global arrays (not indexed by frame) ===")
            global_keys = [k for k in keys if not _is_per_frame_array(data[k], n_frames)]
            if not global_keys:
                _emit(out_handle, "None")
            else:
                for key in global_keys:
                    _emit(out_handle, "")
                    _emit(out_handle, f"{key}:")
                    _emit(out_handle, _value_to_string(data[key]))

            _emit(out_handle, "")
            _emit(out_handle, "=== Per-frame trajectory dump ===")
            per_frame_keys = [k for k in keys if _is_per_frame_array(data[k], n_frames)]
            for i in range(args.start_frame, end_frame + 1):
                step_str = (
                    str(int(data["step"][i]))
                    if "step" in data and data["step"].ndim == 1
                    else f"frame={i}"
                )
                time_str = (
                    _fmt_float(float(data["time"][i]))
                    if "time" in data and _is_per_frame_array(data["time"], n_frames)
                    else "N/A"
                )
                _emit(out_handle, "")
                _emit(out_handle, "=" * 80)
                _emit(out_handle, f"FRAME_INDEX={i} | STEP={step_str} | TIME_FS={time_str}")
                _emit(out_handle, "=" * 80)
                for key in per_frame_keys:
                    value_i = data[key][i]
                    _emit(out_handle, f"{key}:")
                    _emit(out_handle, _value_to_string(value_i))
                    _emit(out_handle, "-" * 80)
    finally:
        if close_out:
            out_handle.close()


if __name__ == "__main__":
    main()
