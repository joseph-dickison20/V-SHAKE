#!/usr/bin/env python3
"""
Evaluate the standard Xu / Hammes–Schiffer overlap correction on PB4-D FHF⁻ data.

  S'_ab = alpha * S_ab ** beta

Default: shared alpha with isotope-specific beta (production parameters), write FINAL_FITS.txt.

Usage:
  python3 opt_param.py
  python3 opt_param.py --alpha 0.56 --beta-h 1.10 --beta-d 0.98 --no-write
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

DATA_CSV = "fhf_data.csv"
REFS_CSV = "fhf_refs.csv"
FINAL_OUT = "FINAL_FITS.txt"

ALPHA_DEFAULT = 0.56
BETA_H_DEFAULT = 1.10
BETA_D_DEFAULT = 0.98

HA_TO_CM = 219474.631371
ISOTOPES = ("H", "D")
DECIMALS = 3


def compute_splitting_standard(S12_raw, E_diab, H_ab_raw, alpha, beta):
    """2×2 generalized eigen-splitting for equal diabatic diagonals (cm^-1)."""
    vs = H_ab_raw / S12_raw
    s_prime = alpha * S12_raw ** beta
    h_ab_corr = s_prime * vs
    # For H = [[E, V], [V, E]], S = [[1, s], [s, 1]]:
    # eigenvalues satisfy (E - λ)^2 = (V - λ s)^2  →  λ = (E ± V) / (1 ± s)
    # splitting = λ+ − λ− = 2 (V - E s) / (1 - s^2)
    denom = 1.0 - s_prime * s_prime
    delta_ha = 2.0 * (h_ab_corr - E_diab * s_prime) / denom
    return abs(delta_ha) * HA_TO_CM


def _f(row, key):
    v = row.get(key, "").strip()
    if v == "":
        return float("nan")
    return float(v)


def load_merged(data_path, refs_path):
    with open(refs_path, newline="") as f:
        refs = {float(r["R_FF"]): r for r in csv.DictReader(f)}
    rows = []
    with open(data_path, newline="") as f:
        for r in csv.DictReader(f):
            rff = float(r["R_FF"])
            ref = refs[rff]
            rows.append(
                {
                    "R_FF": rff,
                    "H_uncorr": _f(r, "H_Splitting_cm-1"),
                    "H_S12": _f(r, "H_S12"),
                    "H_E": _f(r, "H_diabatic_E_Ha"),
                    "H_Hab": _f(r, "H_Hamiltonian_offdiag_Ha"),
                    "H_ref": float(ref["H_Splitting_cm-1"]),
                    "D_uncorr": _f(r, "D_Splitting_cm-1"),
                    "D_S12": _f(r, "D_S12"),
                    "D_E": _f(r, "D_diabatic_E_Ha"),
                    "D_Hab": _f(r, "D_Hamiltonian_offdiag_Ha"),
                    "D_ref": float(ref["D_Splitting_cm-1"]),
                }
            )
    return rows


def is_valid(row, iso):
    return all(
        math.isfinite(row[f"{iso}_{k}"])
        for k in ("uncorr", "S12", "E", "Hab", "ref")
    )


def metrics(calcs, refs):
    if not calcs:
        return None
    err = [c - r for c, r in zip(calcs, refs)]
    rel = [(c - r) / r for c, r in zip(calcs, refs)]
    return {
        "rmse_abs": math.sqrt(sum(e * e for e in err) / len(err)),
        "rmse_rel_pct": math.sqrt(sum(x * x for x in rel) / len(rel)) * 100,
        "max_rel_pct": math.sqrt(max(x * x for x in rel)) * 100,
    }


def format_iso_table(iso, alpha, beta, rows):
    buf = io.StringIO()
    calcs, refs = [], []
    for row in rows:
        if not is_valid(row, iso):
            continue
        c = compute_splitting_standard(
            row[f"{iso}_S12"], row[f"{iso}_E"], row[f"{iso}_Hab"], alpha, beta
        )
        calcs.append(c)
        refs.append(row[f"{iso}_ref"])
    m = metrics(calcs, refs)

    print("=" * 72, file=buf)
    print(
        f"  {iso}:  alpha = {alpha:.{DECIMALS}f}   beta = {beta:.{DECIMALS}f}",
        file=buf,
    )
    if m:
        print(
            f"  rmse_abs = {m['rmse_abs']:.4f} cm^-1   "
            f"rmse_rel = {m['rmse_rel_pct']:.4f} %   "
            f"max_rel = {m['max_rel_pct']:.4f} %",
            file=buf,
        )
    print("=" * 72, file=buf)
    print(
        f"  {'R_FF':>6}  {'Uncorr':>10}  {'Corrected':>10}  {'FGH Ref':>10}  "
        f"{'Corr/Ref':>10}  Fit",
        file=buf,
    )
    for row in rows:
        r = row["R_FF"]
        ref = row[f"{iso}_ref"]
        if not is_valid(row, iso):
            print(
                f"  {r:6.2f}  {'—':>10}  {'—':>10}  {ref:10.4f}  "
                f"{'—':>10}   no",
                file=buf,
            )
            continue
        calc = compute_splitting_standard(
            row[f"{iso}_S12"], row[f"{iso}_E"], row[f"{iso}_Hab"], alpha, beta
        )
        cr = calc / ref if ref else float("nan")
        print(
            f"  {r:6.2f}  {row[f'{iso}_uncorr']:10.4f}  {calc:10.4f}  "
            f"{ref:10.4f}  {cr:10.6f}  yes",
            file=buf,
        )
    return buf.getvalue()


def evaluate(rows, alpha, beta_h, beta_d):
    betas = {"H": beta_h, "D": beta_d}
    return "\n".join(
        format_iso_table(iso, alpha, betas[iso], rows) for iso in ISOTOPES
    ) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--beta-h", type=float, default=BETA_H_DEFAULT)
    parser.add_argument("--beta-d", type=float, default=BETA_D_DEFAULT)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print tables only; do not write FINAL_FITS.txt",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    rows = load_merged(root / DATA_CSV, root / REFS_CSV)
    text = evaluate(rows, args.alpha, args.beta_h, args.beta_d)
    print(text, end="")

    if not args.no_write:
        out = root / FINAL_OUT
        header = (
            "Standard overlap correction — PB4-D FHF⁻\n"
            f"Shared alpha = {args.alpha:.{DECIMALS}f}\n"
            f"beta_H = {args.beta_h:.{DECIMALS}f}   beta_D = {args.beta_d:.{DECIMALS}f}\n"
            "H @ R_FF = 2.65 Å excluded (missing NEO-BL-MSDFT data).\n\n"
        )
        out.write_text(header + text)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
