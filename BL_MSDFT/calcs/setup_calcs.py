#!/usr/bin/env python3
"""Generate PB4-D FHF⁻ OPT/MSDFT inputs under ff_*_ang/{proton,deuteron}/."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
BASIS = ROOT / "basis"

FF_DISTANCES = [2.60, 2.65, 2.70, 2.75, 2.80]

ISOTOPES = [
    ("proton", 1, "h"),
    ("deuteron", 2, "d"),
]

# H @ 2.65 Å excluded from the fit (unstable basis-center optimization)
SKIP = {("2.65", "proton")}

FILES_TO_PATCH = ("opt.in", "msdft.in")


def patch_file(path: Path, half_dist: float, neo_isotope: int, basis_body: str) -> None:
    text = path.read_text()
    dist_str = f"{half_dist:.6f}".rstrip("0").rstrip(".")
    if "." not in dist_str:
        dist_str += ".0"
    text = text.replace("dist_here", dist_str)
    text = text.replace("1_for_H_2_for_D", str(neo_isotope))
    text = text.replace("appropriate_basis_here", basis_body.rstrip())
    path.write_text(text)


def main() -> None:
    n = 0
    for ff in FF_DISTANCES:
        half = ff / 2.0
        ff_label = f"{ff:.2f}".replace(".", "p")
        for iso_label, neo_iso, iso_suffix in ISOTOPES:
            if (f"{ff:.2f}", iso_label) in SKIP:
                continue
            basis_body = (BASIS / f"pb4d_{iso_suffix}.txt").read_text()
            dest = ROOT / f"ff_{ff_label}_ang" / iso_label
            dest.mkdir(parents=True, exist_ok=True)
            for fname in FILES_TO_PATCH:
                shutil.copy2(TEMPLATES / fname, dest / fname)
                patch_file(dest / fname, half, neo_iso, basis_body)
            n += 1
    print(f"Created {n} PB4-D OPT/MSDFT job directories under {ROOT}")


if __name__ == "__main__":
    main()
