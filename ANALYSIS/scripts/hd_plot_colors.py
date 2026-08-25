"""Shared H/D plot colors for master_xyz (step 1), geom (step 2),
diffusion_map_diagnostic (step 3), and jacobian_viz (step 4) PyMOL arrow colors.

Scatter/series markers, MECP dashed lines, MECP point markers, diffusion-map
bars, and displacement arrows use the same color per isotope so H is one red
and D is one blue everywhere those plots are drawn.
"""

from __future__ import annotations

# Tableau red / blue (matplotlib C3 / C0 defaults)
COLOR_H = "#d62728"
COLOR_D = "#1f77b4"

# MECP markers (dashed lines and 2D points) match the isotope series color.
COLOR_H_MECP = COLOR_H
COLOR_D_MECP = COLOR_D

# Aliases used by build_master_xyz
COLOR_H_SCATTER = COLOR_H
COLOR_D_SCATTER = COLOR_D


def series_color(isotope: str) -> str:
    if isotope == "D":
        return COLOR_D
    if isotope == "H":
        return COLOR_H
    raise ValueError(f"isotope must be 'H' or 'D', got {isotope!r}")


def mecp_color(isotope: str) -> str:
    if isotope == "D":
        return COLOR_D_MECP
    if isotope == "H":
        return COLOR_H_MECP
    raise ValueError(f"isotope must be 'H' or 'D', got {isotope!r}")
