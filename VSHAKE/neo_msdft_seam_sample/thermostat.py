"""Thermostat models."""

from __future__ import annotations

import numpy as np

import units


def kinetic_energy_au(masses_me: np.ndarray, velocities: np.ndarray) -> float:
    return 0.5 * float(np.sum(masses_me[:, None] * velocities * velocities))


def instantaneous_temperature(
    masses_me: np.ndarray,
    velocities: np.ndarray,
    dof: int,
) -> float:
    ke = kinetic_energy_au(masses_me, velocities)
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive")
    return 2.0 * ke / (dof * units.KB_HARTREE_PER_K)


def rescale_velocities_to_temperature(
    masses_me: np.ndarray,
    velocities: np.ndarray,
    target_temperature_K: float,
    dof: int,
) -> np.ndarray:
    """Scale velocities so instantaneous temperature matches target_temperature_K."""
    velocities = np.asarray(velocities, dtype=float)
    t_current = instantaneous_temperature(masses_me, velocities, dof)
    if t_current <= 1e-16:
        return velocities.copy()
    scale = np.sqrt(target_temperature_K / t_current)
    return velocities * scale


def heavy_atom_seam_dof(n_heavy_atoms: int) -> int:
    """Hardcoded classical DOF for constrained NEO seam dynamics."""
    dof = 3 * int(n_heavy_atoms) - 7
    if dof <= 0:
        raise ValueError(
            f"Computed non-positive DOF ({dof}) for n_heavy_atoms={n_heavy_atoms}."
        )
    return dof


def stochastic_modified_berendsen_scale(
    dt_au: float,
    tau_au: float,
    target_temperature: float,
    current_temperature: float,
    n_dof: int,
    rng: np.random.Generator,
) -> float:
    """Return stochastic velocity-rescaling factor from Bussi et al. (2007) Eq. (A7).

    This implements the exact analytical propagator for the kinetic energy
    SDE (Eq. 7) over a finite timestep, as derived in the Appendix of
    Bussi, Donadio, and Parrinello, J. Chem. Phys. 126, 014101 (2007).

    Parameters
    ----------
    dt_au : float
        Integration time step in atomic units.
    tau_au : float
        Thermostat relaxation time in atomic units.
    target_temperature : float
        Target temperature in Kelvin.
    current_temperature : float
        Instantaneous temperature in Kelvin.
    n_dof : int
        Number of classical degrees of freedom.
    rng : np.random.Generator
        Random number generator.
    """
    if current_temperature <= 1e-16:
        return 1.0

    ratio = target_temperature / current_temperature
    c = np.exp(-dt_au / tau_au)

    # R_1: a single standard normal draw
    R1 = rng.normal()

    # Sum of (N_f - 1) squared standard normals = gamma(shape=(N_f-1)/2, scale=2)
    if n_dof > 1:
        sum_R_sq = rng.gamma(shape=(n_dof - 1) / 2.0, scale=2.0)
    else:
        sum_R_sq = 0.0

    # Bussi Eq. (A7)
    alpha_sq = (
        c
        + (ratio / n_dof) * (1.0 - c) * (R1**2 + sum_R_sq)
        + 2.0 * np.sqrt(c * ratio * (1.0 - c) / n_dof) * R1
    )

    return float(np.sqrt(max(alpha_sq, 0.0)))
