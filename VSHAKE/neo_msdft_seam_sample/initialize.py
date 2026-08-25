"""Initialization utilities for seam dynamics."""

from __future__ import annotations

import logging

import numpy as np

import units
from qc_call import MSDFTResult, QCCallBase

LOGGER = logging.getLogger("seam_sample")
COMMON_ISOTOPE_MASS_AMU = {
    "H": 1.007825,
    "C": 12.000000,
    "N": 14.003074,
    "O": 15.994915,
    "F": 18.998403,
}


def read_xyz(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    n_atoms = int(lines[0])
    atom_lines = lines[2 : 2 + n_atoms]
    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, units.angstrom_to_bohr(np.array(coords, dtype=float))


def read_velocity_xyz(path: str, expected_symbols: list[str]) -> np.ndarray:
    """Read an XYZ-like velocity file with units matching internal V."""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Velocity file {path} is too short to be valid XYZ-like format.")

    n_atoms = int(lines[0])
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"Velocity file {path} declares {n_atoms} atoms but has {len(atom_lines)} velocity rows."
        )
    if n_atoms != len(expected_symbols):
        raise ValueError(
            f"Velocity file {path} has {n_atoms} atoms but geometry has {len(expected_symbols)} atoms."
        )

    parsed_symbols: list[str] = []
    velocities: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid velocity row in {path}: {line}")
        parsed_symbols.append(parts[0])
        velocities.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if parsed_symbols != expected_symbols:
        raise ValueError(
            "Velocity file atom ordering/symbols do not match geometry symbols. "
            f"Expected {expected_symbols}, got {parsed_symbols}."
        )

    return np.asarray(velocities, dtype=float)


def masses_from_symbols(symbols: list[str]) -> np.ndarray:
    masses = []
    for symbol in symbols:
        if symbol not in COMMON_ISOTOPE_MASS_AMU:
            supported = ", ".join(sorted(COMMON_ISOTOPE_MASS_AMU))
            raise ValueError(
                f"No common-isotope mass configured for '{symbol}'. Supported elements: {supported}."
            )
        masses.append(COMMON_ISOTOPE_MASS_AMU[symbol])
    return np.asarray(masses, dtype=float)


def maxwell_boltzmann_velocities(
    masses_amu: np.ndarray,
    temperature: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Maxwell-Boltzmann velocities in atomic units."""
    masses_amu = np.asarray(masses_amu, dtype=float)
    if masses_amu.ndim != 1:
        raise ValueError("masses_amu must have shape (N_atoms,)")
    rng = np.random.default_rng() if rng is None else rng
    masses_me = masses_amu * units.AMU_TO_ELECTRON_MASS
    sigma_v = np.sqrt(units.KB_HARTREE_PER_K * temperature / masses_me)
    return rng.normal(loc=0.0, scale=sigma_v[:, None], size=(masses_amu.size, 3))


def remove_com_translation(velocities: np.ndarray, masses_me: np.ndarray) -> np.ndarray:
    """Remove center-of-mass translational velocity."""
    velocities = np.asarray(velocities, dtype=float)
    masses_me = np.asarray(masses_me, dtype=float)
    v_com = np.sum(masses_me[:, None] * velocities, axis=0) / np.sum(masses_me)
    return velocities - v_com[None, :]


def remove_angular_momentum(
    velocities: np.ndarray,
    positions: np.ndarray,
    masses_me: np.ndarray,
) -> np.ndarray:
    """Remove net rigid-body rotation about the center of mass.

    With COM-relative positions r_i, L = sum_i m_i (r_i x v_i) and the inertia
    tensor I, solve omega = I^{-1} L and set v_i <- v_i - omega x r_i. Uses
    least squares when I is singular or ill-conditioned (e.g. linear molecules).
    """
    velocities = np.asarray(velocities, dtype=float).copy()
    positions = np.asarray(positions, dtype=float)
    masses_me = np.asarray(masses_me, dtype=float)

    # Positions relative to center of mass
    total_mass = np.sum(masses_me)
    r_com = np.sum(masses_me[:, None] * positions, axis=0) / total_mass
    r = positions - r_com[None, :]

    # Angular momentum: L = sum_i m_i (r_i x v_i)
    L = np.zeros(3, dtype=float)
    for i in range(len(masses_me)):
        L += masses_me[i] * np.cross(r[i], velocities[i])

    # Inertia tensor: I_ab = sum_i m_i (|r_i|^2 delta_ab - r_ia r_ib)
    I = np.zeros((3, 3), dtype=float)
    for i in range(len(masses_me)):
        ri_sq = float(np.dot(r[i], r[i]))
        I += masses_me[i] * (ri_sq * np.eye(3) - np.outer(r[i], r[i]))

    # Planar or linear clusters can make I rank-deficient; avoid unstable solves.
    cond_i = float(np.linalg.cond(I))
    if not np.isfinite(cond_i) or cond_i > 1.0e10:
        LOGGER.warning(
            "Inertia tensor is singular or ill-conditioned (cond=%.3e). "
            "Using least squares for angular momentum removal.",
            cond_i,
        )
        omega = np.linalg.lstsq(I, L, rcond=None)[0]
    else:
        try:
            omega = np.linalg.solve(I, L)
        except np.linalg.LinAlgError:
            LOGGER.warning(
                "Inertia solve failed; using least squares for angular momentum removal."
            )
            omega = np.linalg.lstsq(I, L, rcond=None)[0]

    # Subtract rigid rotation
    for i in range(len(masses_me)):
        velocities[i] -= np.cross(omega, r[i])

    L_after = np.zeros(3, dtype=float)
    for i in range(len(masses_me)):
        L_after += masses_me[i] * np.cross(r[i], velocities[i])
    LOGGER.info(
        "Angular momentum removal: |L_before|=%.3e, |L_after|=%.3e",
        float(np.linalg.norm(L)),
        float(np.linalg.norm(L_after)),
    )

    return velocities


def project_velocity_onto_seam(
    velocities: np.ndarray,
    grad_sigma: np.ndarray,
    masses_me: np.ndarray,
) -> np.ndarray:
    """Project velocities so that dR/dt dot grad_sigma = 0."""
    velocities = np.asarray(velocities, dtype=float)
    grad_sigma = np.asarray(grad_sigma, dtype=float)
    masses_me = np.asarray(masses_me, dtype=float)

    G = -grad_sigma
    MiG = G / masses_me[:, None]
    denom = float(np.sum(G * MiG))
    if denom < 1e-30:
        LOGGER.warning("Velocity projection skipped: near-zero G^T M^-1 G denominator.")
        return velocities
    k = -2.0 * float(np.sum(velocities * G)) / denom
    return velocities + 0.5 * k * MiG


def sample_seam_projected_mb_velocities(
    masses_amu: np.ndarray,
    R: np.ndarray,
    grad_sigma: np.ndarray,
    temperature: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Maxwell-Boltzmann draw with COM, angular-momentum removal, and seam projection."""
    masses_amu = np.asarray(masses_amu, dtype=float)
    masses_me = masses_amu * units.AMU_TO_ELECTRON_MASS
    V = maxwell_boltzmann_velocities(masses_amu, temperature, rng=rng)
    V = remove_com_translation(V, masses_me)
    V = remove_angular_momentum(V, R, masses_me)
    return project_velocity_onto_seam(V, grad_sigma, masses_me)


def seam_tangent_nacv_velocity(
    V_mb: np.ndarray,
    masses_me: np.ndarray,
    msdft_result: MSDFTResult,
) -> np.ndarray | None:
    """Scale seam-tangent NACV to match kinetic energy of V_mb, or None if unavailable."""
    if msdft_result.nacv_d10_all_rows is None:
        return None
    d10_heavy = np.asarray(msdft_result.nacv_d10_all_rows[2:], dtype=float)
    grad_sigma = np.asarray(msdft_result.grad_sigma, dtype=float)
    gs_dot = float(np.sum(grad_sigma * grad_sigma))
    if gs_dot < 1e-30:
        return None
    d10_dot = float(np.sum(d10_heavy * grad_sigma))
    d_parallel = d10_heavy - (d10_dot / gs_dot) * grad_sigma
    if float(np.linalg.norm(d_parallel.ravel())) < 1e-12:
        return None
    ke = 0.5 * float(np.sum(masses_me[:, None] * V_mb * V_mb))
    ke_d = 0.5 * float(np.sum(masses_me[:, None] * d_parallel * d_parallel))
    if ke_d < 1e-30 or ke < 1e-30:
        return None
    alpha = np.sqrt(ke / ke_d)
    return alpha * d_parallel


def check_initial_seam(sigma: float, constraint_threshold: float) -> None:
    """Log whether the initial geometry is on the seam."""
    sigma_abs = abs(float(sigma))
    if sigma_abs > constraint_threshold:
        LOGGER.warning(
            "Initial geometry has |sigma| = %.3e Hartree, which exceeds the constraint threshold %.3e. "
            "The code assumes the initial geometry is on or near the MECP.",
            sigma_abs,
            constraint_threshold,
        )
        return
    LOGGER.info("Initial geometry is on the seam: |sigma| = %.3e Hartree.", sigma_abs)


def initialize_dynamics(
    xyz_path: str,
    masses_amu: np.ndarray,
    temperature: float,
    qc_call: QCCallBase,
    constraint_threshold: float = 4e-5,
    initial_velocity_path: str | None = None,
    initial_velocity_mode: str = "mb",
) -> tuple[list[str], np.ndarray, np.ndarray, MSDFTResult]:
    """Build initial geometry/velocity and perform initial QC call."""
    symbols, R = read_xyz(xyz_path)
    masses_amu = np.asarray(masses_amu, dtype=float)
    if masses_amu.ndim != 1:
        raise ValueError("masses_amu must have shape (N_atoms,)")
    if masses_amu.size != len(symbols):
        raise ValueError("masses_amu length must match atom count in xyz")

    qc_call.set_step_context(0)
    _ = qc_call.eval_diabats(R)
    msdft_result = qc_call.eval_msdft(R)
    check_initial_seam(msdft_result.sigma, constraint_threshold)

    masses_me = masses_amu * units.AMU_TO_ELECTRON_MASS
    mode = initial_velocity_mode.strip().lower()
    if mode not in ("mb", "file", "nacv"):
        raise ValueError(
            f"initial_velocity_mode must be 'mb', 'file', or 'nacv', got '{initial_velocity_mode}'."
        )
    if mode == "file":
        if not initial_velocity_path:
            raise ValueError("initial_velocity_mode is 'file' but initial_velocity_path is not set.")
        V = read_velocity_xyz(initial_velocity_path, expected_symbols=symbols)
        LOGGER.info(
            "Using initial velocities from %s (units must match internal V units).",
            initial_velocity_path,
        )
        V = remove_com_translation(V, masses_me)
        V = remove_angular_momentum(V, R, masses_me)
        V = project_velocity_onto_seam(V, msdft_result.grad_sigma, masses_me)
    elif mode == "nacv":
        if initial_velocity_path:
            LOGGER.warning(
                "initial_velocity_mode is 'nacv'; ignoring initial_velocity_path=%s.",
                initial_velocity_path,
            )
        V_mb = sample_seam_projected_mb_velocities(
            masses_amu, R, msdft_result.grad_sigma, temperature
        )
        V_nacv = seam_tangent_nacv_velocity(V_mb, masses_me, msdft_result)
        if V_nacv is None:
            LOGGER.warning(
                "NACV is entirely along grad_sigma; falling back to MB initialization."
            )
            V = project_velocity_onto_seam(V_mb, msdft_result.grad_sigma, masses_me)
        else:
            V = V_nacv
            ke = 0.5 * float(np.sum(masses_me[:, None] * V * V))
            LOGGER.info(
                "Initial velocities directed along seam-tangent NACV (d_parallel). KE = %.8e Hartree.",
                ke,
            )
    else:
        V = sample_seam_projected_mb_velocities(
            masses_amu, R, msdft_result.grad_sigma, temperature
        )
        LOGGER.info("Sampled initial velocities from Maxwell-Boltzmann distribution.")
    return symbols, R, V, msdft_result
