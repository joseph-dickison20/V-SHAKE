"""Q-Chem call interfaces and result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-14:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


@dataclass
class DiabatResult:
    """Result from two separate NEO-DFT diabatic calculations.

    This is the cheap evaluation used during the SHAKE root search.
    Only the donor and acceptor diabatic energies (and their gradients)
    are computed. No MSDFT diagonalization, coupling, or adiabatic
    quantities are produced.
    """

    sigma: float
    grad_sigma: np.ndarray
    proton_center_donor: np.ndarray
    proton_center_acceptor: np.ndarray
    converged: bool = True


@dataclass
class MSDFTResult:
    """Result from a full NEO-MSDFT calculation at an on-seam geometry.

    This is the expensive evaluation performed once per time step, after
    the constrained geometry has been determined. It includes adiabatic
    energies, the ground-state gradient, diabatic coupling/overlap, and
    tunneling splitting.
    """

    E_D: float
    E_A: float
    grad_E_D: np.ndarray
    grad_E_A: np.ndarray
    E_0: float
    E_1: float
    grad_E_0: np.ndarray
    H_DA: float
    S_DA: float
    proton_center_donor: np.ndarray
    proton_center_acceptor: np.ndarray
    diabatic_contributions_pct: np.ndarray | None = None
    tunneling_splitting_cm1: float | None = None
    msdft_row_labels: np.ndarray | None = None
    grad_E_0_all_rows: np.ndarray | None = None
    grad_E_1_all_rows: np.ndarray | None = None
    grad_H01_all_rows: np.ndarray | None = None
    grad_S01_all_rows: np.ndarray | None = None
    grad_sigma_all_rows: np.ndarray | None = None
    nacv_d10_all_rows: np.ndarray | None = None
    _grad_sigma_direct: np.ndarray | None = None

    @property
    def sigma(self) -> float:
        return self.E_D - self.E_A

    @property
    def grad_sigma(self) -> np.ndarray:
        if self._grad_sigma_direct is not None:
            return self._grad_sigma_direct
        return self.grad_E_D - self.grad_E_A

    @property
    def G(self) -> np.ndarray:
        return -self.grad_sigma

    @property
    def F(self) -> np.ndarray:
        return -self.grad_E_0

    @property
    def tunneling_splitting(self) -> float:
        return self.E_1 - self.E_0


class QCCallBase(ABC):
    def __init__(
        self,
        proton_center_donor: np.ndarray,
        proton_center_acceptor: np.ndarray,
    ):
        """Store initial proton-center guesses in Bohr for warm starts."""
        self.proton_center_donor = np.array(proton_center_donor, dtype=float)
        self.proton_center_acceptor = np.array(proton_center_acceptor, dtype=float)

    def proton_center_separation_bohr(self) -> float:
        """Distance between donor and acceptor proton basis centers (Bohr)."""
        d = self.proton_center_acceptor - self.proton_center_donor
        return float(np.linalg.norm(d))

    def enforce_proton_center_separation(
        self,
        min_separation_bohr: float,
        fallback_direction: np.ndarray | None = None,
    ) -> None:
        """If donor/acceptor centers are closer than ``min_separation_bohr``, separate them.

        Centers are moved symmetrically about their midpoint along the donor→acceptor
        direction. If that direction is degenerate, ``fallback_direction`` (e.g. last
        good donor→acceptor vector) is used when its norm is nonzero; otherwise +x.

        No-op when current separation is already >= ``min_separation_bohr``.
        """
        min_separation_bohr = float(min_separation_bohr)
        if min_separation_bohr <= 0.0:
            return
        d_vec = self.proton_center_acceptor - self.proton_center_donor
        dist = float(np.linalg.norm(d_vec))
        if dist >= min_separation_bohr:
            return

        if dist > 1e-12:
            u = d_vec / dist
        elif fallback_direction is not None:
            fb = np.asarray(fallback_direction, dtype=float).reshape(3)
            if float(np.linalg.norm(fb)) > 1e-12:
                u = _normalize_vector(fb)
            else:
                u = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            u = np.array([1.0, 0.0, 0.0], dtype=float)

        midpoint = 0.5 * (self.proton_center_donor + self.proton_center_acceptor)
        half = 0.5 * min_separation_bohr
        self.proton_center_donor = midpoint - half * u
        self.proton_center_acceptor = midpoint + half * u

    def set_step_context(self, step_num: int) -> None:
        """Optional hook: inform backend that a new trajectory step started."""
        _ = step_num

    def clean_density_guess_files(self) -> None:
        """Remove stale Q-Chem density initial-guess files from the working directory."""
        return

    @abstractmethod
    def eval_diabats(self, R: np.ndarray) -> DiabatResult:
        """Run two NEO-DFT calculations (donor + acceptor) at geometry R.

        Returns only sigma = E_D - E_A and grad_sigma. Used during the
        SHAKE root search to locate the constrained geometry.

        Implementations must use current proton center guesses from this
        object and update them to optimized values before returning.
        """

    @abstractmethod
    def eval_msdft(self, R: np.ndarray) -> MSDFTResult:
        """Run a full NEO-MSDFT calculation at geometry R.

        Returns adiabatic energies, gradients, couplings, tunneling
        splitting, and all quantities needed for dynamics/analysis.
        Called once per time step at the on-seam geometry.

        Implementations must use current proton center guesses from this
        object and update them to optimized values before returning.
        """
