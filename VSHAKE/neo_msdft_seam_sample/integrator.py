"""Constrained seam-sampling velocity-Verlet integrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import units
from initialize import sample_seam_projected_mb_velocities
from qc_call import MSDFTResult, QCCallBase
from thermostat import (
    heavy_atom_seam_dof,
    instantaneous_temperature,
    kinetic_energy_au,
    rescale_velocities_to_temperature,
    stochastic_modified_berendsen_scale,
)

LOGGER = logging.getLogger("seam_sample")


class DiabatOptNotConvergedDuringShakeError(RuntimeError):
    """Raised when eval_diabats fails during an in-progress SHAKE root search."""


@dataclass
class StepInfo:
    g: float
    k: float
    n_shake_iterations: int
    n_diabat_evals: int
    temperature: float
    kinetic_energy: float
    mb_resample_recovery_used: bool = False
    mb_resample_attempt: int = 0


class SeamSampleIntegrator:
    def __init__(
        self,
        qc_call: QCCallBase,
        masses: np.ndarray,
        dt: float = 0.5,
        constraint_threshold: float = 4e-5,
        max_shake_iter: int = 20,
        target_temperature: float = 300.0,
        thermostat_tau: float = 10.0,
        thermostat: str = "berendsen",
        min_proton_center_separation_angstrom: float | None = 0.75,
        max_mb_resamples: int = 20,
    ) -> None:
        self.qc_call = qc_call
        self.masses_amu = np.asarray(masses, dtype=float)
        if self.masses_amu.ndim != 1:
            raise ValueError("masses must have shape (N_atoms,)")
        self.masses_me = self.masses_amu * units.AMU_TO_ELECTRON_MASS
        self.dt_fs = dt
        self.dt_au = dt * units.FS_TO_AU_TIME
        self.constraint_threshold = constraint_threshold
        # Applied independently to each SHAKE attempt (primary step, each MB resample).
        self.max_shake_iter = int(max_shake_iter)
        self.target_temperature = target_temperature
        self.thermostat_tau_au = thermostat_tau * units.FS_TO_AU_TIME
        if thermostat not in ("berendsen", "none"):
            raise ValueError(f"thermostat must be 'berendsen' or 'none', got '{thermostat}'")
        self.thermostat_mode = thermostat
        self.rng = np.random.default_rng()
        self.min_proton_center_separation_angstrom = min_proton_center_separation_angstrom
        self.max_mb_resamples = int(max_mb_resamples)
        if self.max_mb_resamples < 1:
            raise ValueError("max_mb_resamples must be >= 1")

    def _inv_mass_times(self, array: np.ndarray) -> np.ndarray:
        return array / self.masses_me[:, None]

    def _eval_diabats_for_shake(self, R: np.ndarray) -> tuple[float, bool]:
        """Evaluate sigma at R for SHAKE; return (sigma, ok). ok=False on any QC/OPT failure."""
        try:
            result = self.qc_call.eval_diabats(R)
        except RuntimeError as exc:
            LOGGER.warning("Diabat evaluation failed during SHAKE: %s", exc)
            return 0.0, False
        if not result.converged:
            LOGGER.warning(
                "Diabat OPT did not converge during SHAKE (donor and/or acceptor)."
            )
            return 0.0, False
        return result.sigma, True

    def _shake_root_search(
        self,
        R_bar: np.ndarray,
        direction: np.ndarray,
        f0: float,
        fp0: float,
    ) -> tuple[float, int, int]:
        if abs(f0) < self.constraint_threshold:
            return 0.0, 0, 0
        if abs(fp0) < 1e-16:
            raise RuntimeError("SHAKE derivative is near zero; cannot continue")

        calls = 0

        def f_of_g(g):
            nonlocal calls
            calls += 1
            sigma, ok = self._eval_diabats_for_shake(R_bar + g * direction)
            if not ok:
                raise DiabatOptNotConvergedDuringShakeError(
                    "Diabat evaluation failed during SHAKE root search"
                )
            return sigma

        g0 = -f0 / fp0
        f_g0 = f_of_g(g0)
        if abs(f_g0) < self.constraint_threshold:
            return g0, 1, calls

        g_left, g_right = (0.0, g0) if f0 * f_g0 <= 0.0 else (None, None)
        f_left, f_right = (f0, f_g0) if g_left is not None else (None, None)

        if g_left is None:
            step = max(abs(g0), 1.0)
            g_prev, f_prev = g0, f_g0
            for _ in range(self.max_shake_iter):
                g_next = g_prev + np.sign(-f_prev) * step
                f_next = f_of_g(g_next)
                if f_prev * f_next <= 0.0:
                    g_left, g_right = (g_prev, g_next)
                    f_left, f_right = (f_prev, f_next)
                    break
                g_prev, f_prev = g_next, f_next
                step *= 2.0
            if g_left is None:
                raise RuntimeError("Failed to bracket SHAKE root")

        for it in range(1, self.max_shake_iter + 1):
            g_mid = 0.5 * (g_left + g_right)
            f_mid = f_of_g(g_mid)
            if abs(f_mid) < self.constraint_threshold:
                return g_mid, it + 1, calls
            if f_left * f_mid <= 0.0:
                g_right, f_right = g_mid, f_mid
            else:
                g_left, f_left = g_mid, f_mid

        raise RuntimeError("SHAKE did not converge within max iterations")

    def _resample_mb_velocities_preserve_temperature(
        self,
        R: np.ndarray,
        grad_sigma: np.ndarray,
        temperature_keep_K: float,
        n_dof: int,
    ) -> np.ndarray:
        V_mb = sample_seam_projected_mb_velocities(
            self.masses_amu,
            R,
            grad_sigma,
            self.target_temperature,
            rng=self.rng,
        )
        return rescale_velocities_to_temperature(
            self.masses_me, V_mb, temperature_keep_K, n_dof
        )

    def _try_shake_step(
        self,
        R_bar: np.ndarray,
        direction: np.ndarray,
        G_t: np.ndarray,
        *,
        attempt_label: str = "primary",
    ) -> tuple[bool, float, int, int, np.ndarray]:
        LOGGER.debug(
            "SHAKE attempt (%s): up to %d root-search iterations per bracket/bisection phase.",
            attempt_label,
            self.max_shake_iter,
        )
        n_evals = 0
        f0, f0_ok = self._eval_diabats_for_shake(R_bar)
        n_evals = 1
        if not f0_ok:
            LOGGER.warning("SHAKE attempt (%s) failed at initial diabat evaluation.", attempt_label)
            return False, 0.0, 0, n_evals, R_bar.copy()

        fp0 = -float(np.sum(G_t * direction))
        try:
            g, n_shake_iter, root_search_calls = self._shake_root_search(R_bar, direction, f0, fp0)
        except DiabatOptNotConvergedDuringShakeError as exc:
            LOGGER.warning(
                "SHAKE attempt (%s) root search stopped (diabat failure): %s",
                attempt_label,
                exc,
            )
            return False, 0.0, 0, n_evals, R_bar.copy()
        except RuntimeError as exc:
            LOGGER.warning(
                "SHAKE attempt (%s) root search failed (bracketing/convergence): %s",
                attempt_label,
                exc,
            )
            return False, 0.0, 0, n_evals, R_bar.copy()

        n_evals += root_search_calls
        R_new = R_bar + g * direction

        if self.min_proton_center_separation_angstrom is not None:
            min_sep_bohr = units.angstrom_to_bohr(float(self.min_proton_center_separation_angstrom))
            sep = self.qc_call.proton_center_separation_bohr()
            if sep < min_sep_bohr:
                LOGGER.warning(
                    "SHAKE attempt (%s): proton centers too close after SHAKE "
                    "(d(D,A)=%.4f Ang < %.4f Ang).",
                    attempt_label,
                    units.bohr_to_angstrom(sep),
                    self.min_proton_center_separation_angstrom,
                )
                return False, g, n_shake_iter, n_evals, R_new

        return True, g, n_shake_iter, n_evals, R_new

    def step(
        self,
        step_num: int,
        R: np.ndarray,
        V: np.ndarray,
        msdft_current: MSDFTResult,
    ):
        R_input = np.asarray(R, dtype=float).copy()
        V_input = np.asarray(V, dtype=float).copy()

        self.qc_call.set_step_context(step_num)
        R = R_input.copy()
        V = V_input.copy()
        F_t = msdft_current.F
        G_t = msdft_current.G

        n_dof = heavy_atom_seam_dof(R.shape[0])
        t_keep = instantaneous_temperature(self.masses_me, V_input, n_dof)

        h = self.dt_au
        R_bar = R + h * V + 0.5 * h * h * self._inv_mass_times(F_t)
        direction = 0.5 * h * self._inv_mass_times(G_t)

        success, g, n_shake_iter, total_diabat_evals, R_new = self._try_shake_step(
            R_bar, direction, G_t, attempt_label="primary"
        )
        mb_resample_recovery_used = False
        mb_resample_attempt = 0

        if not success:
            LOGGER.warning(
                "Step %d: SHAKE failed (Q-Chem/OPT error, max iterations, or single-well). "
                "Attempting Maxwell-Boltzmann velocity resample recovery.",
                step_num,
            )

            recovery_succeeded = False
            for attempt in range(1, self.max_mb_resamples + 1):
                LOGGER.info(
                    "Step %d: MB resample recovery attempt %d/%d at current geometry.",
                    step_num,
                    attempt,
                    self.max_mb_resamples,
                )
                self.qc_call.proton_center_donor = msdft_current.proton_center_donor.copy()
                self.qc_call.proton_center_acceptor = msdft_current.proton_center_acceptor.copy()
                self.qc_call.clean_density_guess_files()

                V_resampled = self._resample_mb_velocities_preserve_temperature(
                    R,
                    msdft_current.grad_sigma,
                    t_keep,
                    n_dof,
                )
                R_bar_resample = (
                    R + h * V_resampled + 0.5 * h * h * self._inv_mass_times(F_t)
                )
                success_resample, g, n_shake_iter, extra_evals, R_new = self._try_shake_step(
                    R_bar_resample,
                    direction,
                    G_t,
                    attempt_label=f"mb_resample_{attempt}",
                )
                total_diabat_evals += extra_evals

                if success_resample:
                    recovery_succeeded = True
                    mb_resample_recovery_used = True
                    mb_resample_attempt = attempt
                    V = V_resampled
                    LOGGER.info(
                        "Step %d: MB resample recovery succeeded on attempt %d.",
                        step_num,
                        attempt,
                    )
                    break

            if not recovery_succeeded:
                raise RuntimeError(
                    f"Step {step_num}: MB resample recovery failed after "
                    f"{self.max_mb_resamples} attempt(s) at the current geometry "
                    f"(max_mb_resamples={self.max_mb_resamples}). Trajectory terminated."
                )

        msdft_new = self.qc_call.eval_msdft(R_new)
        F_new = msdft_new.F
        G_new = msdft_new.G

        Q = V + 0.5 * self._inv_mass_times(h * F_t + g * G_t)
        V_tilde = Q + 0.5 * h * self._inv_mass_times(F_new)

        denom = float(np.sum(G_new * self._inv_mass_times(G_new)))
        if denom <= 1e-24:
            raise RuntimeError("RATTLE denominator is too small")
        k = -2.0 * float(np.sum(V_tilde * G_new)) / denom
        V_new = V_tilde + 0.5 * k * self._inv_mass_times(G_new)

        if self.thermostat_mode == "berendsen":
            t_inst = instantaneous_temperature(self.masses_me, V_new, n_dof)
            scale = stochastic_modified_berendsen_scale(
                self.dt_au,
                self.thermostat_tau_au,
                self.target_temperature,
                t_inst,
                n_dof,
                self.rng,
            )
            V_new = V_new * scale

        ke = kinetic_energy_au(self.masses_me, V_new)
        t_inst = instantaneous_temperature(self.masses_me, V_new, n_dof)
        step_info = StepInfo(
            g=g,
            k=k,
            n_shake_iterations=n_shake_iter,
            n_diabat_evals=total_diabat_evals,
            temperature=t_inst,
            kinetic_energy=ke,
            mb_resample_recovery_used=mb_resample_recovery_used,
            mb_resample_attempt=mb_resample_attempt,
        )
        return R_new, V_new, msdft_new, step_info
