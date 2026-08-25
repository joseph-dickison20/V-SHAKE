"""Main entrypoint for constrained seam-sampling trajectory generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from initialize import initialize_dynamics, masses_from_symbols, read_xyz
from integrator import SeamSampleIntegrator
from qc_call import QCCallBase
from thermostat import heavy_atom_seam_dof, instantaneous_temperature, kinetic_energy_au
from trajectory import TrajectoryFrame, append_seam_xyz, save_trajectory
import units

LOGGER = logging.getLogger("seam_sample")


def _optional_value(cfg: dict, key: str):
    """Return None for null-like optional config values."""
    value = cfg.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _setup_logging(log_path: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)


def run_from_config(config_path: str, qc_call: QCCallBase) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    save_qchem_outputs = bool(cfg.get("save_qchem_outputs", True))
    output_dir = Path(cfg.get("output_dir", "output"))
    log_filename = cfg.get("log_filename", "output.log")
    log_path = (output_dir / log_filename) if save_qchem_outputs else Path(log_filename)
    _setup_logging(log_path)

    symbols_xyz, _ = read_xyz(cfg["initial_geometry"])
    masses_cfg = _optional_value(cfg, "masses_amu")
    if masses_cfg is not None:
        masses = np.array(masses_cfg, dtype=float)
        LOGGER.info("Using masses_amu from config.")
    else:
        masses = masses_from_symbols(symbols_xyz)
        LOGGER.info("Using common-isotope masses parsed from XYZ symbols.")

    initial_velocity_path = _optional_value(cfg, "initial_velocity_path")
    if "initial_velocity_mode" in cfg and cfg["initial_velocity_mode"] is not None:
        initial_velocity_mode = str(cfg["initial_velocity_mode"]).strip().lower()
    elif initial_velocity_path:
        initial_velocity_mode = "file"
    else:
        initial_velocity_mode = "mb"

    symbols, R, V, msdft_curr = initialize_dynamics(
        xyz_path=cfg["initial_geometry"],
        masses_amu=masses,
        temperature=cfg.get("target_temperature", 300.0),
        qc_call=qc_call,
        constraint_threshold=cfg.get("constraint_threshold", 4e-5),
        initial_velocity_path=initial_velocity_path,
        initial_velocity_mode=initial_velocity_mode,
    )

    if "min_proton_center_separation_angstrom" not in cfg:
        min_proton_center_separation_angstrom = 0.75
    else:
        min_sep_raw = _optional_value(cfg, "min_proton_center_separation_angstrom")
        if min_sep_raw is None:
            min_proton_center_separation_angstrom = None
        else:
            min_proton_center_separation_angstrom = float(min_sep_raw)

    integrator = SeamSampleIntegrator(
        qc_call=qc_call,
        masses=masses,
        dt=cfg.get("dt_fs", 0.5),
        constraint_threshold=cfg.get("constraint_threshold", 4e-5),
        max_shake_iter=cfg.get("max_shake_iter", 20),
        target_temperature=cfg.get("target_temperature", 300.0),
        thermostat_tau=cfg.get("thermostat_tau_fs", 10.0),
        thermostat=cfg.get("thermostat", "berendsen"),
        min_proton_center_separation_angstrom=min_proton_center_separation_angstrom,
        max_mb_resamples=cfg.get("max_mb_resamples", 20),
    )
    if min_proton_center_separation_angstrom is None:
        LOGGER.info(
            "Single-well detection disabled (min_proton_center_separation_angstrom=null)."
        )
    else:
        LOGGER.info(
            "Single-well detection: min donor-acceptor proton center separation=%.4f Ang.",
            min_proton_center_separation_angstrom,
        )

    n_steps = int(cfg["n_steps"])
    save_interval = int(cfg.get("save_interval", 1))
    out_path = Path(cfg.get("trajectory_path", "trajectory.npz"))
    seam_xyz_path = cfg.get("seam_xyz_path", "seam_geometries.xyz")
    Path(seam_xyz_path).write_text("", encoding="utf-8")
    frames: list[TrajectoryFrame] = []

    def _write_trajectory_checkpoint() -> None:
        if frames:
            save_trajectory(str(out_path), frames)
    masses_me = masses * units.AMU_TO_ELECTRON_MASS
    n_dof = heavy_atom_seam_dof(R.shape[0])
    initial_ke = kinetic_energy_au(masses_me, V)
    initial_temp = instantaneous_temperature(masses_me, V, n_dof)
    if initial_velocity_mode == "file":
        LOGGER.info(
            "Initial velocity mode: file (%s), format=symbol vx vy vz, units=Bohr/a.u. time",
            initial_velocity_path,
        )
    elif initial_velocity_mode == "nacv":
        LOGGER.info(
            "Initial velocity mode: nacv (MB kinetic energy, seam-tangent NACV direction), "
            "target_temperature=%.2f K",
            cfg.get("target_temperature", 300.0),
        )
    else:
        LOGGER.info(
            "Initial velocity mode: mb (Maxwell-Boltzmann), units=Bohr/a.u. time, target_temperature=%.2f K",
            cfg.get("target_temperature", 300.0),
        )
    LOGGER.info("=== Step 0 (time 0.000 fs): basis-center optimization + MSDFT ===")
    append_seam_xyz(
        path=seam_xyz_path,
        classical_symbols=symbols,
        R_bohr=R,
        proton_center_donor_bohr=msdft_curr.proton_center_donor,
        proton_center_acceptor_bohr=msdft_curr.proton_center_acceptor,
        step=0,
        time_fs=0.0,
        sigma=msdft_curr.sigma,
        tunneling_splitting_cm1=(
            msdft_curr.tunneling_splitting_cm1
            if msdft_curr.tunneling_splitting_cm1 is not None
            else units.hartree_to_cm1(msdft_curr.tunneling_splitting)
        ),
        E_0=msdft_curr.E_0,
    )
    frames.append(
        TrajectoryFrame(
            step=0,
            time=0.0,
            R=R.copy(),
            V=V.copy(),
            E_D=msdft_curr.E_D,
            E_A=msdft_curr.E_A,
            sigma=msdft_curr.sigma,
            E_0=msdft_curr.E_0,
            E_1=msdft_curr.E_1,
            H_DA=msdft_curr.H_DA,
            S_DA=msdft_curr.S_DA,
            tunneling_splitting=msdft_curr.tunneling_splitting,
            grad_sigma=msdft_curr.grad_sigma.copy(),
            g=0.0,
            k=0.0,
            kinetic_energy=initial_ke,
            total_energy=initial_ke + msdft_curr.E_0,
            temperature=initial_temp,
            n_shake_iterations=0,
            proton_center_donor=msdft_curr.proton_center_donor.copy(),
            proton_center_acceptor=msdft_curr.proton_center_acceptor.copy(),
            diabatic_contributions_pct=(
                msdft_curr.diabatic_contributions_pct.copy()
                if msdft_curr.diabatic_contributions_pct is not None
                else None
            ),
            tunneling_splitting_cm1=msdft_curr.tunneling_splitting_cm1,
            msdft_row_labels=(
                msdft_curr.msdft_row_labels.copy()
                if msdft_curr.msdft_row_labels is not None
                else None
            ),
            grad_E_0_all_rows=(
                msdft_curr.grad_E_0_all_rows.copy()
                if msdft_curr.grad_E_0_all_rows is not None
                else None
            ),
            grad_E_1_all_rows=(
                msdft_curr.grad_E_1_all_rows.copy()
                if msdft_curr.grad_E_1_all_rows is not None
                else None
            ),
            grad_H01_all_rows=(
                msdft_curr.grad_H01_all_rows.copy()
                if msdft_curr.grad_H01_all_rows is not None
                else None
            ),
            grad_S01_all_rows=(
                msdft_curr.grad_S01_all_rows.copy()
                if msdft_curr.grad_S01_all_rows is not None
                else None
            ),
            grad_sigma_all_rows=(
                msdft_curr.grad_sigma_all_rows.copy()
                if msdft_curr.grad_sigma_all_rows is not None
                else None
            ),
            nacv_d10_all_rows=(
                msdft_curr.nacv_d10_all_rows.copy()
                if msdft_curr.nacv_d10_all_rows is not None
                else None
            ),
        )
    )
    _write_trajectory_checkpoint()
    LOGGER.info(
        "step=%d time_fs=%.3f |sigma|=%.3e E0=%.8f K=%.8f Etot=%.8f Delta=%.6f T=%.2f g=%.4e k=%.4e "
        "shake_iters=%d diabat_calls=%d mb_resample_recovery=%d mb_resample_attempt=%d",
        0,
        0.0,
        abs(msdft_curr.sigma),
        msdft_curr.E_0,
        initial_ke,
        initial_ke + msdft_curr.E_0,
        (
            msdft_curr.tunneling_splitting_cm1
            if msdft_curr.tunneling_splitting_cm1 is not None
            else float("nan")
        ),
        initial_temp,
        0.0,
        0.0,
        0,
        1,
        0,
        0,
    )
    LOGGER.info("")

    try:
        for step in range(1, n_steps + 1):
            LOGGER.info("=== Step %d (time %.3f fs) ===", step, step * cfg.get("dt_fs", 0.5))
            R, V, msdft_curr, info = integrator.step(step, R, V, msdft_curr)
            time_fs = step * cfg.get("dt_fs", 0.5)
            append_seam_xyz(
                path=seam_xyz_path,
                classical_symbols=symbols,
                R_bohr=R,
                proton_center_donor_bohr=msdft_curr.proton_center_donor,
                proton_center_acceptor_bohr=msdft_curr.proton_center_acceptor,
                step=step,
                time_fs=time_fs,
                sigma=msdft_curr.sigma,
                tunneling_splitting_cm1=(
                    msdft_curr.tunneling_splitting_cm1
                    if msdft_curr.tunneling_splitting_cm1 is not None
                    else units.hartree_to_cm1(msdft_curr.tunneling_splitting)
                ),
                E_0=msdft_curr.E_0,
            )
            if step % save_interval == 0:
                frames.append(
                    TrajectoryFrame(
                        step=step,
                        time=time_fs,
                        R=R.copy(),
                        V=V.copy(),
                        E_D=msdft_curr.E_D,
                        E_A=msdft_curr.E_A,
                        sigma=msdft_curr.sigma,
                        E_0=msdft_curr.E_0,
                        E_1=msdft_curr.E_1,
                        H_DA=msdft_curr.H_DA,
                        S_DA=msdft_curr.S_DA,
                        tunneling_splitting=msdft_curr.tunneling_splitting,
                        grad_sigma=msdft_curr.grad_sigma.copy(),
                        g=info.g,
                        k=info.k,
                        kinetic_energy=info.kinetic_energy,
                        total_energy=info.kinetic_energy + msdft_curr.E_0,
                        temperature=info.temperature,
                        n_shake_iterations=info.n_shake_iterations,
                        proton_center_donor=msdft_curr.proton_center_donor.copy(),
                        proton_center_acceptor=msdft_curr.proton_center_acceptor.copy(),
                        diabatic_contributions_pct=(
                            msdft_curr.diabatic_contributions_pct.copy()
                            if msdft_curr.diabatic_contributions_pct is not None
                            else None
                        ),
                        tunneling_splitting_cm1=msdft_curr.tunneling_splitting_cm1,
                        msdft_row_labels=(
                            msdft_curr.msdft_row_labels.copy()
                            if msdft_curr.msdft_row_labels is not None
                            else None
                        ),
                        grad_E_0_all_rows=(
                            msdft_curr.grad_E_0_all_rows.copy()
                            if msdft_curr.grad_E_0_all_rows is not None
                            else None
                        ),
                        grad_E_1_all_rows=(
                            msdft_curr.grad_E_1_all_rows.copy()
                            if msdft_curr.grad_E_1_all_rows is not None
                            else None
                        ),
                        grad_H01_all_rows=(
                            msdft_curr.grad_H01_all_rows.copy()
                            if msdft_curr.grad_H01_all_rows is not None
                            else None
                        ),
                        grad_S01_all_rows=(
                            msdft_curr.grad_S01_all_rows.copy()
                            if msdft_curr.grad_S01_all_rows is not None
                            else None
                        ),
                        grad_sigma_all_rows=(
                            msdft_curr.grad_sigma_all_rows.copy()
                            if msdft_curr.grad_sigma_all_rows is not None
                            else None
                        ),
                        nacv_d10_all_rows=(
                            msdft_curr.nacv_d10_all_rows.copy()
                            if msdft_curr.nacv_d10_all_rows is not None
                            else None
                        ),
                        mb_resample_recovery_used=info.mb_resample_recovery_used,
                        mb_resample_attempt=info.mb_resample_attempt,
                    )
                )
                _write_trajectory_checkpoint()
            LOGGER.info(
                "step=%d time_fs=%.3f |sigma|=%.3e E0=%.8f K=%.8f Etot=%.8f Delta=%.6f T=%.2f g=%.4e k=%.4e "
                "shake_iters=%d diabat_calls=%d mb_resample_recovery=%d mb_resample_attempt=%d",
                step,
                time_fs,
                abs(msdft_curr.sigma),
                msdft_curr.E_0,
                info.kinetic_energy,
                info.kinetic_energy + msdft_curr.E_0,
                (
                    msdft_curr.tunneling_splitting_cm1
                    if msdft_curr.tunneling_splitting_cm1 is not None
                    else float("nan")
                ),
                info.temperature,
                info.g,
                info.k,
                info.n_shake_iterations,
                info.n_diabat_evals,
                int(info.mb_resample_recovery_used),
                info.mb_resample_attempt,
            )
            LOGGER.info("")
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; writing partial trajectory")
    finally:
        if frames:
            save_trajectory(str(out_path), frames)
            LOGGER.info("Trajectory written to %s (%d frames)", out_path, len(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained seam-sampling trajectory")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    raise NotImplementedError(
        "Instantiate your QCCallBase implementation and call run_from_config(config, qc_call)."
    )


if __name__ == "__main__":
    main()
