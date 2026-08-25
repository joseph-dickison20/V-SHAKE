"""Trajectory data structures and I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import units


@dataclass
class TrajectoryFrame:
    step: int
    time: float
    R: np.ndarray
    V: np.ndarray
    E_D: float
    E_A: float
    sigma: float
    E_0: float
    E_1: float
    H_DA: float
    S_DA: float
    tunneling_splitting: float
    grad_sigma: np.ndarray
    g: float
    k: float
    kinetic_energy: float
    total_energy: float
    temperature: float
    n_shake_iterations: int
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
    mb_resample_recovery_used: bool = False
    mb_resample_attempt: int = 0


def save_trajectory(path: str, frames: list[TrajectoryFrame]) -> None:
    n = len(frames)
    if n == 0:
        raise ValueError("Cannot save empty trajectory")

    diabatic_contributions_pct = np.stack(
        [
            (
                f.diabatic_contributions_pct
                if f.diabatic_contributions_pct is not None
                else np.full((2, 2), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    grad_E_0_all_rows = np.stack(
        [
            (
                f.grad_E_0_all_rows
                if f.grad_E_0_all_rows is not None
                else np.full((f.R.shape[0] + 2, 3), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    grad_E_1_all_rows = np.stack(
        [
            (
                f.grad_E_1_all_rows
                if f.grad_E_1_all_rows is not None
                else np.full((f.R.shape[0] + 2, 3), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    grad_H01_all_rows = np.stack(
        [
            (
                f.grad_H01_all_rows
                if f.grad_H01_all_rows is not None
                else np.full((f.R.shape[0] + 2, 3), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    grad_S01_all_rows = np.stack(
        [
            (
                f.grad_S01_all_rows
                if f.grad_S01_all_rows is not None
                else np.full((f.R.shape[0] + 2, 3), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    grad_sigma_all_rows = np.stack(
        [
            (
                f.grad_sigma_all_rows
                if f.grad_sigma_all_rows is not None
                else np.full((f.R.shape[0] + 2, 3), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    nacv_d10_all_rows = np.stack(
        [
            (
                f.nacv_d10_all_rows
                if f.nacv_d10_all_rows is not None
                else np.full((f.R.shape[0] + 2, 3), np.nan, dtype=float)
            )
            for f in frames
        ],
        axis=0,
    )
    row_labels = next((f.msdft_row_labels for f in frames if f.msdft_row_labels is not None), None)
    if row_labels is None:
        n_rows = frames[0].R.shape[0] + 2
        row_labels = np.asarray([""] * n_rows, dtype="U8")

    tunneling_splitting_cm1 = np.array(
        [
            (
                f.tunneling_splitting_cm1
                if f.tunneling_splitting_cm1 is not None
                else units.hartree_to_cm1(f.tunneling_splitting)
            )
            for f in frames
        ],
        dtype=float,
    )

    payload = {
        "step": np.array([f.step for f in frames], dtype=int),
        "time / fs": np.array([f.time for f in frames], dtype=float),
        "R": np.stack([f.R for f in frames], axis=0),
        "V / bohr/(a.u. time)": np.stack([f.V for f in frames], axis=0),
        "E_D / E_h": np.array([f.E_D for f in frames], dtype=float),
        "E_A / E_h": np.array([f.E_A for f in frames], dtype=float),
        "sigma": np.array([f.sigma for f in frames], dtype=float),
        "E_0 / E_h": np.array([f.E_0 for f in frames], dtype=float),
        "E_1 / E_h": np.array([f.E_1 for f in frames], dtype=float),
        "H_DA / E_h": np.array([f.H_DA for f in frames], dtype=float),
        "S_DA": np.array([f.S_DA for f in frames], dtype=float),
        "tunneling_splitting / cm-1": tunneling_splitting_cm1,
        "grad_sigma / E_h/bohr": np.stack([f.grad_sigma for f in frames], axis=0),
        "grad_E_0_all_rows / E_h/bohr": grad_E_0_all_rows,
        "grad_E_1_all_rows / E_h/bohr": grad_E_1_all_rows,
        "grad_H01_all_rows / E_h/bohr": grad_H01_all_rows,
        "grad_S01_all_rows / 1/bohr": grad_S01_all_rows,
        "grad_sigma_all_rows / E_h/bohr": grad_sigma_all_rows,
        "nacv_d10_all_rows / 1/bohr": nacv_d10_all_rows,
        "msdft_row_labels": row_labels,
        "g": np.array([f.g for f in frames], dtype=float),
        "k": np.array([f.k for f in frames], dtype=float),
        "kinetic_energy / E_h": np.array([f.kinetic_energy for f in frames], dtype=float),
        "total_energy / E_h": np.array([f.total_energy for f in frames], dtype=float),
        "temperature / K": np.array([f.temperature for f in frames], dtype=float),
        "n_shake_iterations": np.array([f.n_shake_iterations for f in frames], dtype=int),
        "proton_center_donor / bohr": np.stack([f.proton_center_donor for f in frames], axis=0),
        "proton_center_acceptor / bohr": np.stack([f.proton_center_acceptor for f in frames], axis=0),
        "diabatic_contributions_pct": diabatic_contributions_pct,
        "mb_resample_recovery_used": np.array(
            [f.mb_resample_recovery_used for f in frames], dtype=bool
        ),
        "mb_resample_attempt": np.array([f.mb_resample_attempt for f in frames], dtype=int),
    }
    np.savez_compressed(path, **payload)


def load_trajectory(path: str) -> list[TrajectoryFrame]:
    data = np.load(path)
    n = data["step"].shape[0]
    time_key = (
        "time / fs"
        if "time / fs" in data
        else ("time__fs" if "time__fs" in data else "time")
    )
    V_key = (
        "V / bohr/(a.u. time)"
        if "V / bohr/(a.u. time)" in data
        else ("V__bohr_per_a_u_time" if "V__bohr_per_a_u_time" in data else "V")
    )
    E_D_key = "E_D / E_h" if "E_D / E_h" in data else ("E_D__E_h" if "E_D__E_h" in data else "E_D")
    E_A_key = "E_A / E_h" if "E_A / E_h" in data else ("E_A__E_h" if "E_A__E_h" in data else "E_A")
    E_0_key = "E_0 / E_h" if "E_0 / E_h" in data else ("E_0__E_h" if "E_0__E_h" in data else "E_0")
    E_1_key = "E_1 / E_h" if "E_1 / E_h" in data else ("E_1__E_h" if "E_1__E_h" in data else "E_1")
    H_DA_key = (
        "H_DA / E_h" if "H_DA / E_h" in data else ("H_DA__E_h" if "H_DA__E_h" in data else "H_DA")
    )
    split_key = (
        "tunneling_splitting / cm-1"
        if "tunneling_splitting / cm-1" in data
        else (
            "tunneling_splitting__cm_1"
            if "tunneling_splitting__cm_1" in data
            else "tunneling_splitting"
        )
    )
    grad_sigma_key = (
        "grad_sigma / E_h/bohr"
        if "grad_sigma / E_h/bohr" in data
        else ("grad_sigma__E_h_per_bohr" if "grad_sigma__E_h_per_bohr" in data else "grad_sigma")
    )
    ke_key = (
        "kinetic_energy / E_h"
        if "kinetic_energy / E_h" in data
        else ("kinetic_energy__E_h" if "kinetic_energy__E_h" in data else "kinetic_energy")
    )
    etot_key = "total_energy / E_h" if "total_energy / E_h" in data else None
    temp_key = (
        "temperature / K"
        if "temperature / K" in data
        else ("temperature__K" if "temperature__K" in data else "temperature")
    )
    donor_key = (
        "proton_center_donor / bohr"
        if "proton_center_donor / bohr" in data
        else (
            "proton_center_donor__bohr"
            if "proton_center_donor__bohr" in data
            else "proton_center_donor"
        )
    )
    acceptor_key = (
        "proton_center_acceptor / bohr"
        if "proton_center_acceptor / bohr" in data
        else (
            "proton_center_acceptor__bohr"
            if "proton_center_acceptor__bohr" in data
            else "proton_center_acceptor"
        )
    )
    frames = []
    for i in range(n):
        frames.append(
            TrajectoryFrame(
                step=int(data["step"][i]),
                time=float(data[time_key][i]),
                R=data["R"][i],
                V=data[V_key][i],
                E_D=float(data[E_D_key][i]),
                E_A=float(data[E_A_key][i]),
                sigma=float(data["sigma"][i]),
                E_0=float(data[E_0_key][i]),
                E_1=float(data[E_1_key][i]),
                H_DA=float(data[H_DA_key][i]),
                S_DA=float(data["S_DA"][i]),
                tunneling_splitting=float(data[E_1_key][i] - data[E_0_key][i]),
                grad_sigma=data[grad_sigma_key][i],
                g=float(data["g"][i]),
                k=float(data["k"][i]),
                kinetic_energy=float(data[ke_key][i]),
                total_energy=(
                    float(data[etot_key][i])
                    if etot_key is not None
                    else float(data[ke_key][i]) + float(data[E_0_key][i])
                ),
                temperature=float(data[temp_key][i]),
                n_shake_iterations=int(data["n_shake_iterations"][i]),
                proton_center_donor=data[donor_key][i],
                proton_center_acceptor=data[acceptor_key][i],
                diabatic_contributions_pct=(
                    data["diabatic_contributions_pct"][i]
                    if "diabatic_contributions_pct" in data
                    else None
                ),
                tunneling_splitting_cm1=(
                    float(data[split_key][i])
                    if split_key in {"tunneling_splitting__cm_1", "tunneling_splitting / cm-1"}
                    else (
                        units.hartree_to_cm1(float(data[split_key][i]))
                        if split_key in data
                        else None
                    )
                ),
                msdft_row_labels=data["msdft_row_labels"] if "msdft_row_labels" in data else None,
                grad_E_0_all_rows=(
                    data["grad_E_0_all_rows / E_h/bohr"][i]
                    if "grad_E_0_all_rows / E_h/bohr" in data
                    else (
                        data["grad_E_0_all_rows__E_h_per_bohr"][i]
                        if "grad_E_0_all_rows__E_h_per_bohr" in data
                        else None
                    )
                ),
                grad_E_1_all_rows=(
                    data["grad_E_1_all_rows / E_h/bohr"][i]
                    if "grad_E_1_all_rows / E_h/bohr" in data
                    else (
                        data["grad_E_1_all_rows__E_h_per_bohr"][i]
                        if "grad_E_1_all_rows__E_h_per_bohr" in data
                        else None
                    )
                ),
                grad_H01_all_rows=(
                    data["grad_H01_all_rows / E_h/bohr"][i]
                    if "grad_H01_all_rows / E_h/bohr" in data
                    else (
                        data["grad_H01_all_rows__E_h_per_bohr"][i]
                        if "grad_H01_all_rows__E_h_per_bohr" in data
                        else None
                    )
                ),
                grad_S01_all_rows=(
                    data["grad_S01_all_rows / 1/bohr"][i]
                    if "grad_S01_all_rows / 1/bohr" in data
                    else (
                        data["grad_S01_all_rows__1_per_bohr"][i]
                        if "grad_S01_all_rows__1_per_bohr" in data
                        else None
                    )
                ),
                grad_sigma_all_rows=(
                    data["grad_sigma_all_rows / E_h/bohr"][i]
                    if "grad_sigma_all_rows / E_h/bohr" in data
                    else (
                        data["grad_sigma_all_rows__E_h_per_bohr"][i]
                        if "grad_sigma_all_rows__E_h_per_bohr" in data
                        else None
                    )
                ),
                nacv_d10_all_rows=(
                    data["nacv_d10_all_rows / 1/bohr"][i]
                    if "nacv_d10_all_rows / 1/bohr" in data
                    else (
                        data["nacv_d10_all_rows__1_per_bohr"][i]
                        if "nacv_d10_all_rows__1_per_bohr" in data
                        else None
                    )
                ),
                mb_resample_recovery_used=(
                    bool(data["mb_resample_recovery_used"][i])
                    if "mb_resample_recovery_used" in data
                    else False
                ),
                mb_resample_attempt=(
                    int(data["mb_resample_attempt"][i]) if "mb_resample_attempt" in data else 0
                ),
            )
        )
    return frames


def write_xyz(path: str, symbols: list[str], positions_bohr: np.ndarray) -> None:
    """Write a multi-frame XYZ from positions with shape (n_frames, n_atoms, 3)."""
    if positions_bohr.ndim != 3:
        raise ValueError("positions_bohr must have shape (n_frames, n_atoms, 3)")
    with open(path, "w", encoding="utf-8") as handle:
        for frame in positions_bohr:
            handle.write(f"{len(symbols)}\n")
            handle.write("Generated by NEO_MSDFT_SEAM_SAMPLE\n")
            positions_ang = units.bohr_to_angstrom(frame)
            for sym, xyz in zip(symbols, positions_ang):
                handle.write(f"{sym} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")


def append_seam_xyz(
    path: str,
    classical_symbols: list[str],
    R_bohr: np.ndarray,
    proton_center_donor_bohr: np.ndarray,
    proton_center_acceptor_bohr: np.ndarray,
    step: int,
    time_fs: float,
    sigma: float,
    tunneling_splitting_cm1: float,
    E_0: float,
) -> None:
    """Append one frame to a running seam XYZ safety-net file."""
    positions_ang = units.bohr_to_angstrom(np.asarray(R_bohr, dtype=float))
    donor_ang = units.bohr_to_angstrom(np.asarray(proton_center_donor_bohr, dtype=float))
    acceptor_ang = units.bohr_to_angstrom(np.asarray(proton_center_acceptor_bohr, dtype=float))
    n_atoms_total = len(classical_symbols) + 2
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{n_atoms_total}\n")
        handle.write(
            f"step={step} time_fs={time_fs:.3f} sigma={sigma:.3e} "
            f"delta_cm^-1={tunneling_splitting_cm1:.6f} E0={E_0:.8f}\n"
        )
        for sym, xyz in zip(classical_symbols, positions_ang):
            handle.write(f"{sym:<2s} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")
        handle.write(f"Hd {donor_ang[0]: .8f} {donor_ang[1]: .8f} {donor_ang[2]: .8f}\n")
        handle.write(f"Ha {acceptor_ang[0]: .8f} {acceptor_ang[1]: .8f} {acceptor_ang[2]: .8f}\n")
