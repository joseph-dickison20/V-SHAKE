"""Q-Chem interface implementation for V-SHAKE dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import subprocess

import numpy as np

from qc_call import DiabatResult, MSDFTResult, QCCallBase
import units

_QCHEM_SUCCESS_MARKER = "Thank you very much for using Q-Chem.  Have a nice day."
_FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?"
LOGGER = logging.getLogger("seam_sample")
_DENSITY_GUESS_FILES = [
    "Pea_1.txt",
    "Pea_2.txt",
    "Peb_1.txt",
    "Peb_2.txt",
    "Pn_1.txt",
    "Pn_2.txt",
]


@dataclass
class MoleculeSection:
    charge_mult_line: str
    symbols: list[str]


@dataclass
class QChemJob:
    pre_molecule: str
    molecule: MoleculeSection
    post_molecule: str


@dataclass
class QChemTemplate:
    jobs: list[QChemJob]


class QChemCaller(QCCallBase):
    def __init__(
        self,
        proton_center_donor: np.ndarray,
        proton_center_acceptor: np.ndarray,
        opt_template_path: str,
        msdft_template_path: str,
        nprocs: int = 1,
        proton_center_indices: tuple[int, int] = (1, 2),
        workdir: str | None = None,
        output_dir: str | None = None,
        save_qchem_outputs: bool = True,
        msdft_subdir: str = "msdft",
        opt_subdir: str = "opt",
    ):
        super().__init__(proton_center_donor, proton_center_acceptor)
        self.opt_template = self._parse_template(Path(opt_template_path))
        self.msdft_template = self._parse_template(Path(msdft_template_path))
        self.nprocs = int(nprocs)
        self.proton_center_indices_0based = tuple(int(i) - 1 for i in proton_center_indices)
        self.workdir = Path(workdir).resolve() if workdir is not None else Path.cwd()
        self.save_qchem_outputs = bool(save_qchem_outputs)
        self.output_root = Path(output_dir).resolve() if output_dir is not None else self.workdir
        self.msdft_dir = self.output_root / msdft_subdir
        self.opt_dir = self.output_root / opt_subdir
        self.current_step_num = 0
        self._step_opt_call_counter = 0
        self._scratch_counter = 0
        self.last_msdft_extras: dict[str, np.ndarray] = {}
        if self.save_qchem_outputs:
            self.msdft_dir.mkdir(parents=True, exist_ok=True)
            self.opt_dir.mkdir(parents=True, exist_ok=True)

        if len(self.opt_template.jobs) != 1:
            raise ValueError("Optimization template must contain exactly one job.")
        if len(self.msdft_template.jobs) != 3:
            raise ValueError("MSDFT template must contain exactly three jobs.")

    def eval_diabats(self, R: np.ndarray) -> DiabatResult:
        self._step_opt_call_counter += 1
        call_num = self._step_opt_call_counter
        R = np.asarray(R, dtype=float)

        donor_saved = self.proton_center_donor.copy()
        acceptor_saved = self.proton_center_acceptor.copy()

        E_D, opt_proton_donor, donor_ok = self._run_optimization(
            R=R,
            proton_center=self.proton_center_donor,
            donor_or_acceptor="donor",
            call_num=call_num,
            diabat_index=1,
        )
        if donor_ok:
            self.proton_center_donor = opt_proton_donor

        E_A, opt_proton_acceptor, acceptor_ok = self._run_optimization(
            R=R,
            proton_center=self.proton_center_acceptor,
            donor_or_acceptor="acceptor",
            call_num=call_num,
            diabat_index=2,
        )
        if acceptor_ok:
            self.proton_center_acceptor = opt_proton_acceptor

        converged = donor_ok and acceptor_ok
        if not converged:
            self.proton_center_donor = donor_saved
            self.proton_center_acceptor = acceptor_saved
            return DiabatResult(
                sigma=0.0,
                grad_sigma=np.zeros_like(R),
                proton_center_donor=donor_saved,
                proton_center_acceptor=acceptor_saved,
                converged=False,
            )

        sigma = E_D - E_A
        LOGGER.info(
            "  [Q-Chem] step=%d diabat_call=%d sigma=%.8e E_D=%.10f E_A=%.10f",
            self.current_step_num,
            call_num,
            sigma,
            E_D,
            E_A,
        )

        return DiabatResult(
            sigma=sigma,
            grad_sigma=np.zeros_like(R),
            proton_center_donor=self.proton_center_donor.copy(),
            proton_center_acceptor=self.proton_center_acceptor.copy(),
            converged=True,
        )

    def clean_density_guess_files(self) -> None:
        for filename in _DENSITY_GUESS_FILES:
            path = self.workdir / filename
            if path.exists():
                path.unlink()
                LOGGER.info("Removed stale density guess file: %s", filename)

    def eval_msdft(self, R: np.ndarray) -> MSDFTResult:
        R = np.asarray(R, dtype=float)
        R_ang = units.bohr_to_angstrom(R)

        donor_ang = units.bohr_to_angstrom(self.proton_center_donor)
        acceptor_ang = units.bohr_to_angstrom(self.proton_center_acceptor)

        jobs_coords = [
            np.vstack([donor_ang, R_ang]),
            np.vstack([acceptor_ang, R_ang]),
            np.vstack([donor_ang, acceptor_ang, R_ang]),
        ]
        input_text = self._render_template(self.msdft_template, jobs_coords)

        input_path, output_path = self._prepare_paths(
            job_type="msdft",
            basename=f"msdft_{self.current_step_num}",
        )
        input_path.write_text(input_text)

        output_text = self._run_qchem(input_path, output_path, expected_completions=3)
        msdft_idx = output_text.find("NEO-MSDFT RESULTS")
        if msdft_idx < 0:
            raise RuntimeError(f"MSDFT marker not found in {output_path}.")
        msdft_text = output_text[msdft_idx:]

        S = self._parse_2x2_matrix(msdft_text, "Overlap Matrix:")
        H = self._parse_2x2_matrix(msdft_text, "Hamiltonian Matrix:")
        E_0 = self._parse_scalar(msdft_text, r"E_0\s*=\s*(" + _FLOAT_RE + r")", "E_0")
        E_1 = self._parse_scalar(msdft_text, r"E_1\s*=\s*(" + _FLOAT_RE + r")", "E_1")
        tunneling_splitting_cm1 = self._parse_scalar(
            msdft_text,
            r"E_1\s*-\s*E_0\s*=\s*(" + _FLOAT_RE + r")",
            "E_1 - E_0 (cm-1)",
        )
        diabatic_contributions_pct = self._parse_diabatic_contributions(msdft_text)

        n_total = len(self.msdft_template.jobs[2].molecule.symbols)
        grad_E0 = self._parse_gradient_block(msdft_text, "Gradient of E0:", n_total)
        grad_E1 = self._parse_gradient_block(msdft_text, "Gradient of E1:", n_total)
        grad_sigma = self._parse_gradient_block(
            msdft_text, "Gradient of sigma01 (gradH00 - gradH11):", n_total
        )
        grad_H01 = self._parse_gradient_block(msdft_text, "Gradient of H01:", n_total)
        grad_S01 = self._parse_gradient_block(msdft_text, "Gradient of S01:", n_total)
        nacv_d10 = self._parse_gradient_block(msdft_text, "Derivative Coupling d10:", n_total)

        keep_rows = [i for i in range(n_total) if i not in self.proton_center_indices_0based]
        grad_E0_heavy = grad_E0[keep_rows]
        grad_E1_heavy = grad_E1[keep_rows]
        grad_sigma_heavy = grad_sigma[keep_rows]
        grad_H01_heavy = grad_H01[keep_rows]
        grad_S01_heavy = grad_S01[keep_rows]
        nacv_d10_heavy = nacv_d10[keep_rows]

        n_heavy = R.shape[0]
        if grad_E0_heavy.shape != (n_heavy, 3):
            raise RuntimeError(
                f"Parsed heavy-atom gradient shape {grad_E0_heavy.shape} does not match R shape {(n_heavy, 3)}."
            )

        row_labels = np.asarray(
            ["Hd", "Ha", *self.msdft_template.jobs[2].molecule.symbols[2:]], dtype="U8"
        )

        self.last_msdft_extras = {
            "grad_E1": grad_E1_heavy,
            "grad_H01": grad_H01_heavy,
            "grad_S01": grad_S01_heavy,
            "nacv_d10": nacv_d10_heavy,
            "grad_E0_all_rows": grad_E0,
            "grad_E1_all_rows": grad_E1,
            "grad_H01_all_rows": grad_H01,
            "grad_S01_all_rows": grad_S01,
            "grad_sigma_all_rows": grad_sigma,
            "nacv_d10_all_rows": nacv_d10,
            "msdft_row_labels": row_labels,
        }

        result = MSDFTResult(
            E_D=H[0, 0],
            E_A=H[1, 1],
            grad_E_D=np.zeros((n_heavy, 3)),
            grad_E_A=np.zeros((n_heavy, 3)),
            E_0=E_0,
            E_1=E_1,
            grad_E_0=grad_E0_heavy,
            H_DA=H[0, 1],
            S_DA=S[0, 1],
            proton_center_donor=self.proton_center_donor.copy(),
            proton_center_acceptor=self.proton_center_acceptor.copy(),
            diabatic_contributions_pct=diabatic_contributions_pct,
            tunneling_splitting_cm1=tunneling_splitting_cm1,
            msdft_row_labels=row_labels,
            grad_E_0_all_rows=grad_E0,
            grad_E_1_all_rows=grad_E1,
            grad_H01_all_rows=grad_H01,
            grad_S01_all_rows=grad_S01,
            grad_sigma_all_rows=grad_sigma,
            nacv_d10_all_rows=nacv_d10,
            _grad_sigma_direct=grad_sigma_heavy,
        )
        LOGGER.info(
            "  [Q-Chem] step=%d MSDFT E0=%.10f E1=%.10f Delta=%.6f H_DA=%.10f S_DA=%.10f sigma=%.8e",
            self.current_step_num,
            result.E_0,
            result.E_1,
            tunneling_splitting_cm1,
            result.H_DA,
            result.S_DA,
            result.sigma,
        )
        return result

    def _run_optimization(
        self,
        R: np.ndarray,
        proton_center: np.ndarray,
        donor_or_acceptor: str,
        call_num: int,
        diabat_index: int,
    ) -> tuple[float, np.ndarray, bool]:
        R = np.asarray(R, dtype=float)
        proton_center = np.asarray(proton_center, dtype=float)
        proton_center_unchanged = np.asarray(proton_center, dtype=float).copy()
        R_ang = units.bohr_to_angstrom(R)
        proton_ang = units.bohr_to_angstrom(proton_center)
        coords = np.vstack([proton_ang, R_ang])

        input_text = self._render_template(self.opt_template, [coords])
        if diabat_index not in (1, 2):
            raise ValueError(f"diabat_index must be 1 or 2, got {diabat_index}.")
        input_text = input_text.replace("${IDX}", str(diabat_index))
        input_path, output_path = self._prepare_paths(
            job_type="opt",
            basename=f"opt_{donor_or_acceptor}_step_{self.current_step_num}_call_{call_num}",
        )
        input_path.write_text(input_text)

        output_text = self._run_qchem(input_path, output_path, expected_completions=1)
        opt_converged = (
            "OPTIMIZATION CONVERGED" in output_text
            and "MAXIMUM OPTIMIZATION CYCLES REACHED" not in output_text
        )
        if not opt_converged:
            LOGGER.warning(
                "  [Q-Chem] step=%d diabat_call=%d %s OPT did not converge (MAXIMUM OPTIMIZATION CYCLES REACHED). "
                "See %s.",
                self.current_step_num,
                call_num,
                donor_or_acceptor,
                output_path,
            )
            return 0.0, proton_center_unchanged, False

        E = self._parse_scalar(
            output_text, r"Final energy is\s+(" + _FLOAT_RE + r")", "Final energy"
        )
        converged_coords = self._parse_optimization_coordinates(output_text, len(coords))

        R_out_proton = converged_coords[0]
        R_out_heavy = converged_coords[1:]
        shift = R_out_heavy[0] - R_ang[0]
        max_shift_error = np.max(np.abs((R_out_heavy - R_ang) - shift))
        if max_shift_error > 1e-6:
            raise RuntimeError(
                f"Rigid-translation check failed (max deviation {max_shift_error:.3e}) in {output_path}."
            )

        proton_center_optimized = R_out_proton - shift
        proton_center_optimized_bohr = units.angstrom_to_bohr(proton_center_optimized)
        LOGGER.info(
            "    [Q-Chem] step=%d call=%d %s OPT energy=%.10f",
            self.current_step_num,
            call_num,
            donor_or_acceptor,
            E,
        )
        return E, proton_center_optimized_bohr, True

    def set_step_context(self, step_num: int) -> None:
        self.current_step_num = int(step_num)
        self._step_opt_call_counter = 0

    def _prepare_paths(self, job_type: str, basename: str) -> tuple[Path, Path]:
        if self.save_qchem_outputs:
            target_dir = self.msdft_dir if job_type == "msdft" else self.opt_dir
            return target_dir / f"{basename}.in", target_dir / f"{basename}.out"
        self._scratch_counter += 1
        scratch_base = self.workdir / f".qchem_tmp_{self._scratch_counter:08d}_{basename}"
        return scratch_base.with_suffix(".in"), scratch_base.with_suffix(".out")

    def _run_qchem(
        self,
        input_path: Path,
        output_path: Path,
        expected_completions: int = 1,
    ) -> str:
        cmd = ["qchem", "-nt", str(self.nprocs), str(input_path), str(output_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Q-Chem failed with code {proc.returncode}. See {output_path}. stderr:\n{proc.stderr}"
            )
        input_path.unlink(missing_ok=True)

        output_text = output_path.read_text()
        n_completions = output_text.count(_QCHEM_SUCCESS_MARKER)
        if n_completions < expected_completions:
            raise RuntimeError(
                f"Q-Chem completed only {n_completions}/{expected_completions} jobs. See {output_path}."
            )
        if not self.save_qchem_outputs:
            output_path.unlink(missing_ok=True)
        return output_text

    @staticmethod
    def _parse_template(path: Path) -> QChemTemplate:
        text = path.read_text()
        raw_jobs = re.split(r"^\s*@@@\s*$", text, flags=re.MULTILINE)
        jobs: list[QChemJob] = []

        for raw_job in raw_jobs:
            job_text = raw_job.strip()
            if not job_text:
                continue
            lines = job_text.splitlines()
            molecule_start = next(
                (i for i, ln in enumerate(lines) if ln.strip().lower() == "$molecule"),
                None,
            )
            if molecule_start is None:
                raise ValueError(f"$molecule section not found in template {path}.")
            molecule_end = next(
                (
                    i
                    for i in range(molecule_start + 1, len(lines))
                    if lines[i].strip().lower() == "$end"
                ),
                None,
            )
            if molecule_end is None:
                raise ValueError(f"$end for $molecule not found in template {path}.")

            molecule_lines = lines[molecule_start + 1 : molecule_end]
            if not molecule_lines:
                raise ValueError(f"Empty $molecule section in template {path}.")
            charge_mult_line = molecule_lines[0].strip()

            symbols: list[str] = []
            for line in molecule_lines[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid coordinate line in {path}: {line}")
                symbols.append(parts[0])

            pre_molecule = "\n".join(lines[:molecule_start]).rstrip()
            post_molecule = "\n".join(lines[molecule_end + 1 :]).rstrip()
            jobs.append(
                QChemJob(
                    pre_molecule=pre_molecule,
                    molecule=MoleculeSection(charge_mult_line=charge_mult_line, symbols=symbols),
                    post_molecule=post_molecule,
                )
            )

        if not jobs:
            raise ValueError(f"No jobs parsed from template {path}.")
        return QChemTemplate(jobs=jobs)

    @staticmethod
    def _render_template(template: QChemTemplate, jobs_coords: list[np.ndarray]) -> str:
        if len(template.jobs) != len(jobs_coords):
            raise ValueError("Coordinate blocks do not match number of template jobs.")

        rendered: list[str] = []
        for job, coords in zip(template.jobs, jobs_coords):
            coords = np.asarray(coords, dtype=float)
            if coords.shape != (len(job.molecule.symbols), 3):
                raise ValueError(
                    f"Coordinate shape {coords.shape} does not match template symbols "
                    f"({len(job.molecule.symbols)}, 3)."
                )

            parts: list[str] = []
            if job.pre_molecule.strip():
                parts.append(job.pre_molecule)
            parts.append("$molecule")
            parts.append(job.molecule.charge_mult_line)
            for symbol, xyz in zip(job.molecule.symbols, coords):
                parts.append(f"{symbol:<3s}  {xyz[0]: .10f}  {xyz[1]: .10f}  {xyz[2]: .10f}")
            parts.append("$end")
            if job.post_molecule.strip():
                parts.append(job.post_molecule)
            rendered.append("\n".join(parts) + "\n")

        return "\n@@@\n\n".join(rendered)

    @staticmethod
    def _parse_scalar(text: str, pattern: str, label: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f"Could not parse {label}.")
        return float(match.group(1))

    @staticmethod
    def _parse_2x2_matrix(text: str, label: str) -> np.ndarray:
        idx = text.find(label)
        if idx < 0:
            raise RuntimeError(f"Missing section: {label}")
        block = text[idx + len(label) :].lstrip("\n").splitlines()
        if len(block) < 2:
            raise RuntimeError(f"Incomplete matrix section: {label}")

        rows = []
        for line in block[:2]:
            vals = re.findall(_FLOAT_RE, line)
            if len(vals) < 2:
                raise RuntimeError(f"Could not parse matrix row in section {label}: {line}")
            rows.append([float(vals[0]), float(vals[1])])
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _parse_gradient_block(text: str, label: str, n_rows: int) -> np.ndarray:
        idx = text.find(label)
        if idx < 0:
            raise RuntimeError(f"Missing section: {label}")
        lines = text[idx + len(label) :].splitlines()

        rows: list[list[float]] = []
        for line in lines:
            vals = re.findall(_FLOAT_RE, line)
            if len(vals) >= 3:
                rows.append([float(vals[0]), float(vals[1]), float(vals[2])])
                if len(rows) == n_rows:
                    break
            elif rows:
                break
            else:
                continue
        if len(rows) != n_rows:
            raise RuntimeError(f"Expected {n_rows} rows in {label}, found {len(rows)}.")
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _parse_diabatic_contributions(text: str) -> np.ndarray:
        label = "NEO-MSDFT Diabatic Contributions (%):"
        idx = text.find(label)
        if idx < 0:
            raise RuntimeError(f"Missing section: {label}")

        lines = text[idx + len(label) :].splitlines()
        rows: list[list[float]] = []
        pattern = re.compile(
            r"State\s+\d+:\s+Diabat\s+0\s*=\s*(" + _FLOAT_RE + r")%\s*,\s*Diabat\s+1\s*=\s*("
            + _FLOAT_RE
            + r")%"
        )
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if rows:
                    break
                continue
            match = pattern.search(stripped)
            if match is None:
                if rows:
                    break
                continue
            rows.append([float(match.group(1)), float(match.group(2))])
            if len(rows) == 2:
                break

        if len(rows) != 2:
            raise RuntimeError(f"Expected 2 states in {label}, found {len(rows)}.")
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _parse_optimization_coordinates(text: str, n_atoms: int) -> np.ndarray:
        marker = "OPTIMIZATION CONVERGED"
        idx = text.rfind(marker)
        if idx < 0:
            raise RuntimeError("OPTIMIZATION CONVERGED marker not found.")

        tail = text[idx:]
        coord_idx = tail.find("Coordinates (Angstroms)")
        if coord_idx < 0:
            raise RuntimeError("Coordinates (Angstroms) block not found after convergence.")

        lines = tail[coord_idx:].splitlines()
        data_lines = lines[2:]  # drop "Coordinates" + "ATOM X Y Z" header

        coords: list[list[float]] = []
        pattern = re.compile(
            r"^\s*\d+\s+\S+\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s+(" + _FLOAT_RE + r")\s*$"
        )
        for line in data_lines:
            match = pattern.match(line)
            if not match:
                if coords:
                    break
                continue
            coords.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            if len(coords) == n_atoms:
                break

        if len(coords) != n_atoms:
            raise RuntimeError(
                f"Expected {n_atoms} optimized atoms in coordinates block, found {len(coords)}."
            )
        return np.asarray(coords, dtype=float)
