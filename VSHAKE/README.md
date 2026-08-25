# NEO-MSDFT Seam Sampling (V-SHAKE)

Constrained molecular dynamics on the vibronic diabatic energy seam, implemented
as a Python wrapper around Q-Chem. V-SHAKE is an adaptation of the E-SHAKE
algorithm (Cofer-Shabica, DeRosa, and Subotnik, *J. Chem. Theory Comput.* **2026**,
22, 1575–1583) for nuclear-electronic orbital multistate density functional
theory (NEO-MSDFT). It propagates the classical heavy-atom framework on the
ground vibronic adiabatic potential energy surface while enforcing the
constraint

  σ(R) = E_D(R) − E_A(R) = 0,

confining the trajectory to the (3N − 1)-dimensional vibronic diabatic seam.
The quantum proton is **not** a classical degree of freedom — its protonic
basis-center position is optimized by Q-Chem at every geometry evaluation, with
the converged center from the previous evaluation used as a warm start.

---

## What this code does

At each time step the integrator performs a constrained velocity-Verlet step
following Cofer-Shabica et al. eqs. 13–23, with two electronic-structure calls:

1. **SHAKE root search** — repeated cheap diabat evaluations (two NEO-DFT
   geometry optimizations of the protonic basis center per evaluation) until
   the trial geometry satisfies |σ| < threshold. Typically 2–5 root-search
   iterations, i.e. 4–10 OPT jobs per step. If SHAKE fails (single-well collapse
   or diabat OPT non-convergence), **Maxwell–Boltzmann velocity resample**
   recovery is attempted (see dedicated section below) instead of stopping
   immediately.
2. **Full MSDFT call** — a single three-job NEO-MSDFT calculation at the
   converged on-seam geometry. This produces the adiabatic energies E₀ and E₁,
   the ground-state gradient (used as the propagation force), the diabatic
   coupling H_DA and overlap S_DA, the constraint gradient ∇σ (used by
   RATTLE), the nonadiabatic coupling vector d₁₀, and the tunneling splitting
   Δ.
3. **RATTLE velocity correction** (closed-form, no iteration) ensures the
   velocity is tangent to the seam.
4. **Stochastic velocity rescaling** (Bussi, Donadio, and Parrinello,
   *J. Chem. Phys.* **2007**, 126, 014101) — the exact propagator of eq. (A7),
   not the Euler-Maruyama discretization, so it is correct for any τ/Δt.

The accumulated trajectory is written to a NumPy `.npz` archive on every
checkpoint, and a running XYZ "safety net" with the heavy atoms plus both
optimized proton centers is appended to on every step.

---

## Directory layout

The launch script hard-codes the absolute path to the
`neo_msdft_seam_sample/` source directory (see "Installing" below). Once that
is set, **only the per-run files need to live in your working directory.**
A typical run directory looks like:

```
my_run/
├── templates/
│   ├── opt.in            # NEO-DFT optimization template
│   └── msdft.in          # three-job NEO-MSDFT template
├── init_pos.xyz          # initial heavy-atom geometry
├── init_vel.xyz          # (optional) initial velocities
├── config.yaml           # run configuration
├── run_vshake.py         # launch script (one-liner that calls run_from_config)
└── vshake.sh             # SLURM submission script
```

The source code itself lives elsewhere on disk and is imported via
`sys.path.insert(...)` at the top of `run_vshake.py`. After the run finishes
you will additionally have:

```
my_run/
├── seam_geometries.xyz   # one frame per step, written incrementally
├── trajectory.npz        # full per-frame archive, overwritten on each save
└── output/               # per-call Q-Chem inputs and outputs (if save_qchem_outputs=true)
    ├── output.log        # main log
    ├── opt/              # opt_donor_step_N_call_M.in/.out
    └── msdft/            # msdft_N.in/.out
```

For reference, the source tree is:

```
neo_msdft_seam_sample/
├── initialize.py         # geometry/velocity init, COM/rotation cleanup, seam projection
├── integrator.py         # V-SHAKE constrained velocity-Verlet step
├── qc_call.py            # abstract oracle interface, DiabatResult, MSDFTResult
├── qchem_interface.py    # Q-Chem subprocess driver, template parser
├── run.py                # main driver and trajectory loop
├── thermostat.py         # exact Bussi stochastic velocity rescaling
├── trajectory.py         # TrajectoryFrame dataclass, NPZ I/O, XYZ writer
└── units.py              # unit conversions (Bohr, Hartree, fs, AMU, K)
```

---

## Installing

1. Clone the repository to a permanent location on your cluster.
2. Edit the `sys.path.insert` line near the top of `run_vshake.py` to point
   to your `neo_msdft_seam_sample/` directory:

   ```python
   sys.path.insert(0, "/path/to/VSHAKE/neo_msdft_seam_sample")
   ```

3. Create a Python ≥ 3.10 environment with NumPy and PyYAML:

   ```bash
   conda create -n seam_samp_env python=3.10 numpy pyyaml -y
   conda activate seam_samp_env
   ```

4. Make sure `qchem` is on your `PATH` and that your Q-Chem build supports
   NEO-MSDFT with `NEO_BL_MSDFT` and the diagnostic output blocks parsed by
   the driver (see "Required Q-Chem output sections" below).

---

## Quick start

A full run requires four input files in the working directory.

### 1. `init_pos.xyz` — initial heavy-atom geometry (Angstrom)

This is the **classical framework only**, with no quantum proton. The
geometry should be on or near the diabatic seam (i.e., near the MECP).
Atom ordering must match the heavy-atom block of the OPT/MSDFT templates
exactly (the templates list the quantum proton(s) first, then the heavy
atoms in this order).

```
8
Initial heavy-atom framework (Angstrom)
O   0.0000000000 -1.2978408522  1.9929275157
O   0.0000000000  1.2978408522  1.9929275157
C   0.0000000000 -1.2129988522  0.7065245157
C   0.0000000000  1.2129988522  0.7065245157
C   0.0000000000  0.0000000000  0.0000000000
H   0.0000000000  0.0000000000 -1.0775584843
H   0.0000000000 -2.1601138522  0.1662325157
H   0.0000000000  2.1601138522  0.1662325157
```

### 2. `templates/opt.in` — NEO-DFT optimization template

A single Q-Chem job that optimizes the protonic basis-center position with
all heavy atoms frozen. The driver substitutes the coordinate values in the
`$molecule` block on every call and replaces the literal token `${IDX}` in
`$neo_msdft` with `1` (donor) or `2` (acceptor). Everything else is preserved
verbatim.

Requirements:
- `$molecule`: quantum proton H first, then heavy atoms in the same order as
  `init_pos.xyz`.
- `$rem`: must include `JOBTYPE OPT`, `NEO TRUE`, `SYM_IGNORE TRUE`, and
  reasonable `SCF_*` / `NEO_*_CONV` thresholds.
- `$neo_msdft`: must contain `SAVE_DENS ${IDX}`.
- `$OPT FIXED`: must freeze every heavy atom by its 1-based Q-Chem index.
- `$neo_basis`: defines the protonic basis on the quantum proton.

### 3. `templates/msdft.in` — three-job MSDFT template

Three Q-Chem jobs separated by `@@@`. The driver substitutes coordinates in
each `$molecule` block but otherwise touches nothing.

- **Job 1 — donor diabat.** `$molecule`: H + heavy atoms. `$rem`:
  `NEO_BL_MSDFT_DIABAT_IDX 1`.
- **Job 2 — acceptor diabat.** `$molecule`: H + heavy atoms. `$rem`:
  `NEO_BL_MSDFT_DIABAT_IDX 2`.
- **Job 3 — MSDFT coupling.** `$molecule`: H + `@H` + heavy atoms (the
  quantum H and ghost @H must be the first two atoms, in that order).
  `$rem`: `NEO_BL_MSDFT TRUE`, `SKIP_SCFMAN TRUE`. `$neo_basis` defines the
  protonic basis on **both** quantum centers (`H 1` and `H 2`).

All three `$rem` blocks must include `IDERIV 1` (this triggers CPSCF and the
gradient/derivative-coupling output that the driver parses).

### 4. `config.yaml` — run configuration

```yaml
# --- Initial geometry (heavy atoms only, no quantum proton) ---
initial_geometry: "init_pos.xyz"

# Optional. If null, masses are inferred from XYZ symbols using common
# isotope masses. If provided, must be a list with one entry per atom in
# the same order as the XYZ.
masses_amu: null

# --- Quantum proton basis-center initial guesses (Angstrom) ---
proton_center_donor:    [0.0, -0.3367648071, 2.3304949480]
proton_center_acceptor: [0.0,  0.3367648071, 2.3304949480]

# --- Q-Chem interface ---
opt_template: "templates/opt.in"
msdft_template: "templates/msdft.in"
qchem_nprocs: 24
proton_center_indices: [1, 2]   # 1-based indices of H and @H in MSDFT job 3

# --- Dynamics parameters ---
target_temperature: 20.0           # K
initial_velocity_mode: "mb"        # "mb", "file", or "nacv" (see Quick start)
initial_velocity_path: null        # optional XYZ-like file in Bohr/(a.u. time)
dt_fs: 0.1                         # time step (fs)
constraint_threshold: 1.0e-7       # |sigma| threshold (Hartree)
max_shake_iter: 20

# Minimum donor/acceptor proton basis-center distance (Angstrom) after SHAKE;
# null disables single-well detection. See "Single-well and OPT failure recovery".
min_proton_center_separation_angstrom: 0.75

# Maximum MB resample attempts at the current geometry after SHAKE failure
# (see "Single-well and OPT failure recovery")
max_mb_resamples: 20

thermostat: "berendsen"            # "berendsen" or "none"
thermostat_tau_fs: 2.0

# --- Simulation length ---
n_steps: 100
save_interval: 1                   # save trajectory frame every N steps

# --- Output ---
trajectory_path: "trajectory.npz"
seam_xyz_path: "seam_geometries.xyz"
save_qchem_outputs: true
output_dir: "output"
log_filename: "output.log"
msdft_subdir: "msdft"
opt_subdir: "opt"
```

### 5. Initial velocities

Set **`initial_velocity_mode`** in `config.yaml`:

- **`"mb"`** (default): Maxwell-Boltzmann sampling at `target_temperature`, then
  remove center-of-mass translation, remove net rigid-body rotation about the COM
  (`remove_angular_momentum`), then project onto the seam tangent plane so
  **V**·∇σ = 0.
- **`"file"`**: read from `initial_velocity_path` (XYZ-like, Bohr / a.u. time),
  then the same COM removal, angular-momentum removal, and seam projection as
  **`"mb"`**. Atom ordering and symbols must match the geometry file exactly.
- **`"nacv"`**: Maxwell-Boltzmann draw (for kinetic energy only) with COM and
  angular-momentum removal, then replace **V** with **α·d_∥** (seam-tangent NACV
  from the step-0 MSDFT, same KE). If d_∥ is unavailable, fall back to projected
  MB velocities. Useful when the initial geometry is near the edge of the
  double-well region. If `initial_velocity_path` is also set, it is ignored (a
  warning is logged).

Angular-momentum removal uses the initial geometry **R** and atomic masses: it
computes **L** and **I** about the COM, solves **ω** = **I**⁻¹**L** (least squares
if **I** is singular or ill-conditioned), and subtracts **ω** × **r**ᵢ from each
velocity. A log line reports |**L**| before and after.

If `initial_velocity_mode` is omitted, the driver uses **`"file"`** when
`initial_velocity_path` is set and **`"mb"`** otherwise (backward compatible).

Optional **`init_vel.xyz`** example for `"file"` mode:

```
8
Initial velocities in Bohr per atomic unit of time
O   0.00001000   0.00000200  -0.00000300
O  -0.00001000  -0.00000200   0.00000300
...
```

### 6. Run

Interactive:

```bash
python run_vshake.py config.yaml
```

SLURM:

```bash
sbatch vshake.sh
```

The provided `vshake.sh` loads the cluster modules, sources the Q-Chem setup
script, activates the conda environment, and runs `run_vshake.py config.yaml`.
Adjust the module loads and paths for your cluster.

---

## What happens at each step (algorithm summary)

For each step n with current state (R, V) and current MSDFT result containing
F(t) = −∇E₀(t) and G(t) = −∇σ(t):

1. **Unconstrained trial position** (eq. 13):
   R̄ = R + h·V + (h²/2)·M⁻¹·F(t)

2. **SHAKE root search** for the Lagrange multiplier g = h·λ_RR such that
   σ(R̄ + g·(h/2)·M⁻¹·G(t)) = 0:
   - Single Newton step from σ(R̄) and the analytical derivative
     dσ/dg = −(h/2)·G(t)ᵀ·M⁻¹·G(t).
   - If Newton's iterate doesn't bracket the root, expand outward with
     doubling step size.
   - Bisect until |σ| < `constraint_threshold`.
   - Each evaluation calls `eval_diabats`, which runs two independent OPT
     jobs (one per diabat) that re-optimize the protonic basis center.

3. **On-seam geometry** (eq. 15):
   R_new = R̄ + g·(h/2)·M⁻¹·G(t)

   If SHAKE fails because of single-well collapse (proton centers too close)
   or diabat OPT non-convergence, the driver **resamples seam-projected
   Maxwell–Boltzmann velocities** at the current geometry (see
   "Single-well and OPT failure recovery").

4. **Full MSDFT call** at R_new (`eval_msdft`) returns the new E₀, E₁, ∇E₀,
   ∇σ, H_DA, S_DA, d₁₀, and Δ. Define F(t+h) = −∇E₀(t+h) and
   G(t+h) = −∇σ(t+h).

5. **Velocity update** (eq. 16):
   V_new = V + (1/2)·M⁻¹·[h·F(t) + g·G(t) + h·F(t+h) + k·G(t+h)]

6. **RATTLE multiplier** (eq. 23, closed form, no iteration):
   k = −2·(Ṽᵀ·G(t+h)) / (G(t+h)ᵀ·M⁻¹·G(t+h)),
   where Ṽ is the velocity built from steps 1–5 omitting the k term.

7. **Thermostat** (Bussi eq. A7, exact propagator) rescales V_new by
   √α² with α² combining a deterministic e^(−Δt/τ) decay term, a
   gamma-distributed stochastic kick, and the cross term with a single
   Gaussian draw. Skipped if `thermostat: "none"`.

8. **Output**: append the on-seam frame (heavy atoms + Hd + Ha) to the XYZ
   safety file; if step % save_interval == 0, append a `TrajectoryFrame` to
   the in-memory list and rewrite `trajectory.npz`.

The number of classical degrees of freedom used by the thermostat and by the
instantaneous-temperature calculation is hardcoded as **3·N_heavy − 7**.
Here, `N_heavy` is the number of non-quantized (classical) heavy nuclei in the
NEO simulation. The rationale is that a normal PES is 3N − 6 dimensional, and
the additional seam constraint reduces the dimensionality by one more, giving
3N − 7.

---

## Single-well and OPT failure recovery (MB velocity resample)

### Motivation

During SHAKE, σ(R) = E_D − E_A is built from **two independent** NEO-DFT
optimizations of the protonic basis center (donor diabat vs. acceptor diabat).
For some classical geometries R̄, both optimizations can fall into the **same
electronic basin** (single-well collapse): σ → 0 because E_D ≈ E_A while the
diabats are not distinct vibronic states. That is a **false** seam point.

Separately, a diabat OPT job may hit **`MAXIMUM OPTIMIZATION CYCLES REACHED`**
(or exit without **`OPTIMIZATION CONVERGED`**). Then σ cannot be evaluated at R̄
and SHAKE cannot proceed. The driver treats this like single-well collapse and
attempts the same recovery path instead of stopping immediately.

### Detection

- **Single-well:** after a successful SHAKE root search, if
  `min_proton_center_separation_angstrom` is set and the optimized donor–acceptor
  proton center distance is below that threshold.
- **OPT failure:** `eval_diabats` sets `converged=False` when either OPT does not
  converge (nonzero return code still raises immediately as a crash).

### Recovery procedure

When the primary SHAKE attempt fails:

1. Record the instantaneous temperature **T_keep** from the velocities **V**(t) at
   the start of the step (before recovery).
2. Revert proton center warm starts to the current step's MSDFT values. For each
   attempt up to `max_mb_resamples`:
   - Delete stale Q-Chem density guess files (`Pea_*.txt`, `Peb_*.txt`, `Pn_*.txt`,
     if present).
   - Draw new velocities from the same pipeline as **`initial_velocity_mode: "mb"`**
     (Maxwell–Boltzmann at `target_temperature`, remove COM translation and net
     angular momentum, project onto the seam using **∇σ** at R(t)), then **rescale**
     the draw so the instantaneous temperature equals **T_keep**.
   - Recompute **R̄** = R(t) + h·**V**_new + (h²/2)·M⁻¹·**F**(t) with the same
     **F**(t) and **G**(t) from the current MSDFT; retry SHAKE.
3. **Termination.** If every MB resample attempt fails, the trajectory terminates
   with a clear error.

The integrator **never** changes R(t) during recovery: only trial velocities and
the resulting **R̄** change. If SHAKE succeeds, the new on-seam point is the usual
step forward (same step index n+1). When recovery succeeds, **V** used in the
velocity update is the resampled velocity for consistency with the Verlet
predictor that produced R_new.

### Why it works

A random seam-tangent velocity often points toward a basin where donor and
acceptor diabat optimizations remain distinct and the proton potential is
double-welled. Preserving **T_keep** keeps the kinetic energy content of the step
consistent with the thermostat history. With small h and R already on the seam, a
new **R̄** is more likely to land where SHAKE finds a physical root.

### YAML configuration

| Key | Default (if omitted) | Meaning |
|-----|----------------------|---------|
| `min_proton_center_separation_angstrom` | **0.75** | Minimum ‖r_D − r_A‖ (Å) after SHAKE for single-well detection. Set to **`null`** to disable this check (OPT failure recovery still applies). |
| `max_mb_resamples` | **20** | Maximum number of MB velocity resample + SHAKE retries at the **current** geometry before terminating. |

Per dynamics step, the main log line includes **`mb_resample_recovery=`** (0 or 1)
and **`mb_resample_attempt=`** (0 if the primary SHAKE succeeded; 1…N if recovery
succeeded on attempt N).
The NPZ archive stores **`mb_resample_recovery_used`** and **`mb_resample_attempt`**
per saved frame.

---

## Output files

### `seam_geometries.xyz`

A multi-frame XYZ written one frame per step, regardless of `save_interval`.
Each frame contains the heavy atoms followed by the optimized donor proton
center (`Hd`) and acceptor proton center (`Ha`), in Angstrom. The comment
line carries metadata:

```
10
step=1 time_fs=0.100 sigma=1.23e-08 delta_cm^-1=0.00234 E0=-265.66909570
O    0.00000000  -1.29784085   1.99292752
...
Hd   0.00000000  -0.33676481   2.33049495
Ha   0.00000000   0.33676481   2.33049495
```

This is the safety net: every successfully completed step lands in this file
immediately, so if the trajectory dies you still have all the seam
geometries.

### `trajectory.npz`

A NumPy compressed archive rewritten at every checkpoint. Per-frame arrays
are indexed by frame on axis 0. Keys (with units) include:

| Key | Shape | Description |
|---|---|---|
| `step` | (n_frames,) | step index |
| `time / fs` | (n_frames,) | simulation time, fs |
| `R` | (n_frames, N_heavy, 3) | heavy-atom positions, Bohr |
| `V / bohr/(a.u. time)` | (n_frames, N_heavy, 3) | heavy-atom velocities |
| `E_D / E_h`, `E_A / E_h` | (n_frames,) | diabatic energies, Hartree |
| `sigma` | (n_frames,) | E_D − E_A, Hartree |
| `E_0 / E_h`, `E_1 / E_h` | (n_frames,) | adiabatic energies |
| `H_DA / E_h`, `S_DA` | (n_frames,) | diabatic coupling and overlap |
| `tunneling_splitting / cm-1` | (n_frames,) | E_1 − E_0 in cm⁻¹ |
| `grad_sigma / E_h/bohr` | (n_frames, N_heavy, 3) | constraint gradient |
| `grad_E_0_all_rows / E_h/bohr` | (n_frames, N_heavy + 2, 3) | full ∇E₀ including Hd, Ha rows |
| `grad_E_1_all_rows / E_h/bohr` | (n_frames, N_heavy + 2, 3) | full ∇E₁ |
| `grad_H01_all_rows / E_h/bohr` | (n_frames, N_heavy + 2, 3) | full ∇H_DA |
| `grad_S01_all_rows / 1/bohr` | (n_frames, N_heavy + 2, 3) | full ∇S_DA |
| `grad_sigma_all_rows / E_h/bohr` | (n_frames, N_heavy + 2, 3) | full ∇σ |
| `nacv_d10_all_rows / 1/bohr` | (n_frames, N_heavy + 2, 3) | nonadiabatic coupling d₁₀ |
| `msdft_row_labels` | (N_heavy + 2,) | row labels: `["Hd", "Ha", ...]` |
| `g`, `k` | (n_frames,) | SHAKE and RATTLE multipliers |
| `kinetic_energy / E_h` | (n_frames,) | classical KE |
| `total_energy / E_h` | (n_frames,) | KE + E_0 (not strictly conserved) |
| `temperature / K` | (n_frames,) | instantaneous temperature |
| `n_shake_iterations` | (n_frames,) | bisection iterations used |
| `proton_center_donor / bohr` | (n_frames, 3) | optimized donor center |
| `proton_center_acceptor / bohr` | (n_frames, 3) | optimized acceptor center |
| `diabatic_contributions_pct` | (n_frames, 2, 2) | % composition of each adiabatic state |
| `mb_resample_recovery_used` | (n_frames,) | bool: MB resample recovery was used on that step |
| `mb_resample_attempt` | (n_frames,) | int: MB resample attempt that succeeded (0 if primary SHAKE only) |

Load with:

```python
from neo_msdft_seam_sample.trajectory import load_trajectory
frames = load_trajectory("trajectory.npz")
```

or directly with NumPy:

```python
import numpy as np
data = np.load("trajectory.npz")
print(data.files)
sigma = data["sigma"]
delta = data["tunneling_splitting / cm-1"]
```

### `output/` directory

When `save_qchem_outputs: true`, every Q-Chem call's input and output files
are kept under `output_dir`:

- `output/opt/opt_donor_step_<N>_call_<M>.in` / `.out`
- `output/opt/opt_acceptor_step_<N>_call_<M>.in` / `.out`
- `output/msdft/msdft_<N>.in` / `.out`
- `output/output.log` — the driver log (also contains every `[Q-Chem]` line
  with σ, E_D, E_A, E_0, E_1, Δ, H_DA, S_DA reported by the parser, and
  per-step summaries with `mb_resample_recovery=` and `mb_resample_attempt=` when recovery was used).

When `save_qchem_outputs: false`, scratch input/output files are written to
the working directory with `.qchem_tmp_*` names and deleted after parsing.

---

## Inspecting trajectories

A standalone parser is included for verbose per-frame dumps:

```bash
python inspect_trajectory.py --path trajectory.npz --start-frame 0 --end-frame 10
```

This writes a fully formatted text report (default `npz_analysis.out`) listing
every key, every per-frame array slice, and every global array. Use `--out -`
to print to stdout.

---

## Configuration reference

| Key | Type | Default | Description |
|---|---|---|---|
| `initial_geometry` | str | required | Path to heavy-atom XYZ file |
| `masses_amu` | list[float] \| null | null | Atomic masses in AMU. If null, inferred from element symbols (`H`, `C`, `N`, `O`) |
| `proton_center_donor` | list[float] | required | Initial donor proton center [x, y, z] in Å |
| `proton_center_acceptor` | list[float] | required | Initial acceptor proton center [x, y, z] in Å |
| `opt_template` | str | required | Path to OPT template |
| `msdft_template` | str | required | Path to three-job MSDFT template |
| `qchem_nprocs` | int | 1 | CPUs for Q-Chem (`-nt` flag) |
| `proton_center_indices` | list[int] | [1, 2] | 1-based indices of H, @H in MSDFT job 3 |
| `target_temperature` | float | 300.0 | Target temperature, K |
| `initial_velocity_mode` | str | `"mb"` or infer from path | `"mb"`, `"file"`, or `"nacv"` initial velocities |
| `initial_velocity_path` | str \| null | null | Optional initial-velocity file in Bohr / a.u. time (`"file"` mode) |
| `dt_fs` | float | 0.5 | Time step, fs |
| `constraint_threshold` | float | 4e-5 | \|σ\| threshold, Hartree |
| `max_shake_iter` | int | 20 | Max SHAKE root-search iterations **per attempt** (primary step, each MB resample retry) |
| `min_proton_center_separation_angstrom` | float \| null | 0.75 if key omitted; **null disables** | Single-well detection threshold (Å); see "Single-well and OPT failure recovery" |
| `max_mb_resamples` | int | 20 | Maximum MB velocity resample + SHAKE retries at the current geometry before terminating |
| `thermostat` | str | "berendsen" | `"berendsen"` (Bussi exact) or `"none"` |
| `thermostat_tau_fs` | float | 10.0 | Thermostat time constant, fs |
| `n_steps` | int | required | Number of V-SHAKE steps |
| `save_interval` | int | 1 | Save trajectory frame every N steps |
| `trajectory_path` | str | "trajectory.npz" | Path for NPZ archive |
| `seam_xyz_path` | str | "seam_geometries.xyz" | Path for per-step XYZ |
| `save_qchem_outputs` | bool | true | Keep Q-Chem `.in` / `.out` files |
| `output_dir` | str | "output" | Root for Q-Chem outputs and the log |
| `log_filename` | str | "output.log" | Driver log filename |
| `msdft_subdir`, `opt_subdir` | str | "msdft", "opt" | Subdirectories under `output_dir` |

---

## Required Q-Chem output sections

The MSDFT driver greps the Q-Chem output for specific labels. If your
Q-Chem build does not emit any of these blocks, the driver will raise a
`RuntimeError`.

- `NEO-MSDFT RESULTS` — anchor for all subsequent parsing.
- `Overlap Matrix:` — followed by a 2×2 block; `S[0,1]` is `S_DA`.
- `Hamiltonian Matrix:` — followed by a 2×2 block; `H[0,0]`, `H[1,1]` are
  the diabatic energies, `H[0,1]` is `H_DA`.
- `E_0 = ...`, `E_1 = ...`, `E_1 - E_0 = ...` — adiabatic eigenvalues and
  splitting (cm⁻¹).
- `NEO-MSDFT Diabatic Contributions (%):` — followed by two
  `State N: Diabat 0 = X%, Diabat 1 = Y%` lines.
- `Gradient of E0:`, `Gradient of E1:`,
  `Gradient of sigma01 (gradH00 - gradH11):`, `Gradient of H01:`,
  `Gradient of S01:`, `Derivative Coupling d10:` — each followed by
  (N_heavy + 2)-row gradient blocks (rows: Hd, Ha, then heavy atoms).

The OPT driver requires:

- `Final energy is ...` — final SCF energy.
- `OPTIMIZATION CONVERGED` — convergence marker.
- `Coordinates (Angstroms)` — followed by the converged geometry.

---

## Troubleshooting

**"Initial geometry has \|sigma\| = ... which exceeds the constraint threshold."**
The starting geometry is not on the seam. The integrator will still try to
recover on the first SHAKE step, but the dynamics are most well-defined
when you start from an MECP-like geometry. Pre-optimize on the seam (e.g.
with a separate MECP search) before launching V-SHAKE.

**"SHAKE did not converge within max iterations."**
The 1D root search failed to drive |σ| below `constraint_threshold` within
`max_shake_iter` bisections. Most common causes: the time step is too
large, the previous geometry was already off-seam, or the SCF in one of the
diabat OPT jobs is not converging tightly enough. Try reducing `dt_fs`,
tightening `SCF_CONVERGENCE` and `NEO_*_CONV` in the OPT template, or
loosening `constraint_threshold` slightly (1e-5 to 1e-7 Hartree is the
sensible range).

**"Failed to bracket SHAKE root."**
The expansion-with-doubling phase did not find a sign change. This usually
means the local linear approximation underlying the Newton step is wildly
wrong, e.g. the geometry is so distorted that ∇σ no longer points toward
the seam. Reduce `dt_fs` and/or restart from a better seam geometry.

**OPT non-convergence (`MAXIMUM OPTIMIZATION CYCLES REACHED`).**
A diabat OPT job did not converge. The driver logs a warning and attempts
**MB velocity resample** recovery instead of stopping immediately. Open the
corresponding `output/opt/opt_*.out` file if recovery fails repeatedly.
Common causes: poor proton center guesses, tight convergence criteria, or
unphysical geometry. Warm starts revert to the last MSDFT centers on failure.

**`Removed stale density guess file: ...`**
Expected during MB resample recovery: Q-Chem will build fresh density guesses on the
next jobs after single-well collapse or OPT failure.

**"Rigid-translation check failed."**
The driver detected that Q-Chem's reorientation of the converged geometry
is not a pure translation. Since all heavy atoms are `$OPT FIXED`, this
should never happen — it indicates either a parser bug, an unexpected
Q-Chem version-specific reformatting of the converged-geometry block, or
a missing `FIXED` entry in `$OPT`. Inspect the relevant `.out` file.

**"Q-Chem completed only N/3 jobs."**
One of the three MSDFT sub-jobs in `msdft.in` exited without printing the
success marker. The driver counts occurrences of
`Thank you very much for using Q-Chem.` in the output. Open the
corresponding `output/msdft/msdft_<N>.out` and look for SCF failures or
input parsing errors.

**"MSDFT marker not found."**
The string `NEO-MSDFT RESULTS` was not found in the Q-Chem output. Either
job 3 silently completed without computing MSDFT, or your Q-Chem build
labels the output block differently. Inspect the output.

**"MB resample recovery failed after N attempt(s) at the current geometry…"**
Recovery drew new seam-projected Maxwell–Boltzmann velocities (preserving
instantaneous temperature) up to `max_mb_resamples` times; none produced a
successful SHAKE step. Try reducing `dt_fs`, lowering `target_temperature`,
increasing `max_mb_resamples`, or restarting from a geometry more central on the
seam. You can also use `initial_velocity_mode: "nacv"` for an initial velocity
along the seam-tangent NACV direction.

**"Overlap between diabats … is close to or equal to one" / MSDFT linear dependency.**
If this appears despite recovery, increase `min_proton_center_separation_angstrom`
or watch for `mb_resample_recovery=1` in the log. Tune OPT/MSDFT settings if collapse
occurs inside Q-Chem even when centers are separated.

**Persistent large \|sigma\| in the log.**
Check the on-seam |σ| after `eval_msdft`, not just after SHAKE. Because
SHAKE drives σ from the OPT (single-state) energies and MSDFT reports σ
from the diagonal of the diabatic Hamiltonian, small inconsistencies can
appear. If they exceed `constraint_threshold` systematically, tighten the
SCF and integration thresholds in both templates.

---

## How the quantum proton is handled

The quantum proton is **not** a classical degree of freedom. It is described
by a NEO basis whose basis-center position is a parameter of the calculation.
At every diabat OPT call, Q-Chem optimizes that center while holding all
heavy atoms frozen, and the driver reads the optimized center back from the
converged-geometry block. The optimized center from the previous evaluation
is reused as the warm start for the next one (via `self.proton_center_donor`
and `self.proton_center_acceptor` in `QChemCaller`), giving a smooth and
fast-converging stream of OPT jobs.

Because Q-Chem reorients the converged geometry to put the COM at the origin
(or applies a similar rigid translation), the driver computes the translation
applied to the heavy atoms — which must all share the same shift, since they
are frozen — and subtracts that shift from the optimized proton center to
return it to the lab frame. This is the purpose of the rigid-translation
check.

---

## A note on energy conservation and the thermostat

With the thermostat turned off (`thermostat: "none"`), the total energy
`KE + E_0` is a conserved quantity of the constrained dynamics. In
practice, a slow monotonic drift may appear because the protonic
basis-center optimization converges to finite tolerance rather than to the
exact variational minimum. This is the same mechanism discussed by
Dickinson et al. JCP 2026 NEO-ELMD paper: at each step, a small
amount of energy is lost to the incompletely relaxed basis-center degrees
of freedom, and the severity of the drift decreases as the OPT gradient
tolerance is tightened. The drift can be used as a diagnostic — if it is
unacceptably large, tightening `GEOM_OPT_TOL_GRADIENT` in the OPT
template or reducing the time step will help.

With the thermostat on (`thermostat: "berendsen"`), there is no conserved
quantity. The Bussi stochastic velocity rescaling thermostat exchanges
energy with the bath at every step by construction.

We recommend running V-SHAKE **with the thermostat enabled** for production
seam sampling. Without it, the trajectory will naturally settle into local
basins on the seam (near the MECP or other low-energy regions), limiting
the diversity of sampled geometries. The thermostat maintains thermal
fluctuations at the target temperature, driving the trajectory to explore
higher-energy regions of the seam and providing a more representative
ensemble of seam geometries. This is especially important for downstream
analyses that depend on broad coverage of the seam, such as normal-mode
projections, PCA, or tunneling-splitting statistics.

---

## Citation

If you use this code, please cite the original E-SHAKE paper:

- D. V. Cofer-Shabica, J. R. DeRosa, and J. E. Subotnik, "Marcus Theory and
  the Condon Approximation Revisited I: E-SHAKE and Seam Sampling,"
  *J. Chem. Theory Comput.* **2026**, 22, 1575–1583.

The exact stochastic velocity rescaling propagator used here (Eq. A7):

- G. Bussi, D. Donadio, and M. Parrinello, "Canonical sampling through
  velocity rescaling," *J. Chem. Phys.* **2007**, 126, 014101.

And the relevant NEO-MSDFT method papers as appropriate to your study.