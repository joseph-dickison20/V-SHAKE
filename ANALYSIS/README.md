# SIR vibronic-coupling analysis pipeline

Sliced inverse regression (SIR) finds the Cartesian directions along which the
vibronic coupling \(V = \Delta/2\) varies fastest on the seam manifold, and step
4 renders those directions next to the tunneling coordinate at the MECP.

For the theory behind step 3 — what the eigenvalues mean, why PCA is involved,
how "% of spectrum" and the atomic contributions are computed — see
[SIR.md](SIR.md).

## Steps

| Step | Output dir | What it does |
| --- | --- | --- |
| `master_xyz` | `01_master_xyz/` | Gather TRAJS `seam_geometries.xyz`, Kabsch-align to the MECP, write `master.xyz` (frame 0 = MECP) |
| `geom` | `02_geom/` | Donor–acceptor (and CBO CN-bend) geometric scatters |
| `sir` | `03_sir/` | PCA-whiten Cartesians; run SIR on \(V\) for `all`, noise-floor-`filtered`, and classical bootstrap variants |
| `sir_viz` | `04_sir_viz/` | PyMOL renders of the eigenvectors, g⊥, and ∇V, plus cosine bar charts |

There is no dimensionality test and no atom-pair importance analysis. Step 3
always writes unfiltered (`all/`), noise-floor-filtered (`filtered/`), and
bootstrap (`bootstrap_pXX/`) eigenvalue results; step 4 draws eigenvectors from
exactly one of `all` or `filtered` (YAML `sir.viz_source`, default `filtered`).

## Quick start

```bash
cd <this directory>
./run_pipeline.sh --config CBO.yaml
./run_pipeline.sh --config HEO.yaml --force
./run_pipeline.sh --config CBO.yaml --steps master_xyz,geom,sir
```

Slurm — full pipeline for **both** CBO and HEO (a compute node is required
because PyMOL does not run on login nodes):

```bash
cd <this directory>
sbatch run_pipeline.sbatch
```

## Environment

`run_pipeline.sh` activates `seam_samp_env`. Needs numpy, scipy, matplotlib
(Agg), ASE, PyYAML. Step `sir_viz` also needs **PyMOL** and **ImageMagick** on
`PATH`.

## Vectors drawn in step 4

All three are parsed from the reference-batch NEO-MSDFT Q-Chem MECP output
(`{TEMP}/{BATCH}/{ISO}/output/msdft/msdft_0.out`), selected by the
`qchem_mecp` config section.

- **SIR eigenvectors** β₁…βₙ — Cartesian coupling-driving directions from step 3.
  Red for H, blue for D. Their sign is arbitrary.
- **g⊥ (tunneling coordinate)** — the direction perpendicular to the diabatic
  seam space, read from the `Gradient of sigma01 (gradH00 - gradH11):` block.
  Green.
- **∇V** — the gradient of the vibronic coupling. Since \(V = \Delta/2\),
  \(\nabla V = \tfrac{1}{2}(\nabla E_1 - \nabla E_0)\), built from the
  `Gradient of E1:` and `Gradient of E0:` blocks. Orange.

The leading `n_skip_rows` rows of every gradient block are the quantum-nuclear
basis centers and are dropped so the vectors share one atom ordering with
`master.xyz`. The unlisted quantum nucleus is represented by its density mesh
instead of an arrow.

## Orthogonality checks

Three families of cosine similarity are reported, each as a per-atom
decomposition bar chart plus a matching `.txt` table:

| Pair | Where | Question it answers |
| --- | --- | --- |
| cos(g⊥, ∇V) | `gperp_gradV/` | Do the tunneling coordinate and the coupling gradient point the same way? |
| cos(g⊥, βₖ) | `sir{k}/` | Does SIR direction *k* recover the tunneling coordinate? |
| cos(∇V, βₖ) | `sir{k}/` | Does SIR direction *k* recover the coupling gradient? |

Every bar is one atom's contribution \(c_a = (\mathbf{a}_a \cdot
\mathbf{b}_a)/(\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert)\), so the bars sum
exactly to the cos θ printed in the corner of the figure — the chart shows which
atoms create or destroy the alignment, and signed bars can cancel. θ is computed
from |cos θ| because eigenvector signs are arbitrary.

## Nuclear density

Every **static** arrow figure shows the full MECP structure with the transferred
nucleus' density drawn as an isomesh from
`TRAJS/xDENPLT/OUTPUT/{RUN}_{ISO}.cube`. Those cubes sit in the same coordinate
frame as `master.xyz` frame 0, so no alignment is applied. Isovalue, mesh color,
and cube location are configurable; the camera is centered on the molecule alone
so the framing matches the corresponding GIF.

**Swing GIFs deliberately omit the density** and show the bare skeleton.

## Outputs

```text
runs/{CBO,HEO}/
  01_master_xyz/{H,D}/master.xyz
  02_geom/{H,D}/plots/
  02_geom/combo_plots/
  03_sir/{H,D}/
    all/           sir_results.npz, sir_report.txt,
                   sir_eigenvalue_spectrum.png, sir_modes.molden
    filtered/      same (frames with V < noise_floor dropped; MECP kept)
    bootstrap_p75/ sir_bootstrap.npz, sir_report.txt,
                   sir_eigenvalue_spectrum.png   # mean bars + Tukey whiskers
    bootstrap_p85/ ...
    eta1_vs_filtered.txt                         # |cos|/angle of η1 vs filtered
  03_sir/combo_bar_plots/{all,filtered}/
                   sir_eigenvalue_spectrum.png   # grouped H vs D
  03_sir/combo_bar_plots/bootstrap_combo/        # when bootstrap_combo_spectrum
                   sir_eigenvalue_spectrum.png   # 100%+fractions × H/D (6 bars/index)
  04_sir_viz/{H,D}/                              # from sir.viz_source NPZ
    sir{k}/        sir{k}.gif, sir{k}_arrows.png,
                   sir{k}_vs_gperp.txt, sir{k}_vs_gperp_barchart.png,
                   sir{k}_vs_gradV.txt, sir{k}_vs_gradV_barchart.png
    gperp/         gperp.gif, gperp_arrows.png
    gradV/         gradV.gif, gradV_arrows.png
    gperp_gradV/   gperp_gradV_arrows.png,               # both arrows, no GIF
                   gperp_vs_gradV.txt, gperp_vs_gradV_barchart.png
```

## Configuration reference

Beyond `run_name`, `trajectory_root`, `temps`, `isotopes`, `master_xyz`, and
`geom` (all unchanged), the SIR-specific keys are:

```yaml
sir:
  nslices: 10              # SIR slices over the response V
  var: 0.95                # PCA variance cutoff before whitening
  n_eigenvalues_plot: 3    # eigenvalues on the step-3 bar chart
  n_eigenvectors_viz: 3    # eigenvectors rendered in step 4
  noise_floor: 1.0         # cm^-1; frames with V < this dropped for filtered/bootstrap
  bootstrap_fractions: [0.75, 0.85]
  bootstrap_n: 100         # classical bootstrap replicates per fraction
  bootstrap_seed: 0        # reproducible RNG (offset per isotope/fraction)
  bootstrap_combo_spectrum: true  # one mega H/D+fractions combo plot (else per-fraction)
  viz_source: filtered     # filtered | all  -> which NPZ sir_viz reads

qchem_mecp:
  temp: HIGH_TEMP          # batch supplying the MECP msdft_0.out
  batch: BATCH1
n_skip_rows: 2             # quantum-nuclear rows to drop from gradient blocks

sir_viz:
  arrow_scale: 3.0
  arrow_color: {H: "#d62728", D: "#1f77b4"}
  gperp_color: "#2ca02c"
  gradv_color: "#ff7f0e"
  arrow_mol_transparency: 0.2   # static overlays only; 0 opaque, 1 invisible
  amplitude: 0.5                # swing amplitude [Å]
  n_frames: 24
  zoom_center: 3.25
  turn_x: 0
  turn_y: 270
  turn_z: 180
  png_width: 2000
  png_height: 1400
  png_dpi: 600
  ray: 1                        # 1 = ray-traced, 0 = fast draft
  gif_delay: 5
  gif_loop: 0
  gif_extra_args: ""
  density:
    root: TRAJS/xDENPLT/OUTPUT      # cube is {root}/{RUN}_{ISO}.cube
    isovalue: 0.04
    color: [128, 229, 255]
    # paths: {H: ..., D: ...}        # optional explicit cube overrides
  cos_barchart:
    xlim: [-0.5, 13.5]              # CBO has 14 atoms; HEO uses [-0.5, 10.5]
    ylim: null                      # null = symmetric auto-scale
  per_vector:                       # swing/render overrides per vector family
    sir:   {amplitude: 0.50}
    gperp: {amplitude: 0.20}
    gradv: {amplitude: 0.20}
  per_isotope:                      # same overridable fields, per isotope
    H: {}
    D: {}
```
