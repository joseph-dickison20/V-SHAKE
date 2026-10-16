# Sliced Inverse Regression (SIR) for V-SHAKE Seam Data

This document explains, in detail, the sliced inverse regression (SIR) analysis
used to identify the molecular motions that most strongly modulate the vibronic
coupling $V$ within a sampled diabatic seam. It is written for a reader who has
encountered SIR only in passing (e.g. in the Supporting Information of the
V-SHAKE paper) and wants to understand both **what SIR does** and **exactly how
`sir_compute.py` implements it**.

Everything below ties back to V-SHAKE. If you only want the theory, read
[Part 1](#part-1-what-sir-is-and-why-we-use-it). If you want to follow the code,
read [Part 2](#part-2-the-implementation-sir_computepy). The mapping between the
symbols in the paper and the variable names in the code is collected in
[the notation table](#notation-map-paper--code).

---

## Table of contents

- [The problem SIR solves for V-SHAKE](#the-problem-sir-solves-for-v-shake)
- [Part 1: What SIR is and why we use it](#part-1-what-sir-is-and-why-we-use-it)
  - [The inverse-regression idea](#the-inverse-regression-idea)
  - [Why "inverse"?](#why-inverse)
  - [Whitening](#whitening)
  - [Slicing](#slicing)
  - [The SIR matrix and its eigenvectors](#the-sir-matrix-and-its-eigenvectors)
  - [Un-whitening: back to Cartesian motions](#un-whitening-back-to-cartesian-motions)
  - [What the eigenvalues mean](#what-the-eigenvalues-mean)
  - [Assumptions and caveats](#assumptions-and-caveats)
- [Part 2: The implementation (`sir_compute.py`)](#part-2-the-implementation-sir_computepy)
  - [Inputs and outputs](#inputs-and-outputs)
  - [Noise-floor filtering and classical bootstrap](#noise-floor-filtering-and-classical-bootstrap)
  - [Step 0: reading the master XYZ and the response](#step-0-reading-the-master-xyz-and-the-response)
  - [Step 1: Kabsch alignment to the MECP](#step-1-kabsch-alignment-to-the-mecp)
  - [Step 2: PCA whitening](#step-2-pca-whitening)
  - [Step 3: slicing and the SIR matrix](#step-3-slicing-and-the-sir-matrix)
  - [Step 4: un-whitening to Cartesian directions](#step-4-un-whitening-to-cartesian-directions)
  - [Step 5: outputs](#step-5-outputs)
- [Notation map: paper ↔ code](#notation-map-paper--code)
- [Running it](#running-it)
- [Reproducing the paper figures](#reproducing-the-paper-figures)
- [References](#references)

---

## The problem SIR solves for V-SHAKE

A V-SHAKE trajectory samples nuclear configurations $\mathbf{R}$ that lie on the
diabatic seam $\mathcal{Z}(\sigma)$, i.e. the set of geometries for which the
reactant and product diabatic NEO-DFT energies are equal,

$$
\sigma(\mathbf{R}) = E_\mathrm{R}(\mathbf{R}) - E_\mathrm{P}(\mathbf{R}) = 0 .
$$

For every sampled structure we also have the vibronic coupling
$V = \Delta/2$, where $\Delta$ is the tunneling splitting. So the data are a
cloud of points on the seam, each labeled by a scalar coupling.

The seam is a high-dimensional object: it has one dimension fewer than the full
$3N_c$ configuration space (the single constraint $\sigma = 0$ removes one). But
in practice $V$ usually responds strongly to only a **handful** of collective
motions — for the systems in this work, essentially just the donor–acceptor
stretch $\mathbf{R}_\mathrm{DA}$. The question SIR answers is:

> Given only the sampled structures and their couplings, which directions in
> nuclear-displacement space — **staying on the seam** — change $V$ the most?

Crucially, we want this *without* telling the algorithm anything about
$\mathbf{R}_\mathrm{DA}$ or any other chemical coordinate in advance. It should
be purely data-driven. That is exactly what SIR provides, and the fact that it
independently rediscovers $\mathbf{R}_\mathrm{DA}$ as the dominant coordinate for
CBO and HEO is a validation both of the method and of the chemical picture.

---

## Part 1: What SIR is and why we use it

### The inverse-regression idea

The usual regression question is "given a structure, predict the coupling":
model $V$ as a function of the $3N_c$ Cartesian coordinates. That is hard and
overparameterized — we have far more coordinates than we can constrain, and most
of them are irrelevant.

SIR reframes the problem. The core statistical premise (the *dimension-reduction
model*) is that $V$ depends on the full coordinate vector **only through a few
linear combinations of coordinates**:

$$
V \approx f\!\left(\boldsymbol{\eta}_1^{\text{T}}\mathbf{R},\;
\boldsymbol{\eta}_2^{\text{T}}\mathbf{R},\;\dots,\;
\boldsymbol{\eta}_k^{\text{T}}\mathbf{R}\right),
\qquad k \ll 3N_c .
$$

The vectors $\{\boldsymbol{\eta}_i\}$ span what is called the **central
subspace** — the small set of directions that carry all the dependence of $V$ on
geometry. For V-SHAKE, each $\boldsymbol{\eta}_i$ is a $3N_c$-dimensional
Cartesian displacement, i.e. a concerted nuclear motion, and $\boldsymbol{\eta}_1$
is the single motion along which the coupling varies most strongly. This is the
same $\boldsymbol{\eta}_1$ plotted in the SI figure.

We don't need to know the function $f$ — only the directions $\boldsymbol{\eta}_i$.

### Why "inverse"?

Forward regression studies $\mathbb{E}[V \mid \mathbf{R}]$: the average coupling
given a structure. That lives in the high-dimensional coordinate space and is
hard to estimate.

SIR instead studies the **inverse regression curve**
$\mathbb{E}[\mathbf{R} \mid V]$: the average *structure* given a coupling value.
This is a curve in coordinate space parameterized by the single scalar $V$. The
key theorem (Li, 1991) is that, under the model above and a mild condition on the
distribution of $\mathbf{R}$ (see [caveats](#assumptions-and-caveats)), this
inverse curve — after centering — **lies in the central subspace**. So the
directions along which the mean structure moves as you vary $V$ are exactly the
directions $\boldsymbol{\eta}_i$ we are after.

The payoff is enormous: instead of regressing a scalar on thousands of
coordinates, we just track how the *mean geometry* drifts as the coupling
changes. That is something we can estimate reliably from a few thousand sampled
frames.

Concretely, "how the mean geometry drifts with $V$" becomes: sort the frames by
$V$, chop them into bins ("slices"), take the mean structure in each slice, and
look at the directions those slice-means spread along. That is the whole
algorithm. The remaining subtlety is that "spread" has to be measured in a
whitened metric, which we explain next.

### Whitening

Before slicing, SIR *whitens* the coordinates. Whitening is a linear
transformation that removes the overall covariance structure of the sampled
cloud: after whitening, the frames have zero mean and identity covariance (unit
variance in every direction, no correlations).

Why is this necessary? Without it, SIR would confuse "directions along which $V$
varies" with "directions along which the sampling happens to be broad." A floppy,
high-variance motion would dominate the analysis simply because the cloud is
elongated along it, not because $V$ depends on it. Whitening puts every direction
on equal footing so that the slice-mean spread reflects genuine dependence on
$V$, not sampling anisotropy.

In V-SHAKE the raw cloud is *extremely* anisotropic: rigid-body rotation and
translation, and stiff bond directions, carry almost no meaningful variance,
while soft modes carry a lot. We handle this with PCA whitening (see
[Step 2](#step-2-pca-whitening)), which additionally throws away the
near-null-space directions (global rotation/translation and numerically flat
modes) before whitening the rest. This is the "whitening procedure" referred to
in the SI.

### Slicing

This is the step that gives SIR its name. We:

1. Sort all sampled frames in ascending order of their coupling $V$.
2. Partition the sorted list into $N_s$ contiguous **slices** (bins in $V$).
3. Compute the **mean whitened structure** $\bar{\mathbf{a}}_j$ in each slice
   $j$.

Each $\bar{\mathbf{a}}_j$ is a single point in whitened coordinate space: the
average geometry of the frames whose couplings fall in slice $j$. As $j$ runs
from the low-$V$ slice to the high-$V$ slice, the sequence of $\bar{\mathbf{a}}_j$
traces out the (discretized, whitened) inverse-regression curve
$\mathbb{E}[\mathbf{R}\mid V]$.

Slicing is just a nonparametric way to estimate that conditional-mean curve
without assuming any functional form for how geometry depends on $V$. The number
of slices $N_s$ is a smoothing knob: too few and you can't resolve the shape of
the curve; too many and each slice mean is noisy. Ten slices is the default here
and is a standard, robust choice.

Note that because the total frame count $n$ rarely divides evenly by $N_s$, one
slice carries the remainder. In this implementation the slices are built from
evenly spaced index cuts on the $V$-sorted list, so counts are as equal as
integer division allows; the paper phrases this as the lowest-$V$ slice absorbing
the leftover, which is the same idea.

### The SIR matrix and its eigenvectors

The directions the inverse curve spreads along are found by diagonalizing a
weighted covariance of the slice means. This is the matrix $\mathbf{A}$ in the
SI (called `M` in the code):

$$
\mathbf{A} = \sum_{j=1}^{N_s} \frac{n_j}{n}\,
\bar{\mathbf{a}}_j\,\bar{\mathbf{a}}_j^{\text{T}} ,
$$

where $n_j$ is the number of frames in slice $j$ and $n = \sum_j n_j$ is the
total. Each term is the outer product of a slice mean with itself, weighted by
that slice's population. Because the whitened data have zero global mean, the
slice means are already centered, so $\mathbf{A}$ is exactly the (population-
weighted) covariance matrix of the inverse-regression curve.

Diagonalizing $\mathbf{A}$ gives eigenpairs $(\lambda_i, \mathbf{v}_i)$:

- The **eigenvectors** $\mathbf{v}_i$ are the directions (in whitened space)
  along which the slice means — and therefore the mean geometry as a function of
  $V$ — vary most.
- The **eigenvalues** $\lambda_i$ measure how much of that variation each
  direction captures. Large $\lambda_i$ = a direction strongly tied to changes in
  $V$; $\lambda_i \approx 0$ = a direction along which the mean geometry does not
  move as $V$ changes, i.e. irrelevant to the coupling.

The eigenvectors with the largest eigenvalues span the estimated central
subspace. Since they are eigenvectors, each is defined only **up to a sign** —
which is why the SI notes that the orientation of $\boldsymbol{\eta}_1$ (whether
the arrow shows compression or elongation) is arbitrary and only the *axis* is
meaningful.

### Un-whitening: back to Cartesian motions

The eigenvectors $\mathbf{v}_i$ live in the whitened space, which is
mathematically convenient but has no physical units — you can't draw an arrow on
an atom with it. To interpret $\boldsymbol{\eta}_i$ as an actual nuclear motion
we **un-whiten**: apply the inverse of the whitening transform to map
$\mathbf{v}_i$ back to a $3N_c$-dimensional Cartesian displacement. After
un-whitening, $\boldsymbol{\eta}_i$ is a genuine set of per-atom displacement
vectors that can be overlaid on the MECP structure (as in the SI figure) or
animated as a vibration-style GIF.

We then normalize each $\boldsymbol{\eta}_i$ to unit length. All magnitude
information about "how strongly this direction controls $V$" is kept in the
eigenvalue $\lambda_i$, not in the vector, so normalizing the vector loses
nothing and makes the arrows comparable across systems and isotopes.

A metric note that matters for the KIE story: every vector here — the sampled
displacements, the whitening, and the final $\boldsymbol{\eta}_i$ — uses plain,
unweighted Cartesian coordinates with the ordinary Euclidean norm. There is **no
mass weighting in the coordinate metric.** Mass-dependence (the thing that
distinguishes H from D) enters the analysis only through the V-SHAKE constrained
ensemble that generated the structures, because those trajectories were run with
real isotope masses. This keeps the metric consistent end-to-end across the whole
V-SHAKE pipeline and means the H-vs-D differences SIR reports come from the
sampled distribution, not from a coordinate-system choice.

### What the eigenvalues mean

The eigenvalue spectrum is diagnostic. If one eigenvalue dominates and the rest
are small, the coupling is effectively controlled by a **single** collective
motion — a one-dimensional, and likely Condon-friendly, dependence. If several
eigenvalues are comparably large, $V$ responds to multiple independent motions
within the seam, hinting at richer, potentially non-Condon behavior. For CBO and
HEO the spectrum is dominated by $\boldsymbol{\eta}_1$, and that direction turns
out to be donor–acceptor stretch — the expected result, which is why these
systems are used to validate the method rather than to discover anything
surprising. The value of the tool is for more complex systems where the important
in-seam coordinates are not obvious in advance.

One caution: SIR eigenvalues are **not** normalized to sum to one or bounded by
one (unlike, say, diffusion-map eigenvalues). Their absolute scale depends on the
whitening and slicing and is not physically meaningful on its own; only the
*relative* sizes across the spectrum should be interpreted. The code's bar-chart
y-axis is auto-scaled for this reason.

### Assumptions and caveats

- **Linearity condition.** SIR's guarantee that the inverse curve lands in the
  central subspace requires a mild condition on the *predictor* distribution
  (roughly, that the whitened coordinates have linear conditional expectations
  along the relevant directions — satisfied, for instance, by elliptically
  symmetric distributions). Whitening helps, but this is an assumption, not a
  theorem about our specific data. In practice SIR is robust to moderate
  violations.
- **Linear directions only.** SIR finds *linear* collective coordinates. If $V$
  depended on an intrinsically curved coordinate (e.g. a large-amplitude torsion),
  a single $\boldsymbol{\eta}$ would only capture its local, linearized version
  near the MECP. For the small-amplitude, stretch-dominated coupling response
  here, this is not a limitation.
- **Slice count.** Results should be stable to reasonable changes in $N_s$; if the
  leading direction flips character when you halve or double the slice count, the
  signal is weak. This is worth checking for any new system.
- **It characterizes on-seam sensitivity only.** SIR tells you which motions
  change $V$ *while staying on the seam*. It says nothing about barriers to
  reaching those configurations from the equilibrium basin — geometric
  sensitivity is not dynamical accessibility.
- **Sign ambiguity.** As above, each $\boldsymbol{\eta}_i$ is defined up to a
  global sign.

---

## Part 2: The implementation (`sir_compute.py`)

The script is a single, self-contained SIR driver. It takes one or more
MECP-aligned master XYZ files (one per isotope), always runs three families of
analyses per isotope (`all`, `filtered`, and classical bootstrap at each
configured fraction), writes reports + eigenvalue plots + reusable archives, and
— when given more than one isotope — grouped H/D eigenvalue bar charts under
`combo_bar_plots/{variant}/`. All tunable parameters are exposed as command-line
flags (and mirrored in the pipeline YAML `sir:` block).

### Inputs and outputs

**Input.** A "master XYZ" per isotope: a multi-frame XYZ where **frame 0 is the
MECP** and every subsequent frame is a sampled seam structure. Each frame's
comment line carries the metadata `delta_cm^-1=... E0=... step=...`. The response
used by SIR is $V = \delta_{\text{cm}^{-1}} / 2$, i.e. half the tunneling
splitting, read directly from each comment line.

**Outputs (per isotope, under `OUTPUT_DIR/<isotope>/`):**

```text
all/           sir_results.npz, sir_report.txt,
               sir_eigenvalue_spectrum.png, sir_modes.molden
filtered/      same (noise-floor subset; MECP always kept at index 0)
bootstrap_p75/ sir_bootstrap.npz, sir_report.txt,
               sir_eigenvalue_spectrum.png   # mean bars + Tukey box/whiskers
bootstrap_p85/ ...
eta1_vs_filtered.txt   # |cos| and angle of each bootstrap η1 vs filtered η1
```

| File | Contents |
|------|----------|
| `sir_eigenvalue_spectrum.png` | For `all`/`filtered`: plain bar chart of leading $\lambda_i$. For bootstrap: bar at the **mean** over replicates with a Tukey box-and-whisker overlaid (Q1/median/Q3; whiskers to 1.5×IQR). |
| `sir_report.txt` | Human-readable summary (frame counts, response range, PCA/SIR params, spectrum; bootstrap reports also list η₁ `|cos|`/angle vs filtered and mean/std/quartiles). |
| `sir_results.npz` | Full archive for `all`/`filtered`: `symbols`, `ref_coords` (MECP), `y`, `sir_vals`, `betas`, PCA eigenvalues, run parameters. |
| `sir_bootstrap.npz` | Bootstrap stack: `sir_vals_boot` `(N, n_eig)`, `eta1_cos_abs` / `eta1_angle_deg` `(N,)`, plus `fraction`, `n_replicates`, `sample_size`, `noise_floor`, and SIR params. |
| `eta1_vs_filtered.txt` | Per-isotope rollup of η₁ similarity for every bootstrap fraction. |
| `sir_modes.molden` | Top directions as Molden modes (`all`/`filtered` only). |

**Combined output (when >1 isotope):**

- `combo_bar_plots/{all,filtered}/sir_eigenvalue_spectrum.png` — plain grouped H/D bars.
- With `bootstrap_combo_spectrum: true` (default): one
  `combo_bar_plots/bootstrap_combo/sir_eigenvalue_spectrum.png` with **six bars
  per eigenvalue index** in fraction-major order: filtered 100% H/D, then each
  bootstrap fraction descending (e.g. 85%, 75%) × H/D. Filtered bars have no
  whiskers; bootstrap bars show Tukey whiskers. Distinct hatch patterns
  distinguish bootstrap fractions from each other and from the solid 100% bars.
  With the flag off, separate `combo_bar_plots/bootstrap_pXX/` charts are written
  instead.

The `betas` array in the **`all/` or `filtered/`** NPZ is what the downstream
`sir_viz` step consumes (YAML `sir.viz_source`, default `filtered`). Bootstrap
results are analysis-only and are never passed to PyMOL.

### Noise-floor filtering and classical bootstrap

**Noise floor.** Frames with $V$ below `noise_floor` (default $1.0$ cm$^{-1}$) are
dropped for the `filtered` and bootstrap families. Frame 0 (the MECP) is **always
retained** as index 0 so `ref_coords` stay the MECP geometry even if the MECP
itself sits below the floor. If too few frames remain for PCA whitening or for
`nslices` SIR slices, the script fails with a clear error.

**Classical bootstrap (on the filtered set).** For each fraction $f$ in
`bootstrap_fractions` (default $[0.75, 0.85]$):

1. Set sample size $m = \lfloor f \cdot n_{\text{filtered}}\rfloor$.
2. Draw $N=$ `bootstrap_n` replicates (default 100). Each replicate keeps the MECP
   at index 0 and draws $m-1$ indices **with replacement** from the non-MECP
   filtered pool.
3. Run SIR on that sample; collect the eigenvalue spectrum `sir_vals` and the
   leading Cartesian direction $\boldsymbol{\eta}_1$ (`betas[:, 0]`).
4. Write `bootstrap_pXX/` where `XX = round(100 f)` (e.g. $0.75$ → `bootstrap_p75`).

RNG seeding uses `bootstrap_seed` with a deterministic offset per isotope index
and fraction index so re-runs are reproducible.

**η₁ similarity to filtered 100%.** For each bootstrap replicate, compare
$\boldsymbol{\eta}_1^{\text{boot}}$ to $\boldsymbol{\eta}_1^{\text{filtered}}$ with
the sign-invariant overlap $|\cos\theta| = |\boldsymbol{\eta}_1^{\text{filt}}\cdot
\boldsymbol{\eta}_1^{\text{boot}}|$ (both unit-normalized) and
$\theta = \arccos(|\cos\theta|)$ in degrees. Values near $|\cos|=1$ / $\theta=0$
mean the leading coupling-controlling direction is unchanged under subsampling.
Per-replicate tables appear in each `bootstrap_pXX/sir_report.txt` and in the
isotope rollup `eta1_vs_filtered.txt`; arrays `eta1_cos_abs` /
`eta1_angle_deg` are stored in `sir_bootstrap.npz`.

**Reading the whisker plots.** For each eigenvalue index $i$, the bar height is
the mean of $\lambda_i$ over the $N$ replicates. The overlaid box shows Q1,
median, and Q3; whiskers extend to the most extreme replicate within $1.5\times$
IQR of the box (Tukey). Wide whiskers mean that eigenvalue is unstable under
subsampling; a dominant leading $\lambda_1$ with tight whiskers is the usual
signature of a robust coupling-controlling direction. The combined
`bootstrap_combo` figure places filtered 100% bars first (no whiskers), then
bootstrap fractions in descending order, with H and D nested inside each
fraction.

### Step 0: reading the master XYZ and the response

`read_master_xyz` parses each frame, checks that the atom ordering is identical
across all frames (a hard error otherwise, since a scrambled atom order would
silently corrupt the displacement analysis), and returns the symbols, an
`(n_frames, n_atoms, 3)` coordinate array, and the response vector
$y_i = \delta_{\text{cm}^{-1}}^{(i)} / 2$.

This mirrors the paper's definition exactly: on the seam $E_\mathrm{R}=E_\mathrm{P}$,
so the tunneling splitting and the coupling are related by $V = \Delta/2$, and the
splitting in cm$^{-1}$ is what the trajectory writes out.

### Step 1: Kabsch alignment to the MECP

`align_to_mecp` superposes every frame onto frame 0 (the MECP) with the Kabsch
algorithm (`kabsch_align`): center both structures, form the cross-covariance,
take its SVD, and build the optimal proper rotation (the `det` correction
prevents an accidental reflection). This removes global rotation and translation
so that the only variation left in the cloud is genuine internal motion.

This is the Kabsch alignment referenced in the SI. It must happen **before**
whitening — otherwise rigid-body motion would masquerade as real variance and
pollute the whitening and the slice means.

### Step 2: PCA whitening

`pca_whiten` performs the whitening described in
[Part 1](#whitening), but in a numerically safe, dimension-reducing way:

1. Flatten each aligned frame to a $3N_c$ vector and mean-center the cloud.
2. SVD the centered data matrix; the squared singular values give the variance
   (PCA eigenvalues) along each principal direction.
3. Keep the leading principal components that together explain a target fraction
   of the total variance (`--var`, default 0.95), with a floor of 2 components.
   This step **discards the near-null-space** — global rotation/translation
   residue and numerically flat directions — which would otherwise make the
   whitening blow up (dividing by a near-zero variance).
4. Project onto the retained components and divide each by its standard deviation
   $\sqrt{\text{eigval}}$. The result `Z` has identity covariance: it is the
   whitened representation.

The function returns `Z` (whitened scores), `V` (the retained PCA basis, needed
later to un-whiten), the retained PCA eigenvalues, and the number of components
`k`. Reducing to `k` components before whitening is what makes SIR on
$3N_c$-dimensional Cartesian data well-conditioned.

### Step 3: slicing and the SIR matrix

`sir(Z, y, n_slices)` implements the slicing and the $\mathbf{A}$-matrix
construction:

1. Sort the whitened frames by their response `y`.
2. Cut the sorted list into `n_slices` contiguous slices using evenly spaced
   index edges (`np.linspace(0, n, n_slices+1)`), so slice populations are as
   equal as integer division permits.
3. For each slice, compute the mean whitened structure $\bar{\mathbf{a}}_j$ and
   accumulate the population-weighted outer product
   $\frac{n_j}{n}\,\bar{\mathbf{a}}_j\bar{\mathbf{a}}_j^{\text{T}}$ into `M`.
4. Symmetric-eigendecompose `M` (`np.linalg.eigh`) and return the eigenvalues and
   eigenvectors sorted in descending eigenvalue order.

The matrix `M` in the code **is** the matrix $\mathbf{A}$ of SI
eq. (SIR matrix); the returned eigenvalues are the $\lambda_i$ and the
eigenvectors are the whitened-space directions $\mathbf{v}_i$.

### Step 4: un-whitening to Cartesian directions

Back in `run_sir_on_frames` / `compute_sir_core`, the whitened eigenvectors are
mapped to physical Cartesian displacements:

```python
betas = V @ (sir_vecs / np.sqrt(pca_eigvals)[:, None])
betas /= np.linalg.norm(betas, axis=0, keepdims=True)
```

Reading this right to left: `sir_vecs / sqrt(pca_eigvals)` undoes the
variance scaling from whitening, and left-multiplying by `V` (the retained PCA
basis) lifts the result back into the full $3N_c$ Cartesian space. That composite
is exactly the inverse of the whitening transform — the "un-whitening procedure"
in the SI. Each resulting column is then normalized to unit length.

The columns of `betas`, reshaped to `(n_atoms, 3)`, are the
$\boldsymbol{\eta}_i$: per-atom displacement vectors ready to be drawn on the MECP
structure. `betas[:, 0]` is $\boldsymbol{\eta}_1$, the primary coupling-controlling
motion shown in the paper.

### Step 5: outputs

For each of `all` and `filtered`, `run_sir_on_frames` then:

- Plots the eigenvalue spectrum (`plot_eigenvalue_spectrum`) with an auto-scaled
  y-axis, since SIR eigenvalues are unnormalized.
- Writes `sir_report.txt`, including, for each visualized direction, the five
  atoms carrying the largest displacement magnitude — this is the quick,
  text-only way to read off "which atoms move" (e.g. seeing the donor and
  acceptor heavy atoms dominate $\boldsymbol{\eta}_1$).
- Saves `sir_results.npz` for `sir_viz` and for reuse (a cached NPZ is reused on
  re-runs unless `--force` is given).
- Writes `sir_modes.molden`, converting the reference geometry to Bohr (Molden's
  convention) while leaving the dimensionless displacement vectors as-is.

Bootstrap fractions call `run_bootstrap_fraction`, which stacks replicate
eigenvalues into `sir_bootstrap.npz`, records η₁ `|cos|`/angle vs filtered, and
draws mean bars with Tukey whiskers (`plot_bootstrap_eigenvalue_spectrum`).

`main` loops over every `--dataset ISOTOPE XYZ` pair, runs
`run_isotope_variants` (or reuses cached markers), collects spectra / bootstrap
stacks, and if more than one isotope was supplied draws combo plots under
`combo_bar_plots/` (plain bars via `plot_combined_grouped_eigenvalue_bars`;
with `--bootstrap-combo-spectrum`, one mega chart via
`plot_combined_filtered_bootstrap_spectrum`; otherwise per-fraction
`plot_combined_bootstrap_eigenvalue_bars`).

---

## Notation map: paper ↔ code

| Paper / SI symbol | Meaning | Code |
|---|---|---|
| $\sigma(\mathbf{R}) = E_\mathrm{R}-E_\mathrm{P}$ | diabatic energy gap (constraint) | — (defines the seam the input frames lie on) |
| $\mathcal{Z}(\sigma)$ | diabatic seam space | the sampled frames (input XYZ) |
| $V = \Delta/2$ | vibronic coupling (response) | `y = delta_cm_inv / 2` |
| $\delta_{\text{cm}^{-1}}$ | tunneling splitting in cm$^{-1}$ | `delta_cm^-1` in XYZ comment |
| $3N_c$ | classical-nuclear coordinate dimension | `3 * n_atoms` |
| $N_s$ | number of slices | `n_slices` / `--nslices` |
| $n_j,\; n$ | slice population, total frames | slice sizes, `n` |
| $\bar{\mathbf{a}}_j$ | mean whitened structure of slice $j$ | `mean_h` |
| $\mathbf{A}$ | SIR matrix $\sum_j (n_j/n)\bar{\mathbf{a}}_j\bar{\mathbf{a}}_j^{\text{T}}$ | `M` |
| $\lambda_i$ | SIR eigenvalues | `sir_vals` |
| $\mathbf{v}_i$ | whitened-space eigenvectors | `sir_vecs` |
| $\boldsymbol{\eta}_i$ | un-whitened Cartesian direction | `betas[:, i-1]` |
| $\boldsymbol{\eta}_1$ | primary coupling-controlling motion | `betas[:, 0]` |
| MECP | minimum-energy crossing point | frame 0 / `ref_coords` |

---

## Running it

```bash
# Single isotope
python sir_compute.py \
    --dataset H  path/to/HEO_H_master.xyz \
    --output-dir results/heo_sir \
    --run-name HEO

# Both isotopes (also writes the grouped H/D comparison bar chart)
python sir_compute.py \
    --dataset H  path/to/HEO_H_master.xyz \
    --dataset D  path/to/HEO_D_master.xyz \
    --output-dir results/heo_sir \
    --run-name HEO
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dataset ISOTOPE XYZ` | (required) | Isotope label and its master XYZ; repeat once per isotope. |
| `--output-dir` | (required) | Root output directory. |
| `--run-name` | (required) | Label used in reports/titles. |
| `--nslices` | 10 | Number of SIR slices $N_s$. |
| `--var` | 0.95 | Fraction of variance retained by PCA whitening. |
| `--n-eigenvalues-plot` | 3 | How many leading eigenvalues to draw on the bar chart (`sir.n_eigenvalues_plot`). |
| `--n-eigenvectors-viz` | 3 | How many $\boldsymbol{\eta}_i$ to flag for visualization. |
| `--noise-floor` | 1.0 | cm$^{-1}$; frames with $V$ below this are dropped for `filtered`/bootstrap. |
| `--bootstrap-fractions` | 0.75 0.85 | Fractions of the filtered set for classical bootstrap. |
| `--bootstrap-n` | 100 | Number of bootstrap replicates per fraction. |
| `--bootstrap-seed` | 0 | Base RNG seed (offset per isotope/fraction). |
| `--bootstrap-combo-spectrum` / `--no-bootstrap-combo-spectrum` | on | One filtered+bootstrap H/D combo spectrum vs separate per-fraction combo charts. |
| `--force` | off | Recompute even if cached variant markers exist. |

The isotope labels **must** be `H` and `D` to match the shared color convention
in `hd_plot_colors.py` (H = Tableau red `#d62728`, D = Tableau blue `#1f77b4`);
these are the same colors used across the master-XYZ, geometry, and `sir_viz`
steps so the isotopes read consistently in every figure.

## Reproducing the paper figures

The $\boldsymbol{\eta}_1$ vectors shown in the SIR figure of the SI are the
`betas[:, 0]` directions from each isotope's `all/` or `filtered/`
`sir_results.npz` (pipeline default: `filtered`), rendered on the MECP structure
by the `sir_viz` step (arrows) and/or via `sir_modes.molden` (animations).
Grouped eigenvalue bar charts under `combo_bar_plots/` correspond to the
eigenvalue spectra discussed in the text; the `bootstrap_combo` panel (or
per-fraction bootstrap panels) add Tukey whiskers as described above, and
`eta1_vs_filtered.txt` quantifies that $\boldsymbol{\eta}_1$ is stable under
subsampling. Because eigenvectors are sign-ambiguous, an arrow may point "the
other way" than a given rendering; only the axis of motion is physical (the
$|\cos|$ metric already folds out the sign).

---

## References

SIR was introduced by Li and developed further in the references cited in the
paper's SI:

- K.-C. Li, *Sliced Inverse Regression for Dimension Reduction*, **1991**
  (`li_sliced_1991`).
- C. Bernard-Michel et al., Gaussian-process / regularized SIR treatment,
  **2009** (`bernard-michel_gaussian_2009`).
- Gewers et al., PCA/whitening background, **2021** (`gewers_principal_2021`).
- W. Kabsch, structure superposition, **1976** (`kabsch_solution_1976`).

For the V-SHAKE method, the diabatic seam definition, and how the SIR result
supports the Condon analysis and the NEO-GRT thermal-averaging assumption, see
the main paper and its Supporting Information.