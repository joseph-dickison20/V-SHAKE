# NEO-BL-MSDFT Overlap Correction for V-SHAKE

Reparametrization of the NEO-MSDFT overlap correction for the **PB4-D** protonic
basis used in V-SHAKE seam sampling. Benchmarked against FHF⁻ tunneling splittings
(H and D) at F–F distances 2.60–2.80 Å.

## Correction

$$
S'_{ab} = \alpha\, S_{ab}^{\beta}
$$

Here $\alpha$ caps the corrected overlap (important for V-SHAKE stability near MECPs)
and $\beta$ is the power-law scaling.

## Fitting

Corrected NEO-BL-MSDFT tunneling splittings are matched to FGH reference splittings
at each $R_{\mathrm{FF}}$. The objective is a **hybrid of absolute and relative errors,
dominated by the absolute term**:

$$
\begin{aligned}
\mathrm{cost}
&= w_{\mathrm{rel}}\sum_i\left(\frac{\Delta_i^{\mathrm{calc}}-\Delta_i^{\mathrm{ref}}}{\Delta_i^{\mathrm{ref}}}\right)^2 \\
&\quad + w_{\mathrm{abs}}\sum_i\left(\frac{\Delta_i^{\mathrm{calc}}-\Delta_i^{\mathrm{ref}}}{\max_j|\Delta_j^{\mathrm{ref}}|}\right)^2
\end{aligned}
$$

with $w_{\mathrm{abs}}=0.99$, $w_{\mathrm{rel}}=0.01$. Absolute error is scaled by the largest
reference splitting so both terms are dimensionless. Pure relative RMSE overweights
the tiny long-$R_{\mathrm{FF}}$ D splittings; the absolute-heavy hybrid favors the
physically important short-$R$ points.

**Constraints:** shared $\alpha$ for H and D (so the overlap cap cannot invert the isotope
effect); $\beta$ fit independently per isotopologue. H @ 2.65 Å excluded (unstable OPT).

## Final parameters

B3LYP/epc17-2, aug-cc-pVDZ/aug-cc-pV5Z, PB4-D:

| Parameter | Value |
|-----------|-------|
| $\alpha$ (shared H/D) | 0.56 |
| $\beta_{\mathrm{H}}$ | 1.10 |
| $\beta_{\mathrm{D}}$ | 0.98 |

See [`fit/FINAL_FITS.txt`](fit/FINAL_FITS.txt) (reports both absolute and relative RMSE
for these parameters).

## Layout

```
fit/     data CSVs, evaluation script, final tables
calcs/   PB4-D FHF⁻ OPT + MSDFT jobs (inputs + opt/msdft outs)
```

**Excluded:** H @ 2.65 Å (unstable NEO-DFT basis-center opt). No PB5-G jobs or
exploratory archives — those remain under `DEBUG/BL_IMPLEMENT_TESTING/BL_PARAM_FIT`.
