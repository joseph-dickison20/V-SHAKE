# Overlap-correction fit

Inputs:

- `fhf_data.csv` — uncorrected NEO-BL-MSDFT matrix elements / splittings (PB4-D)
- `fhf_refs.csv` — FGH reference splittings

H @ 2.65 Å is blank in `fhf_data.csv` (excluded geometry).

**Objective (used to select $\alpha$, $\beta$):** hybrid absolute + relative error vs FGH refs,
with weights $w_{\mathrm{abs}}=0.99$, $w_{\mathrm{rel}}=0.01$ (absolute term scaled by
$\max|\Delta^{\mathrm{ref}}|$). Shared $\alpha$; $\beta$ per isotopologue. See the top-level README.

`opt_param.py` below **evaluates** the production parameters (does not re-optimize)
and prints absolute / relative RMSE tables.

Evaluate the production parameters and write `FINAL_FITS.txt`:

```bash
python3 opt_param.py
```

Optional: print tables for other ($\alpha$, $\beta_{\mathrm{H}}$, $\beta_{\mathrm{D}}$) without rewriting the file:

```bash
python3 opt_param.py --alpha 0.56 --beta-h 1.10 --beta-d 0.98 --no-write
```
