# FHF⁻ NEO-BL-MSDFT (PB4-D)

Uncorrected ($\alpha = \beta = 1$) NEO-DFT OPT + NEO-BL-MSDFT jobs used to build `../fit/fhf_data.csv`.

| $R_{\mathrm{FF}}$ (Å) | proton | deuteron |
|------------------------|--------|----------|
| 2.60 | yes | yes |
| 2.65 | *excluded* | yes |
| 2.70 | yes | yes |
| 2.75 | yes | yes |
| 2.80 | yes | yes |

Each directory:

- `opt.in` / `opt.out` — NEO-DFT basis-center optimization (one diabat; centers are symmetric)
- `msdft.in` / `msdft.out` — NEO-BL-MSDFT (source of S12 / Hab / splitting)

`basis/` and `templates/{opt,msdft}.in` plus `setup_calcs.py` regenerate the input tree if needed.
