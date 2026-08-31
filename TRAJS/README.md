# Production V-SHAKE trajectories

Seam-sampling archives for **CBO** and **HEO** (H and D; 300 K and 1000 K).
This directory is what [`ANALYSIS/`](../ANALYSIS/) consumes. How to launch a
run is documented in [`VSHAKE/README.md`](../VSHAKE/README.md); how the
pipeline discovers these files is in [`ANALYSIS/README.md`](../ANALYSIS/README.md).

## Layout

```
TRAJS/
├── CBO/
│   ├── sbatch.sh
│   ├── HIGH_TEMP/BATCH{1–8}/{H,D}/
│   └── LOW_TEMP/BATCH{1–8}/{H,D}/
├── HEO/
│   ├── sbatch.sh
│   ├── HIGH_TEMP/BATCH{1–6}/{H,D}/
│   └── LOW_TEMP/BATCH{1–6}/{H,D}/
├── xMECP_OPT/
│   └── {CBO,HEO}_{H,D}_MECP.out
└── xDENPLT/
    └── OUTPUT/{CBO,HEO}_{H,D}.cube
```

`HIGH_TEMP` is 1000 K and `LOW_TEMP` is 300 K (`target_temperature` in each
run's `config.yaml`).

`CBO/sbatch.sh` and `HEO/sbatch.sh` submit every nested `vshake.sh` under that
molecule.

## Per-run directory

Each `{TEMP}/BATCH*/{H,D}/` case is a self-contained V-SHAKE job:

| File | Role |
| --- | --- |
| `seam_geometries.xyz` | Per-step XYZ; this is what ANALYSIS gathers |
| `mecp.xyz` | Initial heavy-atom geometry (MECP) |
| `config.yaml` | Run configuration |
| `opt.in`, `msdft.in` | Q-Chem templates |
| `vshake.sh` | Slurm script |
| `output/` | Per-step Q-Chem dumps (see version control below) |

## Version control (Git)

This repository does **not** store diabat **OPT** job inputs and outputs under
`output/opt/` (see the root `.gitignore`). Those files are large and
regenerable from each run's `opt.in` and `config.yaml` when re-running
V-SHAKE.

Under `output/msdft/`, **`.out` files are kept** (including reference
`msdft_0.out` for g⊥ and ∇V). **Generated `.in` files there are not stored**
(they can be recreated from each run's top-level `msdft.in` template). Run
artifacts such as `seam_geometries.xyz`, configs, and Q-Chem templates remain
in Git. That is enough to run the [`ANALYSIS/`](../ANALYSIS/) pipeline: it
gathers trajectories from `seam_geometries.xyz`, parses
`output/msdft/msdft_0.out` from `HIGH_TEMP/BATCH1/{H,D}`, and reads nuclear
density cubes under `xDENPLT/OUTPUT/`.

The driver can also write `trajectory.npz` (`trajectory_path` in `config.yaml`).
ANALYSIS does not use that archive.

## NEO-MECP optimization outputs

`xMECP_OPT/` holds the Q-Chem output files from the NEO-MECP optimization job
for each system (`{CBO,HEO}_{H,D}_MECP.out`).

## Nuclear density cubes

`xDENPLT/OUTPUT/{CBO,HEO}_{H,D}.cube` are MECP protonic densities for the
PyMOL overlays in ANALYSIS step 4 (`sir_viz.density.root: TRAJS/xDENPLT/OUTPUT`
in [`CBO.yaml`](../ANALYSIS/CBO.yaml) and [`HEO.yaml`](../ANALYSIS/HEO.yaml)).
