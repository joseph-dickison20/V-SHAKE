#!/usr/bin/env python3
"""Launch script for V-SHAKE seam sampling.

Usage:
    python run_vshake.py config.yaml

Or from an sbatch script:
    python run_vshake.py config.yaml
"""

import sys
from pathlib import Path

import numpy as np
import yaml

# Add the code directory to the path
sys.path.insert(0, "/gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/software/NEO_MSDFT_SEAM_SAMPLE/neo_msdft_seam_sample")

import units
from qchem_interface import QChemCaller
from run import run_from_config

if len(sys.argv) < 2:
    print("Usage: python run_vshake.py <config.yaml>")
    sys.exit(1)

config_path = sys.argv[1]

with open(config_path) as f:
    cfg = yaml.safe_load(f)

save_qchem_outputs = bool(cfg.get("save_qchem_outputs", True))
output_dir = Path(cfg.get("output_dir", "output"))
msdft_subdir = cfg.get("msdft_subdir", "msdft")
opt_subdir = cfg.get("opt_subdir", "opt")
qchem_root = output_dir if save_qchem_outputs else Path.cwd()

# Build the Q-Chem oracle with proton center initial guesses
pc_donor = units.angstrom_to_bohr(np.array(cfg["proton_center_donor"], dtype=float))
pc_acceptor = units.angstrom_to_bohr(np.array(cfg["proton_center_acceptor"], dtype=float))

qc = QChemCaller(
    proton_center_donor=pc_donor,
    proton_center_acceptor=pc_acceptor,
    opt_template_path=cfg["opt_template"],
    msdft_template_path=cfg["msdft_template"],
    nprocs=cfg.get("qchem_nprocs", 1),
    proton_center_indices=tuple(cfg.get("proton_center_indices", [1, 2])),
    workdir=str(Path.cwd()),
    output_dir=str(qchem_root),
    save_qchem_outputs=save_qchem_outputs,
    msdft_subdir=msdft_subdir,
    opt_subdir=opt_subdir,
)

# Run the trajectory
run_from_config(config_path, qc)
