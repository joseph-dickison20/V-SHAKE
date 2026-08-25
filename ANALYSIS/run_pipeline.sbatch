#!/bin/bash
#SBATCH --job-name=sir_analysis
#SBATCH --output=sir_analysis.out
#SBATCH --partition=pi_hammes_schiffer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --export=ALL

# Full SIR pipeline for CBO and HEO (all four steps, including PyMOL sir_viz).
# Submit from the analysis directory (the one holding run_pipeline.py):
#   sbatch run_pipeline.sbatch
# Requires a compute node (PyMOL does not work on login nodes).
#
# No path below names the analysis directory, so it can be renamed freely.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate seam_samp_env

# Explicit full pipeline for both systems (master_xyz → geom → sir → sir_viz).
python run_pipeline.py --config CBO.yaml --steps master_xyz,geom,sir,sir_viz --force
python run_pipeline.py --config HEO.yaml --steps master_xyz,geom,sir,sir_viz --force
