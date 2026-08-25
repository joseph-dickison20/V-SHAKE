#!/bin/bash
# Run the SIR vibronic-coupling analysis pipeline (activates seam_samp_env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate seam_samp_env

cd "${SCRIPT_DIR}"
exec python run_pipeline.py "$@"
