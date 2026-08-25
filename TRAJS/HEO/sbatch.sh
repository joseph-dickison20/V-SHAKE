#!/usr/bin/env bash
set -euo pipefail

# Submit a Slurm job for every vshake.sh under this folder (any depth).
#
# Each case directory contains vshake.sh (the batch script with #SBATCH lines).
# vshake.out is the Slurm standard output path set inside vshake.sh, not the
# argument to sbatch. This script runs, from each directory that has the file:
#   sbatch vshake.sh

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
batch_script="${VSHAKE_SBATCH_SCRIPT:-vshake.sh}"

mapfile -d '' -t scripts < <(
  find "$root" -type f -name "$batch_script" -print0 | sort -z
)

if ((${#scripts[@]} == 0)); then
  printf 'no %s files found under %s\n' "$batch_script" "$root" >&2
  exit 1
fi

for script_path in "${scripts[@]}"; do
  case_dir="$(dirname "$script_path")"
  rel="${script_path#"$root"/}"
  printf '==> %s: sbatch %s\n' "$rel" "$batch_script"
  (cd "$case_dir" && sbatch "$batch_script")
done
