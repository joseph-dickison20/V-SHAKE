#!/bin/bash
#SBATCH --job-name=vshake
#SBATCH --out=vshake.out
#SBATCH -p pi_hammes_schiffer
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --time=28-00:00:00
#SBATCH --export=ALL

# Modules and sources for Q-Chem 
module load OpenMPI/4.1.4-GCC-12.2.0
module load HDF5/1.14.0-gompi-2022b
module load OpenBLAS/0.3.12-GCC-10.2.0
module load CMake/3.24.3-GCCcore-12.2.0
module load Subversion
module load Boost/1.86.0-GCC-12.2.0
source /gpfs/gibbs/pi/hammes_schiffer/jad279/BOUCHET/software/.qcsetup

# Initialize conda and activate environment
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate seam_samp_env

# Run V-SHAKE simulation
python run_vshake.py config.yaml
