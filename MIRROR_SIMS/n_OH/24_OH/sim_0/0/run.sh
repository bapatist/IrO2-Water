#!/bin/bash -l
#SBATCH --job-name=M24oh_0_0
#SBATCH --no-requeue
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=72
#SBATCH --mem=240000        # memory limit (16 x 3800M)
#SBATCH --time=24:00:00
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
#SBATCH -D ./
conda activate wfl
module purge
module load gcc/11 mkl/2020.2 gsl/2.4 impi/2021.4 fftw-mpi/3.3.9
module load qt/5.7 openmpi
module load anaconda/3/2021.11
source ${MKLROOT}/bin/mklvars.sh intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mpcdf/soft/SLE_12/packages/x86_64/intel_parallel_studio/2020.2/mkl/lib/intel64/

srun /u/nbapat/software/LAMMPS2/build/lmp -in lammps.inp > out
sed -n '/Time/,/Loop/{ /Loop/! p }' log.lammps > logCut.lammps
