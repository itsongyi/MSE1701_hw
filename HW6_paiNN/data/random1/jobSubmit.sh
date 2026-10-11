#!/bin/bash
#SBATCH -J lmp_1
#SBATCH -p pub_L20
#SBATCH -N 1
#SBATCH --tasks-per-node=4

module load mathlib/gsl/intel/2.6
module load mathlib/hdf5/intel/1.12.0
module load mathlib/netcdf/intel/4.7.4
module load compiler/intel/2021.3.0
module load mpi/intelmpi/2021.3.0
module load mathlib/fftw/3.3.8/7/sharelib


/public/software/apps/lammps/intelmpi/29Oct20/bin/lmp_intelmpi < lammps.in > lammps.out
