#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N ml_docking_dragon
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l filesystems=home:grand:eagle
#PBS -k doe
#PBS -j oe
#PBS -A hpe_dragon_collab
#PBS -q debug
#PBS -V

cd $PBS_O_WORKDIR
./run_driver.sh
