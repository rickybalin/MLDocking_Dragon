#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N ml_docking_dragon
#PBS -l walltime=00:30:00
#PBS -l select=8:ncpus=64:ngpus=4
#PBS -l filesystems=home:grand:eagle
#PBS -k doe
#PBS -j oe
#PBS -A hpe_dragon_collab
#PBS -q debug-scaling
#PBS -V

cd $PBS_O_WORKDIR
./run_driver.sh /eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted-big/tiny
./run_driver.sh /eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted-big/small
./run_driver.sh /eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted-big/med

