#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N dragon
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -A hpe_dragon_collab
##PBS -q prod
##PBS -q preemptable
#PBS -q debug-scaling
#PBS -V

# Set env
source /grand/hpe_dragon_collab/balin/env.sh
export DRAGON_DEFAULT_SEG_SZ=68719476736

# Setup
cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)
DATA_PATH_SMALL=/grand/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-small
DATA_PATH_MED=/grand/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-med
DATA_PATH_LARGE=/grand/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-large
MAX_PROCS=10
echo Running on $NODES nodes

# Loop over the different data loaders for the small data size
echo
echo ====================================
echo ====================================
echo LOOPING OVER THE SMALL PROBLEM SIZE
echo

echo SERIAL CASE
#python data_loader_serial.py --data_path=${DATA_PATH_SMALL}
echo

echo PYTHON MP CASES
for G in "directory" "file"
do
    python data_loader_pools.py --data_path=${DATA_PATH_SMALL} --mp_launch=spawn --max_procs=${MAX_PROCS} --granularity=${G} --validate=no
done
echo

echo DRAGON MP CASES
#for G in "directory" "directory_file" "file"
for G in "directory" "file"
do
    dragon data_loader_pools.py --data_path=${DATA_PATH_SMALL} --mp_launch=dragon --max_procs=${MAX_PROCS} --granularity=${G} --validate=no
done
echo
echo

# Loop over the different data loaders for the medium data size
echo
echo ====================================
echo ====================================
echo LOOPING OVER THE MEDIUM PROBLEM SIZE
echo

echo SERIAL CASE
#python data_loader_serial.py --data_path=${DATA_PATH_MED}
echo

echo PYTHON MP CASES
for G in "directory" "file"
do
    python data_loader_pools.py --data_path=${DATA_PATH_MED} --mp_launch=spawn --max_procs=${MAX_PROCS} --granularity=${G} --validate=no
done
echo

echo DRAGON MP CASES
#for G in "directory" "directory_file" "file"
for G in "directory" "file"
do
    dragon data_loader_pools.py --data_path=${DATA_PATH_MED} --mp_launch=dragon --max_procs=${MAX_PROCS} --granularity=${G} --validate=no
done
echo

# Loop over the different data loaders for the large data size
echo
echo ====================================
echo ====================================
echo LOOPING OVER THE LARGE PROBLEM SIZE
echo

echo SERIAL CASE
#python $EXE_PATH/data_loader_serial.py --data_path=${DATA_PATH_LARGE}
echo

echo PYTHON MP CASES
for G in "directory" "file"
do
    python $EXE_PATH/data_loader_pools.py --data_path=${DATA_PATH_LARGE} --mp_launch=spawn --max_procs=${MAX_PROCS} --granularity=${G} --validate=no
done
echo

echo DRAGON MP CASES
#for G in "directory" "directory_file" "file"
for G in "directory" "file"
do
    dragon $EXE_PATH/data_loader_pools.py --data_path=${DATA_PATH_LARGE} --mp_launch=dragon --max_procs=${MAX_PROCS} --granularity=${G} --validate=no
done
echo



