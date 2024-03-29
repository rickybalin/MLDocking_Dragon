#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N dragon
#PBS -l walltime=00:30:00
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
source /lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/env.sh
export DRAGON_DEFAULT_SEG_SZ=68719476736

# Setup
cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)
DATA_PATH_SMALL=/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-small
DATA_PATH_MED=/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC_22/zinc-22-2d/2d-med
echo Running on $NODES nodes

# Loop over the different data loaders for the small data size
echo
echo ====================================
echo ====================================
echo LOOPING OVER THE SMALL PROBLEM SIZE
echo

echo SERIAL CASE
python data_loader_serial.py --data_path=${DATA_PATH_SMALL}
echo

echo PYTHON MP CASES
for G in "directory" "directory_file" "file"
do
    python data_loader_mp.py --data_path=${DATA_PATH_SMALL} --mp_launch=spawn --granularity=${G}
done
echo

echo DRAGON MP CASES
for G in "directory" "directory_file" "file"
do
    dragon data_loader_mp.py --data_path=${DATA_PATH_SMALL} --mp_launch=dragon --granularity=${G}
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
python data_loader_serial.py --data_path=${DATA_PATH_MED}
echo

echo PYTHON MP CASES
for G in "directory" "directory_file" "file"
do
    python data_loader_mp.py --data_path=${DATA_PATH_MED} --mp_launch=spawn --granularity=${G}
done
echo

echo DRAGON MP CASES
for G in "directory" "directory_file" "file"
do
    dragon data_loader_mp.py --data_path=${DATA_PATH_MED} --mp_launch=dragon --granularity=${G}
done
echo