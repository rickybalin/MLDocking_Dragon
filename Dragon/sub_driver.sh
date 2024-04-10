#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N ml_docking_dragon
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
source /lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/env_st.sh
export DRAGON_DEFAULT_SEG_SZ=68719476736

# Setup
cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)
DATA_PATH=/lus/eagle/clone/g2/projects/hpe_dragon_collab/balin/ZINC-22-2D-smaller_files
MAX_PROCS=10
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $MAX_PROCS max. processes in Pool


dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH} --max_procs=$MAX_PROCS

