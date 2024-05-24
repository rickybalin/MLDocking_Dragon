#!/bin/bash -l
#PBS -S /bin/bash
###PBS -N ml_docking_dragon
#PBS -l walltime=00:30:00
###PBS -l select=2:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle:grand
#PBS -k doe
#PBS -j oe
#PBS -A hpe_dragon_collab
##PBS -q prod
##PBS -q preemptable
###PBS -q debug-scaling
#PBS -V

# Set env
source /grand/hpe_dragon_collab/csimpson/env.sh

# Setup
cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)



DATA_PATH=/lus/eagle/clone/g2/projects/hpe_dragon_collab/csimpson/ZINC-22-presorted-big/

DRIVER_PATH=./
PROCS_PER_NODE=60
MEM_PER_NODE=500
TIMEOUT=10
MANAGERS=1
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $PROCS_PER_NODE max. processes in Pool per Node

export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH

# Run
dragon -l DEBUG ${DRIVER_PATH}/dragon_driver.py --only_load_data=0 --managers_per_node=$MANAGERS --inf_dd_nodes=$NODES --data_path=${DATA_PATH}/tiny --max_procs_per_node=$PROCS_PER_NODE --mem_per_node=$MEM_PER_NODE
#dragon -l DEBUG ${DRIVER_PATH}/dragon_driver.py --only_load_data=1 --managers_per_node=$MANAGERS --inf_dd_nodes=$NODES --data_path=${DATA_PATH}/small --max_procs_per_node=$PROCS_PER_NODE --mem_per_node=$MEM_PER_NODE
#dragon ${DRIVER_PATH}/dragon_driver_loadonly.py --only_load_data=1 --managers_per_node=$MANAGERS --inf_dd_nodes=$NODES --data_path=${DATA_PATH}/med --max_procs_per_node=$PROCS_PER_NODE --mem_per_node=$MEM_PER_NODE


