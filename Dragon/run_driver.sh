#!/bin/bash -l

# Set env
source /grand/hpe_dragon_collab/csimpson/env.sh

# Setup
#cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)

DATA_PATH=/grand/hpe_dragon_collab/test_dataset/data/med
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
dragon -l DEBUG ${DRIVER_PATH}/dragon_driver.py --managers_per_node=$MANAGERS --num_nodes=$NODES --data_path=${DATA_PATH} --max_procs_per_node=$PROCS_PER_NODE --mem_per_node=$MEM_PER_NODE
