#!/bin/bash

module load cuda
module load cudatoolkit

export DOCKING_SIM_DUMMY=1

#export DATA_PATH=/lus/scratch/mendygra/alcf/MLDocking_Dragon/tiny
export DATA_PATH=/lus/scratch/wahlc/dragon/collabs/anl/data/cms-data/med
export DRIVER_PATH=/home/users/klee/home/Repos/fix/MLDocking_Dragon/Dragon/
export CUDNN_LIB_DIR=/lus/scratch/nhill/nccl-tests/cudnn-linux-x86_64-9.5.0.50_cuda12-archive/lib

export LD_LIBRARY_PATH=${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.0

export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH

export DRAGON_DEFAULT_SEG_SZ=34359738368

export FI_HMEM="none"

PROCS_PER_NODE=32
MEM_PER_NODE=200
MANAGERS=4
echo Reading files from $DATA_PATH
echo Running with $PROCS_PER_NODE max. processes in Pool per Node
echo Mem per node $MEM_PER_NODE
echo Managers $MANAGERS

#DEBUG_STR="-l dragon_file=DEBUG -l actor_file=DEBUG"
DEBUG_STR=
#DEBUG_STR="-l DEBUG"
# Run
dragon-cleanup
EXE="dragon $DEBUG_STR ${DRIVER_PATH}/dragon_driver_sequential.py --managers_per_node=$MANAGERS --data_path=${DATA_PATH} --max_procs_per_node=$PROCS_PER_NODE --mem_per_node=$MEM_PER_NODE"
# EXE="dragon -s $DEBUG_STR ${DRIVER_PATH}/dragon_driver_sequential.py \
# --managers_per_node=$MANAGERS \
# --data_path=${DATA_PATH} \
# --max_procs_per_node=$PROCS_PER_NODE \
# --mem_per_node=$MEM_PER_NODE"
echo $EXE
${EXE}

