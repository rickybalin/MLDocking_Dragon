#!/bin/bash -l

# Set env
source /eagle/hpe_dragon_collab/csimpson/env.sh

# Setup
#cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)
#export FI_MR_CACHE_MAX_COUNT=0
export FI_CXI_ODP=1
export DRAGON_HSTA_NO_NET_CONFIG=1
export DRAGON_DEFAULT_SEG_SZ=34359738368
DATA_PATH=/eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted-big/tiny
#DATA_PATH=/eagle/hpe_dragon_collab/csimpson/ZINC-22-presorted/tiny
export DRIVER_PATH=/eagle/hpe_dragon_collab/csimpson/MLDocking_Dragon/Dragon/
export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH

PROCS_PER_NODE=32
MEM_PER_NODE=128
MANAGERS=1
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $PROCS_PER_NODE max. processes in Pool per Node
echo Mem per node $MEM_PER_NODE
echo Managers $MANAGERS

#DEBUG_STR="-l dragon_file=DEBUG -l actor_file=DEBUG"
DEBUG_STR=
#DEBUG_STR="-l DEBUG"
# Run
module list
echo $LD_LIBRARY_PATH
dragon-cleanup
#EXE="dragon ${DRIVER_PATH}/dragon_driver_sequential.py --managers_per_node=$MANAGERS --data_path=${DATA_PATH} --max_procs_per_node=$PROCS_PER_NODE --mem_per_node=$MEM_PER_NODE"
EXE="dragon $DEBUG_STR ${DRIVER_PATH}/dragon_driver_sequential.py \
--managers_per_node=$MANAGERS \
--data_path=${DATA_PATH} \
--max_procs_per_node=$PROCS_PER_NODE \
--mem_per_node=$MEM_PER_NODE"
#--data_dictionary_mem_fraction=0.9"

echo $EXE
${EXE}
