#!/bin/bash -l

# Determine which machine you are on
FULL_HOSTNAME=`hostname -f`
echo $FULL_HOSTNAME
if [[ *"americas"* == $FULL_HOSTNAME ]]; then
    SUNSPOT=1
    echo "Setting up for Sunspot run"
fi
if [[ *"polaris"* == $FULL_HOSTNAME ]]; then
    POLARIS=1
    echo "Setting up for Polaris run"
fi
if [[ *"sirius"* == $FULL_HOSTNAME ]]; then
    SIRIUS=1
    echo "Setting up for Sirius run"
fi

# Set env
if $SUNSPOT; then
    source /gila/Aurora_deployment/csimpson/hpe_dragon_collab/env.sh
fi
echo "Location of dragon:"
which dragon

# Setup
#cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)

# Dragon env vars
#export FI_MR_CACHE_MAX_COUNT=0
export FI_CXI_ODP=1
export DRAGON_HSTA_NO_NET_CONFIG=1
export DRAGON_DEFAULT_SEG_SZ=34359738368

# Machine specific path and resource settings
if $SUNSPOT; then
    export RECEPTOR_FILE=/gila/Aurora_deployment/dragon/receptor_files/3clpro_7bqy.oedu
    DATA_PATH=/gila/Aurora_deployment/dragon/data/tiny
    export DRIVER_PATH=/gila/Aurora_deployment/csimpson/hpe_dragon_collab/MLDocking_Dragon/Dragon/
    export GPU_DEVICES="0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1,4.0,4.1,5.0,5.1"
    export CPU_AFFINITY="list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"
    #"verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:48-51,152-159:56-63,160-167:64-71,168-175:72-79,176-183:80-87,184-191:88-95,192-199"
    export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1
fi

# Other machine specific env vars
# Some env vars from Archit's script
if $SUNSPOT; then
    export PLATFORM_NUM_GPU_TILES=2
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
    export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0
fi

export PYTHONPATH=$DRIVER_PATH:$PYTHONPATH

PROCS_PER_NODE=104
MEM_PER_NODE=128
MANAGERS=1
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $PROCS_PER_NODE max. processes in Pool per Node
echo Mem per node $MEM_PER_NODE
echo Managers $MANAGERS

# Set debug flag
#DEBUG_STR="-l dragon_file=DEBUG -l actor_file=DEBUG"
DEBUG_STR=
#DEBUG_STR="-l DEBUG"

# Run
module list
echo $LD_LIBRARY_PATH
dragon-cleanup
EXE="dragon $DEBUG_STR ${DRIVER_PATH}/dragon_driver_sequential.py \
--managers_per_node=$MANAGERS \
--data_path=${DATA_PATH} \
--max_procs_per_node=$PROCS_PER_NODE \
--mem_per_node=$MEM_PER_NODE"

echo $EXE
${EXE}
