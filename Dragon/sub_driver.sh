#!/bin/bash -l
#PBS -S /bin/bash
###PBS -N ml_docking_dragon
#PBS -l walltime=00:30:00
###PBS -l select=2:ncpus=64:ngpus=4
#PBS -l filesystems=home:eagle:grand
#PBS -k doe
###PBS -j oe
#PBS -A hpe_dragon_collab
##PBS -q prod
##PBS -q preemptable
###PBS -q debug-scaling
#PBS -V

# Set env
source /lus/eagle/clone/g2/projects/hpe_dragon_collab/csimpson/env.sh
export DRAGON_DEFAULT_SEG_SZ=68719476736

# Setup
cd $PBS_O_WORKDIR
NODES=$(cat $PBS_NODEFILE | wc -l)
MAX_PROCS=16

DATA_PATH=/lus/eagle/clone/g2/projects/hpe_dragon_collab/csimpson/ZINC-22-presorted/
TOT_MEM_SIZE=3
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $MAX_PROCS max. processes in Pool
echo Running with $TOT_MEM_SIZE GB for dictionary
echo

#dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/tiny --max_procs_per_node=$MAX_PROCS
dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/small --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE
dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/med --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE
dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/large --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE


DATA_PATH=/lus/eagle/clone/g2/projects/hpe_dragon_collab/csimpson/ZINC-22-presorted-big/
TOT_MEM_SIZE=100
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $MAX_PROCS max. processes in Pool
echo Running with $TOT_MEM_SIZE GB for dictionary
echo

dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/tiny --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE
dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/small --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE
dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH}/med --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE


DATA_PATH=/lus/eagle/clone/g2/projects/hpe_dragon_collab/avasan/ZINC-22-2D-smaller_files/
TOT_MEM_SIZE=2000
echo Running on $NODES nodes
echo Reading files from $DATA_PATH
echo Running with $MAX_PROCS max. processes in Pool
echo Running with $TOT_MEM_SIZE GB for dictionary
echo

dragon dragon_driver.py --num_nodes=$NODES --data_path=${DATA_PATH} --max_procs_per_node=$MAX_PROCS --total_mem_size=$TOT_MEM_SIZE
