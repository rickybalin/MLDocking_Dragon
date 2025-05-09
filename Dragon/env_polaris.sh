#!/bin/bash

WDIR=/eagle/hpe_dragon_collab/csimpson

# Load modules
module use /soft/modulefiles
module load conda
conda activate base

# Load venv
. ${WDIR}/_dragon_env/bin/activate

# Still needed for v0.8
export MPICH_GPU_SUPPORT_ENABLED=0
