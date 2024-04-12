#!/bin/bash -l

qsub -N ml_docking_dragon_2 -q debug -l select=2:ncpus=64:ngpus=4 sub_driver.sh
qsub -N ml_docking_dragon_4 -q debug-scaling -l select=4:ncpus=64:ngpus=4 sub_driver.sh
qsub -N ml_docking_dragon_8 -q preemptable -l select=8:ncpus=64:ngpus=4 sub_driver.sh
qsub -N ml_docking_dragon_16 -q prod -l select=16:ncpus=64:ngpus=4 sub_driver.sh
qsub -N ml_docking_dragon_32 -q prod -l select=32:ncpus=64:ngpus=4 sub_driver.sh
qsub -N ml_docking_dragon_64 -q prod -l select=64:ncpus=64:ngpus=4 sub_driver.sh
qsub -N ml_docking_dragon_256 -q prod -l select=256:ncpus=64:ngpus=4 sub_driver.sh

