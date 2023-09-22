#!/bin/bash

#PBS -q gpu
#PBS -l ngpus=1

module load scientific/pytorch

cd ${PBS_O_WORKDIR:-""}

run-singlegpu.sh pytorch.py
