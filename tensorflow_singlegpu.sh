#!/bin/bash

#PBS -q gpu
#PBS -l ngpus=1

module load scientific/tensorflow

cd ${PBS_O_WORKDIR:-""}

run-singlenode.sh tensorflow_singlegpu.py
