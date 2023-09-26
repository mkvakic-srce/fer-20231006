#!/bin/bash

#PBS -q gpu
#PBS -l ngpus=1

module load scientific/ray

cd ${PBS_O_WORKDIR:-""}

ray-launcher.sh pytorch_ray_train.py
