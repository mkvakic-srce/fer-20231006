#!/bin/bash

#PBS -q gpu
#PBS -l ngpus=1

module load scientific/pytorch

run-singlegpu.sh pytorch_singlegpu.py
