#!/bin/bash

#PBS -q gpu
#PBS -l ngpus=1

module load scientific/tensorflow

run-singlenode.sh tensorflow.py
