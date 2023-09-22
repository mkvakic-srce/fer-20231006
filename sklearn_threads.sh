#!/bin/bash

#PBS -q cpu
#PBS -l select=1:ncpus=16:mem=50GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

$IMAGE_PATH python sklearn_threads.py
