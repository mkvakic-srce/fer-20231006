#!/bin/bash

#PBS -q cpu
#PBS -l select=3:ncpus=16:mem=150GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh sklearn_dask_dask.py
