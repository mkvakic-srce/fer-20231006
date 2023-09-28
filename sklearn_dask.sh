#PBS -q cpu
#PBS -l select=2:ncpus=8

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh sklearn_dask.py
