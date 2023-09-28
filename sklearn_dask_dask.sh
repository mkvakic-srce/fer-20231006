#PBS -q cpu
#PBS -l select=2:ncpus=16:mem=50GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh sklearn_dask_dask.py
