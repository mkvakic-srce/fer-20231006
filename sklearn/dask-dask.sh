#PBS -q cpu-radionica
#PBS -l select=2:ncpus=16:mem=80GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh dask-dask.py
