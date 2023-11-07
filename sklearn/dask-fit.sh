#PBS -q cpu-radionica
#PBS -l select=2:ncpus=8:mem=10GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh dask-fit.py
