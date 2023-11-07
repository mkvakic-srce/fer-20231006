#PBS -q cpu-radionica
#PBS -l select=4:ncpus=8:mem=40GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh dask-dask.py
