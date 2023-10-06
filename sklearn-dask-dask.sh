#PBS -q cpu
#PBS -l select=2:ncpus=16:mem=100GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

dask-launcher.sh sklearn-dask-dask.py
