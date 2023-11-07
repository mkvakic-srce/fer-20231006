#PBS -q cpu-radionica
#PBS -l select=1:ncpus=16:mem=20GB

module load scientific/dask

cd ${PBS_O_WORKDIR:-""}

$IMAGE_PATH python threads.py
