#PBS -q gpu-radionica
#PBS -l select=2:ngpus=1:ncpus=4

module load scientific/ray

cd ${PBS_O_WORKDIR:-""}

ray-launcher.sh train.py
