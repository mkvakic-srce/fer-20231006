#PBS -q gpu
#PBS -l select=2:ngpus=1:ncpus=4

module load scientific/ray

cd ${PBS_O_WORKDIR:-""}

ray-launcher.sh tensorflow-ray-tune.py
