#PBS -q gpu-radionica
#PBS -l select=2:ngpus=1

module load scientific/pytorch

cd ${PBS_O_WORKDIR:-""}

torchrun-multinode.sh torchrun.py
