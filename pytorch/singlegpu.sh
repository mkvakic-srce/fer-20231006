#PBS -q gpu-radionica
#PBS -l ngpus=1

module load scientific/pytorch

cd ${PBS_O_WORKDIR:-""}

run-singlegpu.sh singlegpu.py
