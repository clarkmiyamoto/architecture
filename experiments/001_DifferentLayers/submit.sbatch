#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=${nodes}n_${layers}l
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_${nodes}n_${layers}l_%j.out
#SBATCH --error=slurm_${nodes}n_${layers}l_%j.err

module purge

singularity exec --nv \
  --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; python run.py --nodes 10 --layer 3"