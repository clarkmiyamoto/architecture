from torchvision import datasets, transforms

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./', train=True, download=True)
test_dataset = datasets.MNIST(root='./', train=False, download=True)

### How to run on HPC
'''
srun --pty /bin/bash

singularity exec --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash

source /ext3/env.sh
'''