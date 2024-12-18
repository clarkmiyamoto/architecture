# Read and write to Singularity Env
```
singularity exec \
  --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash

source /ext3/env.sh
```

# Activate (Read Only)
```
singularity exec --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash

source /ext3/env.sh
```
# Activate & Path to Imagenet (Read only)
```
singularity exec --overlay /scratch/cm6627/diffeo_cnn/my_env/overlay-15GB-500K.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash

source /ext3/env.sh
```
First line activates singularity environment.
Second line actviates conda in the singularity environment