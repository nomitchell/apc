#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12  
#SBATCH --mem=64000M      
#SBATCH --time=00-00:05
#SBATCH --output=HPO-cifar-%A-%a.out
#SBATCH --mail-user=nmitc082@uottawa.ca
#SBATCH --mail-type=ALL

module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision torchattacks tqdm --no-index

# $SLURM_ARRAY_TASK_ID

python train_apc.py












