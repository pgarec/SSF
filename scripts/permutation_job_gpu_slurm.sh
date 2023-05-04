#!/bin/bash	

#SBATCH --job-name=mnist-job
#SBATCH --output=./output/mnist_result-%J.out
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00
#SBATCH --mem=5gb
#SBATCH --gres=gpu
#SBATCH --mail-user=polgarciarecasens@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

source ~/.bashrc
module load CUDA/11.8 CUDNN/8.6
conda activate myenv
python tests/clf_nn_mnist.py 

echo "Done: $(date +%F-%R:%S)"