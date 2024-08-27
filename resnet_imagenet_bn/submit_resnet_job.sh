#!/bin/bash
#$ -N ResNet_ImageNet_bn
#$ -cwd
#$ -pe smp 4
#$ -l mem=30G
#$ -l gpu=1
#$ -l h_rt=48:00:00

# Ensure the environment modules system is initialized
source /etc/profile.d/modules.sh

# Load necessary modules
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python/3.9.6-gnu-10.2.0
module load cuda/11.2.0/gnu-10.2.0
module load cudnn/8.1.0.77/cuda-11.2
module load tensorflow/2.11.0/gpu

# Activate virtual environment
source ~/Scratch/resnet_imagenet_bn/resnet_env/bin/activate

# Run the Python script
python3 ~/Scratch/resnet_imagenet_bn/train_resnet_imagenet_bn.py

# Deactivate virtual environment
deactivate
