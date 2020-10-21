#!/bin/bash
#SBATCH -n 2                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p zickler          # Partition to submit to
#SBATCH --mem=10000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1        # Get GPU
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -o output_%j.out  # File to which STDOUT will be written, %j inserts $
#SBATCH -e errors_%j.err  # File to which STDERR will be written, %j inserts $
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kathrynheal@g.harvard.edu
#SBATCH --job-name=train

module load Anaconda3/5.0.1-fasrc01
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01
#conda create -n tf_env python=3.6 numpy six wheel animate
source activate tf_env 
#conda install scikit-learn
#conda install -c conda-forge matplotlib
#conda install ffmpeg

python -W ignore Training.py