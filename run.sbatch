#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=32000M

python3 training.py
