#!/bin/bash
#SBATCH --job-name GTADfK
#SBATCH --array=1
#SBATCH --time=03:59:00
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem 100GB
##SBATCH --qos=ivul
#SBATCH -A conf-gpu-2020.11.23

module load cuda
# module load cudnn
module load miniconda

echo `hostname`
source activate /home/xum/miniconda3/envs/vlg

set -ex

lr=0.0005
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python gtad_train.py --training_lr ${lr}
python gtad_inference.py
python gtad_postprocess.py
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
