#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --mem 90G
#SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=mmvae

# module load py-venv-ml/nightly

source /mnt/projects/debruinz_project/pytorch-nightly-env/bin/activate
export PYTHONPATH=/mnt/home/taylcard/dev/lightning/sci-ml/src:$PYTHONPATH
cd /mnt/home/taylcard/dev/lightning/sci-ml/src/
python3 -m sciml.main fit --config /mnt/home/taylcard/dev/lightning/sci-ml/configs/adv_multi_modal_config.yaml 
