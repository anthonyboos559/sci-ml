#!/bin/bash

log_dir="/mnt/projects/debruinz_project/tony_boos/slurm_logs"
dir_path="/mnt/projects/debruinz_project/tony_boos/sci-ml/src/sciml"

# Extracting filename without directory path
file_in=$(basename "$1")
# Stripping off file extension
filename="${file_in%.*}"

# Create sbatch script
sbatch_script="sbatch_job.sh"

cat > $sbatch_script <<EOL
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --mem 90G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=boosan@mail.gvsu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=$filename
#SBATCH --output=$log_dir/$filename.%j.out
#SBATCH --error=$log_dir/$filename.%j.err

# module load py-venv-ml/nightly

source /mnt/projects/debruinz_project/pytorch-nightly-env/bin/activate
export PYTHONPATH=/mnt/projects/debruinz_project/tony_boos/sci-ml/src:$PYTHONPATH
cd /mnt/projects/debruinz_project/tony_boos/sci-ml/src/
python3 -m sciml.main fit --config /mnt/projects/debruinz_project/tony_boos/sci-ml/configs/multi_modal_config.yaml
# python3 -m sciml.main test --config /mnt/projects/debruinz_project/tony_boos/sci-ml/configs/multi_modal_config.yaml --ckpt_path /mnt/projects/debruinz_project/tensorboard_logs/tony_boos/New_MMVAE/cross_gen_test_3/checkpoints/epoch=29-step=2439840.ckpt
EOL

sbatch $sbatch_script



