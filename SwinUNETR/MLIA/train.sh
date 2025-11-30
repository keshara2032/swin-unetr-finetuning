#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="SWIN-UNETR MLIA Train"
#SBATCH --error="./logs/job-%j-swin_train_script.err"
#SBATCH --output="./logs/job-%j-swin_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

DATA_PATH="/standard/mlia/Team_9/Segmentation_data"
PRETRAINED_DIR="/standard/mlia/Team_9/swin-unetr-finetuning/SwinUNETR/MLIA/pretrained_models"
PRETRAINED_PATH="fold1_f48_ep300_4gpu_dice0_9059/model.pt"
SPATIAL_DIM=2


module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda deactivate &&
conda activate egoems &&

python -u main.py --data_dir="$DATA_PATH" --spatial_dims=$SPATIAL_DIM --pretrained_dir="$PRETRAINED_DIR" --pretrained_model_name="$PRETRAINED_PATH" --logdir=mlia_run --save_checkpoint --val_every=5 --noamp --use_checkpoint --roi_x=96 --roi_y=96 --in_channels=1 --out_channels=4 &&

echo "Done" &&
exit
