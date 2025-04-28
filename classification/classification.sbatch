#!/bin/bash
#SBATCH -o /ictstr01/home/aih/jan.boada/project/codes/results/slurm_out_%j.job
#SBATCH -e /ictstr01/home/aih/jan.boada/project/codes/results/slurm_error_%j.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH --nice=10000

source $HOME/.bashrc

### Helping Guide : https://bioinformatics_core.ascgitlab.helmholtz-muenchen.de/it_hpc_documentation/index.html 

conda activate my_env

# Variables de configuración
DATAROOT=/home/aih/jan.boada/project/codes/classification
DATASET_SELECTION=matek
OUTPUT_DIR=/home/aih/jan.boada/project/codes/results/classification
CLIP_DIRECTORY=/home/aih/jan.boada/project/codes/classification/models/clip.py
FOLD=4
IS_HSV=True
IS_HED=True
IS_RAND_AUG=True
IS_MIX_AUG=True
IS_SYNTH_TRAIN=True
IS_POOLED_FEWSHOT=False
IS_DATASET_WISE=False
MODEL_TYPE=clip
DATASET=matek
EPOCHS=10
WARMUP_EPOCHS=3
LR=1e-4
WD=1e-4
MIN_LR=1e-8
BATCH_SIZE=64
BATCH_SIZE_EVAL=8
DATADREAM_LR=1e-4
DATADREAM_EPOCH=200
N_SHOT=16
FEWSHOT_SEED=seed0
N_IMG_PER_CLS=500
SD_VERSION="sd2.1"
N_TEMPLATE=1
GUIDANCE_SCALE=2.0
LOG=wandb

# Mensajes de depuración
echo "=== INICIO DEL JOB ==="
echo "DATAROOT: $DATAROOT"
echo "DATASET_SELECTION: $DATASET_SELECTION"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "FOLD: $FOLD"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "IS_HSV: $IS_HSV"
echo "IS_HED: $IS_HED"
echo "IS_RAND_AUG: $IS_RAND_AUG"
echo "IS_MIX_AUG: $IS_MIX_AUG"
echo "IS_SYNTH_TRAIN: $IS_SYNTH_TRAIN"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "=== EJECUTANDO MAIN.PY ==="

# Ejecutar el script principal
CUDA_VISIBLE_DEVICES=$GPU python main.py \
--model_type=$MODEL_TYPE \
--dataset=$DATASET \
--output_dir=$OUTPUT_DIR \
--dataroot=$DATAROOT \
--dataset_selection=$DATASET_SELECTION \
--fold=$FOLD \
--is_hsv=$IS_HSV \
--is_hed=$IS_HED \
--is_rand_aug=$IS_RAND_AUG \
--is_mix_aug=$IS_MIX_AUG \
--is_synth_train=$IS_SYNTH_TRAIN \
--is_pooled_fewshot=$IS_POOLED_FEWSHOT \
--is_dataset_wise=$IS_DATASET_WISE \
--datadream_lr=$DATADREAM_LR \
--datadream_epoch=$DATADREAM_EPOCH \
--n_shot=$N_SHOT \
--fewshot_seed=$FEWSHOT_SEED \
--n_img_per_cls=$N_IMG_PER_CLS \
--sd_version=$SD_VERSION \
--n_template=$N_TEMPLATE \
--guidance_scale=$GUIDANCE_SCALE \
--epochs=$EPOCHS \
--warmup_epochs=$WARMUP_EPOCHS \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--batch_size=$BATCH_SIZE \
--batch_size_eval=$BATCH_SIZE_EVAL \
--log=$LOG

# Mensaje al finalizar
echo "=== MAIN.PY FINALIZADO ==="
echo "=== FIN DEL JOB ==="