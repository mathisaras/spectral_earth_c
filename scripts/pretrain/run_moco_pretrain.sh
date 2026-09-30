#!/bin/bash
set -e

# Cap all BLAS/OMP/MKL thread pools to 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Configuration ---

# Dry run mode: set to true to print commands without executing them
DRY_RUN=false

# Number of nodes (1, 2, 4, etc.). Overrides trainer.num_nodes and hydra.launcher.nodes.
NUM_NODES=2

# Environment setting (juwels, jureca, or terrabyte)
ENV="terrabyte"  # Change this to "terrabyte" for terrabyte environment

# Sensors to pretrain on
SENSORS=("enmap" "s2" "lo")

# MoCo hyperparameters
BATCH_SIZE=32
MAX_EPOCHS=100
LEARNING_RATE=0.00001
TEMPERATURE=0.1
WEIGHT_DECAY=0.01
MEMORY_BANK_SIZE=32768
MOCO_MOMENTUM=0.996

# Logging / launcher
WANDB_PROJECT="ssl_pretrain_spectral_earth_mm"

# Environment-specific launcher and paths configuration
if [ "${ENV}" = "juwels" ]; then
  LAUNCHER="slurm_juwels_long"
  PATHS_PROFILE="juwels"
  WANDB_OFFLINE="True"
elif [ "${ENV}" = "jureca" ]; then
  LAUNCHER="slurm_jureca_long"
  PATHS_PROFILE="jureca"
  WANDB_OFFLINE="True"
elif [ "${ENV}" = "terrabyte" ]; then
  LAUNCHER="slurm_long"
  PATHS_PROFILE="default"
  WANDB_OFFLINE="False"
else
  echo "Error: ENV must be one of 'juwels', 'jureca', or 'terrabyte'. Current value: ${ENV}"
  exit 1
fi

# Trainer settings
PRECISION="16-mixed"
STRATEGY="ddp"

# --- Sensor configuration mappings ---
# Patch sizes: 4 for 30m resolution sensors, 12 for 10m resolution sensors
declare -A PATCH_SIZES=(
  ["enmap"]="4"   # 30m resolution
  ["s2"]="12"     # 10m resolution
  ["lo"]="4"      # 30m resolution
)

# Backbones: ENMAP gets both vit_b and spec_vit_b, others only get vit_b
declare -A BACKBONES=(
  ["enmap"]="vit_b_timm spec_vit_b"
  ["s2"]="vit_b_timm"
  ["lo"]="vit_b_timm"
)

# --- Script Logic ---

BASE_CMD="python src/train.py --config-name=pretrain -m"
JOB_COUNT=0

if [ "${DRY_RUN}" = true ]; then
  echo "=========================================="
  echo "  DRY RUN MODE - Commands will be printed"
  echo "=========================================="
  echo ""
fi

echo "=========================================="
echo "  MoCo Pretraining Configuration"
echo "=========================================="
echo "Environment: ${ENV}"
echo "Nodes: ${NUM_NODES}"
echo "Sensors: ${SENSORS[*]}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "Temperature: ${TEMPERATURE}"
echo "Memory Bank Size: ${MEMORY_BANK_SIZE}"
echo "=========================================="
echo ""

for sensor in "${SENSORS[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Sensor: ${sensor^^}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  # Get patch size for this sensor
  PATCH_SIZE="${PATCH_SIZES[${sensor}]}"
  echo "  Patch Size: ${PATCH_SIZE}"
  
  # Get backbones for this sensor
  BACKBONES_STR="${BACKBONES[${sensor}]}"
  echo "  Backbones: ${BACKBONES_STR}"
  echo ""
  
  for backbone in ${BACKBONES_STR}; do
    JOB_COUNT=$((JOB_COUNT + 1))
    
    # Convert sensor to uppercase for data.sensors array
    SENSOR_UPPER=$(echo "${sensor}" | tr '[:lower:]' '[:upper:]')
    
    # Task naming: ssl_pretrain/spectral_earth_mm/${sensor}/moco/${backbone}
    task_name="ssl_pretrain/spectral_earth_mm/${sensor}/moco/${backbone}"
    # Subdir: concise single-level directory with key params not in task_name (lr, temperature, batch_size)
    subdir="lr${LEARNING_RATE}_t${TEMPERATURE}_bs${BATCH_SIZE}"
    
    # Build command
    cmd="${BASE_CMD} \
      experiment=pretrain/moco/base \
      trainer=ddp_ssl \
      trainer.num_nodes=${NUM_NODES} \
      sensor=${sensor} \
      backbone@model.backbone_config=${backbone} \
      backbone_name=${backbone} \
      sensor@model.sensor_config=${sensor} \
      model.backbone_config.patch_size=${PATCH_SIZE} \
      model.lr=${LEARNING_RATE} \
      model.temperature=${TEMPERATURE} \
      model.weight_decay=${WEIGHT_DECAY} \
      model.memory_bank_size=${MEMORY_BANK_SIZE} \
      model.moco_momentum=${MOCO_MOMENTUM} \
      data.batch_size=${BATCH_SIZE} \
      data.sensors=[\"${SENSOR_UPPER}\"] \
      data.num_temporal_views=2 \
      data.normalize=false \
      data.spatial_aug_p=0 \
      data.radiometric_aug_p=0 \
      trainer.max_epochs=${MAX_EPOCHS} \
      trainer.precision=${PRECISION} \
      trainer.strategy=${STRATEGY} \
      logger.wandb.offline=${WANDB_OFFLINE} \
      logger.wandb.project=${WANDB_PROJECT} \
      paths=${PATHS_PROFILE} \
      hydra/launcher=${LAUNCHER} \
      hydra.launcher.nodes=${NUM_NODES} \
      hydra.sweep.subdir=${subdir} \
      task_name=${task_name}"
    
    # Print formatted command
    echo "  [Job ${JOB_COUNT}] ${sensor^^} | ${backbone}"
    echo "  Task: ${task_name}"
    if [ "${DRY_RUN}" = true ]; then
      echo "  ┌─────────────────────────────────────────────────────────────"
      echo "${cmd}" | sed 's/^/  │ /'
      echo "  └─────────────────────────────────────────────────────────────"
    else
      echo "  → Submitting..."
      eval ${cmd} &
    fi
    echo ""
  done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "${DRY_RUN}" = true ]; then
  echo "  DRY RUN COMPLETE"
  echo "  Total jobs that would be submitted: ${JOB_COUNT}"
else
  echo "  All MoCo pretraining jobs submitted"
  echo "  Total jobs submitted: ${JOB_COUNT}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
