#!/bin/bash
set -e

# Cap all BLAS/OMP/MKL thread pools to 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Configuration ---

# Dry run mode: set to true to print commands without executing them
DRY_RUN=false

# Resume mode: when true, search for existing checkpoints and only resubmit runs that
# stopped before completion (epoch < max_epochs - 1). Completed runs are skipped.
RESUME=false

# Number of nodes (1, 2, 4, etc.). Overrides trainer.num_nodes and hydra.launcher.nodes.
NUM_NODES=1

# Environment setting (juwels, jureca, or terrabyte)
ENV="terrabyte"  # Change this to "terrabyte" for terrabyte environment

# Sensors to pretrain on
SENSORS=("enmap")

# MAE hyperparameters/sweeps
MASK_RATIOS=("0.9")

# Normalize patch pixels before loss (try both for remote sensing)
NORM_PIX_OPTIONS=("true" "false")

# Decoder depth (MAE decoder transformer blocks)
DECODER_DEPTH_OPTIONS=(1 4 8)
BATCH_SIZE=64
MAX_EPOCHS=200
LEARNING_RATE=0.00001
WEIGHT_DECAY=0.01

# Logging / launcher
WANDB_PROJECT="ssl_pretrain_spectral_earth_mm"

# Environment-specific launcher and paths configuration
if [ "${ENV}" = "juwels" ]; then
  LAUNCHER="slurm_juwels_long"
  PATHS_PROFILE="juwels"
  WANDB_OFFLINE="True"
  LOG_DIR_BASE="${PWD}/logs/ssl_pretrain/spectral_earth_mm"
elif [ "${ENV}" = "jureca" ]; then
  LAUNCHER="slurm_jureca_long"
  PATHS_PROFILE="jureca"
  WANDB_OFFLINE="True"
  LOG_DIR_BASE="${PWD}/logs/ssl_pretrain/spectral_earth_mm"
elif [ "${ENV}" = "terrabyte" ]; then
  LAUNCHER="slurm_long"
  PATHS_PROFILE="default"
  WANDB_OFFLINE="False"
  LOG_DIR_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/spectral_earth_mm_experiments/logs/ssl_pretrain/spectral_earth_mm"
else
  echo "Error: ENV must be one of 'juwels', 'jureca', or 'terrabyte'. Current value: ${ENV}"
  exit 1
fi

# Trainer settings
PRECISION="16-mixed"
STRATEGY="ddp_find_unused_parameters_true"

# --- Sensor configuration mappings ---
# Patch sizes: 4 for 30m resolution sensors, 12 for 10m resolution sensors
declare -A PATCH_SIZES=(
  ["enmap"]="4"   # 30m resolution
  ["s2"]="12"     # 10m resolution
  ["lo"]="4"      # 30m resolution
)

# Backbones: ENMAP gets both vit_b and spec_vit_b, others only get vit_b
declare -A BACKBONES=(
  ["enmap"]="spec_vit_b"
  ["s2"]="vit_b_timm"
  ["lo"]="vit_b_timm"
)

# --- Resume Logic ---
find_resume_checkpoint() {
  local log_base="$1" task_name="$2" max_epochs="$3" subdir_filter="${4:-}"
  local search_dir="${log_base}/${task_name}"
  [ ! -d "$search_dir" ] && echo "" && return 0
  local max_epoch=-1 best_ckpt=""
  while IFS= read -r ckpt; do
    [ -z "$ckpt" ] && continue
    [ -n "$subdir_filter" ] && [[ "$ckpt" != *"/${subdir_filter}/"* ]] && continue
    epoch=$(basename "$ckpt" .ckpt | sed 's/epoch_0*//')
    if [ -n "$epoch" ] && [ "$epoch" -ge 0 ] 2>/dev/null && [ "$epoch" -gt "$max_epoch" ] 2>/dev/null; then
      max_epoch=$epoch && best_ckpt="$ckpt"
    fi
  done < <(find "$search_dir" -type f -name "epoch_*.ckpt" 2>/dev/null)
  [ -z "$best_ckpt" ] && echo "" && return 0
  [ "$max_epoch" -ge "$((max_epochs - 1))" ] 2>/dev/null && echo "SKIP" && return 0
  echo "$best_ckpt"
}

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
echo "  MAE Pretraining Configuration"
echo "=========================================="
echo "Environment: ${ENV}"
echo "Resume mode: ${RESUME}"
echo "Nodes: ${NUM_NODES}"
echo "Sensors: ${SENSORS[*]}"
echo "Mask Ratios: ${MASK_RATIOS[*]}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
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
    for mr in "${MASK_RATIOS[@]}"; do
      for norm_pix in "${NORM_PIX_OPTIONS[@]}"; do
        for decoder_depth in "${DECODER_DEPTH_OPTIONS[@]}"; do

      np_tag=$([ "$norm_pix" = "true" ] && echo "np1" || echo "np0")

      # Convert sensor to uppercase for data.sensors array
      SENSOR_UPPER=$(echo "${sensor}" | tr '[:lower:]' '[:upper:]')
      
      # Task naming: ssl_pretrain/spectral_earth_mm/${sensor}/mae_${mask_ratio}/${backbone}
      task_name="ssl_pretrain/spectral_earth_mm/${sensor}/mae_${mr}_${np_tag}_dd${decoder_depth}/${backbone}"
      task_search="${sensor}/mae_${mr}_${np_tag}_dd${decoder_depth}/${backbone}"
      # Subdir: concise single-level directory with key params not in task_name (lr, batch_size)
      subdir="lr${LEARNING_RATE}_bs${BATCH_SIZE}"

      # Resume: skip completed runs; add ckpt_path for incomplete runs
      CKPT_OVERRIDE=""
      if [ "${RESUME}" = true ]; then
        CKPT_RESULT=$(find_resume_checkpoint "${LOG_DIR_BASE}" "${task_search}" "${MAX_EPOCHS}" "${subdir}")
        if [ "$CKPT_RESULT" = "SKIP" ]; then
          echo "  [Skip] ${sensor^^} | ${backbone} | mask_ratio=${mr} | np=${norm_pix} | dd=${decoder_depth} (complete)"
          continue
        elif [ -n "$CKPT_RESULT" ]; then
          CKPT_OVERRIDE="ckpt_path=\"${CKPT_RESULT}\""
        fi
      fi

      JOB_COUNT=$((JOB_COUNT + 1))
      
      # Build command
      cmd="${BASE_CMD} \
        experiment=pretrain/mae/base \
        trainer=ddp_ssl \
        trainer.num_nodes=${NUM_NODES} \
        sensor=${sensor} \
        backbone@model.backbone_config=${backbone} \
        backbone_name=${backbone} \
        sensor@model.sensor_config=${sensor} \
        model.backbone_config.patch_size=${PATCH_SIZE} \
        model.norm_pix=${norm_pix} \
        model.decoder_depth=${decoder_depth} \
        model.mask_ratio=${mr} \
        model.lr=${LEARNING_RATE} \
        model.weight_decay=${WEIGHT_DECAY} \
        model.T_max=${MAX_EPOCHS} \
        data.batch_size=${BATCH_SIZE} \
        data.sensors=[\"${SENSOR_UPPER}\"] \
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
      [ -n "${CKPT_OVERRIDE}" ] && cmd="${cmd} ${CKPT_OVERRIDE}"
      
      # Print formatted command
      echo "  [Job ${JOB_COUNT}] ${sensor^^} | ${backbone} | mask_ratio=${mr} | np=${norm_pix} | dd=${decoder_depth}"
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
    done
  done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "${DRY_RUN}" = true ]; then
  echo "  DRY RUN COMPLETE"
  echo "  Total jobs that would be submitted: ${JOB_COUNT}"
else
  echo "  All MAE pretraining jobs submitted"
  echo "  Total jobs submitted: ${JOB_COUNT}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
