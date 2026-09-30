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
NUM_NODES=2

# Environment setting (juwels, jureca, or terrabyte)
ENV="terrabyte"  # Change this to "terrabyte" for terrabyte environment

# Sensors to pretrain on
SENSORS=("enmap")

# DINOv2 hyperparameters (critical ones to tune)
BATCH_SIZE=32
MAX_EPOCHS=100
LEARNING_RATE=0.0001
MIN_LR=0.000001
WARMUP_EPOCHS=10
WEIGHT_DECAY=0.001
WEIGHT_DECAY_END=0.4
N_LOCAL_VIEWS=4
FREEZE_LAST_LAYER_EPOCHS=1

# Optional: sweep over n_local_views (e.g. 4, 6, 8). Use a single value for one job per (sensor, backbone).
# N_LOCAL_VIEWS_LIST=("4" "6" "8")
N_LOCAL_VIEWS_LIST=("4")

# Teacher temp schedule (steps); paper uses ~30k. Uncomment to override.
# TEACHER_TEMP_WARMUP_STEPS=30000

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
STRATEGY="ddp"

# --- Sensor configuration mappings ---
# Patch sizes: 4 for 30m resolution sensors, 12 for 10m resolution sensors
declare -A PATCH_SIZES=(
  ["enmap"]="4"   # 30m resolution
  ["s2"]="12"     # 10m resolution
  ["lo"]="4"      # 30m resolution
)

# Backbones: ENMAP gets both vit_b_timm and spec_vit_b, others only get vit_b_timm
declare -A BACKBONES=(
  ["enmap"]="vit_b_timm spec_vit_b"
  ["s2"]="vit_b_timm"
  ["lo"]="vit_b_timm"
)

# --- Resume Logic ---
# Optional 4th arg: subdir filter (only consider checkpoints under paths containing /subdir/)
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
echo "  DINOv2 Pretraining Configuration"
echo "=========================================="
echo "Environment: ${ENV}"
echo "Resume mode: ${RESUME}"
echo "Nodes: ${NUM_NODES}"
echo "Sensors: ${SENSORS[*]}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "LR: ${LEARNING_RATE} (min_lr: ${MIN_LR})"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "Weight Decay: ${WEIGHT_DECAY} -> ${WEIGHT_DECAY_END}"
echo "N Local Views: ${N_LOCAL_VIEWS_LIST[*]}"
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
    for n_local in "${N_LOCAL_VIEWS_LIST[@]}"; do

      # Convert sensor to uppercase for data.sensors array
      SENSOR_UPPER=$(echo "${sensor}" | tr '[:lower:]' '[:upper:]')
      
      # Task naming: ssl_pretrain/spectral_earth_mm/${sensor}/dino_v2/${backbone}
      # Must match pattern used in configs/experiment/pretrain/dino_v2/base.yaml for logger/grouping
      task_name="ssl_pretrain/spectral_earth_mm/${sensor}/dino_v2/${backbone}"
      task_search="${sensor}/dino_v2/${backbone}"
      # Subdir: key varying params so multirun outputs don't collide (hydra.sweep.subdir)
      subdir="lr${LEARNING_RATE}_wu${WARMUP_EPOCHS}_nlv${n_local}_bs${BATCH_SIZE}"

      # Resume: skip completed runs; add ckpt_path for incomplete runs
      CKPT_OVERRIDE=""
      if [ "${RESUME}" = true ]; then
        CKPT_RESULT=$(find_resume_checkpoint "${LOG_DIR_BASE}" "${task_search}" "${MAX_EPOCHS}" "${subdir}")
        if [ "$CKPT_RESULT" = "SKIP" ]; then
          echo "  [Skip] ${sensor^^} | ${backbone} | n_local_views=${n_local} (complete)"
          continue
        elif [ -n "$CKPT_RESULT" ]; then
          CKPT_OVERRIDE="ckpt_path=\"${CKPT_RESULT}\""
        fi
      fi

      JOB_COUNT=$((JOB_COUNT + 1))
      
      # Build command
      cmd="${BASE_CMD} \
        experiment=pretrain/dino_v2/base \
        trainer=ddp_ssl \
        trainer.num_nodes=${NUM_NODES} \
        sensor=${sensor} \
        backbone@model.backbone_config=${backbone} \
        backbone_name=${backbone} \
        sensor@model.sensor_config=${sensor} \
        model.backbone_config.patch_size=${PATCH_SIZE} \
        model.lr=${LEARNING_RATE} \
        model.min_lr=${MIN_LR} \
        model.warmup_epochs=${WARMUP_EPOCHS} \
        model.weight_decay=${WEIGHT_DECAY} \
        model.weight_decay_end=${WEIGHT_DECAY_END} \
        model.n_local_views=${n_local} \
        model.freeze_last_layer_epochs=${FREEZE_LAST_LAYER_EPOCHS} \
        model.max_epochs=${MAX_EPOCHS} \
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
      [ -n "${CKPT_OVERRIDE}" ] && cmd="${cmd} ${CKPT_OVERRIDE}"
      
      # Optional: override teacher temp warmup steps if set
      if [ -n "${TEACHER_TEMP_WARMUP_STEPS:-}" ]; then
        cmd="${cmd} \
        model.teacher_temp_warmup_steps=${TEACHER_TEMP_WARMUP_STEPS}"
      fi
      
      # Print formatted command
      echo "  [Job ${JOB_COUNT}] ${sensor^^} | ${backbone} | n_local_views=${n_local}"
      echo "  Task: ${task_name}"
      echo "  Subdir: ${subdir}"
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "${DRY_RUN}" = true ]; then
  echo "  DRY RUN COMPLETE"
  echo "  Total jobs that would be submitted: ${JOB_COUNT}"
else
  echo "  All DINOv2 pretraining jobs submitted"
  echo "  Total jobs submitted: ${JOB_COUNT}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
