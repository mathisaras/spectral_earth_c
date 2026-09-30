#!/bin/bash
set -e

# Cap all BLAS/OMP/MKL thread pools to 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Configuration ---

# Dry run mode: set to true to print commands without executing them
DRY_RUN=true

# Resume mode: when true, search for existing checkpoints and only resubmit runs that
# stopped before completion (epoch < max_epochs - 1). Completed runs are skipped.
RESUME=false

# Number of nodes (1, 2, 4, etc.). Overrides trainer.num_nodes and hydra.launcher.nodes.
NUM_NODES=1

# Environment setting (juwels, jureca, or terrabyte)
ENV="terrabyte"  # Change this to "terrabyte" for terrabyte environment

# Sensors to pretrain on
SENSORS=("enmap")

# I-JEPA hyperparameters
BATCH_SIZE=64
MAX_EPOCHS=100
LEARNING_RATE=0.00001
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=10


# I-JEPA Mask Scale Sweeps (critical hyperparameters)
# Format: "enc_min,enc_max,pred_min,pred_max"
# Edit this list to sweep over different configs or use single default
MASK_SCALES=(
  "0.85,1.0,0.15,0.4"   # Default config
  #"0.75,1.0,0.15,0.4"   # More encoder context
  #"0.9,1.0,0.15,0.4"    # Less encoder context (harder)
  #"0.85,1.0,0.15,0.5"   # Larger predictor targets
  #"0.85,1.0,0.2,0.5"    # Larger pred, higher min
  #"0.75,1.0,0.15,0.5"   # More enc + larger pred
)

# Optional VICReg anti-collapse regularization
VICREG_ENABLED="false"
VICREG_WEIGHT=0.0001
VICREG_LAMBDA_PARAM=0.0
VICREG_MU_PARAM=25.0
VICREG_NU_PARAM=0.5
VICREG_GATHER_DISTRIBUTED="false"
VICREG_EPS=0.0001
VICREG_MAX_SAMPLES=8192
VICREG_APPLY_ON="projector"  # projector | z_pred
VICREG_PROJECTOR_HIDDEN_DIM=2048
VICREG_PROJECTOR_OUTPUT_DIM=2048
VICREG_PROJECTOR_NUM_LAYERS=2

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

# Backbones: ENMAP gets both vit_b_timm and spec_vit_b, others only get vit_b_timm
declare -A BACKBONES=(
  ["enmap"]="vit_b_timm spec_vit_b"
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
echo "  I-JEPA Pretraining Configuration"
echo "=========================================="
echo "Environment: ${ENV}"
echo "Resume mode: ${RESUME}"
echo "Nodes: ${NUM_NODES}"
echo "Sensors: ${SENSORS[*]}"
echo "Mask Scale Configs: ${#MASK_SCALES[@]}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "VICReg: ${VICREG_ENABLED} (apply_on=${VICREG_APPLY_ON}, weight=${VICREG_WEIGHT})"
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
    for mask_config in "${MASK_SCALES[@]}"; do
      
      # Parse mask scales
      IFS=',' read -r enc_min enc_max pred_min pred_max <<< "${mask_config}"
      
      # Convert sensor to uppercase for data.sensors array
      SENSOR_UPPER=$(echo "${sensor}" | tr '[:lower:]' '[:upper:]')
      
      # Task naming: include mask scales in task name since they're critical
      # Format: ijepa_timm_e85-100_p15-40 (encoder 0.85-1.0, predictor 0.15-0.40)
      enc_min_str=$(echo "${enc_min}" | sed 's/0\.//' | sed 's/\.//')
      enc_max_str=$(echo "${enc_max}" | sed 's/0\.//' | sed 's/\.//')
      pred_min_str=$(echo "${pred_min}" | sed 's/0\.//' | sed 's/\.//')
      pred_max_str=$(echo "${pred_max}" | sed 's/0\.//' | sed 's/\.//')
      mask_str="e${enc_min_str}-${enc_max_str}_p${pred_min_str}-${pred_max_str}"
      VICREG_TAG=""
      if [ "${VICREG_ENABLED}" = "true" ]; then
        vr_str=$(echo "${VICREG_WEIGHT}" | sed 's/0\.//' | sed 's/\.//')
        [ -z "${vr_str}" ] && vr_str="on"
        vron_tag=$([ "${VICREG_APPLY_ON}" = "projector" ] && echo "p" || echo "z")
        VICREG_TAG="_vr${vr_str}_${vron_tag}"
      fi
      
      task_name="ssl_pretrain/spectral_earth_mm/${sensor}/ijepa_timm_${mask_str}${VICREG_TAG}/${backbone}"
      task_search="${sensor}/ijepa_timm_${mask_str}${VICREG_TAG}/${backbone}"
      # Subdir: concise single-level directory with key params not in task_name (lr, warmup_epochs, batch_size)
      subdir="lr${LEARNING_RATE}_wu${WARMUP_EPOCHS}_bs${BATCH_SIZE}"

      # Resume: skip completed runs; add ckpt_path for incomplete runs
      CKPT_OVERRIDE=""
      if [ "${RESUME}" = true ]; then
        CKPT_RESULT=$(find_resume_checkpoint "${LOG_DIR_BASE}" "${task_search}" "${MAX_EPOCHS}" "${subdir}")
        if [ "$CKPT_RESULT" = "SKIP" ]; then
          echo "  [Skip] ${sensor^^} | ${backbone} | enc:[${enc_min},${enc_max}] pred:[${pred_min},${pred_max}] (complete)"
          continue
        elif [ -n "$CKPT_RESULT" ]; then
          CKPT_OVERRIDE="ckpt_path=\"${CKPT_RESULT}\""
        fi
      fi

      JOB_COUNT=$((JOB_COUNT + 1))
      
      # Build command
      cmd="${BASE_CMD} \
        experiment=pretrain/ijepa/base_timm \
        trainer=ddp_ssl \
        trainer.num_nodes=${NUM_NODES} \
        sensor=${sensor} \
        backbone@model.backbone_config=${backbone} \
        backbone_name=${backbone} \
        sensor@model.sensor_config=${sensor} \
        model.backbone_config.patch_size=${PATCH_SIZE} \
        model.vicreg_enabled=${VICREG_ENABLED} \
        model.vicreg_weight=${VICREG_WEIGHT} \
        model.vicreg_lambda_param=${VICREG_LAMBDA_PARAM} \
        model.vicreg_mu_param=${VICREG_MU_PARAM} \
        model.vicreg_nu_param=${VICREG_NU_PARAM} \
        model.vicreg_gather_distributed=${VICREG_GATHER_DISTRIBUTED} \
        model.vicreg_eps=${VICREG_EPS} \
        model.vicreg_max_samples=${VICREG_MAX_SAMPLES} \
        model.vicreg_apply_on=${VICREG_APPLY_ON} \
        model.vicreg_projector_hidden_dim=${VICREG_PROJECTOR_HIDDEN_DIM} \
        model.vicreg_projector_output_dim=${VICREG_PROJECTOR_OUTPUT_DIM} \
        model.vicreg_projector_num_layers=${VICREG_PROJECTOR_NUM_LAYERS} \
        model.lr=${LEARNING_RATE} \
        model.weight_decay=${WEIGHT_DECAY} \
        model.warmup_epochs=${WARMUP_EPOCHS} \
        model.max_epochs=${MAX_EPOCHS} \
        model.enc_mask_scale=[${enc_min},${enc_max}] \
        model.pred_mask_scale=[${pred_min},${pred_max}] \
        data.batch_size=${BATCH_SIZE} \
        data.sensors=[\"${SENSOR_UPPER}\"] \
        data.num_temporal_views=1 \
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
      echo "  [Job ${JOB_COUNT}] ${sensor^^} | ${backbone} | enc:[${enc_min},${enc_max}] pred:[${pred_min},${pred_max}]"
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "${DRY_RUN}" = true ]; then
  echo "  DRY RUN COMPLETE"
  echo "  Total jobs that would be submitted: ${JOB_COUNT}"
else
  echo "  All I-JEPA pretraining jobs submitted"
  echo "  Total jobs submitted: ${JOB_COUNT}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
