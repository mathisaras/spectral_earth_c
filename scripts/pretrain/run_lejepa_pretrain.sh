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
ENV="terrabyte"  # juwels | jureca | terrabyte

# Sensors to pretrain on
SENSORS=("enmap" "s2" "lo")

# LeJEPA hyperparameters
BATCH_SIZE=32
MAX_EPOCHS=100
LEARNING_RATE=0.0001
WARMUP_EPOCHS=10
WEIGHT_DECAY=0.05
N_LOCAL_VIEWS=6
SIGREG_NUM_VECTORS=256
SIGREG_GATHER_DISTRIBUTED="true"
SIGREG_REF_LAMBDA=0.05
SIGREG_REF_GLOBAL_BATCH=128
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
INVARIANCE_MODES=("coupled_center" "detached_global_center")

# Standardization defaults:
# - enabled for S2/LO by default
# - disabled for ENMAP by default
STANDARDIZATION_MODE="bandwise"  # bandwise | global
STANDARDIZATION_EPS=1.0e-6
DEFAULT_STATS_DIR="${PWD}/data/stats/mm_full_flat_normalized_random_single"
STANDARDIZATION_STATS_SOURCE="${STANDARDIZATION_STATS_SOURCE:-${STANDARDIZATION_STATS_ROOT:-${DEFAULT_STATS_DIR}}}"

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
declare -A PATCH_SIZES=(
  ["enmap"]="4"   # 30m resolution
  ["s2"]="12"     # 10m resolution
  ["lo"]="4"      # 30m resolution
)

# Backbones: ENMAP gets both vit_b_timm and spec_vit_b, others only vit_b_timm
declare -A BACKBONES=(
  ["enmap"]="vit_b_timm spec_vit_b"
  ["s2"]="vit_b_timm"
  ["lo"]="vit_b_timm"
)

find_resume_checkpoint() {
  local log_base="$1" task_name="$2" max_epochs="$3" subdir_filter="${4:-}"
  local search_dir="${log_base}/${task_name}"
  [ ! -d "$search_dir" ] && echo "" && return 0
  local max_epoch=-1 best_ckpt=""
  while IFS= read -r ckpt; do
    [ -z "$ckpt" ] && continue
    [ -n "$subdir_filter" ] && [[ "$ckpt" != *"/${subdir_filter}/"* ]] && continue
    local epoch
    epoch=$(basename "$ckpt" .ckpt | sed 's/epoch_0*//')
    if [ -n "$epoch" ] && [ "$epoch" -ge 0 ] 2>/dev/null && [ "$epoch" -gt "$max_epoch" ] 2>/dev/null; then
      max_epoch=$epoch && best_ckpt="$ckpt"
    fi
  done < <(find "$search_dir" -type f -name "epoch_*.ckpt" 2>/dev/null)
  [ -z "$best_ckpt" ] && echo "" && return 0
  [ "$max_epoch" -ge "$((max_epochs - 1))" ] 2>/dev/null && echo "SKIP" && return 0
  echo "$best_ckpt"
}

sensor_needs_standardization() {
  local sensor="$1"
  case "${sensor}" in
    s2|lo|lt|s1) echo "true" ;;
    *) echo "false" ;;
  esac
}

resolve_stats_source() {
  local source="$1"
  local sensor="$2"
  if [ -d "${source}" ]; then
    local candidate="${source}/${sensor}.yaml"
    if [ ! -f "${candidate}" ]; then
      echo "Error: stats file not found for sensor '${sensor}' in '${source}'." >&2
      exit 1
    fi
    # Module accepts directory directly; we pass directory for cleaner overrides.
    echo "${source}"
    return 0
  fi

  if [ ! -f "${source}" ]; then
    echo "Error: stats source does not exist: ${source}" >&2
    exit 1
  fi
  echo "${source}"
}

short_lamb_tag() {
  local lamb="$1"
  local rounded
  rounded=$(awk -v v="${lamb}" 'BEGIN { printf "%.3f", v }')
  rounded=$(echo "${rounded}" | sed -E 's/0+$//; s/\.$//')
  local tag="${rounded//./p}"
  tag="${tag#0p}"
  echo "${tag}"
}

short_invariance_tag() {
  local mode="$1"
  case "$mode" in
    coupled_center) echo "cc" ;;
    detached_global_center) echo "dgc" ;;
    *) echo "$mode" ;;
  esac
}

compute_sigreg_effective_batch() {
  local local_batch="$1"
  local num_nodes="$2"
  local gpus_per_node="$3"
  local gather="$4"
  if [ "${gather}" = "true" ]; then
    echo $((local_batch * num_nodes * gpus_per_node))
  else
    echo "${local_batch}"
  fi
}

compute_sigreg_lambda() {
  local effective_batch="$1"
  local ref_batch="$2"
  local ref_lambda="$3"
  awk \
    -v n="${effective_batch}" \
    -v ref_n="${ref_batch}" \
    -v ref_l="${ref_lambda}" \
    'BEGIN {
      odds = (ref_l / (1.0 - ref_l)) * ref_n
      lam = odds / (n + odds)
      printf "%.6f", lam
    }'
}

# --- Script Logic ---
BASE_CMD="python src/train.py --config-name=pretrain -m"
JOB_COUNT=0
SIGREG_EFFECTIVE_BATCH=$(compute_sigreg_effective_batch "${BATCH_SIZE}" "${NUM_NODES}" "${GPUS_PER_NODE}" "${SIGREG_GATHER_DISTRIBUTED}")
LAMB=$(compute_sigreg_lambda "${SIGREG_EFFECTIVE_BATCH}" "${SIGREG_REF_GLOBAL_BATCH}" "${SIGREG_REF_LAMBDA}")
LAMB_TAG="$(short_lamb_tag "${LAMB}")"

if [ "${DRY_RUN}" = true ]; then
  echo "=========================================="
  echo "  DRY RUN MODE - Commands will be printed"
  echo "=========================================="
  echo ""
fi

echo "=========================================="
echo "  LeJEPA Pretraining Configuration"
echo "=========================================="
echo "Environment: ${ENV}"
echo "Resume mode: ${RESUME}"
echo "Nodes: ${NUM_NODES}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "Sensors: ${SENSORS[*]}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "LR: ${LEARNING_RATE}"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "SIGReg Gather Distributed: ${SIGREG_GATHER_DISTRIBUTED}"
echo "SIGReg Ref Lambda: ${SIGREG_REF_LAMBDA} @ global batch ${SIGREG_REF_GLOBAL_BATCH}"
echo "SIGReg Effective Batch: ${SIGREG_EFFECTIVE_BATCH}"
echo "Lamb: ${LAMB}"
echo "Invariance Modes: ${INVARIANCE_MODES[*]}"
echo "N Local Views: ${N_LOCAL_VIEWS}"
echo "SIGReg vectors: ${SIGREG_NUM_VECTORS}"
echo "=========================================="
echo ""

for sensor in "${SENSORS[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Sensor: ${sensor^^}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  PATCH_SIZE="${PATCH_SIZES[${sensor}]}"
  BACKBONES_STR="${BACKBONES[${sensor}]}"
  SENSOR_UPPER=$(echo "${sensor}" | tr '[:lower:]' '[:upper:]')

  DO_STANDARDIZE=$(sensor_needs_standardization "${sensor}")
  STD_TAG="nostd"
  STD_OVERRIDES=""
  if [ "${DO_STANDARDIZE}" = "true" ]; then
    STATS_SOURCE=$(resolve_stats_source "${STANDARDIZATION_STATS_SOURCE}" "${sensor}")
    STD_TAG="std_${STANDARDIZATION_MODE}"
    STD_OVERRIDES=" \
      model.standardize=true \
      model.standardization_mode=${STANDARDIZATION_MODE} \
      model.standardization_stats_path=${STATS_SOURCE} \
      model.standardization_eps=${STANDARDIZATION_EPS}"
  else
    STD_OVERRIDES="model.standardize=false"
  fi

  echo "  Patch Size: ${PATCH_SIZE}"
  echo "  Backbones: ${BACKBONES_STR}"
  echo "  Standardization: ${DO_STANDARDIZE}"
  echo ""

  for backbone in ${BACKBONES_STR}; do
    # Baseline handling:
    # - vit_b_timm supports dynamic size local views
    # - spec_vit_b uses fixed input shape -> resize locals to global size
    LOCAL_RESIZE_TO_GLOBAL="false"
    if [ "${backbone}" != "vit_b_timm" ]; then
      LOCAL_RESIZE_TO_GLOBAL="true"
    fi
    LOCAL_TAG=$([ "${LOCAL_RESIZE_TO_GLOBAL}" = "true" ] && echo "lrg" || echo "lrl")

    for invariance_mode in "${INVARIANCE_MODES[@]}"; do
      JOB_COUNT=$((JOB_COUNT + 1))
      INV_TAG=$(short_invariance_tag "${invariance_mode}")

      task_name="ssl_pretrain/spectral_earth_mm/${sensor}/lejepa/${backbone}/${STD_TAG}_${LOCAL_TAG}_l${LAMB_TAG}_im${INV_TAG}"
      task_search="${sensor}/lejepa/${backbone}/${STD_TAG}_${LOCAL_TAG}_l${LAMB_TAG}_im${INV_TAG}"
      subdir="lr${LEARNING_RATE}_wu${WARMUP_EPOCHS}_bs${BATCH_SIZE}_nlv${N_LOCAL_VIEWS}_wd${WEIGHT_DECAY}"

      CKPT_OVERRIDE=""
      if [ "${RESUME}" = true ]; then
        CKPT_RESULT=$(find_resume_checkpoint "${LOG_DIR_BASE}" "${task_search}" "${MAX_EPOCHS}" "${subdir}")
        if [ "$CKPT_RESULT" = "SKIP" ]; then
          echo "  [Skip] ${sensor^^} | ${backbone} | inv=${invariance_mode} | ${STD_TAG} | ${LOCAL_TAG} (complete)"
          continue
        elif [ -n "$CKPT_RESULT" ]; then
          CKPT_OVERRIDE="ckpt_path=\"${CKPT_RESULT}\""
        fi
      fi

      cmd="${BASE_CMD} \
        experiment=pretrain/lejepa/base \
        trainer=ddp_ssl \
        trainer.num_nodes=${NUM_NODES} \
        sensor=${sensor} \
        backbone@model.backbone_config=${backbone} \
        backbone_name=${backbone} \
        sensor@model.sensor_config=${sensor} \
        model.backbone_config.patch_size=${PATCH_SIZE} \
        model.lr=${LEARNING_RATE} \
        model.warmup_epochs=${WARMUP_EPOCHS} \
        model.max_epochs=${MAX_EPOCHS} \
        model.weight_decay=${WEIGHT_DECAY} \
        model.lamb=${LAMB} \
        model.invariance_mode=${invariance_mode} \
        model.n_views=${N_LOCAL_VIEWS} \
        model.sigreg_num_vectors=${SIGREG_NUM_VECTORS} \
        model.sigreg_gather_distributed=${SIGREG_GATHER_DISTRIBUTED} \
        model.local_resize_to_global=${LOCAL_RESIZE_TO_GLOBAL} \
        ${STD_OVERRIDES} \
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

      echo "  [Job ${JOB_COUNT}] ${sensor^^} | ${backbone} | inv=${invariance_mode} | ${STD_TAG} | ${LOCAL_TAG}"
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
  echo "  All LeJEPA pretraining jobs submitted"
  echo "  Total jobs submitted: ${JOB_COUNT}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
