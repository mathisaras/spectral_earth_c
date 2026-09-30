#!/bin/bash
set -e

# Cap all BLAS/OMP/MKL thread pools to 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Configuration ---

# Dry run mode: set to true to print commands without executing them.
# Override via env: DRY_RUN=true ./run_pretrain.sh
DRY_RUN="${DRY_RUN:-false}"

# Resume mode: when true, search for existing checkpoints and only resubmit runs that
# stopped before completion (epoch < max_epochs - 1). Completed runs are skipped.
RESUME="${RESUME:-true}"

# Number of nodes (1, 2, 4, etc.). Overrides trainer.num_nodes and hydra.launcher.nodes.
NUM_NODES="${NUM_NODES:-12}"

# Environment setting (juwels, jureca, or terrabyte)
ENV="${ENV:-terrabyte}"

# Full-dataset target. Kept separate from the paths profiles on purpose so this
# launcher always points at the requested full archive unless explicitly overridden.
FULL_ZARR_PATH="${FULL_ZARR_PATH:-/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/spectral_earth_mm.zarr}"
DATASET_TAG="${DATASET_TAG:-full}"

# Variant handling. By default we launch the lc4-style full run that keeps weak
# sensors in the global context but excludes them from local views only.
# Optionally, enable a second full run for the no-local-exclusion baseline.
RUN_TAG="${RUN_TAG:-lc4lxw}"
LOCAL_EXCLUDED_MODALITIES="${LOCAL_EXCLUDED_MODALITIES:-[S1,LT]}"
RUN_BOTH_LOCAL_VARIANTS="${RUN_BOTH_LOCAL_VARIANTS:-false}"
WEAK_LOCAL_EXCLUDED_MODALITIES="${WEAK_LOCAL_EXCLUDED_MODALITIES:-[S1,LT]}"
WEAK_LOCAL_RUN_TAG="${WEAK_LOCAL_RUN_TAG:-lc4lxw}"

# Sensors to use for multimodal training (uppercase).
SENSOR_SETS=("ENMAP DESIS EMIT S2 LO LT S1")

# Backbone (multi-sensor two stages Hiera)
BACKBONE="multi_sensor_two_stages_hiera_b"
BACKBONE_SHORT="ms_hiera_b"

# Multimodal LeJEPA hyperparameters
BATCH_SIZE=5
MAX_EPOCHS=100
LEARNING_RATE=0.00001
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=20
OUTPUT_DIM=128
PROJECTOR_HIDDEN_DIM=2048
PROJECTOR_NUM_LAYERS=3
PROJECTOR_USE_BN="true"
EMA_TEACHER_ENABLED="true"
TEACHER_MOMENTUM_START=0.996
TEACHER_MOMENTUM_END=1.0
SIGREG_NUM_VECTORS=256
SIGREG_GATHER_DISTRIBUTED="true"
# Keep the original launcher lambda logic based on reference lambda and
# effective global SIGReg batch.
SIGREG_REF_LAMBDA=0.05
SIGREG_REF_GLOBAL_BATCH=128
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
INVARIANCE_MODES=("coupled_center")

# View sampling
GLOBAL_VIEWS=2
# VIEW_PRESETS format:
#   name:global_modalities_per_view:n_local_views:local_modalities_per_view:local_from_global_only:global_crop_scale_override:sensor_drop_prob_override
VIEW_PRESETS=(
  "indep_fixed4lv4:4:4:1:true:default:default"
)

# Batch-level modality dropout (datamodule)
SENSOR_DROP_PROBS=("0.0")
MIN_SENSORS=1

# Model-side augmentations
APPLY_INPUT_NORMALIZATION="true"
STANDARDIZE_MODALITIES="[S2,LO,S1]"
STANDARDIZATION_MODE="bandwise"
STANDARDIZATION_EPS=1.0e-6
DEFAULT_STANDARDIZATION_STATS_SOURCE="${PWD}/data/stats/mm_full_flat_normalized_random_single"
STANDARDIZATION_STATS_SOURCE="${STANDARDIZATION_STATS_SOURCE:-${STANDARDIZATION_STATS_ROOT:-${DEFAULT_STANDARDIZATION_STATS_SOURCE}}}"
# lc4-style crops: less tiny locals and tighter globals.
GLOBAL_CROP_SCALE="[0.4,1.0]"
LOCAL_CROP_SCALE="[0.1,0.4]"
RADIOMETRIC_AUG_P=0.5
RADIOMETRIC_BRIGHTNESS_RANGE="[0.8,1.2]"
RADIOMETRIC_BIAS_RANGE="[-0.1,0.1]"

# Backbone / fusion sweep options
CLS_TOKEN_OPTIONS=("false")
FUSION_PRESETS=(
  "fpa_ppa:projected_attention:projected_attention"
)
ENCODER_PRESETS=("landsat_sentinel_linear")

SPECTRAL_TOKEN_DIM=96
SPECTRAL_FUSION_DIM=192
SPECTRAL_FUSION_HEADS=2
FUSION_PROJECTION_DIM=768
FUSION_ATTENTION_HEADS=2

# Logging / launcher
WANDB_PROJECT="ssl_pretrain_spectral_earth_mm"

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

if [ ! -e "${FULL_ZARR_PATH}" ]; then
  echo "Error: FULL_ZARR_PATH does not exist: ${FULL_ZARR_PATH}"
  exit 1
fi

if [ "${STANDARDIZE_MODALITIES}" != "null" ] && [ ! -e "${STANDARDIZATION_STATS_SOURCE}" ]; then
  echo "Error: STANDARDIZATION_STATS_SOURCE does not exist: ${STANDARDIZATION_STATS_SOURCE}"
  exit 1
fi

PRECISION="16-mixed"
STRATEGY="ddp_find_unused_parameters_true"

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

sensors_to_hydra_list() {
  local sensors="$1"
  local result="["
  local first=true
  for s in $sensors; do
    [ "$first" = true ] && first=false || result="${result},"
    result="${result}${s}"
  done
  result="${result}]"
  echo "$result"
}

short_sensor_tag() {
  local sensors="$1"
  local count
  count=$(echo $sensors | wc -w)
  [ "$count" -eq 7 ] 2>/dev/null && echo "7s" && return
  [ "$count" -eq 1 ] 2>/dev/null && echo "$sensors" | tr '[:upper:]' '[:lower:]' && return
  echo "${count}s"
}

short_pool_tag() {
  local pool="$1"
  case "$pool" in
    projected_attention|proj_attn) echo "pa" ;;
    attention) echo "attn" ;;
    mean) echo "mn" ;;
    max) echo "mx" ;;
    cls) echo "cls" ;;
    *) echo "$pool" ;;
  esac
}

short_fusion_tag() {
  local mode="$1"
  case "$mode" in
    projected_attention|proj_attn) echo "pa" ;;
    projected_weighted|weighted_proj|proj_weighted|weighted_projection) echo "pw" ;;
    multi_query_concat|mq_concat|concat_query) echo "mqc" ;;
    attention) echo "attn" ;;
    weighted) echo "w" ;;
    mean) echo "m" ;;
    max) echo "mx" ;;
    *) echo "$mode" ;;
  esac
}

short_bool_tag() {
  local v="$1"
  [ "$v" = "true" ] && echo "1" || echo "0"
}

short_encoder_tag() {
  local preset="$1"
  case "$preset" in
    landsat_sentinel_linear) echo "elsl" ;;
    legacy_spectral_s2_lo) echo "es2losp" ;;
    *) echo "$preset" ;;
  esac
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

short_task_dataset_tag() {
  local tag="$1"
  case "$tag" in
    full) echo "f" ;;
    *) echo "$tag" ;;
  esac
}

short_task_backbone_tag() {
  local tag="$1"
  case "$tag" in
    ms_hiera_b) echo "mshb" ;;
    *) echo "$tag" ;;
  esac
}

short_task_view_tag() {
  local tag="$1"
  case "$tag" in
    indep_fixed4lv4) echo "i4lv4" ;;
    *) echo "$tag" ;;
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

build_encoder_overrides() {
  local preset="$1"
  case "$preset" in
    landsat_sentinel_linear)
      echo "model.backbone_config.sensor_encoder_types.ENMAP=spectral_transformer \
model.backbone_config.sensor_encoder_types.DESIS=spectral_transformer \
model.backbone_config.sensor_encoder_types.EMIT=spectral_transformer \
model.backbone_config.sensor_encoder_types.S2=linear \
model.backbone_config.sensor_encoder_types.LO=linear \
model.backbone_config.sensor_encoder_types.LT=linear \
model.backbone_config.sensor_encoder_types.S1=linear"
      ;;
    legacy_spectral_s2_lo)
      echo "model.backbone_config.sensor_encoder_types.ENMAP=spectral_transformer \
model.backbone_config.sensor_encoder_types.DESIS=spectral_transformer \
model.backbone_config.sensor_encoder_types.EMIT=spectral_transformer \
model.backbone_config.sensor_encoder_types.S2=spectral_transformer \
model.backbone_config.sensor_encoder_types.LO=spectral_transformer \
model.backbone_config.sensor_encoder_types.LT=linear \
model.backbone_config.sensor_encoder_types.S1=linear"
      ;;
    *)
      echo "Error: unknown encoder preset '${preset}'" >&2
      return 1
      ;;
  esac
}

BASE_CMD="python src/train.py --config-name=pretrain -m"
JOB_COUNT=0
SIGREG_EFFECTIVE_BATCH=$(compute_sigreg_effective_batch "${BATCH_SIZE}" "${NUM_NODES}" "${GPUS_PER_NODE}" "${SIGREG_GATHER_DISTRIBUTED}")
LAMB=$(compute_sigreg_lambda "${SIGREG_EFFECTIVE_BATCH}" "${SIGREG_REF_GLOBAL_BATCH}" "${SIGREG_REF_LAMBDA}")
LAMB_TAG="$(short_lamb_tag "${LAMB}")"
TASK_DATASET_TAG="$(short_task_dataset_tag "${DATASET_TAG}")"
TASK_BACKBONE_TAG="$(short_task_backbone_tag "${BACKBONE_SHORT}")"
RUN_VARIANTS=("${RUN_TAG}:${LOCAL_EXCLUDED_MODALITIES}")
if [ "${RUN_BOTH_LOCAL_VARIANTS}" = "true" ]; then
  weak_variant="${WEAK_LOCAL_RUN_TAG}:${WEAK_LOCAL_EXCLUDED_MODALITIES}"
  if [ "${weak_variant}" != "${RUN_TAG}:${LOCAL_EXCLUDED_MODALITIES}" ]; then
    RUN_VARIANTS+=("${weak_variant}")
  fi
fi

if [ "${DRY_RUN}" = true ]; then
  echo "=========================================="
  echo "  DRY RUN MODE - Commands will be printed"
  echo "=========================================="
  echo ""
fi

echo "=========================================="
echo "  Multimodal EMA-LeJEPA Full-Dataset Run"
echo "=========================================="
echo "Environment: ${ENV}"
echo "Resume mode: ${RESUME}"
echo "Nodes: ${NUM_NODES}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "Backbone: ${BACKBONE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Epochs: ${MAX_EPOCHS}"
echo "LR: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Full Zarr Path: ${FULL_ZARR_PATH}"
echo "Dataset Tag: ${DATASET_TAG}"
echo "EMA Teacher Enabled: ${EMA_TEACHER_ENABLED}"
echo "Teacher Momentum Start: ${TEACHER_MOMENTUM_START}"
echo "Teacher Momentum End: ${TEACHER_MOMENTUM_END}"
echo "SIGReg Gather Distributed: ${SIGREG_GATHER_DISTRIBUTED}"
echo "SIGReg Ref Lambda: ${SIGREG_REF_LAMBDA} @ global batch ${SIGREG_REF_GLOBAL_BATCH}"
echo "SIGReg Effective Batch: ${SIGREG_EFFECTIVE_BATCH}"
echo "Lambda: ${LAMB}"
echo "Invariance Modes: ${INVARIANCE_MODES[*]}"
echo "Global Views: ${GLOBAL_VIEWS}"
echo "Global Crop Scale: ${GLOBAL_CROP_SCALE}"
echo "Local Crop Scale: ${LOCAL_CROP_SCALE}"
echo "Standardize Modalities: ${STANDARDIZE_MODALITIES}"
echo "Standardization Mode: ${STANDARDIZATION_MODE}"
echo "Standardization Stats Source: ${STANDARDIZATION_STATS_SOURCE}"
echo "CLS options: ${CLS_TOKEN_OPTIONS[*]}"
echo "View presets: ${#VIEW_PRESETS[@]}"
echo "Fusion presets: ${#FUSION_PRESETS[@]}"
echo "Encoder presets: ${ENCODER_PRESETS[*]}"
echo "Sensor Drop Prob options: ${SENSOR_DROP_PROBS[*]}"
echo "Min Sensors After Drop: ${MIN_SENSORS}"
echo "Run variants: ${#RUN_VARIANTS[@]}"
for run_variant in "${RUN_VARIANTS[@]}"; do
  IFS=':' read -r variant_tag variant_local_excluded <<< "${run_variant}"
  echo "  ${variant_tag} -> local_excluded_modalities=${variant_local_excluded}"
done
echo "=========================================="
echo ""

for sensor_set in "${SENSOR_SETS[@]}"; do
  SENSORS_HYDRA=$(sensors_to_hydra_list "$sensor_set")
  SENSOR_TAG=$(echo "$sensor_set" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
  SENSOR_SHORT=$(short_sensor_tag "$sensor_set")

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Sensors: ${sensor_set}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for fusion_preset in "${FUSION_PRESETS[@]}"; do
    IFS=':' read -r fusion_label fusion_mode spectral_pool <<< "${fusion_preset}"
    if [ -z "${fusion_label}" ] || [ -z "${fusion_mode}" ] || [ -z "${spectral_pool}" ]; then
      echo "Error: invalid FUSION_PRESET '${fusion_preset}' (expected tag:fusion_mode:spectral_pooling)"
      exit 1
    fi

    for encoder_preset in "${ENCODER_PRESETS[@]}"; do
      ENCODER_OVERRIDES=$(build_encoder_overrides "${encoder_preset}") || exit 1
      ENCODER_TAG=$(short_encoder_tag "${encoder_preset}")

      for use_cls in "${CLS_TOKEN_OPTIONS[@]}"; do
        for view_preset in "${VIEW_PRESETS[@]}"; do
          IFS=':' read -r view_tag global_modalities_per_view n_local_views local_modalities_per_view local_from_global_only global_crop_scale_override sensor_drop_prob_override <<< "${view_preset}"
          if [ -z "${view_tag}" ] || [ -z "${global_modalities_per_view}" ] || [ -z "${n_local_views}" ] || [ -z "${local_modalities_per_view}" ] || [ -z "${local_from_global_only}" ] || [ -z "${global_crop_scale_override}" ] || [ -z "${sensor_drop_prob_override}" ]; then
            echo "Error: invalid VIEW_PRESET '${view_preset}' (expected name:global_modalities_per_view:n_local_views:local_modalities_per_view:local_from_global_only:global_crop_scale_override:sensor_drop_prob_override)"
            exit 1
          fi
          effective_global_crop_scale="${GLOBAL_CROP_SCALE}"
          [ "${global_crop_scale_override}" != "default" ] && effective_global_crop_scale="${global_crop_scale_override}"

          sensor_drop_options=("${SENSOR_DROP_PROBS[@]}")
          [ "${sensor_drop_prob_override}" != "default" ] && sensor_drop_options=("${sensor_drop_prob_override}")

          for sensor_drop_prob in "${sensor_drop_options[@]}"; do
            for invariance_mode in "${INVARIANCE_MODES[@]}"; do
              for run_variant in "${RUN_VARIANTS[@]}"; do
                IFS=':' read -r run_tag local_excluded_modalities <<< "${run_variant}"
                FUSION_MODE_TAG=$(short_fusion_tag "${fusion_mode}")
                POOL_TAG=$(short_pool_tag "${spectral_pool}")
                CLS_TAG=$(short_bool_tag "${use_cls}")
                INV_TAG=$(short_invariance_tag "${invariance_mode}")
                TASK_VIEW_TAG=$(short_task_view_tag "${view_tag}")

                FUSION_TAG=""
                [ "${SPECTRAL_TOKEN_DIM}" != "null" ] && FUSION_TAG="${FUSION_TAG}_sd${SPECTRAL_TOKEN_DIM}"
                [ "${SPECTRAL_FUSION_DIM}" != "null" ] && FUSION_TAG="${FUSION_TAG}_sfd${SPECTRAL_FUSION_DIM}"
                [ "${SPECTRAL_FUSION_HEADS}" != "null" ] && FUSION_TAG="${FUSION_TAG}_sfh${SPECTRAL_FUSION_HEADS}"
                [ "${FUSION_PROJECTION_DIM}" != "null" ] && FUSION_TAG="${FUSION_TAG}_fp${FUSION_PROJECTION_DIM}"
                [ "${FUSION_ATTENTION_HEADS}" != "null" ] && FUSION_TAG="${FUSION_TAG}_fah${FUSION_ATTENTION_HEADS}"
                task_name="ssl_pretrain/spectral_earth_mm/mmel_${TASK_DATASET_TAG}/${TASK_BACKBONE_TAG}/${SENSOR_SHORT}_f${FUSION_MODE_TAG}_p${POOL_TAG}_c${CLS_TAG}_vp${TASK_VIEW_TAG}_lv${n_local_views}_a${run_tag}_l${LAMB_TAG}_im${INV_TAG}_ema1_sdp${sensor_drop_prob}_e${ENCODER_TAG}"
                task_search="mmel_${TASK_DATASET_TAG}/${TASK_BACKBONE_TAG}/${SENSOR_SHORT}_f${FUSION_MODE_TAG}_p${POOL_TAG}_c${CLS_TAG}_vp${TASK_VIEW_TAG}_lv${n_local_views}_a${run_tag}_l${LAMB_TAG}_im${INV_TAG}_ema1_sdp${sensor_drop_prob}_e${ENCODER_TAG}"
                subdir="lr${LEARNING_RATE}_bs${BATCH_SIZE}_wd${WEIGHT_DECAY}_${fusion_label}${FUSION_TAG}"

                CKPT_OVERRIDE=""
                if [ "${RESUME}" = true ]; then
                  CKPT_RESULT=$(find_resume_checkpoint "${LOG_DIR_BASE}" "${task_search}" "${MAX_EPOCHS}" "${subdir}")
                  if [ "$CKPT_RESULT" = "SKIP" ]; then
                    echo "  [Skip] sensors=${SENSOR_TAG} | fusion=${fusion_mode} | pool=${spectral_pool} | cls=${use_cls} | view=${view_tag} | atag=${run_tag} | lexcl=${local_excluded_modalities} | inv=${invariance_mode} | sdp=${sensor_drop_prob} | enc=${encoder_preset} (complete)"
                    continue
                  elif [ -n "$CKPT_RESULT" ]; then
                    CKPT_OVERRIDE="ckpt_path=\"${CKPT_RESULT}\""
                  fi
                fi

                JOB_COUNT=$((JOB_COUNT + 1))

                BACKBONE_FUSION_OVERRIDES=""
                [ "${SPECTRAL_TOKEN_DIM}" != "null" ] && BACKBONE_FUSION_OVERRIDES="${BACKBONE_FUSION_OVERRIDES} model.backbone_config.spectral_token_dim=${SPECTRAL_TOKEN_DIM}"
                [ "${SPECTRAL_FUSION_DIM}" != "null" ] && BACKBONE_FUSION_OVERRIDES="${BACKBONE_FUSION_OVERRIDES} model.backbone_config.spectral_fusion_dim=${SPECTRAL_FUSION_DIM}"
                [ "${SPECTRAL_FUSION_HEADS}" != "null" ] && BACKBONE_FUSION_OVERRIDES="${BACKBONE_FUSION_OVERRIDES} model.backbone_config.spectral_fusion_heads=${SPECTRAL_FUSION_HEADS}"
                [ "${FUSION_PROJECTION_DIM}" != "null" ] && BACKBONE_FUSION_OVERRIDES="${BACKBONE_FUSION_OVERRIDES} model.backbone_config.fusion_projection_dim=${FUSION_PROJECTION_DIM}"
                [ "${FUSION_ATTENTION_HEADS}" != "null" ] && BACKBONE_FUSION_OVERRIDES="${BACKBONE_FUSION_OVERRIDES} model.backbone_config.fusion_attention_heads=${FUSION_ATTENTION_HEADS}"

                  cmd="${BASE_CMD} \
                    experiment=pretrain/multimodal_lejepa/base \
                    trainer=ddp_ssl \
                    trainer.num_nodes=${NUM_NODES} \
                    backbone@model.backbone_config=${BACKBONE} \
                    backbone_name=${BACKBONE} \
                    model.output_dim=${OUTPUT_DIM} \
                    model.projector_hidden_dim=${PROJECTOR_HIDDEN_DIM} \
                    model.projector_num_layers=${PROJECTOR_NUM_LAYERS} \
                    model.projector_use_bn=${PROJECTOR_USE_BN} \
                    model.ema_teacher_enabled=${EMA_TEACHER_ENABLED} \
                    model.teacher_momentum_start=${TEACHER_MOMENTUM_START} \
                    model.teacher_momentum_end=${TEACHER_MOMENTUM_END} \
                    model.lamb=${LAMB} \
                    model.invariance_mode=${invariance_mode} \
                    model.sigreg_num_vectors=${SIGREG_NUM_VECTORS} \
                    model.sigreg_gather_distributed=${SIGREG_GATHER_DISTRIBUTED} \
                    model.global_views=${GLOBAL_VIEWS} \
                    model.global_modalities_per_view=${global_modalities_per_view} \
                    model.n_local_views=${n_local_views} \
                    model.local_modalities_per_view=${local_modalities_per_view} \
                    model.local_modalities_from_global_only=${local_from_global_only} \
                    model.local_excluded_modalities=${local_excluded_modalities} \
                    model.apply_input_normalization=${APPLY_INPUT_NORMALIZATION} \
                    model.standardize_modalities=${STANDARDIZE_MODALITIES} \
                    model.standardization_mode=${STANDARDIZATION_MODE} \
                    model.standardization_stats_path=${STANDARDIZATION_STATS_SOURCE} \
                    model.standardization_eps=${STANDARDIZATION_EPS} \
                    model.global_crop_scale=${effective_global_crop_scale} \
                    model.local_crop_scale=${LOCAL_CROP_SCALE} \
                    model.radiometric_aug_p=${RADIOMETRIC_AUG_P} \
                    model.radiometric_brightness_range=${RADIOMETRIC_BRIGHTNESS_RANGE} \
                    model.radiometric_bias_range=${RADIOMETRIC_BIAS_RANGE} \
                    model.lr=${LEARNING_RATE} \
                    model.weight_decay=${WEIGHT_DECAY} \
                    model.warmup_epochs=${WARMUP_EPOCHS} \
                    model.max_epochs=${MAX_EPOCHS} \
                    model.backbone_config.use_cls_token=${use_cls} \
                    model.backbone_config.fusion_mode=${fusion_mode} \
                    model.backbone_config.spectral_pooling=${spectral_pool} \
                    ${BACKBONE_FUSION_OVERRIDES} \
                    ${ENCODER_OVERRIDES} \
                    data.batch_size=${BATCH_SIZE} \
                    data.sensors=\"${SENSORS_HYDRA}\" \
                    data.sensor_drop_prob=${sensor_drop_prob} \
                    data.min_sensors=${MIN_SENSORS} \
                    data.num_temporal_views=1 \
                    trainer.max_epochs=${MAX_EPOCHS} \
                    trainer.precision=${PRECISION} \
                    trainer.strategy=${STRATEGY} \
                    logger.wandb.offline=${WANDB_OFFLINE} \
                    logger.wandb.project=${WANDB_PROJECT} \
                    paths=${PATHS_PROFILE} \
                    paths.zarr_path=\"${FULL_ZARR_PATH}\" \
                    hydra/launcher=${LAUNCHER} \
                    hydra.launcher.nodes=${NUM_NODES} \
                    hydra.sweep.subdir=${subdir} \
                    task_name=${task_name}"
                  [ -n "${CKPT_OVERRIDE}" ] && cmd="${cmd} ${CKPT_OVERRIDE}"

                  echo "  [Job ${JOB_COUNT}] sensors=${SENSOR_TAG} | fusion=${fusion_mode} | pool=${spectral_pool} | cls=${use_cls} | view=${view_tag} | atag=${run_tag} | lexcl=${local_excluded_modalities} | inv=${invariance_mode} | gm=${global_modalities_per_view} | lv=${n_local_views} lm=${local_modalities_per_view} lfg=${local_from_global_only} | sdp=${sensor_drop_prob} | enc=${encoder_preset}"
                  if [ "${DRY_RUN}" = true ]; then
                    echo "  ┌─────────────────────────────────────────────────────────────"
                    echo "${cmd}" | sed 's/^/  │ /'
                    echo "  └─────────────────────────────────────────────────────────────"
                  else
                    echo "  → Submitting..."
                    eval "${cmd}" &
                  fi
                  echo ""
                done
              done
            done
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
  echo "  All Multimodal EMA-LeJEPA full-dataset jobs submitted"
  echo "  Total jobs submitted: ${JOB_COUNT}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
