#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs/ssl_pretrain_terrabyte/spectral_earth_mm/mmel_fr_36/mshb/7s_fpa_ppa_c0_vpi4lv4_lv4_alc4lxw_r005_lr1u_wu2_l015_imcc_sa1_ema1_sdp0.0_eelsl/multiruns/2026-04-29_15-39-55/lr0.000001_wu2_ep94_bs5_wd0.05_fpa_ppa_sd96_sfd192_sfh2_fp768_fah2/checkpoints/last.backbone.pt}"
DATA_DIR="${DATA_DIR:-/p/scratch/hai_1025/downstream_tasks}"
SPLIT_ROOT="${SPLIT_ROOT:-data/splits}"
STATS_DIR="${STATS_DIR:-data/stats/mm_full_flat_normalized_random_single}"
LOG_DIR="${LOG_DIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOGGER="${LOGGER:-wandb}"

TASKS_CSV="${TASKS_CSV:-cdl,bdforet,bnetd,eurocrops,nlcd,treemap,desis_cdl,eo1_cdl,gaofen5_wuhan,oxhyperminerals_emit_l2a}"
TARGET_BRANCHES_CSV="${TARGET_BRANCHES_CSV:-}"
OUTPUT_SENSOR="${OUTPUT_SENSOR:-}"

MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICES="${DEVICES:-1}"
ACCELERATOR="${ACCELERATOR:-gpu}"
STRATEGY="${STRATEGY:-ddp}"
PRECISION="${PRECISION:-16-mixed}"
SYNC_BATCHNORM="${SYNC_BATCHNORM:-true}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
WANDB_OFFLINE="${WANDB_OFFLINE:-true}"
LINEAR_LR="${LINEAR_LR:-0.0001}"
LINEAR_WD="${LINEAR_WD:-0.001}"
TRAIN="${TRAIN:-true}"
TEST="${TEST:-true}"

EXTRA_ARGS=()
if [ "${FAST_DEV_RUN}" = "true" ]; then
  EXTRA_ARGS+=("+trainer.fast_dev_run=true")
fi
if [ -n "${EXTRA_HYDRA_ARGS:-}" ]; then
  # Space-delimited Hydra overrides, useful for quick local checks.
  read -r -a USER_EXTRA_ARGS <<< "${EXTRA_HYDRA_ARGS}"
  EXTRA_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

IFS=',' read -ra TASKS <<< "${TASKS_CSV}"

task_experiment() {
  local task="$1"
  if [ "${task}" = "oxhyperminerals_emit_l2a" ]; then
    echo "downstream/oxhyperminerals_emit_l2a/10pct"
  else
    echo "downstream/${task}/base"
  fi
}

default_target_branch() {
  case "$1" in
    desis_cdl) echo "DESIS" ;;
    oxhyperminerals_emit_l2a) echo "EMIT" ;;
    *) echo "ENMAP" ;;
  esac
}

lower_sensor() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

run_task() {
  local task="$1"
  local experiment
  local target_csv
  local output_sensor
  experiment=$(task_experiment "${task}")
  target_csv="${TARGET_BRANCHES_CSV:-$(default_target_branch "${task}")}"
  output_sensor="${OUTPUT_SENSOR:-${target_csv%%,*}}"

  local common_args=(
    "--config-name=downstream"
    "experiment=${experiment}"
    "paths.data_dir=${DATA_DIR}"
    "paths.log_dir=${LOG_DIR}"
    "data.split_root=${SPLIT_ROOT}"
    "model.pretrained_weights=${CHECKPOINT_PATH}"
    "model.freeze_backbone=true"
    "model.model_type=lightweight_multitap_seg"
    "model.learning_rate=${LINEAR_LR}"
    "model.weight_decay=${LINEAR_WD}"
    "trainer.max_epochs=${MAX_EPOCHS}"
    "trainer.devices=${DEVICES}"
    "trainer.accelerator=${ACCELERATOR}"
    "trainer.strategy=${STRATEGY}"
    "trainer.precision=${PRECISION}"
    "trainer.sync_batchnorm=${SYNC_BATCHNORM}"
    "data.batch_size=${BATCH_SIZE}"
    "data.num_workers=${NUM_WORKERS}"
    "train=${TRAIN}"
    "test=${TEST}"
    "backbone@model.backbone_config=multi_sensor_two_stages_hiera_b"
    "decoder@model.decoder_config=lightweight_multitap_head"
  )
  common_args+=("logger=${LOGGER}")
  if [ "${LOGGER}" = "wandb" ]; then
    common_args+=("logger.wandb.offline=${WANDB_OFFLINE}")
  fi

  if [[ "${target_csv}" == *,* ]]; then
    common_args+=(
      "+data.target_sensor_names=[${target_csv}]"
      "+data.projection_output_sensor=${output_sensor}"
      "data.standardize=true"
      "data.standardization_stats_path=${STATS_DIR}"
      "+data.projection_standardize_sensors=[S2,LO]"
      "model.backbone_config.canonical_sensor=[${target_csv}]"
    )
  else
    common_args+=(
      "+sensor@data.target_sensor_config=$(lower_sensor "${target_csv}")"
      "model.backbone_config.canonical_sensor=${target_csv}"
    )
  fi

  "${PYTHON_BIN}" src/train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
}

for task in "${TASKS[@]}"; do
  run_task "${task}"
done
