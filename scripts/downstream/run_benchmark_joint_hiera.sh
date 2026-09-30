#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs/ssl_pretrain_terrabyte/spectral_earth_mm/mmel_fr_36/mshb/7s_fpa_ppa_c0_vpi4lv4_lv4_alc4lxw_r005_lr1u_wu2_l015_imcc_sa1_ema1_sdp0.0_eelsl/multiruns/2026-04-29_15-39-55/lr0.000001_wu2_ep94_bs5_wd0.05_fpa_ppa_sd96_sfd192_sfh2_fp768_fah2/checkpoints/last.backbone.pt}"
DATA_DIR="${DATA_DIR:-/p/scratch/hai_1025/downstream_tasks}"
SPLIT_ROOT="${SPLIT_ROOT:-data/splits_joint}"
STATS_DIR="${STATS_DIR:-data/stats/mm_full_flat_normalized_random_single}"
LOG_DIR="${LOG_DIR:-logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOGGER="${LOGGER:-wandb}"

TASKS_CSV="${TASKS_CSV:-cdl,bdforet,eurocrops}"
INPUT_SENSORS_CSV="${INPUT_SENSORS_CSV:-ENMAP,S2,LO}"
OUTPUT_SENSOR="${OUTPUT_SENSOR:-ENMAP}"

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
  read -r -a USER_EXTRA_ARGS <<< "${EXTRA_HYDRA_ARGS}"
  EXTRA_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

IFS=',' read -ra TASKS <<< "${TASKS_CSV}"

for task in "${TASKS[@]}"; do
  args=(
    "--config-name=downstream"
    "experiment=downstream/${task}/base" \
    "paths.data_dir=${DATA_DIR}" \
    "paths.log_dir=${LOG_DIR}" \
    "data.split_root=${SPLIT_ROOT}" \
    "model.pretrained_weights=${CHECKPOINT_PATH}" \
    "model.freeze_backbone=true" \
    "model.model_type=lightweight_multitap_seg" \
    "model.learning_rate=${LINEAR_LR}" \
    "model.weight_decay=${LINEAR_WD}" \
    "trainer.max_epochs=${MAX_EPOCHS}" \
    "trainer.devices=${DEVICES}" \
    "trainer.accelerator=${ACCELERATOR}" \
    "trainer.strategy=${STRATEGY}" \
    "trainer.precision=${PRECISION}" \
    "trainer.sync_batchnorm=${SYNC_BATCHNORM}" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    "+data.input_sensor_names=[${INPUT_SENSORS_CSV}]" \
    "+data.joint_output_sensor=${OUTPUT_SENSOR}" \
    "data.standardize=true" \
    "data.standardization_stats_path=${STATS_DIR}" \
    "+data.input_standardize_sensors=[S2,LO]" \
    "model.backbone_config.canonical_sensor=[${INPUT_SENSORS_CSV}]" \
    "train=${TRAIN}" \
    "test=${TEST}" \
    "backbone@model.backbone_config=multi_sensor_two_stages_hiera_b" \
    "decoder@model.decoder_config=lightweight_multitap_head"
  )
  args+=("logger=${LOGGER}")
  if [ "${LOGGER}" = "wandb" ]; then
    args+=("logger.wandb.offline=${WANDB_OFFLINE}")
  fi
  "${PYTHON_BIN}" src/train.py "${args[@]}" "${EXTRA_ARGS[@]}"
done
