#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="${SCRIPT_DIR}/run_benchmark_multisensor_hiera.sh"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs/ssl_pretrain_terrabyte/spectral_earth_mm/mmel_fr_36/mshb/7s_fpa_ppa_c0_vpi4lv4_lv4_alc4lxw_r005_lr1u_wu2_l015_imcc_sa1_ema1_sdp0.0_eelsl/multiruns/2026-04-29_15-39-55/lr0.000001_wu2_ep94_bs5_wd0.05_fpa_ppa_sd96_sfd192_sfh2_fp768_fah2/checkpoints/last.backbone.pt}"
DATA_DIR="${DATA_DIR:-/p/scratch/hai_1025/downstream_tasks}"
SPLIT_ROOT="${SPLIT_ROOT:-data/splits}"
STATS_DIR="${STATS_DIR:-data/stats/mm_full_flat_normalized_random_single}"
LOG_DIR="${LOG_DIR:-logs}"

RUN_ALL_OPTICAL="${RUN_ALL_OPTICAL:-true}"
RUN_SELECTED_PATHS="${RUN_SELECTED_PATHS:-true}"

ALL_ENMAP_TASKS_CSV="${ALL_ENMAP_TASKS_CSV:-cdl,bdforet,eurocrops,bnetd,nlcd,treemap}"
ALL_DESIS_TASKS_CSV="${ALL_DESIS_TASKS_CSV:-desis_cdl}"
ALL_EMIT_TASKS_CSV="${ALL_EMIT_TASKS_CSV:-oxhyperminerals_emit_l2a}"
ALL_UNSEEN_TASKS_CSV="${ALL_UNSEEN_TASKS_CSV:-eo1_cdl,gaofen5_wuhan}"
SELECTED_TASKS_CSV="${SELECTED_TASKS_CSV:-cdl,bdforet,eurocrops,bnetd,desis_cdl,eo1_cdl,gaofen5_wuhan}"

run_case() {
  local tasks_csv="$1"
  local branches_csv="$2"
  local output_sensor="$3"
  env \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    DATA_DIR="${DATA_DIR}" \
    SPLIT_ROOT="${SPLIT_ROOT}" \
    STATS_DIR="${STATS_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    TASKS_CSV="${tasks_csv}" \
    TARGET_BRANCHES_CSV="${branches_csv}" \
    OUTPUT_SENSOR="${output_sensor}" \
    bash "${BASE_SCRIPT}"
}

if [ "${RUN_ALL_OPTICAL}" = "true" ]; then
  run_case "${ALL_ENMAP_TASKS_CSV}" "ENMAP,DESIS,EMIT,S2,LO" "ENMAP"
  run_case "${ALL_DESIS_TASKS_CSV}" "DESIS,ENMAP,EMIT,S2,LO" "DESIS"
  run_case "${ALL_EMIT_TASKS_CSV}" "EMIT,ENMAP,DESIS,S2,LO" "EMIT"
  run_case "${ALL_UNSEEN_TASKS_CSV}" "ENMAP,DESIS,EMIT,S2,LO" "ENMAP"
fi

if [ "${RUN_SELECTED_PATHS}" = "true" ]; then
  for branch_set in \
    "ENMAP" \
    "DESIS" \
    "EMIT" \
    "S2" \
    "LO" \
    "ENMAP,DESIS" \
    "ENMAP,EMIT" \
    "DESIS,EMIT" \
    "ENMAP,DESIS,EMIT"
  do
    run_case "${SELECTED_TASKS_CSV}" "${branch_set}" "${branch_set%%,*}"
  done
fi
