#!/usr/bin/env bash
# Joint aligned ENMAP/S2/LO fusion ablations for the multimodal Hiera backbone.
#
# This is a thin preset over run_benchmark_joint_hiera.sh. It keeps results in
# dedicated joint_fusion projects and uses the lightweight multi-tap decoder.

set -e

export FAMILY="${FAMILY:-multisensor}"
export JOINT_SPLIT_TAG="${JOINT_SPLIT_TAG:-fuse}"
export DECODERS_CSV="${DECODERS_CSV:-lightweight_multitap_head}"
export JOINT_INPUT_COMBOS_CSV="${JOINT_INPUT_COMBOS_CSV:-enmap;s2;lo;enmap,s2;enmap,lo;s2,lo;enmap,s2,lo}"
export JOINT_STANDARDIZE_SENSORS_CSV="${JOINT_STANDARDIZE_SENSORS_CSV:-s2,lo}"
export LINEAR_LRS="${LINEAR_LRS:-0.001,0.0003,0.0001}"

# Keep the full task/LR sweep by default, but allow quick checks:
#   TASKS_CSV=bdforet LINEAR_LRS=0.0003 MAX_EPOCHS=5 DEBUG_MODE=true bash ...
export TASKS_CSV="${TASKS_CSV:-cdl,bdforet,eurocrops,bnetd}"

bash scripts/downstream/run_benchmark_joint_hiera.sh "$@"
