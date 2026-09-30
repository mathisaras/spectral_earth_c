#!/bin/bash
# Benchmark external / legacy comparison backbones on native downstream tasks.
#
# This script is intentionally separate from the multi-sensor Hiera launcher:
# - no sensor remapping
# - no Lightning checkpoint conversion
# - no Hiera-specific config extraction
# - model-specific input normalization is applied through the datamodule only
#
# Defaults cover the main segmentation/classification downstream tasks. Override:
#   TASKS_CSV=cdl,h2sr BACKBONES_CSV=dofa_b,panopticon_b DEBUG_MODE=true bash scripts/downstream/run_competitor_benchmarks.sh
#
set -e

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

read_csv_array() {
  local csv="$1"
  local -n out_ref=$2
  out_ref=()
  if [ -z "$csv" ]; then
    return 0
  fi
  IFS=',' read -ra out_ref <<< "$csv"
}

# --- Configuration ---

ENV="${ENV:-juwels}"  # "terrabyte" | "juwels"

TASKS=(
  #"bdforet"
  #"bnetd"
  #"cdl"
  #"eurocrops"
  #"desis_cdl"
  #"eo1_cdl"
  #"h2sr"
  "oxhyperminerals_emit_l2a"
  #"gaofen5_wuhan"
  #"nlcd"
  #"treemap"
  #"corine"
)
TASKS_CSV="${TASKS_CSV:-}"

# Base comparison set. Legacy Spectral Earth includes all available old
# checkpoints now that their wrapped patch-embed keys are remapped on load.
BACKBONES=(
  #"spectralearth_old_b"
  #"spectralearth_old_l"
  #"dofa_l"
  #"hypersigma_l"
  #"dofa_b"
  #"hypersigma_b"
  #"panopticon_b"
  #"carl"
  "specaware_b"
)
BACKBONES_CSV="${BACKBONES_CSV:-}"

PROTOCOLS=("linear")
DECODERS=("lightweight_multitap_head")
PROTOCOLS_CSV="${PROTOCOLS_CSV:-}"
DECODERS_CSV="${DECODERS_CSV:-}"
VIT_TAP_INDICES_CSV="${VIT_TAP_INDICES_CSV:-}"

MAX_EPOCHS="${MAX_EPOCHS:-100}"
DEBUG_MODE="${DEBUG_MODE:-false}"
RUN_FAMILY_TAG="${RUN_FAMILY_TAG:-final}"
SUBMISSION_THROTTLE_SECONDS="${SUBMISSION_THROTTLE_SECONDS:-0.15}"

# Match the current multi-sensor Hiera benchmark sweep by default.
LINEAR_LRS="${LINEAR_LRS:-0.003,0.001,0.0003,0.0001,0.00003,0.00001}"
#-0.003,0.001,0.0003,0.0001,0.00003,0.00001
# -0.0003,0.0001,0.00003,0.00001
LINEAR_WD="${LINEAR_WD:-0.001}"
FINETUNE_LRS="${FINETUNE_LRS:-0.0003,0.0001,0.00003,0.00001}"
FINETUNE_WD="${FINETUNE_WD:-0.001}"

DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-8}"
CLASSIFICATION_BATCH_SIZE="${CLASSIFICATION_BATCH_SIZE:-16}"
DEFAULT_NUM_DEVICES="${DEFAULT_NUM_DEVICES:-4}"

if [ -n "$TASKS_CSV" ]; then
  read_csv_array "$TASKS_CSV" TASKS
fi
if [ -n "$BACKBONES_CSV" ]; then
  read_csv_array "$BACKBONES_CSV" BACKBONES
fi
if [ -n "$PROTOCOLS_CSV" ]; then
  read_csv_array "$PROTOCOLS_CSV" PROTOCOLS
fi
if [ -n "$DECODERS_CSV" ]; then
  read_csv_array "$DECODERS_CSV" DECODERS
fi

if [ "${ENV}" = "juwels" ]; then
  LAUNCHER="slurm_juwels"
  PATHS_PROFILE="juwels"
  WANDB_OFFLINE="True"
elif [ "${ENV}" = "terrabyte" ]; then
  LAUNCHER="slurm"
  PATHS_PROFILE="default"
  WANDB_OFFLINE="False"
else
  echo "Error: ENV must be 'juwels' or 'terrabyte'. Got: ${ENV}"
  exit 1
fi

REPO_ROOT="$(pwd -P)"
OFFLINE_ASSET_ROOT="${OFFLINE_ASSET_ROOT:-${REPO_ROOT}/pretrained_models/comparison/offline_assets}"
HF_HOME_SHARED="${HF_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/hf_home}"
HF_HUB_CACHE_SHARED="${HF_HUB_CACHE_SHARED:-${HF_HOME_SHARED}/hub}"
TRANSFORMERS_CACHE_SHARED="${TRANSFORMERS_CACHE_SHARED:-${HF_HOME_SHARED}/transformers}"
SENTENCE_TRANSFORMERS_HOME_SHARED="${SENTENCE_TRANSFORMERS_HOME_SHARED:-${HF_HOME_SHARED}/sentence_transformers}"
TORCH_HOME_SHARED="${TORCH_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/torch_home}"
PANOPTICON_LOCAL_REPO="${PANOPTICON_LOCAL_REPO:-${OFFLINE_ASSET_ROOT}/panopticon_repo}"
PREFETCH_REMOTE_ASSETS="${PREFETCH_REMOTE_ASSETS:-true}"
CARL_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.carl_prefetched"
SPECAWARE_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.specaware_prefetched"
PANOPTICON_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.panopticon_prefetched"

mkdir -p \
  "${HF_HUB_CACHE_SHARED}" \
  "${TRANSFORMERS_CACHE_SHARED}" \
  "${SENTENCE_TRANSFORMERS_HOME_SHARED}" \
  "${TORCH_HOME_SHARED}"

export HF_HOME="${HF_HOME_SHARED}"
export HF_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_SHARED}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME_SHARED}"
export TORCH_HOME="${TORCH_HOME_SHARED}"

BASE_CMD="python src/train.py --config-name=downstream -m"

# --- Task Contracts ---

declare -A task_type_map
task_type_map=(
  ["bdforet"]="segmentation"
  ["bnetd"]="segmentation"
  ["cdl"]="segmentation"
  ["eurocrops"]="segmentation"
  ["desis_cdl"]="segmentation"
  ["eo1_cdl"]="segmentation"
  ["h2sr"]="segmentation"
  ["oxhyperminerals_emit_l2a"]="segmentation"
  ["gaofen5_wuhan"]="segmentation"
  ["nlcd"]="segmentation"
  ["treemap"]="segmentation"
  ["corine"]="classification"
)

declare -A task_source_sensor_map
task_source_sensor_map=(
  ["bdforet"]="enmap"
  ["bnetd"]="enmap"
  ["cdl"]="enmap"
  ["eurocrops"]="enmap"
  ["desis_cdl"]="desis"
  ["eo1_cdl"]="eo1"
  ["h2sr"]="ammis"
  ["oxhyperminerals_emit_l2a"]="emit"
  ["gaofen5_wuhan"]="gaofen5"
  ["nlcd"]="enmap"
  ["treemap"]="enmap"
  ["corine"]="enmap"
)

declare -A task_batch_size_map
task_batch_size_map=(
  ["bdforet"]=$DEFAULT_BATCH_SIZE
  ["bnetd"]=$DEFAULT_BATCH_SIZE
  ["cdl"]=$DEFAULT_BATCH_SIZE
  ["eurocrops"]=$DEFAULT_BATCH_SIZE
  ["desis_cdl"]=$DEFAULT_BATCH_SIZE
  ["eo1_cdl"]=$DEFAULT_BATCH_SIZE
  ["h2sr"]=$DEFAULT_BATCH_SIZE
  ["oxhyperminerals_emit_l2a"]=$DEFAULT_BATCH_SIZE
  ["gaofen5_wuhan"]=$DEFAULT_BATCH_SIZE
  ["nlcd"]=16
  ["treemap"]=16
  ["corine"]=$CLASSIFICATION_BATCH_SIZE
)

declare -A task_num_devices_map
task_num_devices_map=(
  ["bdforet"]=$DEFAULT_NUM_DEVICES
  ["bnetd"]=$DEFAULT_NUM_DEVICES
  ["cdl"]=$DEFAULT_NUM_DEVICES
  ["eurocrops"]=$DEFAULT_NUM_DEVICES
  ["desis_cdl"]=$DEFAULT_NUM_DEVICES
  ["eo1_cdl"]=2
  ["h2sr"]=$DEFAULT_NUM_DEVICES
  ["oxhyperminerals_emit_l2a"]=$DEFAULT_NUM_DEVICES
  ["gaofen5_wuhan"]=$DEFAULT_NUM_DEVICES
  ["nlcd"]=$DEFAULT_NUM_DEVICES
  ["treemap"]=$DEFAULT_NUM_DEVICES
  ["corine"]=$DEFAULT_NUM_DEVICES
)

# supported if explicitly requested with TASKS_CSV.

declare -A task_abbrev_map
task_abbrev_map=(
  ["oxhyperminerals_emit_l2a"]="oxh_l2a"
  ["gaofen5_wuhan"]="gfw"
)

# --- Backbone Contracts ---

declare -A backbone_alias_map
backbone_alias_map=(
  ["old_se_b"]="spectralearth_old_b"
  ["old_se_l"]="spectralearth_old_l"
  ["se_old_b"]="spectralearth_old_b"
  ["se_old_l"]="spectralearth_old_l"
)

declare -A backbone_patch_size_map
backbone_patch_size_map=(
  ["spectralearth_old_b"]="4"
  ["spectralearth_old_l"]="4"
  ["dofa_b"]="16"
  ["dofa_l"]="16"
  ["hypersigma_b"]="8"
  ["hypersigma_l"]="8"
  ["panopticon_b"]="14"
  ["carl"]="8"
  ["specaware_b"]="8"
)

declare -A backbone_img_size_map
backbone_img_size_map=(
  ["spectralearth_old_b"]="128"
  ["spectralearth_old_l"]="128"
  ["dofa_b"]="224"
  ["dofa_l"]="224"
  ["hypersigma_b"]="128"
  ["hypersigma_l"]="128"
  ["panopticon_b"]="224"
  ["carl"]="128"
  ["specaware_b"]="224"
)

declare -A backbone_img_size_field_map
backbone_img_size_field_map=(
  ["spectralearth_old_b"]="img_size"
  ["spectralearth_old_l"]="img_size"
  ["dofa_b"]="input_size"
  ["dofa_l"]="input_size"
  ["hypersigma_b"]="img_size"
  ["hypersigma_l"]="img_size"
  ["panopticon_b"]="img_size"
  ["carl"]="img_size"
  ["specaware_b"]="img_size"
)

declare -A backbone_short_map
backbone_short_map=(
  ["spectralearth_old_b"]="seold_b"
  ["spectralearth_old_l"]="seold_l"
  ["dofa_b"]="dofa_b"
  ["dofa_l"]="dofa_l"
  ["hypersigma_b"]="hsig_b"
  ["hypersigma_l"]="hsig_l"
  ["panopticon_b"]="pano_b"
  ["carl"]="carl"
  ["specaware_b"]="specaw_b"
)

declare -A weight_init_tag_map
weight_init_tag_map=(
  ["spectralearth_old_b"]="old_se_pretrained"
  ["spectralearth_old_l"]="old_se_pretrained"
  ["dofa_b"]="dofa_pretrained"
  ["dofa_l"]="dofa_pretrained"
  ["hypersigma_b"]="hypersigma_pretrained"
  ["hypersigma_l"]="hypersigma_pretrained"
  ["panopticon_b"]="panopticon_pretrained"
  ["carl"]="carl_pretrained"
  ["specaware_b"]="specaware_pretrained"
)

declare -A checkpoint_path_map
checkpoint_path_map=(
  ["spectralearth_old_b"]="pretrained_models/comparison/spectralearth_old/spec_vit_b/mae.pth"
  ["spectralearth_old_l"]="pretrained_models/comparison/spectralearth_old/spec_vit_l/mae.pth"
  ["dofa_b"]="pretrained_models/comparison/dofa_b/DOFA_ViT_base_e100.pth"
  ["dofa_l"]="pretrained_models/comparison/dofa_l/DOFA_ViT_large_e100.pth"
  ["hypersigma_b"]="pretrained_models/comparison/hypersigma_b/spat-vit-base-ultra-checkpoint-1599.pth"
  ["hypersigma_l"]="pretrained_models/comparison/hypersigma_l/spat-vit-large-ultra-checkpoint-1599.pth"
  ["panopticon_b"]="pretrained_models/comparison/panopticon/panopticon_vitb14_teacher.pth"
  ["carl"]="pretrained_models/comparison/carl/carl_ssl_checkpoint.ckpt"
  ["specaware_b"]="pretrained_models/comparison/specaware/SpecAware_Base.pth"
)

declare -A raw_meanstd_stats_path_map
raw_meanstd_stats_path_map=(
  ["enmap"]="data/statistics"
  ["desis"]="data/desis_statistics"
  ["eo1"]="data/eo1_statistics"
  ["emit"]="data/stats/comparison/raw_meanstd/emit"
  ["gaofen5"]="data/stats/comparison/raw_meanstd/gaofen5"
  ["ammis"]="data/stats/comparison/raw_meanstd/ammis"
  ["hyperview"]="data/hyperview_statistics"
)

declare -A dofa_stats_path_map
dofa_stats_path_map=(
  ["enmap"]="data/dofa_statistics"
  ["desis"]="data/desis_statistics"
  ["eo1"]="data/eo1_statistics"
  ["emit"]="data/stats/comparison/raw_meanstd/emit"
  ["gaofen5"]="data/stats/comparison/raw_meanstd/gaofen5"
  ["ammis"]="data/stats/comparison/raw_meanstd/ammis"
  ["hyperview"]="data/hyperview_statistics"
)

declare -A hypersigma_stats_path_map
hypersigma_stats_path_map=(
  ["enmap"]="data/hypersigma_statistics"
  ["desis"]="data/desis_hypersigma_statistics"
  ["eo1"]="data/eo1_hypersigma_statistics"
  ["emit"]="data/stats/comparison/hypersigma_scale/emit"
  ["gaofen5"]="data/stats/comparison/hypersigma_scale/gaofen5"
  ["ammis"]="data/stats/comparison/hypersigma_scale/ammis"
  ["hyperview"]="data/hyperview_hypersigma_statistics"
)

declare -A panopticon_stats_path_map
panopticon_stats_path_map=(
  ["enmap"]="data/panopticon_statistics"
  ["desis"]="data/desis_statistics"
  ["eo1"]="data/eo1_statistics"
  ["emit"]="data/stats/comparison/raw_meanstd/emit"
  ["gaofen5"]="data/stats/comparison/raw_meanstd/gaofen5"
  ["ammis"]="data/stats/comparison/raw_meanstd/ammis"
  ["hyperview"]="data/hyperview_statistics"
)

declare -A decoder_model_type_map
decoder_model_type_map=(
  ["conv_head"]="conv_seg"
  ["multiscale_conv_head"]="learnable_multiscale_conv_seg"
  ["lightweight_multitap_head"]="multi_tap_seg"
  ["upernet_head"]="upernet_seg"
)

declare -A decoder_name_map
decoder_name_map=(
  ["conv_head"]="conv"
  ["multiscale_conv_head"]="msc"
  ["lightweight_multitap_head"]="mtap"
  ["upernet_head"]="uper"
)

# --- Helpers ---

canonical_backbone_name() {
  local backbone="$1"
  if [ -n "${backbone_alias_map[$backbone]}" ]; then
    echo "${backbone_alias_map[$backbone]}"
  else
    echo "$backbone"
  fi
}

get_sensor_img_size() {
  local sensor_name="$1"
  python3 - "$sensor_name" <<'PYEOF'
import sys
import yaml

sensor_name = sys.argv[1]
path = f"configs/sensor/{sensor_name}.yaml"
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
img_size = cfg.get("img_size")
if img_size is None:
    raise SystemExit(f"Sensor config {path} does not define img_size.")
if isinstance(img_size, (list, tuple)):
    print(int(img_size[0]))
else:
    print(int(img_size))
PYEOF
}

get_device_overrides() {
  local num_devices="$1"
  local overrides="trainer.devices=${num_devices} hydra.launcher.tasks_per_node=${num_devices}"
  if [ "${ENV}" = "juwels" ]; then
    overrides="${overrides} hydra.launcher.gres=gpu:${num_devices}"
  else
    overrides="${overrides} hydra.launcher.gpus_per_node=${num_devices}"
  fi
  echo "${overrides}"
}

build_experiment_setup() {
  local task="$1"
  local task_type="$2"
  local run_family_tag="${3:-$RUN_FAMILY_TAG}"
  local base_task="$task"
  if [ -n "$run_family_tag" ]; then
    base_task="${base_task}_${run_family_tag}"
  fi
  if [ "$task_type" = "segmentation" ]; then
    echo "${base_task}_lr"
  else
    echo "${base_task}"
  fi
}

run_online_python() {
  HF_HUB_OFFLINE=0 \
  TRANSFORMERS_OFFLINE=0 \
  python3 "$@"
}

prefetch_timm_model() {
  local model_name="$1"
  echo "Prefetching timm weights for ${model_name} into ${HF_HUB_CACHE_SHARED}"
  run_online_python - "$model_name" <<'PYEOF'
import sys
import timm

model_name = sys.argv[1]
model = timm.create_model(model_name, pretrained=True)
del model
PYEOF
}

prefetch_sentence_transformer() {
  local model_name="$1"
  echo "Prefetching sentence-transformer ${model_name} into ${SENTENCE_TRANSFORMERS_HOME_SHARED}"
  run_online_python - "$model_name" "$SENTENCE_TRANSFORMERS_HOME_SHARED" <<'PYEOF'
import sys
from sentence_transformers import SentenceTransformer

model_name = sys.argv[1]
cache_dir = sys.argv[2]
model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")
del model
PYEOF
}

prefetch_panopticon_repo() {
  local repo_url="https://github.com/Panopticon-FM/panopticon.git"
  if [ -d "${PANOPTICON_LOCAL_REPO}/.git" ]; then
    echo "Using existing Panopticon repo cache at ${PANOPTICON_LOCAL_REPO}"
    return 0
  fi
  echo "Cloning Panopticon repo into ${PANOPTICON_LOCAL_REPO}"
  mkdir -p "$(dirname "${PANOPTICON_LOCAL_REPO}")"
  git clone --depth 1 "${repo_url}" "${PANOPTICON_LOCAL_REPO}"
}

declare -A prepared_remote_assets=()

ensure_backbone_offline_assets() {
  local backbone="$1"
  if [ "$PREFETCH_REMOTE_ASSETS" != "true" ]; then
    return 0
  fi
  if [ -n "${prepared_remote_assets[$backbone]}" ]; then
    return 0
  fi

  case "$backbone" in
    carl)
      if [ ! -f "${CARL_PREFETCH_MARKER}" ]; then
        prefetch_timm_model "vit_small_patch14_dinov2.lvd142m" || return 1
        prefetch_timm_model "vit_base_patch14_dinov2.lvd142m" || return 1
        prefetch_timm_model "timm/eva02_base_patch14_224.mim_in22k" || return 1
        touch "${CARL_PREFETCH_MARKER}"
      fi
      ;;
    specaware_b)
      if [ ! -f "${SPECAWARE_PREFETCH_MARKER}" ]; then
        prefetch_sentence_transformer "sentence-transformers/all-MiniLM-L6-v2" || return 1
        touch "${SPECAWARE_PREFETCH_MARKER}"
      fi
      ;;
    panopticon_b)
      if [ ! -f "${PANOPTICON_PREFETCH_MARKER}" ]; then
        prefetch_panopticon_repo || return 1
        touch "${PANOPTICON_PREFETCH_MARKER}"
      fi
      ;;
  esac

  prepared_remote_assets["$backbone"]=1
  return 0
}

enable_offline_runtime_env() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

hash_group_name() {
  local group_name="$1"
  python3 - "$group_name" <<'PYEOF'
import hashlib
import sys

print(hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest()[:10])
PYEOF
}

shorten_wandb_group_name() {
  local group_name="$1"

  if [ "${#group_name}" -le 128 ]; then
    echo "$group_name"
    return 0
  fi

  group_name=$(echo "$group_name" | sed \
    -e 's|^downstream/|ds/|' \
    -e 's|/linear/|/lin/|g' \
    -e 's|/finetune/|/ft/|g' \
    -e 's|_final_remap_lr|_frm_lr|g' \
    -e 's|_final_lr|_flr|g' \
    -e 's|_final_remap|_frm|g' \
    -e 's|_final|_fin|g' \
    -e 's|/old_se_pretrained|/oldse|g' \
    -e 's|/dofa_pretrained|/dofa|g' \
    -e 's|/hypersigma_pretrained|/hsig|g' \
    -e 's|/panopticon_pretrained|/pano|g' \
    -e 's|/carl_pretrained|/carl|g' \
    -e 's|/specaware_pretrained|/specaw|g')

  if [ "${#group_name}" -le 128 ]; then
    echo "$group_name"
    return 0
  fi

  local hash_suffix
  hash_suffix=$(hash_group_name "$group_name")
  local keep_chars=$((128 - ${#hash_suffix} - 1))
  if [ "$keep_chars" -lt 1 ]; then
    keep_chars=1
  fi
  echo "${group_name:0:keep_chars}-${hash_suffix}"
}

resolve_backbone_img_size() {
  local backbone="$1"
  local sensor="$2"
  local img_size="${backbone_img_size_map[$backbone]}"
  if [[ "$backbone" == hypersigma_* && "$sensor" = "emit" ]]; then
    echo "64"
    return 0
  fi
  if [ -n "$img_size" ]; then
    echo "$img_size"
    return 0
  fi
  get_sensor_img_size "$sensor"
}

validate_existing_path() {
  local label="$1"
  local path="$2"
  if [ -z "$path" ]; then
    echo "Error: missing path for ${label}." >&2
    return 1
  fi
  if [ ! -e "$path" ]; then
    echo "Error: ${label} path does not exist: ${path}" >&2
    return 1
  fi
  return 0
}

submit_command() {
  local full_cmd="$1"
  echo "Running Command:"
  echo "$full_cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  (
    eval "$full_cmd"
  ) &
  sleep "$SUBMISSION_THROTTLE_SECONDS"
}

build_weight_overrides() {
  local backbone="$1"
  local ckpt_path="${checkpoint_path_map[$backbone]}"
  validate_existing_path "${backbone} checkpoint" "$ckpt_path" || return 1

  case "$backbone" in
    spectralearth_old_*|dofa_*)
      echo "model.pretrained_weights=${ckpt_path}"
      ;;
    hypersigma_*)
      echo "model.backbone_config.pretrained=${ckpt_path}"
      ;;
    panopticon_b)
      echo "model.backbone_config.checkpoint_path=${ckpt_path} model.backbone_config.hub_repo=${PANOPTICON_LOCAL_REPO}"
      ;;
    carl)
      echo "model.backbone_config.ssl_ckpt_path=${ckpt_path}"
      ;;
    specaware_b)
      echo "model.backbone_config.pretrained=${ckpt_path}"
      ;;
    *)
      echo "Error: no checkpoint loading rule for backbone '${backbone}'." >&2
      return 1
      ;;
  esac
}

build_input_norm_overrides() {
  local backbone="$1"
  local sensor="$2"
  local stats_path=""

  case "$backbone" in
    spectralearth_old_*)
      stats_path="${raw_meanstd_stats_path_map[$sensor]}"
      validate_existing_path "${backbone}/${sensor} stats" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    dofa_*)
      stats_path="${dofa_stats_path_map[$sensor]}"
      validate_existing_path "${backbone}/${sensor} stats" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    hypersigma_*)
      stats_path="${hypersigma_stats_path_map[$sensor]}"
      validate_existing_path "${backbone}/${sensor} stats" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    panopticon_b)
      stats_path="${panopticon_stats_path_map[$sensor]}"
      validate_existing_path "${backbone}/${sensor} stats" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    carl|specaware_b)
      echo "data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
      ;;
    *)
      echo "Error: no input normalization rule for backbone '${backbone}'." >&2
      return 1
      ;;
  esac
}

build_spatial_overrides() {
  local backbone="$1"
  local img_size="$2"
  local img_size_field="${backbone_img_size_field_map[$backbone]}"
  local overrides="data.img_size=${img_size}"
  if [ -n "$img_size_field" ]; then
    overrides="${overrides} model.backbone_config.${img_size_field}=${img_size}"
  fi
  echo "$overrides"
}

build_multitap_overrides() {
  local model_type="$1"
  if [ "$model_type" != "multi_tap_seg" ]; then
    echo ""
    return 0
  fi

  # Competitor backbones are ViT-style in this script.  Their default taps are
  # inferred from depth in Python: 12-block models use [3,7,11]; deeper models
  # get evenly spaced semantic taps including the last layer.
  local overrides="model.multi_tap_is_vit=true"
  if [ -n "$VIT_TAP_INDICES_CSV" ]; then
    overrides="${overrides} model.tap_indices=[${VIT_TAP_INDICES_CSV}]"
  fi
  echo "$overrides"
}

build_protocol_hparams() {
  local protocol="$1"
  case "$protocol" in
    linear)
      echo "${LINEAR_LRS}|${LINEAR_WD}|true"
      ;;
    finetune)
      echo "${FINETUNE_LRS}|${FINETUNE_WD}|false"
      ;;
    *)
      echo "Error: unknown protocol '${protocol}'." >&2
      return 1
      ;;
  esac
}

if [ "$DEBUG_MODE" = "true" ]; then
  echo "========================================="
  echo "  DEBUG MODE - commands printed only"
  echo "========================================="
fi

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  task_type="${task_type_map[$task]}"
  source_sensor="${task_source_sensor_map[$task]}"
  if [ -z "$task_type" ] || [ -z "$source_sensor" ]; then
    echo "Warning: task '${task}' is not configured in this script. Skipping."
    continue
  fi

  bs="${task_batch_size_map[$task]}"
  num_devices="${task_num_devices_map[$task]}"
  if [ -z "$bs" ] || [ -z "$num_devices" ]; then
    echo "Warning: task '${task}' is missing batch/device settings. Skipping."
    continue
  fi

  experiment_setup=$(build_experiment_setup "$task" "$task_type")
  task_group_name="${task_abbrev_map[$task]:-$task}"
  group_experiment_setup=$(build_experiment_setup "$task_group_name" "$task_type")
  device_overrides=$(get_device_overrides "$num_devices")

  echo "===== Task: ${task} | sensor: ${source_sensor} | type: ${task_type} ====="

  for requested_backbone in "${BACKBONES[@]}"; do
    backbone=$(canonical_backbone_name "$requested_backbone")
    if [ ! -f "configs/backbone/${backbone}.yaml" ]; then
      echo "Warning: backbone config not found: configs/backbone/${backbone}.yaml. Skipping."
      continue
    fi
    ensure_backbone_offline_assets "$backbone" || {
      echo "Warning: failed to prefetch offline assets for ${backbone}. Skipping."
      continue
    }

    patch_size="${backbone_patch_size_map[$backbone]}"
    if [ -z "$patch_size" ]; then
      echo "Warning: no patch-size contract for backbone '${backbone}'. Skipping."
      continue
    fi

    img_size=$(resolve_backbone_img_size "$backbone" "$source_sensor") || {
      echo "Warning: could not resolve img_size for ${backbone}/${source_sensor}. Skipping."
      continue
    }
    input_norm_overrides=$(build_input_norm_overrides "$backbone" "$source_sensor") || continue
    weight_overrides=$(build_weight_overrides "$backbone") || continue
    spatial_overrides=$(build_spatial_overrides "$backbone" "$img_size")

    backbone_short="${backbone_short_map[$backbone]:-$backbone}"
    weight_init_tag="${weight_init_tag_map[$backbone]:-pretrained}"

    echo "--- Backbone: ${backbone} (patch_size: ${patch_size}, img_size: ${img_size}) ---"

    for protocol in "${PROTOCOLS[@]}"; do
      hparams=$(build_protocol_hparams "$protocol") || exit 1
      current_lrs="${hparams%%|*}"
      rest="${hparams#*|}"
      current_wd="${rest%%|*}"
      freeze_backbone_flag="${rest##*|}"

      current_lrs="${current_lrs// /}"
      current_wd="${current_wd// /}"

      if [ "$task_type" = "segmentation" ]; then
        for decoder in "${DECODERS[@]}"; do
          model_type="${decoder_model_type_map[$decoder]}"
          decoder_short="${decoder_name_map[$decoder]:-$decoder}"
          if [ -z "$model_type" ]; then
            echo "Warning: unknown decoder '${decoder}'. Skipping."
            continue
          fi
          multitap_overrides=$(build_multitap_overrides "$model_type")
          task_name="downstream/${group_experiment_setup}/${source_sensor}/${decoder_short}/${backbone_short}/${protocol}/${weight_init_tag}"
          wandb_group_name=$(shorten_wandb_group_name "$task_name")
          full_cmd="${BASE_CMD} experiment=downstream/${task}/base \
            backbone@model.backbone_config=${backbone} \
            sensor=${source_sensor} \
            sensor@data.sensor_config=${source_sensor} \
            experiment_setup=${experiment_setup} \
            protocol=${protocol} \
            weight_init=${weight_init_tag} \
            backbone_name=${backbone} \
            decoder_name=${decoder} \
            decoder@model.decoder_config=${decoder} \
            model.model_type=${model_type} \
            model.freeze_backbone=${freeze_backbone_flag} \
            model.learning_rate=${current_lrs} \
            model.weight_decay=${current_wd} \
            model.backbone_config.patch_size=${patch_size} \
            data.batch_size=${bs} \
            trainer.max_epochs=${MAX_EPOCHS} \
            logger.wandb.offline=${WANDB_OFFLINE} \
            logger.wandb.group=${wandb_group_name} \
            paths=${PATHS_PROFILE} \
            hydra.sweep.subdir=lrwd_'\${hydra.job.num}'_bs-${bs} \
            hydra/launcher=${LAUNCHER} \
            task_name=${task_name} \
            ${device_overrides} \
            ${spatial_overrides} \
            ${input_norm_overrides} \
            ${multitap_overrides} \
            ${weight_overrides}"

          submit_command "$full_cmd"
        done
      else
        task_name="downstream/${group_experiment_setup}/${source_sensor}/${backbone_short}/${protocol}/${weight_init_tag}"
        wandb_group_name=$(shorten_wandb_group_name "$task_name")
        full_cmd="${BASE_CMD} experiment=downstream/${task}/base \
          backbone@model.backbone_config=${backbone} \
          sensor=${source_sensor} \
          sensor@data.sensor_config=${source_sensor} \
          experiment_setup=${experiment_setup} \
          protocol=${protocol} \
          weight_init=${weight_init_tag} \
          backbone_name=${backbone} \
          model.freeze_backbone=${freeze_backbone_flag} \
          model.learning_rate=${current_lrs} \
          model.weight_decay=${current_wd} \
          model.backbone_config.patch_size=${patch_size} \
          data.batch_size=${bs} \
          trainer.max_epochs=${MAX_EPOCHS} \
          logger.wandb.offline=${WANDB_OFFLINE} \
          logger.wandb.group=${wandb_group_name} \
          paths=${PATHS_PROFILE} \
          hydra.sweep.subdir=lrwd_'\${hydra.job.num}'_bs-${bs} \
          hydra/launcher=${LAUNCHER} \
          task_name=${task_name} \
          ${device_overrides} \
          ${spatial_overrides} \
          ${input_norm_overrides} \
          ${weight_overrides}"

        submit_command "$full_cmd"
      fi
    done
  done
done
echo "===== All competitor downstream sweep launchers started. ====="
