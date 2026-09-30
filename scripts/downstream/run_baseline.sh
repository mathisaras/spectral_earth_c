#!/bin/bash
set -e

# cap all BLAS/OMP/MKL thread‐pools to 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Configuration ---

# Environment setting (juwels or terrabyte)
ENV="terrabyte"  # Change this to "terrabyte" for terrabyte environment

# 1. Specify the downstream tasks to run (including corine for classification)
#    Tasks with fixed native sensors (desis_cdl, eo1_cdl, h2sr, gaofen5_wuhan,
TASKS=(
    "oxhyperminerals_emit_l2a"
    #"gaofen5_wuhan"
)
TASKS_CSV="${TASKS_CSV:-}"

# 2. Define Hyperparameters per Task (using Bash Associative Arrays)
#    - Keys are the task names defined above.
#    - Values are comma-separated strings for LRs, single value for WD.
#    - Keep only 3 learning rates per protocol as requested.

declare -A linear_lrs_map
linear_lrs_map=(
    ["bdforet"]="0.03,0.01,0.003"
    ["bnetd"]="0.03,0.01,0.003"
    ["cdl"]="0.03,0.01,0.003"
    ["desis_cdl"]="0.03,0.01,0.003"
    ["eo1_cdl"]="0.03,0.01,0.003"
    ["eurocrops"]="0.03,0.01,0.003"
    ["gaofen5_wuhan"]="0.03,0.01,0.003"
    ["h2sr"]="0.03,0.01,0.003"
    ["oxhyperminerals_emit_l2a"]="0.03,0.01,0.003"
    ["nlcd"]="0.03,0.01,0.003"
    ["treemap"]="0.03,0.01,0.003"
    ["corine"]="0.03,0.01,0.003"
)

declare -A finetune_lrs_map
finetune_lrs_map=(
    ["bdforet"]="0.0003,0.0001,0.00003,0.00001"
    ["bnetd"]="0.0003,0.0001,0.00003,0.00001"
    ["cdl"]="0.0003,0.0001,0.00003,0.00001"
    ["desis_cdl"]="0.0003,0.0001,0.00003,0.00001"
    ["eo1_cdl"]="0.0003,0.0001,0.00003,0.00001"
    ["eurocrops"]="0.0003,0.0001,0.00003,0.00001"
    ["gaofen5_wuhan"]="0.0003,0.0001,0.00003,0.00001"
    ["h2sr"]="0.0003,0.0001,0.00003,0.00001"
    ["oxhyperminerals_emit_l2a"]="0.0003,0.0001,0.00003,0.00001"
    ["nlcd"]="0.0003,0.0001,0.00003,0.00001"
    ["treemap"]="0.0003,0.0001,0.00003,0.00001"
    ["corine"]="0.0003,0.0001,0.00003,0.00001"
)

declare -A linear_wd_map
linear_wd_map=(
    ["bdforet"]="0.00001"
    ["bnetd"]="0.00001"
    ["cdl"]="0.00001"
    ["desis_cdl"]="0.00001"
    ["eo1_cdl"]="0.00001"
    ["eurocrops"]="0.00001"
    ["gaofen5_wuhan"]="0.00001"
    ["h2sr"]="0.00001"
    ["oxhyperminerals_emit_l2a"]="0.00001"
    ["nlcd"]="0.00001"
    ["treemap"]="0.00001"
    ["corine"]="0.00001"
)

declare -A finetune_wd_map
finetune_wd_map=(
    ["bdforet"]="0.0001"
    ["bnetd"]="0.0001"
    ["cdl"]="0.0001"
    ["desis_cdl"]="0.0001"
    ["eo1_cdl"]="0.0001"
    ["eurocrops"]="0.0001"
    ["gaofen5_wuhan"]="0.0001"
    ["h2sr"]="0.0001"
    ["oxhyperminerals_emit_l2a"]="0.0001"
    ["nlcd"]="0.0001"
    ["treemap"]="0.0001"
    ["corine"]="0.0001"
)

# 3. Define Sensors, Backbones, Protocols, and Decoders
#    SENSORS is only used for tasks without a fixed native sensor.
SENSORS=("enmap")
BACKBONES=("spec_vit_b")  # Base names, will be mapped to sensor-specific names
PROTOCOLS=("linear" "finetune")
# Available decoder options for segmentation tasks:
# - conv_head: Standard single-scale decoder
# - multiscale_conv_head: Multi-scale decoder with channel reduction
# - lightweight_multitap_head: Shared lightweight multi-tap decoder
# - upernet_head: UperNet decoder with PSP and FPN modules
DECODERS=("conv_head")
SENSORS_CSV="${SENSORS_CSV:-}"
BACKBONES_CSV="${BACKBONES_CSV:-}"
PROTOCOLS_CSV="${PROTOCOLS_CSV:-}"
DECODERS_CSV="${DECODERS_CSV:-}"

# 4. Fixed weight initialization (random weights for baseline)
WEIGHT_INIT="${WEIGHT_INIT:-random}"
ENABLE_SENSOR_REMAP="${ENABLE_SENSOR_REMAP:-false}"
TARGET_SENSOR="${TARGET_SENSOR:-enmap}"
DEBUG_MODE="${DEBUG_MODE:-false}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi
if [ -n "$SENSORS_CSV" ]; then
  IFS=',' read -ra SENSORS <<< "$SENSORS_CSV"
fi
if [ -n "$BACKBONES_CSV" ]; then
  IFS=',' read -ra BACKBONES <<< "$BACKBONES_CSV"
fi
if [ -n "$PROTOCOLS_CSV" ]; then
  IFS=',' read -ra PROTOCOLS <<< "$PROTOCOLS_CSV"
fi
if [ -n "$DECODERS_CSV" ]; then
  IFS=',' read -ra DECODERS <<< "$DECODERS_CSV"
fi

# 5. Environment-specific launcher and paths configuration
if [ "${ENV}" = "juwels" ]; then
  LAUNCHER="slurm_juwels"
  PATHS_PROFILE="juwels"
  WANDB_OFFLINE="True"
elif [ "${ENV}" = "terrabyte" ]; then
  LAUNCHER="slurm"  # Remove "juwels" from launcher name
  PATHS_PROFILE="default"  # Use default paths instead of juwels
  WANDB_OFFLINE="False"
else
  echo "Error: ENV must be either 'juwels' or 'terrabyte'. Current value: ${ENV}"
  exit 1
fi

# 6. Sensor-specific configurations
declare -A sensor_backbone_map
sensor_backbone_map=(
    ["enmap_vit_s"]="vit_s"
    ["enmap_vit_b"]="spec_vit_b"
    ["enmap_spec_vit_b"]="spec_vit_b"
    ["desis_vit_b"]="spec_vit_b"
    ["desis_spec_vit_b"]="spec_vit_b"
    ["eo1_vit_b"]="spec_vit_b"
    ["eo1_spec_vit_b"]="spec_vit_b"
    ["emit_vit_b"]="spec_vit_b"
    ["emit_spec_vit_b"]="spec_vit_b"
    ["gaofen5_vit_b"]="spec_vit_b"
    ["gaofen5_spec_vit_b"]="spec_vit_b"
    ["ammis_vit_b"]="spec_vit_b"
    ["ammis_spec_vit_b"]="spec_vit_b"
    ["l8_vit_s"]="vit_s"
    ["l8_vit_b"]="vit_b"
    ["s2_vit_s"]="vit_s"
    ["s2_vit_b"]="vit_b"
)

declare -A sensor_patch_size_map
sensor_patch_size_map=(
    ["enmap"]="4"
    ["desis"]="4"
    ["eo1"]="4"
    ["emit"]="4"
    ["gaofen5"]="4"
    ["ammis"]="4"
    ["l8"]="4"
    ["s2"]="12"
)

# Competitor/foundation backbones have model-native token patch and image-size
# contracts.  Do not let the sensor-specific SpectralEarth patch-size defaults
# silently override those values when a comparison backbone is selected here.
declare -A backbone_patch_size_map
backbone_patch_size_map=(
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
    ["dofa_b"]="input_size"
    ["dofa_l"]="input_size"
    ["hypersigma_b"]="img_size"
    ["hypersigma_l"]="img_size"
    ["panopticon_b"]="img_size"
    ["carl"]="img_size"
    ["specaware_b"]="img_size"
)

# Model-specific input normalization contracts for comparison backbones.  These
# overrides stay in the benchmark script so the native SpectralEarth-MM data path
# remains untouched.
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

declare -A backbone_default_weight_init_map
backbone_default_weight_init_map=(
    ["panopticon_b"]="panopticon_pretrained"
)

# 7. Task type mapping (segmentation vs classification)
declare -A task_type_map
task_type_map=(
    ["bdforet"]="segmentation"
    ["bnetd"]="segmentation"
    ["cdl"]="segmentation"
    ["desis_cdl"]="segmentation"
    ["eo1_cdl"]="segmentation"
    ["gaofen5_wuhan"]="segmentation"
    ["eurocrops"]="segmentation"
    ["h2sr"]="segmentation"
    ["oxhyperminerals_emit_l2a"]="segmentation"
    ["nlcd"]="segmentation"
    ["treemap"]="segmentation"
    ["corine"]="classification"
)

# Tasks with a fixed native sensor should not be routed through the generic
# SENSORS list above.
declare -A task_source_sensor_map
task_source_sensor_map=(
    ["desis_cdl"]="desis"
    ["eo1_cdl"]="eo1"
    ["oxhyperminerals_emit_l2a"]="emit"
    ["gaofen5_wuhan"]="gaofen5"
    ["h2sr"]="ammis"
)

# Tasks listed here keep native and remapped baseline runs under the same
# downstream project namespace so they are easy to compare side by side.
declare -A task_project_mode_map
task_project_mode_map=(
    ["eo1_cdl"]="remap_project"
    ["gaofen5_wuhan"]="remap_project"
    ["h2sr"]="remap_project"
)

# Some fixed-sensor tasks should stay native even in broad sweeps where
# ENABLE_SENSOR_REMAP=true is set globally.
declare -A task_disable_sensor_remap_map
task_disable_sensor_remap_map=(
    ["oxhyperminerals_emit_l2a"]="true"
)

# Abbreviations used only in task_name / W&B group names.
declare -A task_abbrev_map
task_abbrev_map=(
    ["oxhyperminerals_emit_l2a"]="oxh_l2a"
)


# 8. Decoder to model type mapping
declare -A decoder_model_type_map
decoder_model_type_map=(
    ["conv_head"]="conv_seg"
    ["multiscale_conv_head"]="multiscale_conv_seg"
    ["lightweight_multitap_head"]="multi_tap_seg"
    ["upernet_head"]="upernet_seg"
)

resolve_backbone_config_name() {
  local sensor="$1"
  local backbone_base="$2"
  local backbone_key="${sensor}_${backbone_base}"

  if [ -n "${sensor_backbone_map[$backbone_key]}" ]; then
    echo "${sensor_backbone_map[$backbone_key]}"
    return 0
  fi

  if [ -f "configs/backbone/${backbone_base}.yaml" ]; then
    echo "${backbone_base}"
    return 0
  fi

  return 1
}

validate_remap_metadata() {
  local source_sensor="$1"
  local target_sensor="$2"
  python3 - "$source_sensor" "$target_sensor" <<'PYEOF'
import sys
import yaml

for sensor_name in sys.argv[1:]:
    path = f"configs/sensor/{sensor_name}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    spectral = cfg.get("spectral") or {}
    processed = spectral.get("processed_band_metadata_path")
    if not processed:
        raise SystemExit(
            f"Sensor '{sensor_name}' is missing spectral.processed_band_metadata_path "
            f"in {path}. Cross-sensor remapping requires validated spectral metadata."
        )
PYEOF
}

get_task_sensors() {
  local task="$1"
  if [ -n "${task_source_sensor_map[$task]}" ]; then
    printf '%s\n' "${task_source_sensor_map[$task]}"
  else
    printf '%s\n' "${SENSORS[@]}"
  fi
}

build_experiment_setup() {
  local task="$1"
  local task_type="$2"
  local remap_enabled="${3:-$ENABLE_SENSOR_REMAP}"
  if [ "$task_type" = "classification" ]; then
    echo "${task}"
    return 0
  fi
  if [ "$remap_enabled" = "true" ] || [ "${task_project_mode_map[$task]}" = "remap_project" ]; then
    echo "${task}_remap_lr"
  else
    echo "${task}_lr"
  fi
}

build_sensor_tag() {
  local source_sensor="$1"
  local target_sensor="$2"
  local remap_enabled="${3:-$ENABLE_SENSOR_REMAP}"
  if [ "$remap_enabled" = "true" ] && [ "$source_sensor" != "$target_sensor" ]; then
    echo "${source_sensor}2${target_sensor}"
  else
    echo "${source_sensor}"
  fi
}

build_model_input_overrides() {
  local backbone_config_name="$1"
  local source_sensor="$2"
  local target_sensor="$3"
  local remap_enabled="$4"

  if [ "$backbone_config_name" = "panopticon_b" ]; then
    if [ "$remap_enabled" = "true" ] && [ "$source_sensor" != "$target_sensor" ]; then
      echo "Error: panopticon_b should run on the native sensor grid, not through ENABLE_SENSOR_REMAP." >&2
      echo "       Panopticon consumes raw sensor values plus wavelength metadata; raw spectral remapping would be incoherent." >&2
      return 1
    fi

    local stats_path="${panopticon_stats_path_map[$source_sensor]}"
    if [ -z "$stats_path" ]; then
      echo "Error: Missing Panopticon stats mapping for sensor '${source_sensor}'." >&2
      return 1
    fi
    if [ ! -e "$stats_path" ]; then
      echo "Error: Missing Panopticon stats at '${stats_path}' for sensor '${source_sensor}'." >&2
      echo "       Stage compatible bandwise mean/std stats at that path before running Panopticon." >&2
      return 1
    fi

    echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
    return 0
  fi

  echo ""
  return 0
}

resolve_weight_init_tag() {
  local backbone_config_name="$1"
  local requested_weight_init="$2"
  local default_tag="${backbone_default_weight_init_map[$backbone_config_name]}"
  if [ "$requested_weight_init" = "random" ] && [ -n "$default_tag" ]; then
    echo "$default_tag"
  else
    echo "$requested_weight_init"
  fi
}

# --- Script Logic ---

BASE_CMD="python src/train.py --config-name=downstream -m"

for task in "${TASKS[@]}"; do
  echo "===== Processing Task: $task ====="

  # Check if hyperparameters are defined for the task
  if [[ -z "${linear_lrs_map[$task]}" ]] || [[ -z "${finetune_lrs_map[$task]}" ]] || \
     [[ -z "${linear_wd_map[$task]}" ]] || [[ -z "${finetune_wd_map[$task]}" ]]; then
    echo "Skipping task $task: Hyperparameters not defined."
    continue
  fi

  # Get task type
  task_type="${task_type_map[$task]}"
  if [ -z "$task_type" ]; then
    echo "Skipping task $task: Task type not defined."
    continue
  fi

  mapfile -t task_sensors < <(get_task_sensors "$task")
  for source_sensor in "${task_sensors[@]}"; do
    task_enable_sensor_remap="${ENABLE_SENSOR_REMAP}"
    if [ "${task_disable_sensor_remap_map[$task]}" = "true" ]; then
      task_enable_sensor_remap="false"
    fi

    target_sensor="$source_sensor"
    if [ "$task_enable_sensor_remap" = "true" ]; then
      if [ -z "$TARGET_SENSOR" ]; then
        echo "Error: ENABLE_SENSOR_REMAP=true requires TARGET_SENSOR to be set."
        exit 1
      fi
      target_sensor="$TARGET_SENSOR"
      validate_remap_metadata "$source_sensor" "$target_sensor"
    fi

    sensor_tag=$(build_sensor_tag "$source_sensor" "$target_sensor" "$task_enable_sensor_remap")
    task_group_name="${task_abbrev_map[$task]:-$task}"
    echo "===== Processing Sensor: ${sensor_tag} ====="

    declare -A seen_backbone_configs=()

    for backbone_base in "${BACKBONES[@]}"; do
      backbone_config_name=$(resolve_backbone_config_name "$target_sensor" "$backbone_base") || {
        echo "Skipping unknown backbone combination: ${target_sensor}_${backbone_base}"
        continue
      }

      if [ -n "${seen_backbone_configs[$backbone_config_name]}" ]; then
        echo "Skipping duplicate resolved backbone config: ${backbone_base} -> ${backbone_config_name}"
        continue
      fi
      seen_backbone_configs["$backbone_config_name"]=1

      backbone_patch_size="${backbone_patch_size_map[$backbone_config_name]}"
      if [ -n "$backbone_patch_size" ]; then
        patch_size="$backbone_patch_size"
      else
        # Default SpectralEarth baselines still use the sensor-specific token patch size.
        patch_size="${sensor_patch_size_map[$target_sensor]}"
      fi
      img_size_override="${backbone_img_size_map[$backbone_config_name]}"
      img_size_field="${backbone_img_size_field_map[$backbone_config_name]}"
      if [[ "$backbone_config_name" == hypersigma_* && "$target_sensor" = "emit" ]]; then
        # EMIT patches are native 64x64, which also matches the HyperSIGMA
        # checkpoint's 8x8 token positional grid with patch_size=8.
        img_size_override="64"
      fi
      
      if [ -z "$patch_size" ]; then
        echo "Skipping unknown sensor: $target_sensor"
        continue
      fi

      backbone_contract_msg="patch_size: $patch_size"
      if [ -n "$img_size_override" ]; then
        backbone_contract_msg="${backbone_contract_msg}, img_size: ${img_size_override}"
      fi
      echo "--- Backbone: $backbone_config_name (${backbone_contract_msg}) ---"
      effective_weight_init=$(resolve_weight_init_tag "$backbone_config_name" "$WEIGHT_INIT")

      for protocol in "${PROTOCOLS[@]}"; do
        echo "--- Protocol: $protocol ---"

        # Select LRs and WD based on protocol and task
        current_lrs=""
        current_wd=""
        freeze_backbone_flag=""

        if [ "$protocol" = "linear" ]; then
          current_lrs="${linear_lrs_map[$task]}"
          current_wd="${linear_wd_map[$task]}"
          freeze_backbone_flag="true"
        elif [ "$protocol" = "finetune" ]; then
          current_lrs="${finetune_lrs_map[$task]}"
          current_wd="${finetune_wd_map[$task]}"
          freeze_backbone_flag="false"
        else
          echo "Unknown protocol: $protocol"
          continue
        fi

        # Define experiment setup string (used for organizing outputs/logs / W&B project)
        experiment_setup=$(build_experiment_setup "$task" "$task_type" "$task_enable_sensor_remap")
        group_experiment_setup=$(build_experiment_setup "$task_group_name" "$task_type" "$task_enable_sensor_remap")

        # Loop over decoders for segmentation tasks
        if [ "$task_type" = "segmentation" ]; then
          for decoder in "${DECODERS[@]}"; do
            echo "--- Decoder: $decoder ---"
            
            # Get model type for this decoder
            model_type="${decoder_model_type_map[$decoder]}"
            if [ -z "$model_type" ]; then
              echo "Skipping unknown decoder: $decoder"
              continue
            fi

            # Manually loop over learning rates and weight decays
            IFS=',' read -ra lr_list <<< "${current_lrs}"
            IFS=',' read -ra wd_list <<< "${current_wd}"
            for lr in "${lr_list[@]}"; do
              for wd in "${wd_list[@]}"; do
                # Build the sweep subdir with decoder, lr- and wd- prefixes
                subdir="${sensor_tag}/${experiment_setup}/${backbone_config_name}/${decoder}/${protocol}/lr-${lr}/wd-${wd}"
                echo "Subdir: ${subdir}"

                # Construct the base command with common parameters
                base_cmd="${BASE_CMD} experiment=downstream/${task}/base \
                  backbone@model.backbone_config=${backbone_config_name} \
                  sensor=${target_sensor} \
                  sensor@data.sensor_config=${source_sensor} \
                  experiment_setup=${experiment_setup} \
                  protocol=${protocol} \
                  weight_init=${effective_weight_init} \
                  backbone_name=${backbone_config_name} \
                  decoder_name=${decoder} \
                  task_name=downstream/${group_experiment_setup}/${sensor_tag}/${decoder}/${backbone_config_name}/${protocol}/${effective_weight_init} \
                  model.freeze_backbone=${freeze_backbone_flag} \
                  model.learning_rate=${lr} \
                  model.weight_decay=${wd} \
                  model.backbone_config.patch_size=${patch_size} \
                  trainer.max_epochs=${MAX_EPOCHS} \
                  logger.wandb.offline=${WANDB_OFFLINE} \
                  paths=${PATHS_PROFILE} \
                  hydra.sweep.subdir=${subdir} \
                  hydra/launcher=${LAUNCHER}"
                if [ -n "$img_size_override" ]; then
                  base_cmd="${base_cmd} data.img_size=${img_size_override}"
                fi
                if [ -n "$img_size_override" ] && [ -n "$img_size_field" ]; then
                  base_cmd="${base_cmd} model.backbone_config.${img_size_field}=${img_size_override}"
                fi
                input_norm_overrides=$(build_model_input_overrides "$backbone_config_name" "$source_sensor" "$target_sensor" "$task_enable_sensor_remap") || exit 1
                if [ -n "$input_norm_overrides" ]; then
                  base_cmd="${base_cmd} ${input_norm_overrides}"
                fi

                # Add decoder-specific parameters
                full_cmd="${base_cmd} \
                  decoder@model.decoder_config=${decoder} \
                  model.model_type=${model_type}"
                if [ "$task_enable_sensor_remap" = "true" ] && [ "$source_sensor" != "$target_sensor" ]; then
                  full_cmd="${full_cmd} +sensor@data.target_sensor_config=${target_sensor}"
                fi

                # Print and run the command (run in background with '&')
                echo "Running Command:"
                echo "$full_cmd"
                if [ "$DEBUG_MODE" = "true" ]; then
                  echo "[DEBUG] Skipping execution."
                else
                  eval $full_cmd &
                fi

                # Optional: Add a small delay between job submissions if needed
                # sleep 2
              done
            done
          done # End decoder loop
        else
          # Classification tasks: no decoder loop needed
          # Manually loop over learning rates and weight decays
          IFS=',' read -ra lr_list <<< "${current_lrs}"
          IFS=',' read -ra wd_list <<< "${current_wd}"
          for lr in "${lr_list[@]}"; do
            for wd in "${wd_list[@]}"; do
              # Build the sweep subdir with lr- and wd- prefixes
              subdir="${sensor_tag}/${experiment_setup}/${backbone_config_name}/${protocol}/lr-${lr}/wd-${wd}"
              echo "Subdir: ${subdir}"

              # Construct the base command with common parameters
              base_cmd="${BASE_CMD} experiment=downstream/${task}/base \
                backbone@model.backbone_config=${backbone_config_name} \
                sensor=${target_sensor} \
                sensor@data.sensor_config=${source_sensor} \
                experiment_setup=${experiment_setup} \
                protocol=${protocol} \
                weight_init=${effective_weight_init} \
                backbone_name=${backbone_config_name} \
                task_name=downstream/${group_experiment_setup}/${sensor_tag}/${backbone_config_name}/${protocol}/${effective_weight_init} \
                model.freeze_backbone=${freeze_backbone_flag} \
                model.learning_rate=${lr} \
                model.weight_decay=${wd} \
                model.backbone_config.patch_size=${patch_size} \
                trainer.max_epochs=${MAX_EPOCHS} \
                logger.wandb.offline=${WANDB_OFFLINE} \
                paths=${PATHS_PROFILE} \
                hydra.sweep.subdir=${subdir} \
                hydra/launcher=${LAUNCHER}"
              if [ -n "$img_size_override" ]; then
                base_cmd="${base_cmd} data.img_size=${img_size_override}"
              fi
              if [ -n "$img_size_override" ] && [ -n "$img_size_field" ]; then
                base_cmd="${base_cmd} model.backbone_config.${img_size_field}=${img_size_override}"
              fi
              input_norm_overrides=$(build_model_input_overrides "$backbone_config_name" "$source_sensor" "$target_sensor" "$task_enable_sensor_remap") || exit 1
              if [ -n "$input_norm_overrides" ]; then
                base_cmd="${base_cmd} ${input_norm_overrides}"
              fi

              full_cmd="${base_cmd}"
              if [ "$task_enable_sensor_remap" = "true" ] && [ "$source_sensor" != "$target_sensor" ]; then
                full_cmd="${full_cmd} +sensor@data.target_sensor_config=${target_sensor}"
              fi

              # Print and run the command (run in background with '&')
              echo "Running Command:"
              echo "$full_cmd"
              if [ "$DEBUG_MODE" = "true" ]; then
                echo "[DEBUG] Skipping execution."
              else
                eval $full_cmd &
              fi

              # Optional: Add a small delay between job submissions if needed
              # sleep 2
            done
          done
        fi
      done # End protocol loop
    done # End backbone loop
  done # End sensor loop
done # End task loop

echo "===== All baseline downstream jobs submitted. ====="
# Wait for all background jobs to finish (optional, remove if you want the script to exit immediately)
# wait 
