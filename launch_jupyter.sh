# export TMPDIR="$TMPDIR/$SLURM_JOB_ID/${SLURM_PROCID}"
# export TEMP="$TMPDIR"
# export TMP="$TMPDIR"
# mkdir -p "$TMPDIR"
# echo "Rank $SLURM_PROCID: Using TMPDIR=$TMPDIR for training"
export PYTHONUNBUFFERED=1 

#export PYTHONPATH=$(pwd)/resources/torchtitan:$PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "SCRIPT_DIR: $SCRIPT_DIR"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/assets/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
unset HF_HUB_OFFLINE  # only if you need downloads

HF_CACHE_PREFERRED="/p/home/jusers/benassou1/juwels/.cache/huggingface"
HF_CACHE_FALLBACK="${SCRIPT_DIR}/assets/hf"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Use local FLUX.2 autoencoder weights when available (offline-friendly).
AE_DEFAULT="${SCRIPT_DIR}/assets/hf/FLUX.2-dev/ae.safetensors"
if [ -z "${AE_MODEL_PATH:-}" ] && [ -f "$AE_DEFAULT" ]; then
    export AE_MODEL_PATH="$AE_DEFAULT"
fi

QWEN3_4B_DEFAULT="${SCRIPT_DIR}/flux2_klein/text_encoder"
if [ -z "${FLUX2_QWEN3_4B_MODEL_PATH:-}" ] && [ -d "$QWEN3_4B_DEFAULT" ]; then
    export FLUX2_QWEN3_4B_MODEL_PATH="$QWEN3_4B_DEFAULT"
fi


export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}/torchtitan/models/flux2:${PYTHONPATH}"

DUMP_FOLDER="${DUMP_FOLDER:-${SCRIPT_DIR}/outputs/${SLURM_JOB_ID:-local}}"
CHECKPOINT_ARGS=()
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    CHECKPOINT_ARGS+=(--checkpoint.initial_load_path "$RESUME_CHECKPOINT")
    CHECKPOINT_ARGS+=(--checkpoint.load_step "${CHECKPOINT_LOAD_STEP:--1}")
elif [ -n "${CHECKPOINT_LOAD_STEP:-}" ]; then
    CHECKPOINT_ARGS+=(--checkpoint.load_step "$CHECKPOINT_LOAD_STEP")
fi
if [ -n "${CHECKPOINT_EXCLUDE_FROM_LOADING:-}" ]; then
    CHECKPOINT_ARGS+=(--checkpoint.exclude_from_loading "$CHECKPOINT_EXCLUDE_FROM_LOADING")
fi

python -m torch.distributed.run --nnodes=${SLURM_JOB_NUM_NODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --rdzv_backend c10d \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
  --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  -m torchtitan.models.flux2.trainer \
  --module "$MODEL" \
  --config "$CONFIG" \
  --dump_folder "$DUMP_FOLDER" \
  "${CHECKPOINT_ARGS[@]}" \
  --training.local_batch_size "${BS:-8}" \
  --training.steps "${STEPS:-30000}" \
  --metrics.log_freq "${LOG_FREQ:-100}" \
  --encoder.text_encoder_cache_mode "${TEXT_ENCODER_CACHE_MODE:-off}" \
  --encoder.text_encoder_cache_dir "${TEXT_ENCODER_CACHE_DIR:-${SCRIPT_DIR}/storage/text_encoder_cache}"
