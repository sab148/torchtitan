#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# export TMPDIR="$TMPDIR/$SLURM_JOB_ID/${SLURM_PROCID}"
# export TEMP="$TMPDIR"
# export TMP="$TMPDIR" #/p/project1/atmlaml/benassou1/torchtitan/torchtitan/models/flux/train_configs/flux_schnell_model.toml
# mkdir -p "$TMPDIR"
# echo "Rank $SLURM_PROCID: Using TMPDIR=$TMPDIR for training". #./torchtitan/models/llama3/train_configs/llama3_8b.toml
export PYTHONUNBUFFERED=1

HF_ROOT="${HF_HOME:-/p/scratch/atmlaml/benassou1/.cache}"
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

AE_DEFAULT="${SCRIPT_DIR}/assets/hf/FLUX.2-dev/ae.safetensors"
if [ -z "${AE_MODEL_PATH:-}" ] && [ -f "$AE_DEFAULT" ]; then
    export AE_MODEL_PATH="$AE_DEFAULT"
fi

# This used to add NVIDIA CUDA runtime libraries to LD_LIBRARY_PATH.
NPKG=$(python -c "import site, os; d=[p for p in site.getsitepackages() if p.endswith('site-packages')][0]+'/nvidia'; print(d)")
# Only touch LD_LIBRARY_PATH if that directory exists.
# if [ -d "$NPKG" ]; then
#     export LD_LIBRARY_PATH="$NPKG/nvjitlink/lib:$NPKG/cusparse/lib:${LD_LIBRARY_PATH:-}"
#     export LD_LIBRARY_PATH="$NPKG/nvshmem/lib:$LD_LIBRARY_PATH"
#     export NVSHMEM_DIR="$NPKG/nvshmem"
#     export PATH="${NVSHMEM_DIR}/bin:$PATH"
# fi
unset NPKG

NGPU=${NGPU:-"8"}
GPUS_PER_NODE=${GPUS_PER_NODE:-${NGPU}}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-${MODEL:-"flux2"}}
CONFIG=${CONFIG:-"flux2_klein_4b"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_JOB_ID=${SLURM_JOB_ID:-local}

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

extra_args=("$@")

echo "Launching module=${MODULE} config=${CONFIG}"
if [ "${#extra_args[@]}" -gt 0 ]; then
    printf 'Extra args:'
    printf ' %q' "${extra_args[@]}"
    printf '\n'
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_conf=is_host=$(if (( SLURM_NODEID )); then echo 0; else echo 1; fi) \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    "${extra_args[@]}"
