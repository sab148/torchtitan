#!/usr/bin/env bash

export PYTHONUNBUFFERED=1 

export HF_HOME=/p/scratch/nxtaim-1/benassou1/clean_repo/torchtitan/assets/hf
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
unset HF_HUB_OFFLINE  # only if you need downloads

# export HF_HOME=/p/project1/atmlaml/$USER/hf
# export HF_DATASETS_CACHE=$HF_HOME/datasets

# Default to local HF cache if present (used for offline runs).
# Prefer the real HuggingFace cache if it exists.
HF_CACHE_PREFERRED="/p/home/jusers/benassou1/juwels/.cache/huggingface"
HF_CACHE_FALLBACK="/p/scratch/nxtaim-1/benassou1/clean_repo/torchtitan/assets/hf"
# Prefer the real HF cache if it exists. Override any pre-set HF_HOME to avoid mismatches.
if [ -d "$HF_CACHE_PREFERRED/hub" ]; then
    export HF_HOME="$HF_CACHE_PREFERRED"
elif [ -d "$HF_CACHE_FALLBACK" ]; then
    export HF_HOME="$HF_CACHE_FALLBACK"
fi
if [ -n "${HF_HOME:-}" ] && [ -z "${HF_DATASETS_CACHE:-}" ]; then
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
fi
# Keep all HF caches colocated so offline lookups work reliably.
if [ -n "${HF_HOME:-}" ]; then
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/hub"
    echo "HF_HOME=$HF_HOME"
    echo "HF_HUB_CACHE=$HF_HUB_CACHE"
fi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Use local FLUX.2 autoencoder weights when available (offline-friendly).
AE_DEFAULT="/p/scratch/nxtaim-1/benassou1/clean_repo/torchtitan/assets/hf/FLUX.2-dev/ae.safetensors"
if [ -z "${AE_MODEL_PATH:-}" ] && [ -f "$AE_DEFAULT" ]; then
    export AE_MODEL_PATH="$AE_DEFAULT"
fi

# this one used to add the NVIDIA's CUDA Runtime Libraries to the LD_LIBRARY_PATH
NPKG=$(python -c "import site, os; d=[p for p in site.getsitepackages() if p.endswith('site-packages')][0]+'/nvidia'; print(d)")
# Only touch LD_LIBRARY_PATH if that directory exists
# if [ -d "$NPKG" ]; then
#     export LD_LIBRARY_PATH="$NPKG/nvjitlink/lib:$NPKG/cusparse/lib:${LD_LIBRARY_PATH:-}"
#     export LD_LIBRARY_PATH="$NPKG/nvshmem/lib:$LD_LIBRARY_PATH"
#     export NVSHMEM_DIR="$NPKG/nvshmem"
#     export PATH="${NVSHMEM_DIR}/bin:$PATH"

#     export LD_LIBRARY_PATH="$(
#     printf '%s' "${LD_LIBRARY_PATH:-}" \
#     | tr ':' '\n' \
#     | grep -vE '/stages/20[0-9]{2}/software/CUDA' \
#     | awk 'NF' \
#     | paste -sd: -
#     )"
# fi



NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODEL=${MODEL:-"flux2"}
CONFIG=${CONFIG:-"flux2_klein_4b_dataconsolidation"}
NNODES=${SLURM_JOB_NUM_NODES:-1}
NPROC_PER_NODE=${GPUS_PER_NODE:-$NGPU}
NODE_RANK=${SLURM_NODEID:-0}
RDZV_ID=${SLURM_JOB_ID:-flux2-local}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export PYTHONPATH=/p/scratch/nxtaim-1/benassou1/clean_repo/torchtitan:${PYTHONPATH}

# Keep the cluster scientific stack consistent: prefer the module-provided
# SciPy/numpy/h5py paths ahead of the venv site-packages to avoid ABI mismatches.
export PYTHONPATH="$(python3 - <<'PYTHONPATH_REORDER'
import os

paths = [p for p in os.environ.get('PYTHONPATH', '').split(':') if p]
priority_markers = (
    '/software/h5py/',
    '/software/SciPy-bundle/',
)
front = []
rest = []
seen = set()
for path in paths:
    if path in seen:
        continue
    seen.add(path)
    if any(marker in path for marker in priority_markers):
        front.append(path)
    else:
        rest.append(path)
print(':'.join(front + rest))
PYTHONPATH_REORDER
)"

PYTORCH_ALLOC_CONF="expandable_segments:True" \

export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_backend c10d \
    --rdzv_id="$RDZV_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_conf=is_host=$(if ((NODE_RANK)); then echo 0; else echo 1; fi) \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    torchtitan/models/flux2/trainer.py \
    --module "$MODEL" \
    --config "$CONFIG" \
    --training.local_batch_size "${BS:-4}" \
    --training.steps "${STEPS:-100}" \
    --metrics.log_freq "${LOG_FREQ:-1}"
