#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --nodes=1
#SBATCH --partition=develbooster
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32  # 80 physical cores per node.
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH -o cache/slurm_output/%j_%a.log  # %j will be replaced by the job ID, %a by array index
#SBATCH --array=1

set -x
ulimit -c 0

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

# Without this, srun does not inherit cpus-per-task from sbatch.
echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=7010
export GPUS_PER_NODE=4

echo "MASTER_ADDR:MASTER_PORT=""$MASTER_ADDR":"$MASTER_PORT"
echo "----------------------------------"
export DEVICES_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_NODES="$SLURM_JOB_NUM_NODES"
export GLOO_SOCKET_IFNAME=ib0

# Try to reduce link flips.
export NCCL_IB_TIMEOUT=100
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10

echo "Job id: $SLURM_JOB_ID"
# We activate our environemnt
source sc_venv_template/activate.sh


# Set a default TMPDIR if not already set
export WANDB_MODE=offline

# train_configs_folder="launch_scripts/benchmark/optg-5T/configs"
# train_config="moe_opt_g_5T.toml"
RUN_NAME=${RUN_NAME:-"flux2-train"}
export MODULE=${MODULE:-"flux2"}
export CONFIG=${CONFIG:-"flux2_klein_4b"}

# it will run on 4096 seqlen, local BS = 12, steps = 79473.  Should run 256 GPUS to be total of 1T tokens 

steps=100
# BS=9
# gradient_accumulation_steps=3
BS=10
gradient_accumulation_steps=1
total_BS=$((BS * gradient_accumulation_steps * GPUS_PER_NODE * NUM_NODES))
# total_BS=10368

lr=0.0761092



mkdir -p storage
mkdir -p storage/jobs_outputs/${RUN_NAME}/
mkdir -p storage/exp_${RUN_NAME}/
exp_name=storage/exp_${RUN_NAME}/${RUN_NAME}/${RUN_NAME}-lr.${lr}-bs.${BS}-global_bs.${total_BS}-steps.${steps}-${SLURM_NNODES}
LOG_NAME="storage/jobs_outputs/${RUN_NAME}/${RUN_NAME}-lr.${lr}-bs.${BS}-global_bs.${total_BS}-steps.${steps}-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${SLURM_NNODES}.log"
mkdir -p "$(dirname "$LOG_NAME")"


# exec >>"$LOG_NAME" 2>&1

echo "start training 8"

# params=" \
#   --job.config_file $train_configs_folder/$train_config \
#   --job.dump_folder $exp_name \
#   --training.steps $steps \
#   --optimizer.lr $lr \
#   --training.local_batch_size $BS \
#   --training.global_batch_size $total_BS \
#   --parallelism.data_parallel_replicate_degree 1 \
#   --metrics.log_freq 5 \
#   --debug.seed 1012  \
#   --model.flavor 30bA3b
# "
# export params
#   --metrics.log_freq 1  \
# echo "params: $params"
echo "Launching training..."

export HF_HOME="${HF_HOME:-/p/scratch/atmlaml/benassou1/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# export WANDB_PROJECT=opt-g_5T
# export WANDB_GROUP="${RUN_NAME}"
# export WANDB_NAME="${RUN_NAME}-lr.${lr}-bs.${BS}-global_bs.${total_BS}-steps.${steps}-JOB-${SLURM_ARRAY_TASK_ID}"
# export WANDB_RUN_ID="$WANDB_NAME"

export LOGLEVEL=INFO
export LOG_RANK="0"
export TORCH_CUDA_ARCH_LIST="80"  # For A100 GPUs
echo "Address: ${MASTER_ADDR}:${MASTER_PORT}"


echo " <<<< torchtitan     git commit $(git -C resources/torchtitan rev-parse HEAD)"
echo " <<<< configurations git commit $(git -C reproduce_cfgs rev-parse HEAD)"


echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

# this one used to add the NVIDIA's CUDA Runtime Libraries to the LD_LIBRARY_PATH
NPKG=$(python -c "import site, os; d=[p for p in site.getsitepackages() if p.endswith('site-packages')][0]+'/nvidia'; print(d)")
# Only touch LD_LIBRARY_PATH if that directory exists
if [ -d "$NPKG" ]; then
    export LD_LIBRARY_PATH="$NPKG/nvjitlink/lib:$NPKG/cusparse/lib:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$NPKG/nvshmem/lib:$LD_LIBRARY_PATH"
    export NVSHMEM_DIR="$NPKG/nvshmem"
    export PATH="${NVSHMEM_DIR}/bin:$PATH"

    export LD_LIBRARY_PATH="$(
    printf '%s' "${LD_LIBRARY_PATH:-}" \
    | tr ':' '\n' \
    | grep -vE '/stages/20[0-9]{2}/software/CUDA' \
    | awk 'NF' \
    | paste -sd: -
    )"
fi

srun bash launch.sh 
#   --encoder.text_encoder_cache_mode=read \
#   --encoder.text_encoder_cache_dir=/p/scratch/nxtaim-1/benassou1/flux2_text_cache

echo "RUNNING DONE!"
