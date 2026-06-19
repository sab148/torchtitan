#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --nodes=32
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72  # 80 physical cores per node.
#SBATCH --time=00:04:00
#SBATCH --gres=gpu:4
#SBATCH -o %j_%a.log  # %j will be replaced by the job ID, %a by array index
#SBATCH --array=1

set -x
ulimit -c 0

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
source ./llm_env/activate.sh

# Set a default TMPDIR if not already set
export WANDB_MODE=offline

train_configs_folder="launch_scripts/benchmark/llama3"
train_config="llama3_8b.toml"
RUN_NAME="[benchmark]-llama3-jupiter"

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
exec >>"$LOG_NAME" 2>&1




#--job.dump_folder $exp_name \
#--training.steps $steps \
#--optimizer.lr $lr \
#--training.local_batch_size $BS \
#--training.global_batch_size $total_BS \
#--parallelism.data_parallel_replicate_degree 1 \
#--metrics.log_freq 5 \
#--debug.seed 1012  \
#--model.flavor 30bA3b

params=" \
  --job.config_file $train_configs_folder/$train_config \
  --training.local_batch_size=2 \
"
export params
#   --metrics.log_freq 1  \
echo "params: $params"
echo "Launching training..."

path=$(grep -oP '(?<=--job.dump_folder )[^ ]+' <<< "$params")

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_PROJECT=opt-g_5T
export WANDB_GROUP="${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}-lr.${lr}-bs.${BS}-global_bs.${total_BS}-steps.${steps}-JOB-${SLURM_ARRAY_TASK_ID}"
export WANDB_RUN_ID="$WANDB_NAME"

export LOGLEVEL=INFO
export LOG_RANK="0"
echo "Address: ${MASTER_ADDR}:${MASTER_PORT}"


echo " <<<< torchtitan     git commit $(git -C torchtitan rev-parse HEAD)"
##echo " <<<< torchtitan     git commit $(git -C torchtitan rev-parse HEAD)"

echo " <<<< configurations git commit $(git -C reproduce_cfgs rev-parse HEAD)"

srun bash launch.sh
echo "RUNNING DONE!"