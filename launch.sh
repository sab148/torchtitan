# export TMPDIR="$TMPDIR/$SLURM_JOB_ID/${SLURM_PROCID}"
# export TEMP="$TMPDIR"
# export TMP="$TMPDIR"
# mkdir -p "$TMPDIR"
# echo "Rank $SLURM_PROCID: Using TMPDIR=$TMPDIR for training"
export PYTHONUNBUFFERED=1 

#export PYTHONPATH=$(pwd)/resources/torchtitan:$PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH


torchrun --nnodes=${SLURM_JOB_NUM_NODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --rdzv_backend c10d \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
  --rdzv_conf=is_host=$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  torchtitan/train.py $params