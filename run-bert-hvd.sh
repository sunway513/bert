#!/bin/bash

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO

NUM_GPUS=${NUM_GPUS:-1}
MODEL_NAME=${MODEL_NAME:-bert_large}

# remove the previous checkpoints
rm -rf /tmp/pretraining_output/*

mpirun -np ${NUM_GPUS} \
    -H localhost:${NUM_GPUS} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 run_pretraining.py \
    --input_file=./bert-data/*.tfrecord \
    --output_dir=/tmp/pretraining_output \
    --do_train=True \
    --do_eval=False \
    --use_horovod=True \
    --bert_config_file=./configs/${MODEL_NAME}/bert_config.json \
    --train_batch_size=6 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --num_train_steps=$((20*NUM_GPUS)) \
    --num_warmup_steps=10 \
    --learning_rate=2e-5 |& tee bert-large-perf-${NUM_GPUS}gpus.txt

# remove the generated checkpoints
rm -rf /tmp/pretraining_output/*

# calculate and print out per GPU performance
./bert-calculate-perf.sh bert-large-perf-${NUM_GPUS}gpus.txt
