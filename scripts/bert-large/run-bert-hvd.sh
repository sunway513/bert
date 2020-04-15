#!/bin/bash

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO

NUM_GPUS=${NUM_GPUS:-8}
MODEL_NAME=${MODEL_NAME:-bert_large}

# remove the previous checkpoints
rm -rf /tmp/pretraining_output/*

cd ../..

mpirun -np ${NUM_GPUS} \
    -H localhost:${NUM_GPUS} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 run_pretraining.py \
    --input_file=/data/wikipedia/seq512/*.tfrecord \
    --output_dir=/tmp/pretraining_output \
    --do_train=True \
    --do_eval=True \
    --use_horovod=True \
    --bert_config_file=./configs/${MODEL_NAME}/bert_config.json \
    --train_batch_size=6 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --num_train_steps=1000000 \
    --num_warmup_steps=1000 \
    --learning_rate=2e-5 
