#!/bin/bash
export SEQ=128
NUM_GPUS=1 ./run-bert-hvd.sh
NUM_GPUS=2 ./run-bert-hvd.sh
NUM_GPUS=4 ./run-bert-hvd.sh
NUM_GPUS=8 ./run-bert-hvd.sh
