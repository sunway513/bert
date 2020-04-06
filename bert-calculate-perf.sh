#!/bin/bash
NUM_GPUS=${NUM_GPUS:-1}
echo "average examples/sec performance per GPU: "
cat $1 | grep 'py:2308' | sed -e 's/.*://' | tail -n +$((NUM_GPUS+1)) |  awk '{ total += $1; count++ } END { print total/count }'
