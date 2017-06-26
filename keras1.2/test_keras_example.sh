#!/usr/bin/env bash
# This script is for benchmarking keras examples.
# Change backend and number of gpus for different configuration
backend="tensorflow mxnet"
export MXNET_KERAS_TEST_MACHINE="GPU"
for gpu_num in 1; do
    export GPU_NUM=$gpu_num
    for back in $backend; do
        export KERAS_BACKEND=$back
        python utils/benchmark_keras_example.py
    done
done
