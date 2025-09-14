#!/bin/bash

export UCX_TLS=cuda,sm
export UCX_NET_DEVICES=lo
export SGLANG_ENABLE_TORCH_INFERENCE_MODE=1

idx=${1:-0}

args=(
    --watchdog-timeout 86400
    --random-seed 0
    # --log-level debug

    ## select models
    # --model-path ministral/Ministral-3b-instruct
    --model-path meta-llama/Llama-3.1-8B-Instruct
    # --model-path Qwen/Qwen3-32B

    --trust-remote-code
    --host 0.0.0.0
    --port $((8100 + ${idx}))
    --device cuda
    --base-gpu-id ${idx}
    --mem-fraction-static 0.85
    --max-running-requests 1024
    --max-prefill-tokens 16384
    --context-length 16384
    --chunked-prefill-size 16384
    --schedule-policy fcfs
    # --schedule-conservativeness 0.1

    --page-size 1
    --attention-backend flashinfer
    --sampling-backend flashinfer
    --grammar-backend xgrammar
    --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 160 192 224 256 320 384 448 512 576 640 704 768 832 896 960 1024 1088 1152 1216 1280 1344 1408 1472 1536 1600 1664 1728 1792 1856 1920 1984 2048
    --disable-radix-cache

    # only for prefill
    --disable-overlap-schedule

    --force-cuda-graph-prefill
    --collect-extra-batch-info
    --monitor-iteration-metrics
)

python -m sglang.launch_server "${args[@]}"
