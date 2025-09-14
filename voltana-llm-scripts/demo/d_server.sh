#!/bin/bash

export UCX_TLS=cuda,sm
export UCX_NET_DEVICES=lo
export SGLANG_ENABLE_TORCH_INFERENCE_MODE=1
export DEFAULT_FORCE_STREAM_INTERVAL=1


GPU=${1:-0}
PORT=$((8200+$GPU))

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
    --port $PORT
    --device cuda
    --base-gpu-id $GPU
    --mem-fraction-static 0.85
    --max-running-requests 1024
    --max-prefill-tokens 16384
    --context-length 16384
    --chunked-prefill-size 16384
    --schedule-policy fcfs
    # --schedule-conservativeness 0.1

    --page-size 1
    --attention-backend triton
    --sampling-backend flashinfer
    --grammar-backend xgrammar
    --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 160 192 224 256 320 384 448 512 576 640 704 768 832 896 960 1024 1088 1152 1216 1280 1344 1408 1472 1536 1600 1664 1728 1792 1856 1920 1984 2048
    --disable-radix-cache

    --disaggregation-mode decode
    --disaggregation-transfer-backend fake
    # --disaggregation-ib-device enP62544s1
    --num-reserved-decode-tokens 8

    ## energy & freq monitor
    --enable-energy-monitor
    --energy-monitor-interval 0

    ## freq manager
    --enable-freq-manager

    # --freq-manager-dummy-run  # dry run
    --freq-manager-interval 0
    --freq-manager-latency-lookup-table-path ./prediction/latency-prefill-8b.csv

    # settings
    --freq-manager-slo 50
    --freq-manager-slo margin 0.35
    --freq-manager-f-list 1005 1410

    # rps & tps
    --freq-manager-tps-threshold 100000000
    --freq-manager-tps-window 5
    --freq-manager-rps-threshold 1000000
    --freq-manager-rps-window 5

    --monitor-iteration-metrics
)

python -m sglang.launch_server "${args[@]}"
