#!/bin/bash

export TILE_SIZE=128

CONFIG_FILE=$1

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

source $CONFIG_FILE

args=(
    --host 0.0.0.0
    --port $PORT
    --prefill "${prefill_urls[@]}"
    --decode "${decode_urls[@]}"
    --fake-transfer-v2
    --echo-lb
    --policy rr
    --echo-log-path /dev/null
    --echo-log-server-info-path log/default.log
    --echo-refresh-server-info-interval 0.01
    --latency-lookuptable-path ./prediction/latency-prefill-8b.csv
    --energy-lookuptable-path ./prediction/energy-prefill-8b.csv
)

python -m sglang.srt.disaggregation.launch_lb "${args[@]}"
