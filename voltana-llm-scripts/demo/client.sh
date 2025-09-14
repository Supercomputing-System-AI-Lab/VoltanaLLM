#!/bin/bash

CONFIG_FILE=$1
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

model=meta-llama/Llama-3.1-8B-Instruct
dataset=sharegpt

source $CONFIG_FILE
wait_for_server localhost:$PORT


for rps in "${rps_list[@]}"; do
for slo in "${slo_list[@]}"; do
for enable_tile_scheduler in "${scheduler_list[@]}"; do
for policy in "${policy_list[@]}"; do
for port in "${port_list[@]}"; do
for freq in "${freq_list[@]}"; do

    n_prompt=$((time * rps))

    echo "=============================="
    echo "Running benchmark with LBPORT=$port FREQ = $freq RPS = $rps"
    echo "=============================="

    log_file=$log_dir/slo_${slo}_policy_${policy}_freq${freq}_rps${rps}_time_${time}.log
    args=(
        --model $model
	--host localhost
	--seed $seed
        --port $port
        --backend sglang
        --pd-separated
	--log-file $log_file

        --dataset-name $dataset
        --sharegpt-context-len 3000

        --lmsys-context-len 3000

	--num-prompt $n_prompt
        --request-rate $rps
        # --burstiness 0.01
        --random-range-ratio 0
        --random-sort
        --random-input-len 100
        --random-output-len 500
        --disable-stream
        --warmup-requests 0
        --output-file $log_dir/slo_${slo}_${freq}_${policy}_${enable_tile_scheduler}.jsonl
        --output-details
    )

    if [ "$freq" = "auto" ]; then
	    curl -X POST http://localhost:$port/set_freq_manager_state -H "Content-Type: application/json" -d "{\"dummy_run\": false, \"slo_p50\": $slo}"
    else
	    curl -X POST http://localhost:$port/set_freq_manager_state -H "Content-Type: application/json" -d "{\"dummy_run\": true, \"slo_p50\": $slo}"
        curl -X POST http://localhost:$port/set_freq \
	    -H "Content-Type: application/json" \
	    -d "{\"freq\": $freq}"
    fi
    # set fixed freq
    sleep 1

    # start benchmark
    curl -X POST "http://localhost:$port/start_benchmark" \
    -H "Content-Type: application/json" \
    -d "{
        \"policy\": \"${policy}\",
        \"server_info_log_file\": \"$log_file\"
    }"
    sleep 1
    # set scheduler
    curl -X POST "http://localhost:$port/set_tile_scheduler" \
    -H "Content-Type: application/json" \
    -d "{
        \"enable_tile_scheduler\": \"${enable_tile_scheduler}\"
    }"
    python -m sglang.bench_serving_hpca "${args[@]}"
    #end benchmark
    curl -X POST "http://localhost:$port/end_benchmark"
    echo "Finished LB_PORT=$port FREQ=$freq RPS = $rps"
    echo
done
done
done
done
done
done
