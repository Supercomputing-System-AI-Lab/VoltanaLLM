#!/bin/bash

log_dir=8B-2p2d-80G
mkdir -p $log_dir

seed=13

# ======================== BEGIN ROUTER PARAMETERS =====================
# 0 for multiple ports
# 1 for multiple ips
use_ip=0
decode_ips=(
    0.0.0.0
)
prefill_ips=(
    0.0.0.0
)
prefill_ports=(
    $((PORT+100))
    $((PORT+101))
)
decode_ports=(
    $((PORT+200))
    $((PORT+201))
)

# ======================== END ROUTER PARAMETERS =====================


# ======================== BEGIN BENCHMARK PARAMETERS =====================
slo_list=(45)
PORT=8000
time=120

# tile_aware(EchoRoute) rr(RoundRobin)
policy_list=(tile_aware)

rps_list=(60)

# 1410 1005 auto(EchoFreq)
freq_list=(auto)

port_list=($PORT)
scheduler_list=(false)
# ======================== END BENCHMARK PARAMETERS =====================


# ========================DO NOT MODIFY ============================ #
if [ "$use_ip" -eq 1 ]; then
        for ip in "${decode_ips[@]}"; do
            decode_urls+=(http://$ip:8200)
        done

        for ip in "${prefill_ips[@]}"; do
            prefill_urls+=(http://$ip:8100)
        done
else
        for port in "${decode_ports[@]}"; do
            decode_urls+=(http://0.0.0.0:$port)
        done

        for port in "${prefill_ports[@]}"; do
            prefill_urls+=(http://0.0.0.0:$port)
        done
fi

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s $port/generate > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

for ip in ${decode_urls[@]}; do
  echo wait_for_$ip
  wait_for_server $ip
done

for ip in ${prefill_urls[@]}; do
  echo wait_for_$ip
  wait_for_server $ip
done

# ========================DO NOT MODIFY ============================ #

