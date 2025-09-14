import os
import time
from typing import List, Tuple

import orjson
import requests
import tqdm

os.environ["USE_FREQ_SETTINGS"] = "1"
os.environ["SKIP_NUM_NEW_TOKENS_CHECK"] = "1"

from sglang.bench_serving_voltanallm import get_parser, run_benchmark

GPU_IDX = 0

# MODEL = "ministral/Ministral-3b-instruct"
# MAX_TOKENS = 1210369
# MAX_LENGTH = 8192
# TP = 1

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 457600
MAX_LENGTH = 8192
TP = 1

# MODEL = "Qwen/Qwen3-32B"
# MAX_TOKENS = 325037
# MAX_LENGTH = 8192
# TP = 2

def bench_single(freq: int, bs: int, gpu_idx: int, warmup: int = 0):
    inlen = 1
    outlen = min(MAX_LENGTH, round(MAX_TOKENS * 0.95 / bs))

    model_name = MODEL.split("/")[-1]
    prefix = f"{model_name}-A100-80G-400W-triton-"
    file_id = f"bs-{bs}-outlen-{outlen}"
    print(file_id)
    output_file_id = f"{prefix}{file_id}.json"
    output_extra_file_id = f"{prefix}{file_id}-extra.json"

    args = get_parser().parse_args()

    extra_request_body = {
        "stream_options": {
            "include_extra_batch_info": True
        },
        "bootstrap_host": "localhost",  # work with fake transfer backend
        "bootstrap_port": 8100,
        "bootstrap_room": 10,
    }
    extra_request_body = orjson.dumps(extra_request_body).decode('utf-8')

    args.seed = 0
    args.backend = "sglang"
    args.model = MODEL
    args.host = "localhost"
    args.port = 8200 + gpu_idx
    args.dataset_name = "random"
    args.random_input_len = inlen
    args.random_output_len = outlen
    args.random_range_ratio_input = 1
    args.random_range_ratio_output = 1
    args.num_prompts = bs
    args.request_rate = float("inf")
    args.pd_separated = True
    args.output_details = True
    args.output_file = f"{freq}/{output_file_id}"
    args.output_extra_batch_info_file = f"{freq}/{output_extra_file_id}"
    args.extra_request_body = extra_request_body
    args.warmup_requests = warmup

    if TP == 1:
        args.freq_settings = f"{gpu_idx}:{freq}"
    elif TP == 2:
        args.tp_size = TP
        args.freq_settings = f"{gpu_idx}:{freq},{gpu_idx + 1}:{freq}"
    else:
        raise NotImplementedError

    start = time.perf_counter()
    run_benchmark(args)
    end = time.perf_counter()

    print(f"Benchmark completed in {end - start:.2f} seconds.")


def bench_latency():

    bs_list = [8 * k for k in range(1, 16)] + [8 * k + 1 for k in range(0, 16)] \
            + [128 * k for k in range(1, 9)] + [128 * k + 1 for k in range(1, 8)] # [8k, 8k + 1] + [128k, 128k + 1]
    bs_list = sorted(bs_list)
    bs_list = [bs_list[-1]] + bs_list[:-1]

    freq_list = [1410, 1305, 1200, 1095, 1005]

    print("Starting profiling...")

    warmuped = False
    for freq in tqdm.tqdm(freq_list):
        print(f"Profiling frequency {freq} ...")
        os.system(f"mkdir -p {freq}")
        for bs in bs_list:
            if warmuped:
                warmup = 0
            else:
                warmuped = True
                warmup = 1
            bench_single(freq, bs, GPU_IDX, warmup=warmup)

    print("Profiling completed.")

if __name__ == "__main__":
    bench_latency()
