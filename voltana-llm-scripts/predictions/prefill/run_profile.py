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

def bench_latency():

    freq_list = [1410, 1305, 1200, 1095, 1005]
    rps_list = [10, 20, 30, 40, 50, 60, 70, 80]

    warmuped = False

    for freq in freq_list:

        print(f"freq: {freq}")
        os.system(f"mkdir -p {freq}")

        for rps in rps_list:

            model_name = MODEL.split("/")[-1]
            prefix = f"{model_name}-A100-80G-400W-flashinfer-"
            file_id = f"rps-{rps}-freq-{freq}"
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
            # args.host = "localhost"
            args.host = "gpub060"
            args.port = 8100 + GPU_IDX
            args.dataset_name = "sharegpt"
            args.sharegpt_output_len = 1
            args.num_prompts = 60 * rps
            args.request_rate = rps
            args.pd_separated = True
            args.output_details = True
            args.output_file = f"{freq}/{output_file_id}"
            args.output_extra_batch_info_file = f"{freq}/{output_extra_file_id}"
            args.extra_request_body = extra_request_body

            if TP == 1:
                args.freq_settings = f"{GPU_IDX}:{freq}"
            elif TP == 2:
                args.tp_size = TP
                args.freq_settings = f"{GPU_IDX}:{freq},{GPU_IDX + 1}:{freq}"
            else:
                raise NotImplementedError

            if not warmuped:
                args.warmup_requests = 1
                warmuped = True
            else:
                args.warmup_requests = 0

            start = time.perf_counter()
            run_benchmark(args)
            end = time.perf_counter()

            print(f"Benchmark completed in {end - start:.2f} seconds.")

if __name__ == "__main__":
    bench_latency()
