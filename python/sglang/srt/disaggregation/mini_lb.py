"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import dataclasses
import logging
import random
import time
import urllib
from itertools import chain
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.managers.io_struct import SetFreqManagerReq, SetFreqReq
from sglang.srt.disaggregation.launch_lb import LBArgs
from sglang.srt.disaggregation.utils import PDRegistryRequest
import aiofiles


def setup_logger():
    logger = logging.getLogger("pdlb")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def logger_add_file_output(logger: logging.Logger, log_file_path: str):
    file_handler = logging.FileHandler(log_file_path, mode="a")
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


logger = setup_logger()


@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None


class NRequestCounterGuard:

    def __init__(self, n_req: Dict[str, int], server: str):
        self.n_req = n_req
        self.server = server

    def __enter__(self):
        self.n_req[self.server] += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.n_req[self.server] -= 1


class MiniLoadBalancer:
    def __init__(self, prefill_configs: List[PrefillConfig], decode_servers: List[str], fake_transfer: bool = False, fake_transfer_v2: bool = False):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers
    
        self.freq_servers = self.decode_servers + self.prefill_servers
        # self.freq_servers = self.prefill_servers
        self.n_req = {server: 0 for server in self.decode_servers + self.prefill_servers}
        
        
        self.fake_transfer = fake_transfer
        self.fake_transfer_v2 = fake_transfer_v2
        self.on_benchmark = False

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(new_decode_server)

    def select_pair(self, request_data: Optional[dict] = None):
        # TODO: return some message instead of panic
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        prefill_config = random.choice(self.prefill_configs)
        decode_server = random.choice(self.decode_servers)
        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        with NRequestCounterGuard(self.n_req, decode_server):

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=3600
                )  # Add timeout for request reliability
            ) as session:

                arrival_time = time.time()

                if self.fake_transfer:
                    decode_response = await session.post(f"{decode_server}/{endpoint}", json=modified_request)
                elif self.fake_transfer_v2:
                    prefill_response = await session.post(f"{prefill_server}/{endpoint}", json=modified_request)
                    logger.debug("waiting for prefill")
                    prefill_json = await prefill_response.json()
                    await prefill_response.wait_for_close()
                    decode_response = await session.post(f"{decode_server}/{endpoint}", json=modified_request)
                else:
                    tasks = [
                        session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                        session.post(f"{decode_server}/{endpoint}", json=modified_request),
                    ]

                    # Wait for both responses to complete. Prefill should end first.
                    prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    raise NotImplementedError()

                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()

                    # merge `meta_info.input_token_logprobs` from prefill to decode
                    if "meta_info" in ret_json:
                        if "input_token_logprobs" in ret_json["meta_info"]:
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                prefill_json["meta_info"]["input_token_logprobs"]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                else:
                    ret_json = await decode_response.json()
                    if self.fake_transfer_v2:
                        first_token_ts = ret_json["non_stream_metrics"]["first_token_ts"]
                        ret_json["non_stream_metrics"]["ttft"] = first_token_ts - arrival_time
                        # print("ret_json:", ret_json)
                        # print("prefill_json:", prefill_json)
                        # ret_json["non_stream_metrics"]["ttft"] = prefill_json["meta_info"]["e2e_latency"]  # overwrite the TTFT from decoding

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():

            with NRequestCounterGuard(self.n_req, decode_server):

                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(
                        total=3600
                    )  # Add timeout for request reliability
                ) as session:
                    if self.fake_transfer:
                        decode_response = await session.post(f"{decode_server}/{endpoint}", json=modified_request)
                    elif self.fake_transfer_v2:
                        prefill_response = await session.post(f"{prefill_server}/{endpoint}", json=modified_request)
                        logger.debug("waiting for prefill")
                        await prefill_response.wait_for_close()
                        decode_response = await session.post(f"{decode_server}/{endpoint}", json=modified_request)
                    else:
                        # Create the tasks for both prefill and decode requests
                        tasks = [
                            session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                            session.post(f"{decode_server}/{endpoint}", json=modified_request),
                        ]
                        # Wait for both responses to complete. Since this is streaming, they return immediately.
                        prefill_response, decode_response = await asyncio.gather(*tasks)

                    if modified_request.get("return_logprob", False):
                        raise NotImplementedError()

                        prefill_chunks = []
                        async for chunk in prefill_response.content:
                            prefill_chunks.append(chunk)

                        first_prefill_chunk = (
                            prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                        )
                        first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                        async for chunk in decode_response.content:
                            # Note: This is inefficient
                            # merge prefill input_token_logprobs, output_token_logprobs to decode
                            decoded_chunk = chunk.decode("utf-8")
                            if (
                                decoded_chunk
                                and decoded_chunk.startswith("data:")
                                and "[DONE]" not in decoded_chunk
                            ):
                                ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                                ret_json["meta_info"]["input_token_logprobs"] = (
                                    first_prefill_chunk_json["meta_info"][
                                        "input_token_logprobs"
                                    ]
                                    + ret_json["meta_info"]["input_token_logprobs"]
                                )

                                yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                            else:
                                yield chunk
                    else:
                        async for chunk in decode_response.content:
                            # decoded_chunk = chunk.decode("utf-8")

                            # if "[DONE]" in decoded_chunk:
                            #     self.n_req[decode_server] -= 1

                            yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):

    from sglang.srt.disaggregation.echo_lb import EchoLoadBalancer

    if isinstance(load_balancer, EchoLoadBalancer):
        asyncio.create_task(
            load_balancer.decode_router.refresh_server_info_loop()
        )

    logger.info("🚀 Load balancer started.")

    yield

    print("🛑 Load balancer shutting down...")


app = FastAPI(lifespan=lifespan)
load_balancer: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_check():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    async with aiohttp.ClientSession() as session:
        for server in chain(prefill_servers):
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in chain(decode_servers):
            server_info = await session.get(f"{server}/get_server_info")
            decode_infos.append(await server_info.json())

    return {"prefill": prefill_infos, "decode": decode_infos}


@app.post("/start_benchmark")
async def start_benchmark(request_data: dict):
    policy = request_data["policy"]
    server_info_log_file = request_data["server_info_log_file"]
    load_balancer.decode_router.policy = policy
    load_balancer.log_server_info_path = server_info_log_file
    # clear previous log file if exists
    if load_balancer.on_benchmark:
        if load_balancer.decode_router.log_file is not None:
            await load_balancer.decode_router.log_file.close()
            load_balancer.decode_router.log_file = None
    load_balancer.on_benchmark = True
    return Response(status_code=200)


@app.post("/end_benchmark")
async def end_benchmark():
    load_balancer.on_benchmark = False
    return Response(status_code=200)


async def _forward_to_server(request_data: Optional[dict], url: str, method: str) -> bool:
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method,
            url,
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as response:
            return response.status == 200
    return False


@app.post("/set_freq_manager_state")
async def set_freq_manager_state(request_data: dict):
    decode_tasks = [
        _forward_to_server(request_data, f"{server}/set_freq_manager_state", "POST")
        for server in load_balancer.decode_servers
    ]
    prefill_request_data = request_data.copy()
    prefill_request_data["slo_p50"] *= 10
    prefill_tasks = [
        _forward_to_server(prefill_request_data, f"{server}/set_freq_manager_state", "POST")
        for server in load_balancer.prefill_servers
    ]
    all_results = await asyncio.gather(*(decode_tasks + prefill_tasks))

    if not all(all_results):
        num_decode_tasks = len(decode_tasks)
        if not all(all_results[:num_decode_tasks]):
            return Response(status_code=500, content="Failed on decode servers")
        else:
            return Response(status_code=500, content="Failed on prefill servers")

    return Response(status_code=200)



@app.post("/set_tile_scheduler")
async def set_tile_scheduler(request_data: dict):
    tasks = [_forward_to_server(request_data, f"{server}/set_tile_scheduler", "POST") for server in load_balancer.freq_servers]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    if not all(results):
        return Response(status_code=500, content="Failed to set tile scheduler on all servers")
    return Response(status_code=200)

@app.post("/set_freq")
async def set_freq(request_data: dict):
    tasks = [_forward_to_server(request_data, f"{server}/set_freq", "POST") for server in load_balancer.freq_servers]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    if not all(results):
        return Response(status_code=500, content="Failed to set frequency on all servers")
    return Response(status_code=200)


@app.post("/unset_freq")
async def unset_freq():
    tasks = [_forward_to_server(None, f"{server}/unset_freq", "POST") for server in load_balancer.freq_servers]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    if not all(results):
        return Response(status_code=500, content="Failed to unset frequency on all servers")
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair(request_data)

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await load_balancer.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
async def register(obj: PDRegistryRequest):
    if obj.mode == "prefill":
        load_balancer.add_prefill_server(
            PrefillConfig(obj.registry_url, obj.bootstrap_port)
        )
        logger.info(
            f"Registered prefill server: {obj.registry_url} with bootstrap port: {obj.bootstrap_port}"
        )
    elif obj.mode == "decode":
        load_balancer.add_decode_server(obj.registry_url)
        logger.info(f"Registered decode server: {obj.registry_url}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be either PREFILL or DECODE.",
        )

    logger.info(
        f"#Prefill servers: {len(load_balancer.prefill_configs)}, "
        f"#Decode servers: {len(load_balancer.decode_servers)}"
    )

    return Response(status_code=200)


def run(prefill_configs, decode_addrs, host, port):
    global load_balancer
    load_balancer = MiniLoadBalancer(prefill_configs, decode_addrs)
    uvicorn.run(
        app,
        host=host,
        port=port,
        backlog=100000,
        timeout_keep_alive=3600,
        loop="uvloop"
    )

def run_echo_lb(
    prefill_configs,
    lb_args: LBArgs,
):
    from sglang.srt.disaggregation.echo_lb_new import EchoLoadBalancer
    global load_balancer
    load_balancer = EchoLoadBalancer(
        prefill_configs = prefill_configs,
        decode_servers = lb_args.decode_infos,
        policy=lb_args.policy,
        fake_transfer=lb_args.fake_transfer,
        fake_transfer_v2=lb_args.fake_transfer_v2,
        log_path=lb_args.echo_log_path,
        log_server_info_path=lb_args.echo_log_server_info_path,
        refresh_server_info_interval=lb_args.echo_refresh_server_info_interval,
        latency_lookuptable_path=lb_args.latency_lookuptable_path,
        energy_lookuptable_path=lb_args.energy_lookuptable_path,
    )
    uvicorn.run(
        app,
        host=lb_args.host,
        port=lb_args.port,
        backlog=100000,
        timeout_keep_alive=3600,
        loop="uvloop"
    )


if __name__ == "__main__":
    # FIXME: remove this, use the unified entry point: sglang.srt.disaggregation.launch_lb
    from sglang.srt.disaggregation.launch_lb import main

    main()
