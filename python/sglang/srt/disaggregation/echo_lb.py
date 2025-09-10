"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import os
import logging
import asyncio
import logging
import random
from itertools import chain
import time
from typing import Dict, List, Optional

import aiohttp
import aiofiles
import orjson

from sglang.srt.managers.lookup_table import EnergyLookupTable, LatencyLookupTable
from sglang.srt.disaggregation.mini_lb import MiniLoadBalancer, logger_add_file_output, setup_logger, PrefillConfig
from sglang.bench_serving_voltanallm import get_tokenizer
from functools import cmp_to_key


logger = setup_logger()
TILE_SIZE = int(os.environ.get("TILE_SIZE", "128"))

class DecodeRouter():
    def __init__(
        self,
        lb: MiniLoadBalancer,
        decode_servers: List[str],
        policy: str = "rr",
        refresh_server_info_interval: float = 0.01,
        latency_lookuptable_path: Optional[str] = None,
        energy_lookuptable_path: Optional[str] = None,
        log_server_info_path: Optional[str] = None,
    ):
        self.lb = lb
        self.decode_servers = decode_servers

        if policy not in ["rr", "ln", "tile_aware", "fadr"]:
            raise ValueError(f"Invalid policy: {policy}. Supported policies: rr, lfl, hfl, hl, ll, tile_aware")

        self.policy = policy
        if latency_lookuptable_path:
            self.latency_lookup_table = LatencyLookupTable(latency_lookuptable_path)
        if energy_lookuptable_path:
            self.energy_lookup_table = EnergyLookupTable(energy_lookuptable_path)

        self.servers_info: Dict[str, Dict] = {}  # refresh in the background loop
        self.servers_info_history: Dict[str, List[Dict]] = {
            server: [] for server in self.lb.freq_servers
        }
        self.refresh_server_info_interval = refresh_server_info_interval
        self.log_server_info_path = log_server_info_path

        self.energy = {}
        for server in decode_servers:
            self.energy[server] = 0

        self.log_file = None

        # misc
        self.decode_index = -1
        self.tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    async def refresh_server_info_loop(self):
        logger.info(f"Refreshing server info every {self.refresh_server_info_interval}s")

        # assert self.log_server_info_path is not None, "log_server_info_path must be set to enable server info logging"

        self.log_file = None

        while True:
            if self.lb.on_benchmark:
                if self.log_file is None:
                    self.log_file = await aiofiles.open(self.lb.log_server_info_path, mode="wb")
                await self._refresh_server_info_once(self.log_file)
                await self.log_file.flush()
            else:
                if self.log_file is not None:
                    await self.log_file.close()
                    self.log_file = None
            await asyncio.sleep(self.refresh_server_info_interval)

        # aiofiles.os.sendfile(self.log_server_info_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND)
        # async with aiofiles.open(self.log_server_info_path, mode="wb") as log_file:
        #     while True:
        #         if self.lb.on_benchmark:
        #             await self._refresh_server_info_once(log_file)
        #             await log_file.flush()  # Ensure the log file is flushed after each write
        #         await asyncio.sleep(self.refresh_server_info_interval)

    async def _refresh_server_info_once(self, log_file = None):
        tasks = [self._get_server_info(server) for server in self.lb.freq_servers]
        results = await asyncio.gather(*tasks)

        # now = time.perf_counter()

        for server, info in results:
            if not info:
                continue

            item_to_append = {
                # "time": info["time"],
                "time": time.perf_counter(),  # use current time instead of server time
                "n_req": self.lb.n_req[server],
                "lookup_itl": self.latency_lookup_table.lookup(
                    bs=info["bs"],
                    tokens=info["num_running_tokens"],
                    freq=info["freq"],
                ),
            }

            for key in info.keys():
                if key not in ["time"]:
                    item_to_append[key] = info[key]

            # item_to_append["last_itl"] *= 1000

            self.servers_info[server] = item_to_append
            self.servers_info_history[server].append(item_to_append)

            if log_file:
                record = {server: item_to_append}
                record_json = orjson.dumps(record)
                await log_file.write(record_json + b"\n")

    async def _get_server_info(self, decode_server):

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{decode_server}/get_server_info") as response:
                if response.status != 200:
                    logger.error(f"Failed to get server info from {decode_server}: {response.status}")
                    return decode_server, None
                else:
                    info = await response.json()
                    if not info or "internal_states" not in info or not info["internal_states"]:
                        logger.error(f"Invalid server info from {decode_server}: {info}")
                        return decode_server, None
                    return decode_server, info["internal_states"][0]

    def route(self, request_data: Optional[dict] = None):

        # initiaze the energy for decode servers
        for decode_server, info in self.servers_info.items():
            if self.energy.get(decode_server, 0) == 0:
                self.energy[decode_server] = info["gpu_energy"]
        if self.policy == "rr":
            decode_server = self._round_robin_decode_server()
        elif self.policy == 'ln':
            decode_server = self._lowest_nreq_decode_server()
        elif self.policy == 'tile_aware':
            decode_server = self._tile_aware_sche()
        elif self.policy == 'fadr':
            decode_server = self._tile_aware_sche2(request_data)
        else:
            raise ValueError(f"Invalid policy: {self.policy}")

        logger.debug(f"Selected decode server: {decode_server} with policy {self.policy}")

        return decode_server

    def _tile_aware_sche2(self, request_data: dict):
        servers_info  = [info for info in self.servers_info.items() if info[1]["is_prefill"] is False]
        reqs = [self.lb.n_req[server] for server, _ in servers_info]
        if max(reqs) < TILE_SIZE:
            sorted_servers = sorted(
                servers_info,
                key=lambda x: (x[1]["freq"], self.lb.n_req[x[0]])
            )
            return sorted_servers[0][0]
        tokens = self.tokenizer.encode(request_data["text"])
        slo = servers_info[0][1]["slo"]


        freq_list = [1005, 1095, 1200, 1305, 1410]
        def boundary(bs, load):
            for freq in freq_list:
                itl_ = self.latency_lookup_table.lookup(bs=bs+1, tokens=load + len(tokens)+50*bs, freq=freq)
                if itl_ is not None and itl_ <= slo:
                    return freq, itl_

            return 1410, -1
        def get_cur_freq(bs, load):
            for freq in freq_list:
                itl_ = self.latency_lookup_table.lookup(bs=bs, tokens=load+2*bs, freq=freq)
                if itl_ is not None and itl_ <= slo:
                    return freq, itl_
            return 1410, -1

        freq_list = [1005, 1095, 1200, 1305, 1410]
        next_freq = {}
        for server, info in servers_info:
            nreq = self.lb.n_req[server]
            freq, itl1= get_cur_freq(nreq, info["load"])
            freq_, itl2 = boundary(nreq, info["load"])
            if freq_ > freq or freq_ == 1410:
                next_freq[server] = (freq_, nreq, itl2)
            else:
                next_freq[server] = (-freq, nreq, itl1)


        freq_list = [-1305, -1200, -1095, -1005, 1005, 1095, 1200, 1305, 1410]
        pos = {f: i for i, f in enumerate(freq_list)}

        # freq level diff < 2
        sorted_servers = sorted(next_freq.items(), key=lambda x: (abs(x[1][0]),x[1][0],(x[1][1] + 127)//128, -x[1][1]))
        if pos[abs(sorted_servers[-1][1][0])]  - pos[abs(sorted_servers[0][1][0])] > 1:
            return sorted_servers[0][0]

        def cmp_servers(a, b):
            fa, na, _ = a[1]
            fb, nb, _ = b[1]
            if fa == fb:
                t1_a, t1_b = (na + 127) // 128, (nb + 127) // 128
                if t1_a != t1_b:
                    # ascend
                    return -1 if t1_a < t1_b else 1
                return -1 if na > nb else 1

            return -1 if fa < fb else 1

        sorted_servers = sorted(next_freq.items(), key=cmp_to_key(cmp_servers))
        # print(sorted_servers)
        return sorted_servers[0][0]


    def _tile_aware_sche(self):
        servers_info  = [info for info in self.servers_info.items() if info[1]["is_prefill"] is False]
        # print(servers_info, self.servers_info)
        reqs = [self.lb.n_req[server] for server, _ in servers_info]

        if max(reqs) < TILE_SIZE:
            sorted_servers = sorted(
                servers_info,
                key=lambda x: (x[1]["freq"], self.lb.n_req[x[0]])
            )
            return sorted_servers[0][0]

        sorted_servers = sorted(
            servers_info,
            key=lambda x: ((self.lb.n_req[x[0]] + (TILE_SIZE - 1)) // TILE_SIZE, -self.lb.n_req[x[0]])
        )
        for server in sorted_servers:
            if self.lb.n_req[server[0]] == 0 or  self.lb.n_req[server[0]] % TILE_SIZE > 0:
                return server[0]
        return sorted_servers[0][0]

    def _round_robin_decode_server(self):
        self.decode_index = (self.decode_index+1) % len(self.decode_servers)
        return self.decode_servers[self.decode_index]

    def _random_decode_server(self, servers):
        return random.choice(self.decode_servers)

    def _lowest_nreq_decode_server(self):
        servers_info  = self.get_decode_server_info()
        slo = servers_info[0][1]["slo"]

        sorted_servers = sorted(
            servers_info,
            key=lambda x: self.lb.n_req[x[0]],  # Sort by load
        )

        return sorted_servers[0][0]



    def _predict_output_length(self, request_data: dict):
        return request_data["sampling_params"]["max_new_tokens"]  # oracle length prediction for now


class EchoLoadBalancer(MiniLoadBalancer):
    def __init__(
        self,
        prefill_configs: List[PrefillConfig],
        decode_servers: List[str],
        policy: str = "rr",
        fake_transfer: bool = False,
        fake_transfer_v2: bool = False,
        refresh_server_info_interval: float = 0.05,
        log_path: Optional[str] = None,
        log_server_info_path: Optional[str] = None,
        latency_lookuptable_path: Optional[str] = None,
        energy_lookuptable_path: Optional[str] = None,
    ):
        super().__init__(prefill_configs, decode_servers, fake_transfer, fake_transfer_v2)

        self.log_server_info_path = log_server_info_path

        self.decode_router = DecodeRouter(
            lb = self,
            decode_servers = decode_servers,
            policy=policy,
            refresh_server_info_interval=refresh_server_info_interval,
            latency_lookuptable_path=latency_lookuptable_path,
            energy_lookuptable_path=energy_lookuptable_path,
            log_server_info_path=log_server_info_path,
        )

        if log_path:
            logger_add_file_output(logger, log_path)

        logger.info(f"EchoLoadBalancer initialized: policy={policy}, interval={refresh_server_info_interval}s")

    def select_pair(self, request_data: Optional[dict] = None):

        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        prefill_config = random.choice(self.prefill_configs)
        decode_server = self.decode_router.route(request_data)

        return prefill_config.url, prefill_config.bootstrap_port, decode_server
