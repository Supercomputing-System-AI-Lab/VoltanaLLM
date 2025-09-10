import dataclasses
import logging
import os
import subprocess
import sys
import msgspec
import time
import signal
import statistics
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import OrderedDict
from typing import Dict, List, Optional, Union, Tuple

import atomics
import requests
import pandas as pd
import psutil
import setproctitle
import zmq

from sglang.srt.managers.io_struct import MetricsForFreq, SetFreqManagerReq, SetFreqReq, UnsetFreqReq
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.lookup_table import LatencyLookupTable, EnergyLookupTable, LatencyLookupTablePrefill
from sglang.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sglang.utils import (
    get_exception_traceback,
    get_bool_env_var,
)

VOLTANA_DEBUG = get_bool_env_var("VOLTANA_DEBUG", "false")

logger = logging.getLogger(__name__)

class FreqManager:

    def __init__(
        self,
        server_args: ServerArgs,  # including freq manager args
        port_args: PortArgs,
        **kwargs,
    ):
        if server_args.disaggregation_mode == "prefill":
            logger.info("Initializing FreqManager (Prefill)...")
        elif server_args.disaggregation_mode == "decode":
            logger.info("Initializing FreqManager (Decode)...")
        else:
            raise ValueError(f"FreqManager does not support in co-location mode")

        self.server_args = server_args
        self.port_args = port_args

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()

        context = zmq.Context(2)
        # communication with scheduler to get system load metrics
        self.recv_from_scheduler_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.metric_monitor_ipc_name, True
        )
        # communication with http server to get settings update
        self.recv_from_http_server = get_zmq_socket(
            context, zmq.PULL, port_args.control_freq_manager_ipc_name, True
        )

        # devices
        self.device_ids = [server_args.base_gpu_id + i * server_args.gpu_id_step for i in range(server_args.tp_size)]

        # arg
        self.enabled = server_args.enable_freq_manager
        self.is_prefill = server_args.disaggregation_mode == "prefill"
        self.dummy_run = server_args.freq_manager_dummy_run
        self.interval = server_args.freq_manager_interval
        self.latency_lookup_table_path = server_args.freq_manager_latency_lookup_table_path
        self.energy_lookup_table_path = server_args.freq_manager_energy_lookup_table_path
        self.use_bs_total = server_args.freq_manager_use_bs_total

        self.slo = server_args.freq_manager_slo * (1 + server_args.freq_manager_slo_margin)
        logger.info(f"Using SLO: {self.slo} ms (base: {server_args.freq_manager_slo} ms, margin: {server_args.freq_manager_slo_margin * 100}%)")

        self.tps_threshold = server_args.freq_manager_tps_threshold
        self.tps_window = server_args.freq_manager_tps_window
        self.rps_threshold = server_args.freq_manager_rps_threshold
        self.rps_window = server_args.freq_manager_rps_window

        assert len(server_args.freq_manager_f_list) >= 2, "Frequency list must contain at least two frequencies."
        self.f_list = sorted(server_args.freq_manager_f_list)

        # atomics for gpu
        self.gpu_freq_shm = shared_memory.SharedMemory(name=kwargs["gpu_freq_shm_name"])
        self.gpu_power_shm = shared_memory.SharedMemory(name=kwargs["gpu_power_shm_name"])
        self.gpu_energy_shm = shared_memory.SharedMemory(name=kwargs["gpu_energy_shm_name"])
        self.gpu_freq_buf = self.gpu_freq_shm.buf[:4]
        self.gpu_power_buf = self.gpu_power_shm.buf[:4]
        self.gpu_energy_buf = self.gpu_energy_shm.buf[:4]

        # init freq setter
        self._init_freq_setter(context)

        # init states
        self._init_state()

    def _init_freq_setter(self, context):
        zmq_to_freq_setter = f"ipc:///tmp/srt_freq_setter_input_{self.device_ids[0]}"
        zmq_from_freq_setter = f"ipc:///tmp/srt_freq_setter_output_{self.device_ids[0]}"
        self.send_to_freq_setter = get_zmq_socket(
            context, zmq.PUSH, zmq_to_freq_setter, False
        )
        self.recv_from_freq_setter = get_zmq_socket(
            context, zmq.PULL, zmq_from_freq_setter, True
        )
        setter_path = os.path.join(os.path.dirname(__file__), "freq_setter.py")
        # run in a separate process with root privileges
        cmd = ["sudo", "-E", "python", setter_path, "--gpu-index"] + [str(x) for x in self.device_ids] + ["--zmq-input", zmq_to_freq_setter, "--zmq-output", zmq_from_freq_setter]
        self.setter_proc = subprocess.Popen(cmd)
        logger.info(f"Started frequency setter process with command: {' '.join(cmd)}")
        time.sleep(3)  # wait for the process to start
        os.system(f"sudo chown {os.getuid()}:{os.getgid()} /tmp/srt_freq_setter_input_{self.device_ids[0]}") # make sure the user can write to the socket
        self.send_to_freq_setter.send_pyobj("echo")


    def _init_state(self) -> None:
        # table
        if self.is_prefill:
            self.latency_lookup_table = LatencyLookupTablePrefill(self.latency_lookup_table_path)
        else:
            self.latency_lookup_table = LatencyLookupTable(self.latency_lookup_table_path)
        self.energy_lookup_table = EnergyLookupTable(self.energy_lookup_table_path)

        # state
        self.f_target = self.f_list[0]  # start from low freq
        self.f_last = self.f_target
        self.ts_last_with_metrics = 0

        # arrival metrics
        self.rps_timestamps: List[float] = []
        self.tps_timestamps: List[Tuple[float, int]] = []

        self.cur_tps = 0.0
        self.cur_rps = 0.0

        # metrics for this iteration
        self.iteration_metrics_list: List[MetricsForFreq] = []

        # init frequency
        self.set_frequency(self.f_target)

    def __del__(self):
        if not self.dummy_run:
            self.reset_frequency()
        self.setter_proc.terminate()

    def _get_f_last(self):
        with atomics.atomicview(buffer=self.gpu_freq_buf, atype=atomics.UINT) as a:
            self.f_last = a.load()

    def reset_frequency(self, force: bool = False) -> None:
        if (not force) and self.dummy_run:
            if VOLTANA_DEBUG:
                logger.info("Dummy run, not resetting frequency.")
            return
        self.send_to_freq_setter.send_pyobj(0)

    def set_frequency(self, frequency: int, force: bool = False) -> None:
        if (not force) and self.dummy_run:
            if VOLTANA_DEBUG:
                logger.info(f"Dummy run, not setting frequency to {frequency}")
            return
        self.send_to_freq_setter.send_pyobj(frequency)

    def _update_settings(self, req: SetFreqManagerReq) -> None:
        logger.info(f"Updating frequency manager settings: {req}")
        if req.enabled is not None:
            self.enabled = req.enabled
        if req.dummy_run is not None:
            self.dummy_run = req.dummy_run
        if req.interval is not None:
            self.interval = req.interval
        if req.latency_lookup_table_path is not None:
            self.latency_lookup_table_path = req.latency_lookup_table_path
        if req.energy_lookup_table_path is not None:
            self.energy_lookup_table_path = req.energy_lookup_table_path
        if req.f_list:
            assert len(req.f_list) >= 2, "Frequency list must contain at least two frequencies."
            self.f_list = sorted(req.f_list)
        if req.slo is not None:
            self.slo = req.slo
        if req.rps_threshold is not None:
            self.rps_threshold = req.rps_threshold
        if req.rps_window is not None:
            self.rps_window = req.rps_window
        if req.tps_threshold is not None:
            self.tps_threshold = req.tps_threshold
        if req.tps_window is not None:
            self.tps_window = req.tps_window
        if req.use_bs_total is not None:
            self.use_bs_total = req.use_bs_total

        self._init_state()

    def event_loop(self):

        while True:

            # check if there are requests from the HTTP server to change settings
            recv_obj = None
            while self.recv_from_http_server.poll(timeout=0) != 0:
                recv_obj = self.recv_from_http_server.recv_pyobj() # read the least obj

            if recv_obj:
                if isinstance(recv_obj, SetFreqManagerReq):
                    logger.info(f"Received request to update settings: {recv_obj}")
                    self._update_settings(recv_obj)
                elif isinstance(recv_obj, SetFreqReq):
                    freq = recv_obj.freq
                    logger.info(f"Received request to set frequency: {freq}")
                    assert freq in self.f_list, f"Frequency {freq} is not in the frequency list {self.f_list}"
                    self.f_last = freq
                    self.set_frequency(freq, force=True)
                    self.dummy_run = True
                elif isinstance(recv_obj, UnsetFreqReq):
                    logger.info("Received request to unset frequency.")
                    self.f_last = self.f_list[-1]
                    self.reset_frequency(force=True)
                    self.dummy_run = True
                else:
                    logger.error(
                        f"Received unknown object type: {type(recv_obj)}. "
                        f"Expected SetFreqManagerReq or SetFreqReq."
                    )

            time.sleep(self.interval)

            if not self.enabled:
                logger.info("Frequency manager is disabled, skipping iteration.")
                continue

            # clear history metrics
            self.iteration_metrics_list.clear()

            # pulling data form scheduler
            while self.recv_from_scheduler_tokenizer.poll(timeout=0) != 0:
                recv_obj = self.recv_from_scheduler_tokenizer.recv_pyobj()
                if isinstance(recv_obj, Tuple):  # from tokenizer
                    recv_ts, inlen = recv_obj
                    self.rps_timestamps.append(recv_ts)
                    self.tps_timestamps.append(recv_obj)
                elif isinstance(recv_obj, MetricsForFreq):  # from scheduler
                    self.iteration_metrics_list.append(recv_obj)
                else:
                    logger.warning(
                        f"Received unknown object type: {type(recv_obj)}. "
                        f"Expected float or MetricsForFreq."
                    )
            else:
                logger.debug("No new metrics received from scheduler or tokenizer.")

            now = time.perf_counter()

            # tps window
            if self.tps_threshold is not None:
                tps_in_window = [x for x in self.tps_timestamps if (now - x[0] < self.tps_window)]
                self.cur_tps = sum(x[1] for x in tps_in_window) / self.tps_window
                self.tps_timestamps = tps_in_window
                if VOLTANA_DEBUG:
                    logger.info(f"Current TPS: {self.cur_tps}, Threshold: {self.tps_threshold}")

            # rps window
            if self.rps_threshold is not None:
                reqs_in_window = [t for t in self.rps_timestamps if (now - t < self.rps_window)]
                self.cur_rps = len(reqs_in_window) / self.rps_window
                self.rps_timestamps = reqs_in_window
                if VOLTANA_DEBUG:
                    logger.info(f"Current RPS: {self.cur_rps}, Threshold: {self.rps_threshold}")

            if self.dummy_run:
                if VOLTANA_DEBUG:
                    logger.info("Dummy run, skipping frequency management.")
                continue

            self.f_target = self.get_f_target()

            if len(self.iteration_metrics_list) == 0:
                if now - self.ts_last_with_metrics > 5:
                    logger.info("No metrics for too long (5s), resetting frequency.")
                    self.f_target = self.f_list[0]  # reset to low freq
            else:
                self.ts_last_with_metrics = now

            if VOLTANA_DEBUG:
                logger.info(f"f_target for this iteration: {self.f_target}")

            if (self.f_target is not None) and (self.f_target != self.f_last):
                if VOLTANA_DEBUG:
                    logger.info(f"Updating f_last: {self.f_last} -> {self.f_target}")
                self.f_last = self.f_target
                self.set_frequency(self.f_target)

    def get_f_target(self) -> Optional[int]:

        # tps is high -> max freq
        if (self.tps_threshold is not None) and (self.cur_tps > self.tps_threshold):
            if VOLTANA_DEBUG:
                logger.info(f"TPS ({self.cur_tps}) > threshold ({self.tps_threshold}), setting f to max ({self.f_list[-1]}).")
            return self.f_list[-1] # max freq

        # rps is high -> max freq
        if (self.rps_threshold is not None) and (self.cur_rps > self.rps_threshold):
            if VOLTANA_DEBUG:
                logger.info(f"RPS ({self.cur_rps}) > threshold ({self.rps_threshold}), setting f to max ({self.f_list[-1]}).")
            return self.f_list[-1]  # max freq

        # skip if no metrics available
        if len(self.iteration_metrics_list) == 0:
            if VOLTANA_DEBUG:
                logger.debug("No metrics available, cannot determine f_target.")
            return None

        # exist waiting requests -> max freq
        num_waiting_reqs = self.iteration_metrics_list[-1].len_waiting_queue
        if num_waiting_reqs > 0:
            if VOLTANA_DEBUG:
                logger.info(f"Existing waiting reqs ({num_waiting_reqs}), setting f max ({self.f_list[-1]}).")
            return self.f_list[-1]  # max freq

        # get metrics
        if not self.is_prefill:
            if self.use_bs_total:
                bs_list = [m.total_bs for m in self.iteration_metrics_list]
            else:
                bs_list = [m.bs for m in self.iteration_metrics_list]
        tokens_list = [m.num_running_tokens for m in self.iteration_metrics_list if m.num_running_tokens not in [0, None]]  # remove 0s for prefill

        if ((not self.is_prefill) and (len(bs_list) == 0)) or (len(tokens_list) == 0):
            if VOLTANA_DEBUG:
                logger.warning("No metrics available, cannot determine frequency target.")
            return None

        # calculate metrics for lookup table
        if not self.is_prefill:
            if len(bs_list) > 1:
                bs_median = statistics.median(bs_list)
            else:
                bs_median = bs_list[0]
        else:
            bs_median = 1
        if len(tokens_list) > 1:
            tokens_median = statistics.median(tokens_list)
        else:
            tokens_median = tokens_list[0]

        # adjustable time of TTFT = SLO - waiting time
        if self.iteration_metrics_list[-1].ttft_slo_offsets:
            ttft_slo_offset = max(self.iteration_metrics_list[-1].ttft_slo_offsets) * 1000 # convert to ms
        else:
            ttft_slo_offset = 0.0

        # find the lowest frequency that meets the SLO
        for freq in self.f_list[:-1]:
            latency_pred = self.latency_lookup_table.lookup(freq, bs_median, tokens_median)
            if latency_pred is None:
                if VOLTANA_DEBUG:
                    logger.warning(f"Latency lookup table returned None for f ({freq}), skipping.")
                continue
            elif latency_pred <= (self.slo - ttft_slo_offset):
                if VOLTANA_DEBUG:
                    logger.info(f"f ({freq}) meets SLO ({self.slo_p50} (-{ttft_slo_offset})) with predicted latency ({latency_pred}).")
                return freq
            else:
                if VOLTANA_DEBUG:
                    logger.debug(f"f ({freq}) does not meet SLO ({self.slo_p50} (-{ttft_slo_offset})) with predicted latency ({latency_pred}).")
                pass

        # if no frequency meets the SLO, return the highest frequency
        if VOLTANA_DEBUG:
            logger.info(f"No frequency meets SLO ({self.slo} (-{ttft_slo_offset})), setting to f max {self.f_list[-1]}.")
        return self.f_list[-1]


def run_freq_manager_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    **kwargs,
) -> None:
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::freq_manager_process")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        logger.info("Starting FreqManager process...")
        manager = FreqManager(server_args, port_args, **kwargs)
        logger.info("FreqManager process started successfully.")
        manager.event_loop()
    except Exception as e:
        traceback = get_exception_traceback()
        logger.error(f"FreqManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
