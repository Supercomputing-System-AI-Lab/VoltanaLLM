import argparse
import logging
import os
import sys
import time
from typing import List

import zmq
import pynvml

from sglang.utils import get_bool_env_var

VOLTANA_DEBUG = get_bool_env_var("VOLTANA_DEBUG", "false")

logger = logging.getLogger(__name__)

def event_loop(gpu_idx: List[int], zmq_input: str, zmq_output: str):
    if os.geteuid() != 0:
        logger.error("This script requires root privileges. Please run with 'sudo'.")
        sys.exit(1)

    context = zmq.Context()

    buf_size = int(0.5 * 1024**3)

    socket_input = context.socket(zmq.PULL)
    socket_input.bind(zmq_input)
    socket_input.setsockopt(zmq.RCVHWM, 0)
    socket_input.setsockopt(zmq.RCVBUF, buf_size)

    socket_output = context.socket(zmq.PUSH)
    socket_output.connect(zmq_output)
    socket_output.setsockopt(zmq.SNDHWM, 0)
    socket_output.setsockopt(zmq.SNDBUF, buf_size)

    f_last = None

    pynvml.nvmlInit()
    pynvml_handles = []
    for idx in gpu_idx:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            pynvml_handles.append(handle)
            logger.debug(f"Initialized NVML handle for GPU {idx}.")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML handle for GPU {idx}: {e}")
            sys.exit(1)

    logger.info(f"Listening for frequency requests on {zmq_input}...")

    while True:

        try:
            recv_obj = socket_input.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            continue

        if VOLTANA_DEBUG:
            logger.debug(f"Received object: {recv_obj}")

        if not isinstance(recv_obj, int):
            if VOLTANA_DEBUG:
                logger.warning(f"Received unexpected object type: {type(recv_obj)}. Expected int.")
            continue

        f_target = recv_obj
        if VOLTANA_DEBUG:
            logger.debug(f"Received target frequency: {f_target} MHz")

        if f_target == f_last:
            if VOLTANA_DEBUG:
                logger.debug(f"Frequency {f_target} MHz is the same as last, skipping.")
            continue

        no_errors = True

        if f_target == 0:
            if VOLTANA_DEBUG:
                logger.debug("Received frequency 0, resetting GPU clocks.")
            for handle in pynvml_handles:
                try:
                    pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                    if VOLTANA_DEBUG:
                        logger.debug(f"Reset GPU clocks for handle {handle}.")
                except pynvml.NVMLError as e:
                    if VOLTANA_DEBUG:
                        logger.error(f"Failed to reset clocks for handle {handle}: {e}")
                    no_errors = False
                    continue
            f_last = None
            socket_output.send_pyobj(no_errors, flags=zmq.NOBLOCK, copy=True)
            continue

        for handle in pynvml_handles:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, f_target, f_target)

        assert no_errors
        f_last = f_target
        if VOLTANA_DEBUG:
            logger.debug(f"Set frequency to {f_target} MHz without errors.")

        # send ack
        # socket_output.send_pyobj(no_errors, flags=zmq.NOBLOCK, copy=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency Setter Event Loop")
    parser.add_argument("--gpu-index", type=int, nargs="+", required=True)
    parser.add_argument("--zmq-input", type=str, required=True)
    parser.add_argument("--zmq-output", type=str, required=True)
    args = parser.parse_args()
    event_loop(args.gpu_index, args.zmq_input, args.zmq_output)
