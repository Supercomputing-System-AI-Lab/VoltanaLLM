import logging
import time
import signal
from multiprocessing import shared_memory


import atomics
import psutil
import pynvml
import setproctitle

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_itself_when_parent_died,
)
from sglang.utils import (
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

class EnergyMonitor:

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        **kwargs,
    ):
        logger.info("Initializing EnergyMonitor...")

        self.server_args = server_args
        self.port_args = port_args

        self.interval = server_args.energy_monitor_interval

        # atomics for gpu
        self.gpu_freq_shm = shared_memory.SharedMemory(name=kwargs["gpu_freq_shm_name"])
        self.gpu_power_shm = shared_memory.SharedMemory(name=kwargs["gpu_power_shm_name"])
        self.gpu_energy_shm = shared_memory.SharedMemory(name=kwargs["gpu_energy_shm_name"])
        self.gpu_freq_buf = self.gpu_freq_shm.buf[:4]
        self.gpu_power_buf = self.gpu_power_shm.buf[:4]
        self.gpu_energy_buf = self.gpu_energy_shm.buf[:4]

        self.device_ids = [server_args.base_gpu_id + i * server_args.gpu_id_step for i in range(server_args.tp_size)]

        pynvml.nvmlInit()
        self.nvml_handles = {
            device_id: pynvml.nvmlDeviceGetHandleByIndex(device_id)
            for device_id in self.device_ids
        }

    def _get_freq(self, device_id: int) -> int:
        handle = self.nvml_handles[device_id]
        freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        return freq

    def _get_energy(self, device_id: int) -> int:
        handle = self.nvml_handles[device_id]
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return energy

    def _get_power(self, device_id: int) -> int:
        handle = self.nvml_handles[device_id]
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power

    def event_loop(self):

        while True:

            time.sleep(self.interval)
            energy_total = 0
            power_total = 0
            freq = 0

            for device_id in self.device_ids:

                energy_total += self._get_energy(device_id)
                power_total += self._get_power(device_id)
                freq = self._get_freq(device_id)

            energy_total %= (2 ** 32)

            with atomics.atomicview(buffer=self.gpu_energy_buf, atype=atomics.UINT) as a:
                a.store(energy_total)
            with atomics.atomicview(buffer=self.gpu_power_buf, atype=atomics.UINT) as a:
                a.store(power_total)
            with atomics.atomicview(buffer=self.gpu_freq_buf, atype=atomics.UINT) as a:
                a.store(freq)

def run_energy_monitor_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    **kwargs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::energy_monitor_process")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        monitor = EnergyMonitor(
            server_args=server_args,
            port_args=port_args,
            **kwargs,
        )
        monitor.event_loop()
    except Exception as e:
        traceback = get_exception_traceback()
        logger.error(f"EnergyMonitor hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
