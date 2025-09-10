import sys
import logging
import dataclasses
from typing import List, Dict, Optional

import pandas as pd

from sglang.utils import get_bool_env_var

logger = logging.getLogger(__name__)

VOLTANA_DEBUG = get_bool_env_var("VOLTANA_DEBUG", "false")


@dataclasses.dataclass
class LookupTableItem:
    bs_start: Optional[float]
    bs_end: Optional[float]
    coef_bs: Optional[float]
    coef_tokens: float
    coef_intercept: float
    r_2: Optional[float] = None


class BaseLookupTable:

    _logging_header: str = ""

    def __init__(self, table_csv_path: Optional[str] = None):
        if table_csv_path is None:
            logger.warning(f"{self._logging_header}table_csv_path is None, using empty table.")
            self.tables = {}
            return

        table = pd.read_csv(table_csv_path)
        self.tables = dict()
        for freq in table["freq"].unique():
            table_freq = table[table["freq"] == freq]
            if "bs_start" in table_freq.columns:
                table_freq = table_freq.sort_values(by=["bs_start"])
            items = []
            # convert to list of tuples
            for i in range(len(table_freq)):
                item = LookupTableItem(
                    bs_start=float(table_freq.iloc[i]["bs_start"]) if "bs_start" in table_freq.columns else None,
                    bs_end=float(table_freq.iloc[i]["bs_end"]) if "bs_end" in table_freq.columns else None,
                    coef_bs=float(table_freq.iloc[i]["coef_bs"]) if "coef_bs" in table_freq.columns else None,
                    coef_tokens=float(table_freq.iloc[i]["coef_tokens"]),
                    coef_intercept=float(table_freq.iloc[i]["coef_intercept"]),
                    r_2=float(table_freq.iloc[i]["r_2"]) if "r_2" in table_freq.columns else None,
                )
                items.append(item)

            self.tables[freq] = items
        self.freq_list = sorted(self.tables.keys())

    def _find_table(self, freq: int, bs: float, tokens: float) -> Optional[LookupTableItem]:
        if freq not in self.tables:
            logger.warning(f"{self._logging_header}freq: {freq}, bs: {bs}, tokens: {tokens} => not found")
            return None
        table = self.tables[freq]
        if bs is None:
            if len(table) != 1:
                logger.warning(f"There are multiple items for freq: {freq} without bs, this is not expected")
            return table[0]
        for item in table:
            if item.bs_start <= bs <= item.bs_end:
                return item
        return None

    def _check_zero(self, bs: float, tokens: float) -> bool:
        if bs <= 0:
            logger.debug(f"{self._logging_header}bs = {bs} <= 0 detected, return 0")
            return True
        if tokens <= 0:
            logger.debug(f"{self._logging_header}tokens = {tokens} <= 0 detected, return 0")
            tokens = True
        return False

    def _calc_interploation(self, freq: int, freq_left: int, freq_right: int, bs: float, tokens: float) -> Optional[float]:
        assert freq_left < freq < freq_right
        value_left = self.lookup(freq_left, bs, tokens)
        value_right = self.lookup(freq_right, bs, tokens)
        if value_left is None or value_right is None:
            return None
        slope = (value_right - value_left) / (freq_right - freq_left)
        value = value_left + slope * (freq - freq_left)
        return float(value)

    def _freq_interpolate(self, freq: int, bs: float, tokens: float) -> Optional[float]:
        if VOLTANA_DEBUG:
            logger.debug(f"{self._logging_header}freq {freq} not found, trying interpolation")
        assert freq not in self.freq_list
        if freq < self.freq_list[0]:
            return self._calc_interploation(freq, self.freq_list[0], self.freq_list[1], bs, tokens)
        if freq > self.freq_list[-1]:
            return self._calc_interploation(freq, self.freq_list[-2], self.freq_list[-1], bs, tokens)
        for i in range(len(self.freq_list) - 1):
            if self.freq_list[i] < freq < self.freq_list[i + 1]:
                return self._calc_interploation(freq, self.freq_list[i], self.freq_list[i + 1], bs, tokens)
        return None

    def lookup(self, freq: int, bs: float, tokens: float) -> Optional[float]:
        raise NotImplementedError("This method should be implemented in subclasses")


class LatencyLookupTable(BaseLookupTable):

    def __init__(self, table_csv_path: Optional[str] = None):
        self._logging_header = "[Latency Lookup Table Decoding]: "
        super().__init__(table_csv_path)

    def lookup(self, freq: int, bs: float, tokens: float) -> Optional[float]:
        if self._check_zero(bs, tokens):
            return 0

        if freq not in self.freq_list:
            return self._freq_interpolate(freq, bs, tokens)

        item = self._find_table(freq, bs, tokens)
        if item is None:
            return None

        latency = item.coef_bs * bs + item.coef_tokens * tokens + item.coef_intercept

        if latency <= 0:
            if VOLTANA_DEBUG:
                logger.fatal(f"{self._logging_header}freq: {freq}, tokens: {tokens} => latency: {latency} <= 0, this is not expected")
            latency = 20

        if VOLTANA_DEBUG:
            logger.info(f"{self._logging_header}freq: {freq}, bs: {bs}, tokens: {tokens} => latency: {latency}")
        return latency


class LatencyLookupTablePrefill(BaseLookupTable):

    def __init__(self, table_csv_path: Optional[str] = None):
        self._logging_header = "[Latency Lookup Table Prefill]: "
        super().__init__(table_csv_path)

    def lookup(self, freq: int, bs: float, tokens: float) -> Optional[float]:
        if self._check_zero(1.0, tokens):
            return 0

        if freq not in self.freq_list:
            return self._freq_interpolate(freq, bs, tokens)

        item = self._find_table(freq, None, tokens)
        if item is None:
            return None

        latency = item.coef_tokens * tokens + item.coef_intercept

        if latency <= 0:
            if VOLTANA_DEBUG:
                logger.fatal(f"{self._logging_header}freq: {freq}, tokens: {tokens} => latency: {latency} <= 0, this is not expected")
            latency = 20

        if VOLTANA_DEBUG:
            logger.info(f"{self._logging_header}freq: {freq}, tokens: {tokens} => latency: {latency}")
        return latency


class EnergyLookupTable(BaseLookupTable):

    def __init__(self, table_csv_path: Optional[str] = None):
        self._logging_header = "[Energy Lookup Table]: "
        super().__init__(table_csv_path)

    def lookup(self, freq: int, bs: float, tokens: float) -> Optional[float]:
        if self._check_zero(bs, tokens):
            return 0

        if freq not in self.freq_list:
            return self._freq_interpolate(freq, bs, tokens)

        item = self._find_table(freq, bs, tokens)
        if item is None:
            return None

        energy = item.coef_bs * bs + item.coef_tokens * tokens + item.coef_intercept

        if energy <= 0:
            logger.fatal(f"{self._logging_header}freq: {freq}, bs: {bs}, tokens: {tokens} => energy: {energy} <= 0, this is not expected")
            sys.exit(1)

        if VOLTANA_DEBUG:
            logger.info(f"{self._logging_header}freq: {freq}, bs: {bs}, tokens: {tokens} => energy: {energy}")
        return energy
