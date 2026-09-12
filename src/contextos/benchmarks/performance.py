"""Low-overhead, standard-library performance measurement helpers."""

from __future__ import annotations

import ctypes
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class PerformanceProbe:
    """Measurements populated when a profiled block exits, including on failure."""

    wall_time_ms: float = 0.0
    peak_process_memory_bytes: int | None = None


def peak_process_memory_bytes() -> int | None:
    """Return the process-lifetime peak resident set using native OS counters."""
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set_size)
        return None
    try:
        resource = importlib.import_module("resource")
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    return peak_rss if sys.platform == "darwin" else peak_rss * 1_024


@contextmanager
def measure_performance() -> Iterator[PerformanceProbe]:
    """Measure a block's wall time and native process-lifetime peak resident memory."""
    probe = PerformanceProbe()
    started = perf_counter()
    try:
        yield probe
    finally:
        probe.wall_time_ms = (perf_counter() - started) * 1_000
        probe.peak_process_memory_bytes = peak_process_memory_bytes()
