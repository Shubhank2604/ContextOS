"""Tests for benchmark performance measurement primitives."""

import pytest

from contextos.benchmarks.performance import measure_performance, peak_process_memory_bytes


def test_probe_records_wall_time_and_peak_process_memory() -> None:
    with measure_performance() as probe:
        allocated = bytearray(4_096)

    assert allocated
    assert probe.wall_time_ms >= 0.0
    assert probe.peak_process_memory_bytes is not None
    assert probe.peak_process_memory_bytes >= len(allocated)


def test_probe_finishes_when_profiled_operation_fails() -> None:
    with pytest.raises(RuntimeError, match="fixture failure"), measure_performance() as probe:
        raise RuntimeError("fixture failure")

    assert probe.wall_time_ms >= 0.0
    assert probe.peak_process_memory_bytes is not None


def test_native_peak_memory_counter_is_available() -> None:
    first = peak_process_memory_bytes()
    second = peak_process_memory_bytes()

    assert first is not None and first > 0
    assert second is not None and second >= first
