"""Tests for Chapter 9.4: Compositing Performance Profiling & Local APIC Timer Simulation."""

from pathlib import Path
from PIL import Image
import pytest

from common import (
    LAPICTimerRegisterSim,
    CompositingProfilerSim,
    MouseObserverLatencySim,
    plot_fig9_10_lapic_timer_architecture_and_mmio_registers,
    plot_fig9_11_full_redraw_overhead_and_mouse_latency_timeline,
    plot_fig9_12_compositing_performance_profiling_and_complexity_analysis,
)


def test_lapic_timer_register_decoding():
    """Verify Local APIC Divide Configuration and LVT Timer bitfield decoding."""
    timer = LAPICTimerRegisterSim()

    # Default divide_config in MikanOS is 0b1011 (divide by 1: 1:1)
    assert timer.divide_config == 0b1011
    assert timer.divider == 1

    # Test other divider ratios per Intel SDM specification
    assert LAPICTimerRegisterSim.decode_divide_config(0b0000) == 2
    assert LAPICTimerRegisterSim.decode_divide_config(0b0001) == 4
    assert LAPICTimerRegisterSim.decode_divide_config(0b0010) == 8
    assert LAPICTimerRegisterSim.decode_divide_config(0b0011) == 16
    assert LAPICTimerRegisterSim.decode_divide_config(0b1000) == 32
    assert LAPICTimerRegisterSim.decode_divide_config(0b1001) == 64
    assert LAPICTimerRegisterSim.decode_divide_config(0b1010) == 128
    assert LAPICTimerRegisterSim.decode_divide_config(0b1011) == 1

    # LVT Timer default: (0b001 << 16) | 32
    # bit 16: masked = 1, bit 18:17: mode = 00 (One-shot), bit 7:0: vector = 32
    assert timer.is_masked is True
    assert timer.timer_mode == "One-shot"
    assert timer.vector == 32


def test_lapic_timer_countdown_and_elapsed():
    """Verify Local APIC Timer countdown mechanics and elapsed count calculation."""
    timer = LAPICTimerRegisterSim()
    timer.initialize(divide_code=0b1011, vector=32, masked=True)

    assert timer.elapsed() == 0
    assert timer.is_running is False

    # Start timer with max count 0xFFFFFFFF
    timer.start()
    assert timer.is_running is True
    assert timer.current_count == 0xFFFFFFFF
    assert timer.elapsed() == 0

    # Advance 50,000 CPU cycles (with 1:1 divider)
    timer.tick_cycles(50_000)
    assert timer.elapsed() == 50_000
    assert timer.current_count == 0xFFFFFFFF - 50_000

    # Advance another 150,000 cycles
    timer.tick_cycles(150_000)
    assert timer.elapsed() == 200_000
    assert timer.current_count == 0xFFFFFFFF - 200_000

    # Stop timer
    timer.stop()
    assert timer.is_running is False
    assert timer.initial_count == 0


def test_lapic_timer_divided_clock():
    """Verify Local APIC Timer with 1:16 clock divider (0b0011)."""
    timer = LAPICTimerRegisterSim()
    timer.initialize(divide_code=0b0011, vector=40, masked=False)

    assert timer.divider == 16
    assert timer.is_masked is False
    assert timer.vector == 40

    timer.start()
    # Advance 160 CPU cycles -> 160 / 16 = 10 timer ticks
    timer.tick_cycles(160)
    assert timer.elapsed() == 10
    assert timer.current_count == 0xFFFFFFFF - 10


def test_compositing_profiler_sim():
    """Verify CompositingProfilerSim calculations for full layer rendering."""
    profiler = CompositingProfilerSim(screen_width=1024, screen_height=768, bpp=4)

    assert profiler.total_screen_pixels == 786_432
    assert profiler.frame_buffer_bytes == 786_432 * 4  # 3,145,728 bytes (3 MiB)

    # Profile: Desktop background (1024x768) + Console (640x400) + Mouse (16x24)
    layers = [
        (1024, 768),
        (640, 400),
        (16, 24),
    ]
    res = profiler.profile_layers(layers)

    expected_scanned = (1024 * 768) + (640 * 400) + (16 * 24)
    assert res["num_layers"] == 3
    assert res["scanned_pixels"] == expected_scanned
    assert res["vram_bytes"] == expected_scanned * 4
    assert res["vram_mb"] == pytest.approx((expected_scanned * 4) / (1024 * 1024), rel=1e-3)
    assert res["estimated_lapic_ticks"] == expected_scanned * 12
    assert res["estimated_time_ms"] > 0
    assert res["max_fps"] > 0


def test_full_vs_partial_redraw_speedup():
    """Verify speedup ratio of partial (dirty rect) redraw over full compositing."""
    profiler = CompositingProfilerSim(screen_width=1024, screen_height=768, bpp=4)
    comp = profiler.compare_full_vs_partial(mouse_w=16, mouse_h=24)

    assert comp["screen_pixels"] == 786_432
    assert comp["mouse_pixels"] == 384
    assert comp["dirty_rect_pixels"] == 768  # Old + New position union
    # Speedup should be around 1000x!
    assert comp["speedup_ratio"] > 1000.0
    assert comp["pixel_reduction_pct"] > 99.9


def test_mouse_observer_latency_simulation():
    """Verify queue backlog and latency growth when redraw takes longer than event interval."""
    # Redraw takes 25ms, mouse events arrive every 10ms -> queue backlog builds up!
    sim = MouseObserverLatencySim(render_time_ms=25.0, event_interval_ms=10.0)
    records = sim.simulate(num_events=5)

    assert len(records) == 5

    # Event 1: arrives at 0ms, starts at 0ms, finishes at 25ms, latency 25ms
    assert records[0]["arrival_ms"] == 0.0
    assert records[0]["start_ms"] == 0.0
    assert records[0]["finish_ms"] == 25.0
    assert records[0]["latency_ms"] == 25.0

    # Event 2: arrives at 10ms, must wait until 25ms, finishes at 50ms, latency 40ms
    assert records[1]["arrival_ms"] == 10.0
    assert records[1]["start_ms"] == 25.0
    assert records[1]["finish_ms"] == 50.0
    assert records[1]["latency_ms"] == 40.0

    # Latencies should strictly increase due to server overload
    for i in range(1, len(records)):
        assert records[i]["latency_ms"] > records[i - 1]["latency_ms"]
        assert records[i]["queue_backlog"] >= 1


def test_figure_generation_outputs(tmp_path):
    """Verify that all 3 Chapter 9.4 figures generate valid 300 DPI images."""
    p10 = tmp_path / "fig9_10_test.png"
    p11 = tmp_path / "fig9_11_test.png"
    p12 = tmp_path / "fig9_12_test.png"

    out10 = plot_fig9_10_lapic_timer_architecture_and_mmio_registers(p10)
    out11 = plot_fig9_11_full_redraw_overhead_and_mouse_latency_timeline(p11)
    out12 = plot_fig9_12_compositing_performance_profiling_and_complexity_analysis(p12)

    for p in (out10, out11, out12):
        assert Path(p).is_file()
        with Image.open(p) as img:
            assert img.width > 1000
            assert img.height > 800
