"""Unit tests for Chapter 0 Section 0.4: How to Enjoy OS Development."""

import math
from pathlib import Path
import subprocess
import pytest

from common import (
    MIKANOS_KEY_MILESTONES,
    BareMetalDebugSimulator,
    DebugInspectionStep,
    DebugStrategy,
    DevelopmentMethod,
    HardwareEnvironmentFeature,
    QemuVsHardwareComparator,
    SmallStepPhase,
    SmallStepProgressTracker,
)


def test_debug_strategy_enums() -> None:
    """Verify debug strategies and development methodologies enums."""
    assert DebugStrategy.FRAMEBUFFER_COLOR.value == "framebuffer_color"
    assert DebugStrategy.HALT_LOOP.value == "halt_loop"
    assert DebugStrategy.SERIAL_PORT.value == "serial_port"
    assert DebugStrategy.QEMU_MONITOR.value == "qemu_monitor"
    assert DebugStrategy.BINARY_SEARCH.value == "binary_search"

    assert DevelopmentMethod.BIG_BANG.value == "big_bang"
    assert DevelopmentMethod.SMALL_STEP.value == "small_step"


def test_small_step_milestones() -> None:
    """Verify predefined MikanOS incremental development milestones."""
    assert len(MIKANOS_KEY_MILESTONES) >= 10

    first = MIKANOS_KEY_MILESTONES[0]
    assert first.day == "day01a"
    assert first.chapter == 1
    assert first.added_lines > 0
    assert "UEFI" in first.title or "UEFI" in first.primary_feature

    # Verify chapters are monotonically non-decreasing
    chapters = [m.chapter for m in MIKANOS_KEY_MILESTONES]
    assert chapters == sorted(chapters)


def test_small_step_progress_tracker() -> None:
    """Verify cost model comparing Big-Bang vs Small-Step development."""
    tracker = SmallStepProgressTracker()
    total_lines = tracker.total_cumulative_lines()
    assert total_lines > 500

    cost_bb = tracker.calculate_debugging_cost(total_lines, DevelopmentMethod.BIG_BANG)
    cost_ss = tracker.calculate_debugging_cost(total_lines, DevelopmentMethod.SMALL_STEP)

    assert cost_bb["worst_case_search_lines"] == float(total_lines)
    assert cost_ss["worst_case_search_lines"] < cost_bb["worst_case_search_lines"]
    assert cost_ss["feedback_delay_days"] < cost_bb["feedback_delay_days"]

    # Small-step search space should be roughly total_lines / K
    k = tracker.total_milestones()
    expected_delta = total_lines / k
    assert math.isclose(cost_ss["worst_case_search_lines"], expected_delta, rel_tol=1e-5)


def test_bare_metal_debug_simulator_vram_and_serial() -> None:
    """Verify VRAM visual marker and serial port bare-metal debugging."""
    sim = BareMetalDebugSimulator(total_lines=100, bug_line=45)

    # Checkpoint 1 (line 20) is before bug: passes
    ok1 = sim.set_vram_color(checkpoint_line=20, rgb=(255, 0, 0))
    assert ok1 is True
    assert sim.vram_color == (255, 0, 0)

    # Serial character output before bug
    ok_serial1 = sim.write_serial(checkpoint_line=30, char="A")
    assert ok_serial1 is True
    assert sim.serial_buffer == ["A"]

    # Checkpoint 2 (line 60) is after bug line 45: crashes
    ok2 = sim.set_vram_color(checkpoint_line=60, rgb=(0, 255, 0))
    assert ok2 is False
    # VRAM color remains RED (last successful checkpoint before crash)
    assert sim.vram_color == (255, 0, 0)

    ok_serial2 = sim.write_serial(checkpoint_line=50, char="B")
    assert ok_serial2 is False
    assert sim.serial_buffer == ["A"]


def test_bare_metal_debug_simulator_binary_search() -> None:
    """Verify binary search debugging accurately pinpoints bug lines."""
    total_lines = 120
    test_bug_lines = [1, 2, 42, 77, 119, 120]

    for bug in test_bug_lines:
        sim = BareMetalDebugSimulator(total_lines=total_lines, bug_line=bug)
        steps = sim.locate_bug_by_binary_search()
        assert len(steps) > 0

        # Theoretical upper bound is ceil(log2(N))
        max_theoretical_steps = math.ceil(math.log2(total_lines))
        assert len(steps) <= max_theoretical_steps

        # Last step narrow down
        last_step = steps[-1]
        found_line = (last_step.probe_line + 1) if last_step.outcome == "PASS" else last_step.probe_line
        assert found_line == bug, f"Failed to locate bug {bug}: got {found_line}"


def test_qemu_vs_hardware_comparator() -> None:
    """Verify QEMU vs real hardware feature comparison matrix."""
    table = QemuVsHardwareComparator.get_comparison_table()
    assert len(table) >= 4

    categories = [f.category for f in table]
    assert any("時間" in c for c in categories)
    assert any("デバッグ" in c for c in categories)
    assert any("安全" in c for c in categories)

    for feature in table:
        assert isinstance(feature, HardwareEnvironmentFeature)
        assert len(feature.qemu_ovmf) > 0
        assert len(feature.real_hardware) > 0
        assert len(feature.impact_on_os_dev) > 0


def test_cpp_debug_strategies_binary() -> None:
    """Verify C++20 standalone binary executes with return code 0 and valid output."""
    project_root = Path(__file__).resolve().parent.parent
    bin_path = project_root / "build" / "bin" / "ch00_os_debug_strategies"

    if not bin_path.exists():
        pytest.skip("ch00_os_debug_strategies binary not built yet.")

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert res.returncode == 0
    assert "Section 0.4" in res.stdout
    assert "VRAM" in res.stdout
    assert "Binary Search" in res.stdout
    assert "All C++20 Section 0.4 verifications PASSED" in res.stdout


def test_generated_figures_exist() -> None:
    """Verify generated Section 0.4 architecture and debugging figures exist."""
    project_root = Path(__file__).resolve().parent.parent
    result_dir = project_root / "0" / "result"

    fig1 = result_dir / "fig0_7_small_step_development_cycle.png"
    fig2 = result_dir / "fig0_8_os_debugging_strategies.png"

    assert fig1.exists(), f"Missing {fig1}"
    assert fig2.exists(), f"Missing {fig2}"
    assert fig1.stat().st_size > 50_000, "Figure 0.7 file too small"
    assert fig2.stat().st_size > 50_000, "Figure 0.8 file too small"

