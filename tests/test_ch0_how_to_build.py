"""Unit tests for Chapter 0 Section 0.1: How to Build an OS."""

from pathlib import Path
import subprocess
import pytest

from common import HISTORICAL_OS_METRICS, MIKANOS_GROWTH_METRICS, OSCodebaseMetric


def test_historical_os_metrics_values() -> None:
    """Verify historical OS LOC numbers match the textbook specification."""
    by_name = {m.name: m for m in HISTORICAL_OS_METRICS}

    haribote = by_name["Haribote OS (30-day OS)"]
    linux_001 = by_name["Linux 0.01"]
    mikanos = by_name["MikanOS (Final Kernel)"]
    linux_410 = by_name["Linux 4.10"]

    assert haribote.lines_of_code == 4_249
    assert linux_001.lines_of_code == 10_239
    assert mikanos.lines_of_code == 11_071
    assert linux_410.lines_of_code == 21_000_000

    # MikanOS is roughly the same scale as Linux 0.01 (~1.08x)
    ratio = mikanos.lines_of_code / linux_001.lines_of_code
    assert pytest.approx(ratio, rel=1e-2) == 1.08125

    # Average LOC per day over 30 days is ~369.03 lines/day
    assert pytest.approx(mikanos.lines_per_day(30), rel=1e-3) == 369.0333


def test_os_codebase_metric_invalid_days() -> None:
    """Verify ValueError on non-positive days in lines_per_day."""
    metric = OSCodebaseMetric("TestOS", 2026, 1000, "x86-64", "Educational")
    with pytest.raises(ValueError):
        metric.lines_per_day(0)
    with pytest.raises(ValueError):
        metric.lines_per_day(-5)


def test_mikanos_growth_curve_monotonicity() -> None:
    """Verify MikanOS official tag LOC and file counts grow monotonically from day02b to day30f."""
    assert len(MIKANOS_GROWTH_METRICS) >= 7
    prev_files = 0
    prev_lines = 0
    for tag, num_files, num_lines, desc in MIKANOS_GROWTH_METRICS:
        assert tag.startswith("day")
        assert len(desc) > 0
        assert num_files >= prev_files
        assert num_lines > prev_lines
        prev_files = num_files
        prev_lines = num_lines


def test_cpp20_os_scale_analyzer_execution(tmp_path: Path) -> None:
    """Compile and execute standalone C++20 src/ch00/os_scale_analyzer.cpp."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch00" / "os_scale_analyzer.cpp"
    assert cpp_src.exists()

    bin_path = tmp_path / "os_scale_analyzer"
    compile_res = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", str(cpp_src), "-o", str(bin_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

    run_res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=False)
    assert run_res.returncode == 0
    assert "MikanOS (Final Kernel)" in run_res.stdout
    assert "11071" in run_res.stdout
