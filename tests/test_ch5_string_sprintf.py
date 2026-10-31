"""Tests for Chapter 5.4: String Rendering and sprintf (osbook_day05d)."""

import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import pytest

from common import (
    StringRendererSim,
    NewlibStubAnalyzer,
    VarargArgument,
    SystemVVarargsSimulator,
    MiniSprintfSim,
    plot_fig5_07_string_rendering_coordinate_advance,
    plot_fig5_08_newlib_sbrk_and_varargs_abi,
)


def test_string_renderer_glyph_positions():
    """Verify glyph coordinate calculation and 8-pixel stride."""
    sample = "Hello"
    positions = StringRendererSim.calculate_glyph_positions(x=10, y=50, s=sample)

    assert len(positions) == 5
    for i, p in enumerate(positions):
        assert p["index"] == i
        assert p["char"] == sample[i]
        assert p["x"] == 10 + 8 * i
        assert p["y"] == 50
        assert p["width"] == 8
        assert p["height"] == 16
        assert p["bbox"] == (10 + 8 * i, 50, 10 + 8 * i + 8, 66)

    assert StringRendererSim.get_string_width(sample) == 40
    assert StringRendererSim.get_string_bounding_box(10, 50, sample) == (10, 50, 40, 16)


def test_string_renderer_rendering_on_buffer():
    """Verify string rendering onto a simulated FrameBuffer numpy array."""
    renderer = StringRendererSim()
    fb = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White background

    rendered_count = renderer.render_string(fb, x=0, y=10, s="Hi", color=(0, 0, 255))
    assert rendered_count == 2

    # Check that at least some pixels in the glyph areas were painted blue
    blue_pixels = np.sum((fb[:, :, 0] == 0) & (fb[:, :, 1] == 0) & (fb[:, :, 2] == 255))
    assert blue_pixels > 0


def test_newlib_stub_analyzer_metadata():
    """Verify Newlib required stubs metadata and minimal implementations."""
    sbrk_info = NewlibStubAnalyzer.get_stub_info("sbrk")
    assert sbrk_info is not None
    assert "caddr_t sbrk" in sbrk_info["signature"]
    assert "sprintf" in sbrk_info["required_for"][0]
    assert "NULL" in sbrk_info["minimal_impl"]

    write_info = NewlibStubAnalyzer.get_stub_info("write")
    assert write_info is not None
    assert "printf" in write_info["required_for"]


def test_newlib_sbrk_simulation():
    """Verify sbrk stub behavior in bare-metal and heap-enabled environments."""
    # Day05d bare-metal: no heap allocated
    result, msg = NewlibStubAnalyzer.simulate_sbrk_stub(current_break=0, increment=1024, max_heap=0)
    assert result is None
    assert "NULL" in msg

    # Heap-enabled environment: successful break allocation
    prev_brk, msg = NewlibStubAnalyzer.simulate_sbrk_stub(
        current_break=0x1000, increment=512, max_heap=0x8000
    )
    assert prev_brk == 0x1000
    assert "Allocated 512 bytes" in msg

    # Heap-enabled environment: out of memory
    oom_brk, msg = NewlibStubAnalyzer.simulate_sbrk_stub(
        current_break=0x7E00, increment=1024, max_heap=0x8000
    )
    assert oom_brk is None
    assert "ENOMEM" in msg


def test_system_v_varargs_abi_registers():
    """Verify System V AMD64 ABI register allocation and va_list state."""
    sim = SystemVVarargsSimulator()

    # sprintf(char *buf, const char *fmt, int a, int b, double f1, int c, int d, int e, int f)
    arg1 = sim.pass_argument("buf", 0x1000, "char*")
    assert arg1.passed_via == "RDI"
    assert not arg1.is_float

    arg2 = sim.pass_argument("fmt", 0x2000, "char*")
    assert arg2.passed_via == "RSI"

    arg3 = sim.pass_argument("a", 3, "int")
    assert arg3.passed_via == "RDX"

    arg4 = sim.pass_argument("b", 4, "int")
    assert arg4.passed_via == "RCX"

    arg5 = sim.pass_argument("f1", 3.14, "double")
    assert arg5.passed_via == "XMM0"
    assert arg5.is_float
    assert sim.al_register == 1

    arg6 = sim.pass_argument("c", 5, "int")
    assert arg6.passed_via == "R8"

    arg7 = sim.pass_argument("d", 6, "int")
    assert arg7.passed_via == "R9"

    # Overflow onto stack (7th GP integer argument)
    arg8 = sim.pass_argument("e", 7, "int")
    assert arg8.passed_via == "Stack [RSP+0]"

    # Verify va_list state
    state = sim.get_va_list_state(num_fixed_gp=2, num_fixed_fp=0)
    assert state["gp_offset"] == 16  # 2 fixed GP args (buf, fmt) consumed 16 bytes
    assert state["fp_offset"] == 48  # start of FP area in reg_save_area
    assert state["al_register"] == 1
    assert state["total_stack_args"] == 1


def test_mini_sprintf_formatting():
    """Verify format string parsing and conversion specifiers."""
    fmt = "Result: %d + %x = %s (Char: %c, Esc: %%)"
    formatted, steps = MiniSprintfSim.format(fmt, 10, 255, "valid", ord('Z'))

    assert formatted == "Result: 10 + ff = valid (Char: Z, Esc: %)"
    assert len(steps) >= 5

    # Check day05d exact expression
    d5_fmt = "1 + 2 = %d"
    d5_res, _ = MiniSprintfSim.format(d5_fmt, 1 + 2)
    assert d5_res == "1 + 2 = 3"


def test_figure_generation_outputs():
    """Verify Figure 5.7 and Figure 5.8 generation and properties."""
    fig5_07_path = Path("5/result/fig5_07_string_rendering_coordinate_advance.png")
    fig5_08_path = Path("5/result/fig5_08_newlib_sbrk_and_varargs_abi.png")

    plot_fig5_07_string_rendering_coordinate_advance(str(fig5_07_path))
    plot_fig5_08_newlib_sbrk_and_varargs_abi(str(fig5_08_path))

    assert fig5_07_path.exists()
    assert fig5_08_path.exists()

    with Image.open(fig5_07_path) as img:
        assert img.width >= 1200
        assert img.height >= 800

    with Image.open(fig5_08_path) as img:
        assert img.width >= 1200
        assert img.height >= 800


def test_cpp20_string_sprintf_sim_standalone():
    """Verify C++20 standalone executable compilation and execution."""
    cpp_src = Path("src/ch05/string_sprintf_sim.cpp")
    bin_path = Path("src/ch05/string_sprintf_sim")

    assert cpp_src.exists()

    compile_cmd = [
        "g++", "-std=c++20", "-O2", "-Wall", "-Wextra",
        str(cpp_src), "-o", str(bin_path)
    ]
    res_compile = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert res_compile.returncode == 0, f"Compilation failed:\n{res_compile.stderr}"

    run_cmd = [str(bin_path)]
    res_run = subprocess.run(run_cmd, capture_output=True, text=True)
    assert res_run.returncode == 0, f"Execution failed:\n{res_run.stderr}"
    assert "ALL TESTS PASSED" in res_run.stdout
    assert "WriteString" in res_run.stdout
    assert "1 + 2 = 3" in res_run.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify Chapter 5.4 notebook exists and is error-free."""
    nb_path = Path("5/5.4_String_Rendering_and_sprintf.ipynb")
    assert nb_path.exists(), "Notebook 5.4_String_Rendering_and_sprintf.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, "Notebook should have at least 10 rich cells"

    # Verify Cell 0 is Google Colab automatic environment setup
    cell0_source = "".join(cells[0].get("source", []))
    assert "Google Colab" in cell0_source
    assert "study-mikanos" in cell0_source

    # Check for execution errors in code cells
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", (
                    f"Cell {i} has execution error: {output.get('ename')}: {output.get('evalue')}"
                )
