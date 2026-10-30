"""Tests for Chapter 4.1: Introduction to Make & Bare-metal Build Automation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    MakeRule,
    MakeDependencyGraph,
    CompilerFlagAnalyzer,
    plot_fig4_01_make_dag_and_timestamp_evaluation,
    plot_fig4_02_compiler_and_linker_flag_architecture,
)


def test_make_rule_and_automatic_variable_expansion():
    """Verify MakeRule automatic variable expansions ($@, $<, $^, $?, $*)."""
    rule = MakeRule(
        target="kernel.elf",
        prerequisites=["main.o", "graphics.o", "Makefile"],
        recipes=["ld.lld $(LDFLAGS) -o $@ $^"]
    )
    # Expand with all prerequisites
    cmds = rule.expand_recipe("kernel.elf", ["main.o", "graphics.o", "Makefile"])
    assert len(cmds) == 1
    assert cmds[0] == "ld.lld $(LDFLAGS) -o kernel.elf main.o graphics.o Makefile"

    # Pattern rule with stem
    prule = MakeRule(
        target="%.o",
        prerequisites=["%.cpp", "Makefile"],
        recipes=["clang++ $(CXXFLAGS) -c $< -o $@"],
        is_pattern=True
    )
    pcmds = prule.expand_recipe("main.o", ["main.cpp", "Makefile"], newer_prerequisites=["main.cpp"], stem="main")
    assert len(pcmds) == 1
    assert pcmds[0] == "clang++ $(CXXFLAGS) -c main.cpp -o main.o"


def test_make_dependency_graph_dag_and_topological_sort():
    """Verify DAG adjacency list, topological ordering, and cycle detection."""
    graph = MakeDependencyGraph()
    graph.add_rule("all", ["kernel.elf"], [])
    graph.add_rule("kernel.elf", ["main.o", "Makefile"], ["ld.lld -o $@ $^"])
    graph.add_pattern_rule("%.o", "%.cpp", ["clang++ -c $<"])
    graph.mark_phony("all")

    graph.set_file("main.cpp", 100.0)
    graph.set_file("Makefile", 100.0)

    dag = graph.get_dependencies_dag("all")
    assert dag["all"] == ["kernel.elf"]
    assert dag["kernel.elf"] == ["main.o", "Makefile"]
    assert dag["main.o"] == ["main.cpp"]

    order = graph.topological_sort("all")
    # Prerequisites must precede targets
    idx_main_cpp = order.index("main.cpp")
    idx_main_o = order.index("main.o")
    idx_makefile = order.index("Makefile")
    idx_kernel_elf = order.index("kernel.elf")
    idx_all = order.index("all")

    assert idx_main_cpp < idx_main_o
    assert idx_main_o < idx_kernel_elf
    assert idx_makefile < idx_kernel_elf
    assert idx_kernel_elf < idx_all

    # Test cyclic dependency detection
    cyclic = MakeDependencyGraph()
    cyclic.add_rule("a", ["b"], [])
    cyclic.add_rule("b", ["c"], [])
    cyclic.add_rule("c", ["a"], [])
    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        cyclic.topological_sort("a")


def test_make_timestamp_incremental_build():
    """Verify incremental build decision tree based on file timestamps."""
    graph = MakeDependencyGraph()
    graph.add_rule("all", ["kernel.elf"], [])
    graph.add_rule("kernel.elf", ["main.o", "Makefile"], ["ld.lld -o $@ $^"])
    graph.add_pattern_rule("%.o", "%.cpp", ["clang++ -c $<"])
    graph.mark_phony("all")

    graph.set_file("main.cpp", 100.0)
    graph.set_file("Makefile", 100.0)

    # 1. Full initial build
    cmds1 = graph.build("all", current_time=1000.0)
    assert len(cmds1) == 2
    assert "clang++ -c main.cpp" in cmds1[0]
    assert "ld.lld -o kernel.elf main.o Makefile" in cmds1[1]
    assert graph.file_exists("main.o")
    assert graph.file_exists("kernel.elf")

    # 2. No changes -> up to date
    cmds2 = graph.build("all", current_time=2000.0)
    assert len(cmds2) == 0

    # 3. Modify main.cpp -> incremental rebuild of main.o and kernel.elf
    graph.set_file("main.cpp", 2100.0)
    cmds3 = graph.build("all", current_time=3000.0)
    assert len(cmds3) == 2
    assert "clang++ -c main.cpp" in cmds3[0]
    assert "ld.lld -o kernel.elf main.o Makefile" in cmds3[1]

    # 4. Modify Makefile -> only kernel.elf rebuilt (main.o depends on main.cpp only in pattern)
    graph.set_file("Makefile", 4000.0)
    cmds4 = graph.build("all", current_time=5000.0)
    assert len(cmds4) == 1
    assert "ld.lld -o kernel.elf main.o Makefile" in cmds4[0]


def test_phony_target_protection():
    """Verify phony targets execute regardless of existing filesystem files."""
    graph = MakeDependencyGraph()
    graph.add_rule("clean", [], ["rm -rf *.o kernel.elf"])
    graph.mark_phony("clean")

    # Virtual file 'clean' with future timestamp
    graph.set_file("clean", 999999.0)

    # Must still execute
    cmds = graph.build("clean", current_time=1000.0)
    assert len(cmds) == 1
    assert cmds[0] == "rm -rf *.o kernel.elf"


def test_compiler_and_linker_flag_analyzer():
    """Verify architectural inspection of MikanOS bare-metal compiler & linker flags."""
    flags = CompilerFlagAnalyzer.FLAGS
    assert len(flags) == 13

    # Check key flags exist
    flag_freestanding = CompilerFlagAnalyzer.get_flag("-ffreestanding")
    assert "自立" in flag_freestanding.purpose or "フリースタンディング" in flag_freestanding.purpose
    assert "スタートアップ" in flag_freestanding.baremetal_necessity or "libc" in flag_freestanding.baremetal_necessity

    flag_redzone = CompilerFlagAnalyzer.get_flag("-mno-red-zone")
    assert "128" in flag_redzone.purpose
    assert "割り込み" in flag_redzone.baremetal_necessity or "スタック" in flag_redzone.baremetal_necessity

    flag_image_base = CompilerFlagAnalyzer.get_flag("--image-base 0x100000")
    assert "0x100000" in flag_image_base.purpose or "1MiB" in flag_image_base.purpose

    flag_entry = CompilerFlagAnalyzer.get_flag("--entry KernelMain")
    assert "KernelMain" in flag_entry.purpose

    categories = CompilerFlagAnalyzer.get_flags_by_category()
    assert "Optimization" in categories
    assert "Target Architecture" in categories
    assert "Runtime Environment" in categories
    assert "Linker Specification" in categories


def test_figure_generation_outputs(tmp_path: Path):
    """Verify Figure 4-1 and Figure 4-2 generate valid high-DPI PNGs."""
    p1 = plot_fig4_01_make_dag_and_timestamp_evaluation(tmp_path / "fig4_01.png")
    assert p1.exists()
    assert p1.stat().st_size > 50_000
    with Image.open(p1) as img:
        assert img.format == "PNG"
        assert img.width > 1000

    p2 = plot_fig4_02_compiler_and_linker_flag_architecture(tmp_path / "fig4_02.png")
    assert p2.exists()
    assert p2.stat().st_size > 50_000
    with Image.open(p2) as img:
        assert img.format == "PNG"
        assert img.width > 1000


def test_cpp20_make_sim_standalone():
    """Verify standalone C++20 make_sim binary runs without error."""
    bin_path = Path("build/bin/ch04_make_sim")
    assert bin_path.exists(), f"Binary {bin_path} not found. Run cmake build first."

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "All C++20 Make & Bare-metal Build tests passed successfully!" in res.stdout


def test_notebook_structure_and_error_free_execution():
    """Verify 4.1 notebook exists, contains required sections, and has zero execution errors."""
    nb_path = Path("4/4.1_Introduction_to_Make.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist."

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 10, "Notebook should have at least 10 cells."

    # Verify zero errors
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", (
                    f"Cell {idx} has execution error: {out.get('ename')}: {out.get('evalue')}"
                )

    # Check key terms in notebook source
    full_text = "".join("".join(c.get("source", [])) for c in cells)
    assert "kernel/Makefile" in full_text
    assert "DAG" in full_text
    assert "Topological" in full_text or "トポロジカル" in full_text
    assert "$@" in full_text
    assert "$<" in full_text
    assert "$^" in full_text
    assert ".PHONY" in full_text
    assert "-ffreestanding" in full_text
    assert "-mno-red-zone" in full_text
    assert "--image-base 0x100000" in full_text
