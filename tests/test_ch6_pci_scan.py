"""Tests for Chapter 6.3: PCI Configuration Space & Bus Scanning Simulation."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    PCIAddress,
    PCIDeviceSim,
    PCIConfigSpaceSimulator,
    plot_fig6_05_pci_config_address_and_header_layout,
    plot_fig6_06_pci_bus_tree_and_recursive_scan_algorithm,
)


def test_pci_address_bit_manipulation():
    """Verify 32-bit CONFIG_ADDRESS bitfield packing and unpacking."""
    # Bus 0, Dev 0, Func 0, Reg 0 -> 0x80000000 (Enable bit only)
    addr0 = PCIAddress(bus=0, device=0, function=0, reg_addr=0x00, enable=True)
    assert addr0.to_uint32() == 0x80000000

    # Bus 1, Dev 2, Func 3, Reg 0x0C
    # Bus 1: (1 << 16) = 0x00010000
    # Dev 2: (2 << 11) = 0x00001000
    # Func 3: (3 << 8) = 0x00000300
    # Reg 0x0C: 0x0C
    # Enable: 0x80000000
    # Sum: 0x8001130C
    addr1 = PCIAddress(bus=1, device=2, function=3, reg_addr=0x0C, enable=True)
    assert addr1.to_uint32() == 0x8001130C

    # Round trip from uint32
    decoded = PCIAddress.from_uint32(0x8001130C)
    assert decoded.bus == 1
    assert decoded.device == 2
    assert decoded.function == 3
    assert decoded.reg_addr == 0x0C
    assert decoded.enable is True

    # 4-byte DWORD alignment masking check
    unaligned = PCIAddress(bus=0, device=0, function=0, reg_addr=0x0F, enable=True)
    assert unaligned.to_uint32() & 0x03 == 0
    assert unaligned.to_uint32() & 0xFC == 0x0C


def test_pci_device_class_code_and_flags():
    """Verify PCI device class code derivation and helper flags."""
    # xHCI: Base Class 0x0C, Sub Class 0x03, Prog IF 0x30, Rev 0x01
    xhci = PCIDeviceSim(
        bus=0, device=4, function=0,
        vendor_id=0x1B36, device_id=0x000D,
        base_class=0x0C, sub_class=0x03, prog_if=0x30, revision_id=0x01,
        header_type=0x00
    )
    assert xhci.class_code == 0x0C033001
    assert xhci.is_xhci is True
    assert xhci.is_single_function is True
    assert xhci.is_pci_bridge is False

    # PCI-to-PCI Bridge: Base Class 0x06, Sub Class 0x04
    bridge = PCIDeviceSim(
        bus=0, device=5, function=0,
        vendor_id=0x1B36, device_id=0x0001,
        base_class=0x06, sub_class=0x04, prog_if=0x00, revision_id=0x00,
        header_type=0x01, secondary_bus=1, subordinate_bus=1
    )
    assert bridge.is_pci_bridge is True
    assert bridge.is_xhci is False
    assert bridge.is_single_function is True

    # Multi-Function Device (header_type bit 7 = 1)
    multi_dev = PCIDeviceSim(
        bus=0, device=1, function=0,
        vendor_id=0x8086, device_id=0x7000,
        header_type=0x80
    )
    assert multi_dev.is_single_function is False


def test_pci_config_simulator_io_ports():
    """Verify reading and writing to simulated I/O ports 0x0CF8 and 0x0CFC."""
    sim = PCIConfigSpaceSimulator()
    dev = PCIDeviceSim(
        bus=0, device=2, function=0,
        vendor_id=0x1234, device_id=0x5678,
        command=0x0003, status=0x0210,
        base_class=0x03, sub_class=0x00, prog_if=0x00, revision_id=0x02,
        header_type=0x00
    )
    sim.register_device(dev)

    # 1. Vendor ID & Device ID (Offset 0x00)
    assert sim.read_vendor_id(0, 2, 0) == 0x1234
    assert sim.read_device_id(0, 2, 0) == 0x5678

    # 2. Non-existent device returns 0xFFFF
    assert sim.read_vendor_id(0, 31, 0) == 0xFFFF
    assert sim.read_data() == 0xFFFFFFFF

    # 3. Class Code (Offset 0x08)
    assert sim.read_class_code(0, 2, 0) == 0x03000002

    # 4. Header Type (Offset 0x0C)
    assert sim.read_header_type(0, 2, 0) == 0x00


def test_pci_single_vs_multi_function_scan():
    """Verify that ScanAllBus only queries functions 1..7 when bit 7 of Header Type is set."""
    sim = PCIConfigSpaceSimulator()
    # Host bridge
    sim.register_device(PCIDeviceSim(0, 0, 0, 0x8086, 0x1237, header_type=0x00))

    # Single-function device at 0.1.0
    sim.register_device(PCIDeviceSim(0, 1, 0, 0x1111, 0x2222, header_type=0x00))
    # Phantom device at 0.1.1 (should NOT be detected because 0.1.0 is single-function)
    sim.register_device(PCIDeviceSim(0, 1, 1, 0x1111, 0x3333, header_type=0x00))

    discovered, _ = sim.scan_all_bus()
    funcs = [(d.bus, d.device, d.function) for d in discovered]
    assert (0, 0, 0) in funcs
    assert (0, 1, 0) in funcs
    assert (0, 1, 1) not in funcs

    # Multi-function device at 0.2.0 (header_type = 0x80)
    sim.register_device(PCIDeviceSim(0, 2, 0, 0x4444, 0x5555, header_type=0x80))
    sim.register_device(PCIDeviceSim(0, 2, 1, 0x4444, 0x6666, header_type=0x00))
    sim.register_device(PCIDeviceSim(0, 2, 3, 0x4444, 0x7777, header_type=0x00))

    discovered2, _ = sim.scan_all_bus()
    funcs2 = [(d.bus, d.device, d.function) for d in discovered2]
    assert (0, 2, 0) in funcs2
    assert (0, 2, 1) in funcs2
    assert (0, 2, 3) in funcs2


def test_pci_bridge_recursive_scan():
    """Verify that discovering a PCI-to-PCI bridge triggers a recursive scan on the secondary bus."""
    sim = PCIConfigSpaceSimulator()
    sim.register_device(PCIDeviceSim(0, 0, 0, 0x8086, 0x1237, header_type=0x00))
    # Bridge on Bus 0, Dev 5 leading to Secondary Bus 1
    sim.register_device(PCIDeviceSim(
        0, 5, 0, 0x1B36, 0x0001,
        base_class=0x06, sub_class=0x04,
        header_type=0x01, secondary_bus=1, subordinate_bus=1
    ))
    # Device on Bus 1, Dev 2
    sim.register_device(PCIDeviceSim(1, 2, 0, 0x8086, 0x2415, header_type=0x00))

    discovered, logs = sim.scan_all_bus()
    funcs = [(d.bus, d.device, d.function) for d in discovered]
    assert (0, 5, 0) in funcs
    assert (1, 2, 0) in funcs
    assert any("Found PCI-to-PCI Bridge! Recursively scanning secondary bus 1" in l for l in logs)


def test_xhci_device_detection():
    """Verify exact identification of the xHCI USB controller among arbitrary devices."""
    sim = PCIConfigSpaceSimulator.create_qemu_standard_pci_space()
    discovered, _ = sim.scan_all_bus()

    xhci_devs = [d for d in discovered if d.is_xhci]
    assert len(xhci_devs) == 1
    xhci = xhci_devs[0]
    assert xhci.bus == 0
    assert xhci.device == 4
    assert xhci.function == 0
    assert xhci.vendor_id == 0x1B36
    assert xhci.device_id == 0x000D


def test_qemu_standard_topology_scan():
    """Verify full discovery of 9 standard QEMU devices across Bus 0 and Bus 1."""
    sim = PCIConfigSpaceSimulator.create_qemu_standard_pci_space()
    discovered, _ = sim.scan_all_bus()
    assert len(discovered) == 9

    # Verify device types present
    vendors = {d.vendor_id for d in discovered}
    assert 0x8086 in vendors  # Intel
    assert 0x1234 in vendors  # QEMU VGA
    assert 0x1B36 in vendors  # Red Hat / QEMU xHCI and Bridge


def test_cpp_standalone_simulation_execution():
    """Verify C++20 standalone simulator builds and passes all tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cpp_src = repo_root / "src" / "ch06" / "pci_scan_sim.cpp"
    assert cpp_src.exists()

    bin_path = repo_root / "build" / "bin" / "ch06_pci_scan_sim"
    if not bin_path.exists():
        subprocess.run(
            ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", str(cpp_src), "-o", "/tmp/ch06_pci_scan_sim"],
            check=True
        )
        bin_path = Path("/tmp/ch06_pci_scan_sim")

    res = subprocess.run([str(bin_path), "--test"], capture_output=True, text=True, check=True)
    assert "All Chapter 6.3 PCI scan simulation tests PASSED" in res.stdout


def test_figure_generation_and_validity(tmp_path):
    """Verify generation of Figure 6.05 and 6.06 with high DPI and valid image properties."""
    fig1_path = tmp_path / "fig6_05.png"
    fig2_path = tmp_path / "fig6_06.png"

    plot_fig6_05_pci_config_address_and_header_layout(str(fig1_path))
    plot_fig6_06_pci_bus_tree_and_recursive_scan_algorithm(str(fig2_path))

    assert fig1_path.exists()
    assert fig2_path.exists()
    assert fig1_path.stat().st_size > 100_000
    assert fig2_path.stat().st_size > 100_000

    with Image.open(fig1_path) as img1:
        assert img1.size[0] >= 3000
        assert img1.size[1] >= 2000

    with Image.open(fig2_path) as img2:
        assert img2.size[0] >= 3000
        assert img2.size[1] >= 2000


def test_notebook_structure_and_error_free_execution():
    """Verify 6/6.3_Scanning_PCI_Configuration_Space.ipynb exists, is valid JSON, and has no execution errors."""
    nb_path = Path("6/6.3_Scanning_PCI_Configuration_Space.ipynb")
    assert nb_path.exists(), "Notebook 6.3 must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10, "Notebook must contain at least 10 cells"

    # Check for execution errors in cell outputs
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", f"Cell {idx} contains an error output: {output}"
