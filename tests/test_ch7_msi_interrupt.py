"""Tests for Chapter 7.5: PCI MSI (Message Signaled Interrupts) Configuration."""

import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from common import (
    MSITriggerMode,
    MSIDeliveryMode,
    CapabilityHeader,
    MSICapability,
    PCIDeviceMSISimulator,
    format_msi_address,
    format_msi_data,
    plot_msi_vs_legacy_interrupts,
    plot_pci_capability_linked_list_and_msi_register,
)


def test_msi_constants_and_enums():
    """Verify MSI trigger mode, delivery mode, and PCI capability constants."""
    assert MSITriggerMode.kEdge == 0
    assert MSITriggerMode.kLevel == 1

    assert MSIDeliveryMode.kFixed == 0b000
    assert MSIDeliveryMode.kLowestPriority == 0b001
    assert MSIDeliveryMode.kSMI == 0b010
    assert MSIDeliveryMode.kNMI == 0b100
    assert MSIDeliveryMode.kINIT == 0b101
    assert MSIDeliveryMode.kExtINT == 0b111


def test_capability_header_serialization():
    """Verify 32-bit CapabilityHeader serialization and deserialization."""
    header = CapabilityHeader(cap_id=0x05, next_ptr=0x68, cap=0x0081)
    assert header.cap_id == 0x05
    assert header.next_ptr == 0x68
    assert header.cap == 0x0081

    expected_raw = 0x05 | (0x68 << 8) | (0x0081 << 16)
    assert header.raw_value == expected_raw

    unpacked = CapabilityHeader.from_raw(expected_raw)
    assert unpacked.cap_id == 0x05
    assert unpacked.next_ptr == 0x68
    assert unpacked.cap == 0x0081


def test_msi_capability_bitfields():
    """Verify MSICapability header serialization."""
    msi = MSICapability(
        cap_id=0x05,
        next_ptr=0x00,
        msi_enable=True,
        multi_msg_capable=3,
        multi_msg_enable=2,
        addr_64_capable=True,
        per_vector_mask_capable=False,
    )
    raw = msi.header_raw
    assert (raw & 0xFF) == 0x05
    assert ((raw >> 8) & 0xFF) == 0x00
    assert bool((raw >> 16) & 0x1) is True
    assert ((raw >> 17) & 0x7) == 3
    assert ((raw >> 20) & 0x7) == 2
    assert bool((raw >> 23) & 0x1) is True
    assert bool((raw >> 24) & 0x1) is False


def test_format_msi_address():
    """Verify Local APIC Message Address calculation on x86-64."""
    # BSP (apic_id = 0)
    addr_bsp = format_msi_address(apic_id=0)
    assert addr_bsp == 0xFEE00000

    # AP core 1 (apic_id = 1)
    addr_ap1 = format_msi_address(apic_id=1)
    assert addr_ap1 == 0xFEE01000

    # AP core 15 (apic_id = 15)
    addr_ap15 = format_msi_address(apic_id=15)
    assert addr_ap15 == 0xFEE0F000
    assert (addr_ap15 >> 20) == 0xFEE
    assert ((addr_ap15 >> 12) & 0xFF) == 15
    assert (addr_ap15 & 0x0FFF) == 0


def test_format_msi_data():
    """Verify Local APIC Message Data calculation."""
    # Edge Trigger, Fixed Delivery, Vector 0x40 (xHCI)
    data_edge = format_msi_data(vector=0x40, delivery_mode=MSIDeliveryMode.kFixed, trigger_mode=MSITriggerMode.kEdge)
    assert data_edge == 0x0040

    # Level Trigger, Fixed Delivery, Vector 0x40 (MikanOS standard)
    data_level = format_msi_data(vector=0x40, delivery_mode=MSIDeliveryMode.kFixed, trigger_mode=MSITriggerMode.kLevel)
    assert data_level == 0xC040
    assert (data_level & 0xFF) == 0x40
    assert (data_level & 0xC000) == 0xC000


def test_pci_device_msi_simulator_scan_and_config():
    """Verify linked list scanning and MSI register configuration in simulator."""
    sim = PCIDeviceMSISimulator(bus=0, device=20, function=0)

    # Setup linked list: 0x34 -> 0x50 (Power Management) -> 0x70 (MSI) -> 0x00
    sim.write_conf_reg(0x34, 0x50)
    sim.add_capability(cap_addr=0x50, cap_id=0x01, next_ptr=0x70)
    sim.setup_msi_capability(cap_addr=0x70, next_ptr=0x00, addr_64_capable=True)

    # Configure MSI for BSP, Vector 0x40
    ok, msg = sim.configure_msi_fixed_destination(
        apic_id=0,
        trigger_mode=MSITriggerMode.kLevel,
        delivery_mode=MSIDeliveryMode.kFixed,
        vector=0x40,
        num_vector_exponent=0,
    )
    assert ok is True
    assert "Configured MSI at 0x70" in msg

    # Inspect configured registers
    msi_read = sim.read_msi_capability(0x70)
    assert msi_read.msi_enable is True
    assert msi_read.msg_addr == 0xFEE00000
    assert msi_read.msg_upper_addr == 0x00000000
    assert msi_read.msg_data == 0xC040


def test_pci_device_msi_simulator_no_msi():
    """Verify error handling when PCI device does not have MSI capability."""
    sim = PCIDeviceMSISimulator(bus=0, device=31, function=2)
    sim.write_conf_reg(0x34, 0x40)
    sim.add_capability(cap_addr=0x40, cap_id=0x01, next_ptr=0x00)

    ok, msg = sim.configure_msi_fixed_destination(
        apic_id=0,
        trigger_mode=MSITriggerMode.kLevel,
        delivery_mode=MSIDeliveryMode.kFixed,
        vector=0x40,
    )
    assert ok is False
    assert "No MSI or MSI-X capability found" in msg


def test_figure_generation_and_dimensions(tmp_path):
    """Verify Figure 7.09 and 7.10 generation and image properties."""
    fig7_09 = tmp_path / "fig7_09.png"
    fig7_10 = tmp_path / "fig7_10.png"

    plot_msi_vs_legacy_interrupts(str(fig7_09))
    plot_pci_capability_linked_list_and_msi_register(str(fig7_10))

    assert fig7_09.exists()
    assert fig7_10.exists()

    with Image.open(fig7_09) as img:
        assert img.width > 3000
        assert img.height > 1800

    with Image.open(fig7_10) as img:
        assert img.width > 3000
        assert img.height > 1800


def test_cpp20_msi_sim_execution():
    """Verify C++20 standalone binary execution."""
    bin_path = Path("build/bin/ch07_msi_interrupt_sim")
    if not bin_path.exists():
        pytest.skip("ch07_msi_interrupt_sim binary not built")

    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "ALL 7 TESTS PASSED SUCCESSFULLY!" in res.stdout
    assert "MikanOS Section 7.5: PCI MSI Configuration Verification" in res.stdout


def test_notebook_structure_and_execution():
    """Verify Section 7.5 Jupyter Notebook existence, valid JSON structure, and execution."""
    nb_path = Path("7/7.5_Configuring_PCI_MSI_Interrupts.ipynb")
    assert nb_path.exists(), "7/7.5_Configuring_PCI_MSI_Interrupts.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 8

    # Verify Cell 0 has Google Colab setup
    first_cell = nb_data["cells"][0]
    first_cell_src = "".join(first_cell.get("source", []))
    assert "google.colab" in first_cell_src
    assert "study-mikanos" in first_cell_src

    # Verify key topics exist in markdown cells
    full_text = "".join("".join(c.get("source", [])) for c in nb_data["cells"])
    assert "MSI" in full_text
    assert "0xFEE00000" in full_text
    assert "Capabilities Pointer" in full_text
    assert "ReadCapabilityHeader" in full_text
    assert "ConfigureMSIFixedDestination" in full_text
