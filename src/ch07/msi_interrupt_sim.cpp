/**
 * @file msi_interrupt_sim.cpp
 * @brief C++20 Standalone Verification of PCI MSI (Message Signaled Interrupts)
 *        Configuration and Local APIC Integration (MikanOS Chapter 7.5 / osbook_day07a)
 *
 * This program verifies:
 * 1. PCI CapabilityHeader union and MSICapability bitfield structure
 * 2. PCI Configuration Space (0x00..0xFF) and linked list traversal from 0x34
 * 3. Local APIC Message Address calculation: 0xFEE00000 | (apic_id << 12)
 * 4. Local APIC Message Data calculation: (delivery_mode << 8) | vector | trigger_mode
 * 5. Full MikanOS MSI configuration flow:
 *    - ReadCapabilityHeader
 *    - ConfigureMSI
 *    - ConfigureMSIFixedDestination
 * 6. 32-bit vs 64-bit address capability register offset handling (+8 vs +12)
 * 7. Error handling for devices without MSI (kNoPCIMSI)
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic msi_interrupt_sim.cpp -o msi_interrupt_sim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <optional>

// ============================================================================
// 1. MikanOS PCI / MSI Data Structures (kernel/pci.hpp)
// ============================================================================

namespace pci {

  struct ClassCode {
    uint8_t base, sub, interface;

    constexpr bool Match(uint8_t b, uint8_t s, uint8_t i) const noexcept {
      return (base == b) && (sub == s) && (interface == i);
    }
  };

  struct Device {
    uint8_t bus{0}, device{0}, function{0}, header_type{0};
    ClassCode class_code{0, 0, 0};
  };

  /** @brief PCI Capability Header Union (4 bytes) */
  union CapabilityHeader {
    uint32_t data;
    struct {
      uint32_t cap_id : 8;    // bits 0..7: Capability ID (0x05=MSI, 0x11=MSI-X)
      uint32_t next_ptr : 8;  // bits 8..15: Offset to next capability (0=End)
      uint32_t cap : 16;      // bits 16..31: Capability specific control/status
    } __attribute__((packed)) bits;
  } __attribute__((packed));

  static_assert(sizeof(CapabilityHeader) == 4, "CapabilityHeader must be exactly 4 bytes");

  constexpr uint8_t kCapabilityMSI = 0x05;
  constexpr uint8_t kCapabilityMSIX = 0x11;

  /** @brief MSI Capability Structure matching MikanOS MSICapability */
  struct MSICapability {
    union {
      uint32_t data;
      struct {
        uint32_t cap_id : 8;                  // bits 0..7
        uint32_t next_ptr : 8;                // bits 8..15
        uint32_t msi_enable : 1;              // bit 16
        uint32_t multi_msg_capable : 3;       // bits 17..19 (2^N messages capable)
        uint32_t multi_msg_enable : 3;        // bits 20..22 (2^M messages enable)
        uint32_t addr_64_capable : 1;         // bit 23
        uint32_t per_vector_mask_capable : 1; // bit 24
        uint32_t : 7;                         // bits 25..31 (reserved)
      } __attribute__((packed)) bits;
    } __attribute__((packed)) header;

    uint32_t msg_addr{0};
    uint32_t msg_upper_addr{0};
    uint32_t msg_data{0};
    uint32_t mask_bits{0};
    uint32_t pending_bits{0};
  } __attribute__((packed));

  enum class MSITriggerMode {
    kEdge = 0,
    kLevel = 1
  };

  enum class MSIDeliveryMode {
    kFixed          = 0b000,
    kLowestPriority = 0b001,
    kSMI            = 0b010,
    kNMI            = 0b100,
    kINIT           = 0b101,
    kExtINT         = 0b111,
  };

  enum class ErrorCode {
    kSuccess,
    kNoPCIMSI,
    kNotImplemented,
    kIndexOutOfRange,
  };

  struct Error {
    ErrorCode code;
    constexpr bool IsSuccess() const noexcept { return code == ErrorCode::kSuccess; }
    constexpr const char* Name() const noexcept {
      switch (code) {
        case ErrorCode::kSuccess: return "kSuccess";
        case ErrorCode::kNoPCIMSI: return "kNoPCIMSI";
        case ErrorCode::kNotImplemented: return "kNotImplemented";
        case ErrorCode::kIndexOutOfRange: return "kIndexOutOfRange";
        default: return "Unknown";
      }
    }
  };

} // namespace pci

// ============================================================================
// 2. Mock PCI Configuration Space & Simulator
// ============================================================================

class MockPCIDevice {
public:
  pci::Device dev_info;
  std::array<uint8_t, 256> config_space{};

  MockPCIDevice(uint8_t b, uint8_t d, uint8_t f, uint8_t header_type = 0) {
    dev_info.bus = b;
    dev_info.device = d;
    dev_info.function = f;
    dev_info.header_type = header_type;
    config_space.fill(0);
  }

  uint32_t ReadConfReg(uint8_t reg_addr) const {
    uint8_t aligned = reg_addr & 0xFC;
    uint32_t val = 0;
    std::memcpy(&val, &config_space[aligned], sizeof(uint32_t));
    return val;
  }

  void WriteConfReg(uint8_t reg_addr, uint32_t val) {
    uint8_t aligned = reg_addr & 0xFC;
    std::memcpy(&config_space[aligned], &val, sizeof(uint32_t));
  }

  void SetCapPtr(uint8_t offset) {
    config_space[0x34] = offset;
  }

  void AddCapabilityHeader(uint8_t offset, uint8_t cap_id, uint8_t next_ptr, uint16_t cap_ctrl = 0) {
    pci::CapabilityHeader header{};
    header.bits.cap_id = cap_id;
    header.bits.next_ptr = next_ptr;
    header.bits.cap = cap_ctrl;
    WriteConfReg(offset, header.data);
  }
};

// Global simulated device table for pci::ReadConfReg / WriteConfReg
static MockPCIDevice* g_sim_dev = nullptr;

namespace pci {

  uint32_t ReadConfReg(const Device&, uint8_t reg_addr) {
    assert(g_sim_dev != nullptr);
    return g_sim_dev->ReadConfReg(reg_addr);
  }

  void WriteConfReg(const Device&, uint8_t reg_addr, uint32_t value) {
    assert(g_sim_dev != nullptr);
    g_sim_dev->WriteConfReg(reg_addr, value);
  }

  CapabilityHeader ReadCapabilityHeader(const Device& dev, uint8_t addr) {
    CapabilityHeader header;
    header.data = ReadConfReg(dev, addr);
    return header;
  }

  MSICapability ReadMSICapability(const Device& dev, uint8_t cap_addr) {
    MSICapability msi_cap{};
    msi_cap.header.data = ReadConfReg(dev, cap_addr);
    msi_cap.msg_addr = ReadConfReg(dev, cap_addr + 4);

    uint8_t msg_data_addr = cap_addr + 8;
    if (msi_cap.header.bits.addr_64_capable) {
      msi_cap.msg_upper_addr = ReadConfReg(dev, cap_addr + 8);
      msg_data_addr = cap_addr + 12;
    }

    msi_cap.msg_data = ReadConfReg(dev, msg_data_addr);

    if (msi_cap.header.bits.per_vector_mask_capable) {
      msi_cap.mask_bits = ReadConfReg(dev, msg_data_addr + 4);
      msi_cap.pending_bits = ReadConfReg(dev, msg_data_addr + 8);
    }

    return msi_cap;
  }

  void WriteMSICapability(const Device& dev, uint8_t cap_addr,
                          const MSICapability& msi_cap) {
    WriteConfReg(dev, cap_addr, msi_cap.header.data);
    WriteConfReg(dev, cap_addr + 4, msi_cap.msg_addr);

    uint8_t msg_data_addr = cap_addr + 8;
    if (msi_cap.header.bits.addr_64_capable) {
      WriteConfReg(dev, cap_addr + 8, msi_cap.msg_upper_addr);
      msg_data_addr = cap_addr + 12;
    }

    WriteConfReg(dev, msg_data_addr, msi_cap.msg_data);

    if (msi_cap.header.bits.per_vector_mask_capable) {
      WriteConfReg(dev, msg_data_addr + 4, msi_cap.mask_bits);
      WriteConfReg(dev, msg_data_addr + 8, msi_cap.pending_bits);
    }
  }

  Error ConfigureMSIRegister(const Device& dev, uint8_t cap_addr,
                             uint32_t msg_addr, uint32_t msg_data,
                             unsigned int num_vector_exponent) {
    auto msi_cap = ReadMSICapability(dev, cap_addr);

    if (msi_cap.header.bits.multi_msg_capable <= num_vector_exponent) {
      msi_cap.header.bits.multi_msg_enable =
        msi_cap.header.bits.multi_msg_capable;
    } else {
      msi_cap.header.bits.multi_msg_enable = num_vector_exponent;
    }

    msi_cap.header.bits.msi_enable = 1;
    msi_cap.msg_addr = msg_addr;
    msi_cap.msg_data = msg_data;

    WriteMSICapability(dev, cap_addr, msi_cap);
    return {ErrorCode::kSuccess};
  }

  Error ConfigureMSIXRegister(const Device&, uint8_t, uint32_t, uint32_t, unsigned int) {
    return {ErrorCode::kNotImplemented};
  }

  Error ConfigureMSI(const Device& dev, uint32_t msg_addr, uint32_t msg_data,
                     unsigned int num_vector_exponent) {
    uint8_t cap_addr = ReadConfReg(dev, 0x34) & 0xffu;
    uint8_t msi_cap_addr = 0, msix_cap_addr = 0;
    while (cap_addr != 0) {
      auto header = ReadCapabilityHeader(dev, cap_addr);
      if (header.bits.cap_id == kCapabilityMSI) {
        msi_cap_addr = cap_addr;
      } else if (header.bits.cap_id == kCapabilityMSIX) {
        msix_cap_addr = cap_addr;
      }
      cap_addr = header.bits.next_ptr;
    }

    if (msi_cap_addr) {
      return ConfigureMSIRegister(dev, msi_cap_addr, msg_addr, msg_data, num_vector_exponent);
    } else if (msix_cap_addr) {
      return ConfigureMSIXRegister(dev, msix_cap_addr, msg_addr, msg_data, num_vector_exponent);
    }
    return {ErrorCode::kNoPCIMSI};
  }

  Error ConfigureMSIFixedDestination(
      const Device& dev, uint8_t apic_id,
      MSITriggerMode trigger_mode, MSIDeliveryMode delivery_mode,
      uint8_t vector, unsigned int num_vector_exponent) {
    uint32_t msg_addr = 0xfee00000u | (apic_id << 12);
    uint32_t msg_data = (static_cast<uint32_t>(delivery_mode) << 8) | vector;
    if (trigger_mode == MSITriggerMode::kLevel) {
      msg_data |= 0xc000;
    }
    return ConfigureMSI(dev, msg_addr, msg_data, num_vector_exponent);
  }

} // namespace pci

// ============================================================================
// 3. Verification Test Suite
// ============================================================================

void TestCapabilityHeaderLayout() {
  std::cout << "[Test 1] PCI CapabilityHeader Bitfield Layout & Union Integrity..." << std::endl;

  pci::CapabilityHeader h{};
  h.bits.cap_id = 0x05;    // MSI
  h.bits.next_ptr = 0x60;  // next at 0x60
  h.bits.cap = 0x0081;     // 64-bit capable, MSI enable

  assert(sizeof(pci::CapabilityHeader) == 4);
  assert(h.bits.cap_id == 0x05);
  assert(h.bits.next_ptr == 0x60);
  assert(h.bits.cap == 0x0081);

  // Check raw integer serialization
  uint32_t expected = 0x05 | (0x60 << 8) | (0x0081 << 16);
  assert(h.data == expected);

  std::cout << "  -> CapabilityHeader size = " << sizeof(pci::CapabilityHeader)
            << " bytes, raw data = 0x" << std::hex << std::setfill('0')
            << std::setw(8) << h.data << std::dec << " [OK]" << std::endl;
}

void TestMSICapabilityBitfields() {
  std::cout << "[Test 2] MSICapability Bitfield Structure & Control Bits..." << std::endl;

  pci::MSICapability msi{};
  msi.header.bits.cap_id = pci::kCapabilityMSI; // 0x05
  msi.header.bits.next_ptr = 0x00;              // End of list
  msi.header.bits.msi_enable = 1;
  msi.header.bits.multi_msg_capable = 3;        // 2^3 = 8 messages capable
  msi.header.bits.multi_msg_enable = 2;         // 2^2 = 4 messages enabled
  msi.header.bits.addr_64_capable = 1;          // 64-bit address support
  msi.header.bits.per_vector_mask_capable = 0;

  assert(msi.header.bits.cap_id == 0x05);
  assert(msi.header.bits.next_ptr == 0x00);
  assert(msi.header.bits.msi_enable == 1);
  assert(msi.header.bits.multi_msg_capable == 3);
  assert(msi.header.bits.multi_msg_enable == 2);
  assert(msi.header.bits.addr_64_capable == 1);
  assert(msi.header.bits.per_vector_mask_capable == 0);

  std::cout << "  -> MSI Enable=" << msi.header.bits.msi_enable
            << ", MultiMsgCapable=2^" << msi.header.bits.multi_msg_capable
            << ", 64bCapable=" << msi.header.bits.addr_64_capable << " [OK]" << std::endl;
}

void TestLocalAPICMessageAddress() {
  std::cout << "[Test 3] Local APIC Message Address Calculation..." << std::endl;

  // Address: 0xFEE00000 | (apic_id << 12)
  for (uint8_t apic_id : {0, 1, 2, 7, 15, 255}) {
    uint32_t addr = 0xfee00000u | (apic_id << 12);
    // Base address verification (bits 31..20 must be 0xFEE)
    assert((addr >> 20) == 0xFEE);
    // Destination ID verification (bits 19..12 must match apic_id)
    assert(((addr >> 12) & 0xFF) == apic_id);
    // Lower bits 11..0 should be zero (RH=0, DM=0)
    assert((addr & 0x0FFF) == 0);
  }

  uint32_t bsp_addr = 0xfee00000u | (0 << 12);
  assert(bsp_addr == 0xFEE00000u);

  uint32_t ap1_addr = 0xfee00000u | (1 << 12);
  assert(ap1_addr == 0xFEE01000u);

  std::cout << "  -> BSP Addr: 0x" << std::hex << bsp_addr
            << ", Core 1 Addr: 0x" << ap1_addr << std::dec << " [OK]" << std::endl;
}

void TestLocalAPICMessageData() {
  std::cout << "[Test 4] Local APIC Message Data Formulation (Vector & Trigger)..." << std::endl;

  // Edge Trigger, Fixed Delivery, Vector 0x40
  uint8_t vector = 0x40; // InterruptVector::kXHCI
  auto delivery = pci::MSIDeliveryMode::kFixed;
  uint32_t data_edge = (static_cast<uint32_t>(delivery) << 8) | vector;
  assert((data_edge & 0xFF) == 0x40);
  assert(((data_edge >> 8) & 0x7) == 0);
  assert(data_edge == 0x0040);

  // Level Trigger, Fixed Delivery, Vector 0x40 (MikanOS standard)
  uint32_t data_level = data_edge | 0xc000;
  assert(data_level == 0xC040);
  assert((data_level & (1 << 15)) != 0); // Trigger Mode = Level (1)
  assert((data_level & (1 << 14)) != 0); // Level = Assert (1)

  std::cout << "  -> Edge Data: 0x" << std::hex << data_edge
            << ", Level Data: 0x" << data_level << std::dec << " [OK]" << std::endl;
}

void TestCapabilityLinkedListScan() {
  std::cout << "[Test 5] PCI Capability Linked List Traversal..." << std::endl;

  MockPCIDevice dev(0, 20, 0); // xHCI bus 0, dev 20, func 0
  g_sim_dev = &dev;

  // Set CapPtr at offset 0x34 -> 0x50
  dev.SetCapPtr(0x50);

  // Offset 0x50: Capability 0x01 (PCI Power Management), Next -> 0x70
  dev.AddCapabilityHeader(0x50, 0x01, 0x70);

  // Offset 0x70: Capability 0x05 (MSI), Next -> 0x90
  // Control: 64-bit capable (bit 7 of message control = 1 -> cap = 0x0080)
  dev.AddCapabilityHeader(0x70, pci::kCapabilityMSI, 0x90, 0x0080);

  // Offset 0x90: Capability 0x10 (PCI Express), Next -> 0x00 (End)
  dev.AddCapabilityHeader(0x90, 0x10, 0x00);

  // Run ConfigureMSIFixedDestination
  auto err = pci::ConfigureMSIFixedDestination(
      dev.dev_info, 0 /* bsp apic_id */,
      pci::MSITriggerMode::kLevel, pci::MSIDeliveryMode::kFixed,
      0x40 /* InterruptVector::kXHCI */, 0 /* 2^0 = 1 vector */);

  assert(err.IsSuccess());

  // Verify that MSI Capability at 0x70 was correctly configured
  auto msi_cap = pci::ReadMSICapability(dev.dev_info, 0x70);
  assert(msi_cap.header.bits.cap_id == pci::kCapabilityMSI);
  assert(msi_cap.header.bits.msi_enable == 1);
  assert(msi_cap.msg_addr == 0xFEE00000u);
  assert(msi_cap.msg_upper_addr == 0x00000000u);
  assert(msi_cap.msg_data == 0xC040u);

  std::cout << "  -> Found MSI at 0x70: Enabled=" << msi_cap.header.bits.msi_enable
            << ", MsgAddr=0x" << std::hex << msi_cap.msg_addr
            << ", MsgData=0x" << msi_cap.msg_data << std::dec << " [OK]" << std::endl;
}

void TestDeviceWithoutMSIErrorHandling() {
  std::cout << "[Test 6] Error Handling for Device without MSI (kNoPCIMSI)..." << std::endl;

  MockPCIDevice dev(0, 31, 2);
  g_sim_dev = &dev;

  // CapPtr points to 0x40, which only has Power Management (0x01)
  dev.SetCapPtr(0x40);
  dev.AddCapabilityHeader(0x40, 0x01, 0x00); // Only PM capability

  auto err = pci::ConfigureMSIFixedDestination(
      dev.dev_info, 0,
      pci::MSITriggerMode::kLevel, pci::MSIDeliveryMode::kFixed,
      0x40, 0);

  assert(!err.IsSuccess());
  assert(err.code == pci::ErrorCode::kNoPCIMSI);

  std::cout << "  -> Correctly returned error: " << err.Name() << " [OK]" << std::endl;
}

void Test32BitVs64BitMSIOffset() {
  std::cout << "[Test 7] 32-bit vs 64-bit MSI Capability Register Offset..." << std::endl;

  // Case A: 32-bit Address capable (addr_64_capable = 0)
  // Header(+0), MsgAddr(+4), MsgData(+8)
  {
    MockPCIDevice dev32(0, 1, 0);
    g_sim_dev = &dev32;
    dev32.SetCapPtr(0x80);
    // cap = 0x0000 (addr_64_capable = 0)
    dev32.AddCapabilityHeader(0x80, pci::kCapabilityMSI, 0x00, 0x0000);

    auto err = pci::ConfigureMSIFixedDestination(
        dev32.dev_info, 0,
        pci::MSITriggerMode::kEdge, pci::MSIDeliveryMode::kFixed,
        0x35, 0);
    assert(err.IsSuccess());

    // MsgData should be at offset 0x80 + 8 = 0x88
    uint32_t data = dev32.ReadConfReg(0x88);
    assert((data & 0xFFFF) == 0x0035);
  }

  // Case B: 64-bit Address capable (addr_64_capable = 1)
  // Header(+0), MsgAddr(+4), MsgUpperAddr(+8), MsgData(+12)
  {
    MockPCIDevice dev64(0, 2, 0);
    g_sim_dev = &dev64;
    dev64.SetCapPtr(0x80);
    // cap = 0x0080 (addr_64_capable = 1)
    dev64.AddCapabilityHeader(0x80, pci::kCapabilityMSI, 0x00, 0x0080);

    auto err = pci::ConfigureMSIFixedDestination(
        dev64.dev_info, 0,
        pci::MSITriggerMode::kEdge, pci::MSIDeliveryMode::kFixed,
        0x35, 0);
    assert(err.IsSuccess());

    // MsgUpperAddr should be at 0x80 + 8 = 0x88 (0x00000000)
    uint32_t upper = dev64.ReadConfReg(0x88);
    assert(upper == 0);

    // MsgData should be at offset 0x80 + 12 = 0x8C
    uint32_t data = dev64.ReadConfReg(0x8C);
    assert((data & 0xFFFF) == 0x0035);
  }

  std::cout << "  -> 32-bit (MsgData at +8) and 64-bit (MsgData at +12) offsets verified [OK]" << std::endl;
}

int main() {
  std::cout << "============================================================" << std::endl;
  std::cout << "  MikanOS Section 7.5: PCI MSI Configuration Verification" << std::endl;
  std::cout << "============================================================" << std::endl;

  TestCapabilityHeaderLayout();
  TestMSICapabilityBitfields();
  TestLocalAPICMessageAddress();
  TestLocalAPICMessageData();
  TestCapabilityLinkedListScan();
  TestDeviceWithoutMSIErrorHandling();
  Test32BitVs64BitMSIOffset();

  std::cout << "============================================================" << std::endl;
  std::cout << "  ALL 7 TESTS PASSED SUCCESSFULLY!" << std::endl;
  std::cout << "============================================================" << std::endl;

  return 0;
}
