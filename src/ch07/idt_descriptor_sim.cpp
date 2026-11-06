/**
 * @file idt_descriptor_sim.cpp
 * @brief C++20 Standalone Verification of x86-64 IDT (Interrupt Descriptor Table)
 *        and 16-byte Gate Descriptors (MikanOS Chapter 7.4 / osbook_day07a)
 *
 * This program verifies:
 * 1. 16-byte (128-bit) Interrupt Gate Descriptor layout:
 *    - offset_low (bits 0..15), segment_selector (bits 16..31), attr (bits 32..47),
 *      offset_middle (bits 48..63), offset_high (bits 64..95), reserved (bits 96..127)
 *    - Strict struct size (sizeof == 16) and member byte offsets using offsetof
 * 2. InterruptDescriptorAttribute bitfield union (2 bytes):
 *    - IST (3 bits), Type (4 bits: 0xE for Interrupt Gate), DPL (2 bits), Present (1 bit)
 *    - Type-safety with DescriptorType enum class
 * 3. MakeIDTAttr constexpr helper function and SetIDTEntry bit shifting logic
 * 4. 64-bit virtual address decomposition and lossless reconstruction:
 *    - handler_rip = (offset_high << 32) | (offset_middle << 16) | offset_low
 * 5. 10-byte (80-bit) IDTR (Interrupt Descriptor Table Register) layout:
 *    - limit = sizeof(idt) - 1 = 4095 (0x0FFF)
 *    - base = linear base address (&idt[0])
 * 6. CPU hardware interrupt dispatch simulation (Vector 0x40 MSI interrupt lookup)
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic idt_descriptor_sim.cpp -o idt_descriptor_sim
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
// 1. MikanOS IDT Gate Descriptor Data Structures (kernel/interrupt.hpp)
// ============================================================================

enum class DescriptorType : uint8_t {
  kUpper8Bytes   = 0,
  kLDT           = 2,
  kTSSAvailable  = 9,
  kTSSBusy       = 11,
  kCallGate      = 12,
  kInterruptGate = 14,  // 0xE: 64-bit Interrupt Gate (Clears IF flag automatically)
  kTrapGate      = 15,  // 0xF: 64-bit Trap Gate (Preserves IF flag)
};

constexpr std::string_view DescriptorTypeToString(DescriptorType type) noexcept {
  switch (type) {
    case DescriptorType::kUpper8Bytes:   return "Upper 8 Bytes";
    case DescriptorType::kLDT:           return "LDT Descriptor";
    case DescriptorType::kTSSAvailable:  return "TSS (Available)";
    case DescriptorType::kTSSBusy:       return "TSS (Busy)";
    case DescriptorType::kCallGate:      return "Call Gate";
    case DescriptorType::kInterruptGate: return "Interrupt Gate (0xE)";
    case DescriptorType::kTrapGate:      return "Trap Gate (0xF)";
  }
  return "Unknown";
}

#pragma pack(push, 1)

union InterruptDescriptorAttribute {
  uint16_t data;
  struct {
    uint16_t interrupt_stack_table : 3;      // bits 0..2: IST index (0 = RSP, 1..7 = IST1..7 in TSS)
    uint16_t : 5;                           // bits 3..7: Reserved (must be 0)
    DescriptorType type : 4;                // bits 8..11: Gate Type (0xE = Interrupt Gate)
    uint16_t : 1;                           // bit 12: S flag (0 = system descriptor)
    uint16_t descriptor_privilege_level : 2;// bits 13..14: DPL (0 = Ring 0, 3 = Ring 3)
    uint16_t present : 1;                   // bit 15: P flag (1 = segment present)
  } __attribute__((packed)) bits;
} __attribute__((packed));

static_assert(sizeof(InterruptDescriptorAttribute) == 2,
              "InterruptDescriptorAttribute must be exactly 2 bytes (16 bits)");

struct InterruptDescriptor {
  uint16_t offset_low;        // Bytes 0..1: bits 0..15 of handler RIP
  uint16_t segment_selector;  // Bytes 2..3: Target Code Segment Selector (CS)
  InterruptDescriptorAttribute attr; // Bytes 4..5: Gate attributes (IST, Type, DPL, Present)
  uint16_t offset_middle;     // Bytes 6..7: bits 16..31 of handler RIP
  uint32_t offset_high;       // Bytes 8..11: bits 32..63 of handler RIP
  uint32_t reserved;          // Bytes 12..15: Reserved (must be zero)
} __attribute__((packed));

static_assert(sizeof(InterruptDescriptor) == 16,
              "InterruptDescriptor must be exactly 16 bytes (128 bits)");

struct IDTR {
  uint16_t limit;  // Size of IDT in bytes - 1 (256 * 16 - 1 = 4095)
  uint64_t base;   // 64-bit linear base address of IDT
} __attribute__((packed));

static_assert(sizeof(IDTR) == 10,
              "IDTR register structure must be exactly 10 bytes (80 bits)");

#pragma pack(pop)

// Member offset compile-time checks
static_assert(offsetof(InterruptDescriptor, offset_low) == 0, "offset_low offset != 0");
static_assert(offsetof(InterruptDescriptor, segment_selector) == 2, "segment_selector offset != 2");
static_assert(offsetof(InterruptDescriptor, attr) == 4, "attr offset != 4");
static_assert(offsetof(InterruptDescriptor, offset_middle) == 6, "offset_middle offset != 6");
static_assert(offsetof(InterruptDescriptor, offset_high) == 8, "offset_high offset != 8");
static_assert(offsetof(InterruptDescriptor, reserved) == 12, "reserved offset != 12");

// ============================================================================
// 2. MikanOS IDT Helper Functions (kernel/interrupt.hpp & interrupt.cpp)
// ============================================================================

constexpr InterruptDescriptorAttribute MakeIDTAttr(
    DescriptorType type,
    uint8_t descriptor_privilege_level,
    bool present = true,
    uint8_t interrupt_stack_table = 0) {
  InterruptDescriptorAttribute attr{};
  attr.bits.interrupt_stack_table = interrupt_stack_table;
  attr.bits.type = type;
  attr.bits.descriptor_privilege_level = descriptor_privilege_level;
  attr.bits.present = present;
  return attr;
}

void SetIDTEntry(InterruptDescriptor& desc,
                 InterruptDescriptorAttribute attr,
                 uint64_t offset,
                 uint16_t segment_selector) {
  desc.attr = attr;
  desc.offset_low = offset & 0xffffu;
  desc.offset_middle = (offset >> 16) & 0xffffu;
  desc.offset_high = offset >> 32;
  desc.segment_selector = segment_selector;
  desc.reserved = 0;
}

uint64_t GetHandlerOffset(const InterruptDescriptor& desc) noexcept {
  return (static_cast<uint64_t>(desc.offset_high) << 32) |
         (static_cast<uint64_t>(desc.offset_middle) << 16) |
         static_cast<uint64_t>(desc.offset_low);
}

// ============================================================================
// 3. Hardware Interrupt Dispatch Simulation
// ============================================================================

struct DispatchResult {
  bool success;
  std::string message;
  uint8_t vector;
  uint64_t handler_address;
  uint16_t segment_selector;
  uint8_t ist;
  DescriptorType type;
  uint8_t dpl;
  bool present;
};

class SimulatedCPU {
 public:
  IDTR idtr{0, 0};

  void LoadIDT(uint16_t limit, uint64_t base) noexcept {
    idtr.limit = limit;
    idtr.base = base;
  }

  DispatchResult DispatchInterrupt(uint8_t vector, const InterruptDescriptor* idt_memory) const {
    const uint32_t entry_offset = static_cast<uint32_t>(vector) * sizeof(InterruptDescriptor);
    // Limit check: (vector * 16 + 15) must be <= idtr.limit
    if (entry_offset + sizeof(InterruptDescriptor) - 1 > idtr.limit) {
      return {
        .success = false,
        .message = "#GP(Vector): Vector offset exceeds IDTR limit",
        .vector = vector,
        .handler_address = 0,
        .segment_selector = 0,
        .ist = 0,
        .type = DescriptorType::kUpper8Bytes,
        .dpl = 0,
        .present = false,
      };
    }

    const InterruptDescriptor& desc = idt_memory[vector];
    if (!desc.attr.bits.present) {
      return {
        .success = false,
        .message = "#NP(Segment Not Present): Entry P bit is 0",
        .vector = vector,
        .handler_address = 0,
        .segment_selector = desc.segment_selector,
        .ist = static_cast<uint8_t>(desc.attr.bits.interrupt_stack_table),
        .type = desc.attr.bits.type,
        .dpl = static_cast<uint8_t>(desc.attr.bits.descriptor_privilege_level),
        .present = false,
      };
    }

    return {
      .success = true,
      .message = "OK: Successfully dispatched to interrupt handler",
      .vector = vector,
      .handler_address = GetHandlerOffset(desc),
      .segment_selector = desc.segment_selector,
      .ist = static_cast<uint8_t>(desc.attr.bits.interrupt_stack_table),
      .type = desc.attr.bits.type,
      .dpl = static_cast<uint8_t>(desc.attr.bits.descriptor_privilege_level),
      .present = true,
    };
  }
};

// ============================================================================
// 4. Test Suite and Verifications
// ============================================================================

void TestStructureLayoutAndSizes() {
  std::cout << "[Test 1] Structure Sizes & Byte Offsets Verification..." << std::endl;
  assert(sizeof(InterruptDescriptor) == 16);
  assert(sizeof(InterruptDescriptorAttribute) == 2);
  assert(sizeof(IDTR) == 10);
  assert(offsetof(InterruptDescriptor, offset_low) == 0);
  assert(offsetof(InterruptDescriptor, segment_selector) == 2);
  assert(offsetof(InterruptDescriptor, attr) == 4);
  assert(offsetof(InterruptDescriptor, offset_middle) == 6);
  assert(offsetof(InterruptDescriptor, offset_high) == 8);
  assert(offsetof(InterruptDescriptor, reserved) == 12);
  std::cout << "  ✓ All sizes and offsets verified with static_assert and runtime assert." << std::endl;
}

void TestAttributeBitfields() {
  std::cout << "[Test 2] InterruptDescriptorAttribute Bitfield Serialization..." << std::endl;
  // Type = 0xE (Interrupt Gate), DPL = 0, Present = 1, IST = 0
  auto attr = MakeIDTAttr(DescriptorType::kInterruptGate, 0, true, 0);
  assert(attr.bits.type == DescriptorType::kInterruptGate);
  assert(attr.bits.descriptor_privilege_level == 0);
  assert(attr.bits.present == 1);
  assert(attr.bits.interrupt_stack_table == 0);

  // Raw word verification:
  // bit 0..2 = 0
  // bit 8..11 = 14 (0xE) -> 0x0E00
  // bit 13..14 = 0
  // bit 15 = 1 -> 0x8000
  // Expected: 0x8E00
  assert(attr.data == 0x8E00);
  std::cout << "  ✓ Standard kernel Interrupt Gate attribute raw value: 0x"
            << std::hex << std::uppercase << attr.data << std::dec << " (matches 0x8E00)" << std::endl;

  // Test with non-zero IST and DPL = 3 (User-mode Call/Trap gate)
  auto user_trap_attr = MakeIDTAttr(DescriptorType::kTrapGate, 3, true, 2);
  assert(user_trap_attr.bits.interrupt_stack_table == 2);
  assert(user_trap_attr.bits.type == DescriptorType::kTrapGate);
  assert(user_trap_attr.bits.descriptor_privilege_level == 3);
  assert(user_trap_attr.bits.present == 1);
  // Expected: bit0..2=2, bit8..11=15 (0xF), bit13..14=3, bit15=1
  // 0x0002 | (0xF << 8) | (3 << 13) | (1 << 15) = 2 | 0x0F00 | 0x6000 | 0x8000 = 0xEF02
  assert(user_trap_attr.data == 0xEF02);
  std::cout << "  ✓ User-level Trap Gate with IST attribute raw value: 0x"
            << std::hex << std::uppercase << user_trap_attr.data << std::dec << " (matches 0xEF02)" << std::endl;
}

void TestSetIDTEntryAndReconstruction() {
  std::cout << "[Test 3] 64-bit Linear Address Split and Lossless Reconstruction..." << std::endl;
  InterruptDescriptor desc{};
  const uint64_t test_address = 0x123456789ABCDEF0ULL;
  const uint16_t cs = 0x0008;  // Kernel CS selector
  auto attr = MakeIDTAttr(DescriptorType::kInterruptGate, 0);

  SetIDTEntry(desc, attr, test_address, cs);

  assert(desc.offset_low == 0xDEF0);
  assert(desc.offset_middle == 0x9ABC);
  assert(desc.offset_high == 0x12345678);
  assert(desc.segment_selector == 0x0008);
  assert(desc.reserved == 0);

  const uint64_t reconstructed = GetHandlerOffset(desc);
  assert(reconstructed == test_address);
  std::cout << "  ✓ Original Address: 0x" << std::hex << test_address << "\n"
            << "  ✓ offset_low:    0x" << desc.offset_low << "\n"
            << "  ✓ offset_middle: 0x" << desc.offset_middle << "\n"
            << "  ✓ offset_high:   0x" << desc.offset_high << "\n"
            << "  ✓ Reconstructed: 0x" << reconstructed << std::dec << std::endl;

  // Test Canonical Higher-Half address (Kernel Space 0xFFFF800000102030)
  const uint64_t higher_half_addr = 0xFFFF800000102030ULL;
  SetIDTEntry(desc, attr, higher_half_addr, cs);
  assert(GetHandlerOffset(desc) == higher_half_addr);
  assert(desc.offset_high == 0xFFFF8000);
  assert(desc.offset_middle == 0x0010);
  assert(desc.offset_low == 0x2030);
  std::cout << "  ✓ Canonical Higher-Half address successfully tested: 0x"
            << std::hex << higher_half_addr << std::dec << std::endl;
}

void TestSimulatedCPUDispatch() {
  std::cout << "[Test 4] Simulated CPU IDT Loading and Vector 0x40 Dispatch..." << std::endl;
  alignas(16) std::array<InterruptDescriptor, 256> idt{};
  SimulatedCPU cpu;

  // Load IDT with full 256 entries
  const uint16_t limit = sizeof(idt) - 1;  // 256 * 16 - 1 = 4095 (0x0FFF)
  assert(limit == 4095);
  cpu.LoadIDT(limit, reinterpret_cast<uint64_t>(idt.data()));

  // Unconfigured vector dispatch test (Not Present)
  auto res_uninit = cpu.DispatchInterrupt(0x40, idt.data());
  assert(!res_uninit.success);
  assert(!res_uninit.present);
  std::cout << "  ✓ Uninitialized vector 0x40 correctly raises exception: " << res_uninit.message << std::endl;

  // Configure Vector 0x40 (xHCI Interrupt Handler in MikanOS)
  const uint64_t handler_xhci = 0x0000000000105420ULL;
  const uint16_t kernel_cs = 0x0008;
  SetIDTEntry(idt[0x40], MakeIDTAttr(DescriptorType::kInterruptGate, 0), handler_xhci, kernel_cs);

  auto res_xhci = cpu.DispatchInterrupt(0x40, idt.data());
  assert(res_xhci.success);
  assert(res_xhci.vector == 0x40);
  assert(res_xhci.handler_address == handler_xhci);
  assert(res_xhci.segment_selector == kernel_cs);
  assert(res_xhci.type == DescriptorType::kInterruptGate);
  assert(res_xhci.dpl == 0);
  assert(res_xhci.present);
  std::cout << "  ✓ Configured Vector 0x40 dispatched successfully to 0x"
            << std::hex << res_xhci.handler_address << " with CS=0x"
            << res_xhci.segment_selector << std::dec << std::endl;

  // Test limit protection: Simulate truncated IDT limit (only 32 entries loaded)
  cpu.LoadIDT(32 * sizeof(InterruptDescriptor) - 1, reinterpret_cast<uint64_t>(idt.data()));
  auto res_out_of_limit = cpu.DispatchInterrupt(0x40, idt.data());
  assert(!res_out_of_limit.success);
  std::cout << "  ✓ Vector 0x40 with limit=511 correctly blocked by limit check: "
            << res_out_of_limit.message << std::endl;
}

int main() {
  std::cout << "====================================================================\n"
            << " MikanOS Chapter 7.4: IDT & 16-byte Gate Descriptors C++20 Verifier\n"
            << "====================================================================\n\n";

  TestStructureLayoutAndSizes();
  std::cout << "\n";
  TestAttributeBitfields();
  std::cout << "\n";
  TestSetIDTEntryAndReconstruction();
  std::cout << "\n";
  TestSimulatedCPUDispatch();

  std::cout << "\n====================================================================\n"
            << " All Chapter 7.4 C++20 Verification Tests PASSED Successfully!\n"
            << "====================================================================\n";
  return 0;
}
