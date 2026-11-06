/**
 * @file interrupt_flow_sim.cpp
 * @brief C++20 Standalone Verification of End-to-End Interrupt Pipeline and Lifecycle
 *        (MikanOS Chapter 7.6 / osbook_day07a)
 *
 * This program comprehensively verifies:
 * 1. x86-64 InterruptFrame memory layout and 40-byte binary packing
 * 2. Local APIC Memory-Mapped Register offsets (ID: 0xFEE00020, EOI: 0xFEE000B0)
 * 3. PCIe MSI Address (0xFEE00000 | (apic_id << 12)) and Data packing
 * 4. IDT 16-byte Interrupt Gate descriptor resolution and dispatch
 * 5. Complete 12-stage end-to-end interrupt pipeline emulation:
 *    Mouse Event -> xHCI Event Ring -> PCIe MSI -> LAPIC IRR -> CPU Accept (IF=0) ->
 *    Hardware Stack Push -> IDT Lookup -> Handler Prologue -> ProcessEvent ->
 *    NotifyEndOfInterrupt (EOI) -> Handler Epilogue -> IRETQ Stack Restore
 * 6. Hardware/Software responsibility boundary and architectural invariants
 * 7. Multi-event batch processing and stack alignment consistency
 *
 * Compile:
 *   g++ -std=c++20 -O2 -Wall -Wextra -pedantic interrupt_flow_sim.cpp -o interrupt_flow_sim
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
#include <deque>
#include <map>
#include <algorithm>

// ============================================================================
// 1. MikanOS Hardware / Architectural Data Structures
// ============================================================================

namespace os {

  /**
   * @brief x86-64 Hardware-pushed stack frame during an interrupt
   */
  struct __attribute__((packed)) InterruptFrame {
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
  };

  static_assert(sizeof(InterruptFrame) == 40, "InterruptFrame must be exactly 40 bytes");
  static_assert(offsetof(InterruptFrame, rip) == 0, "rip offset must be 0");
  static_assert(offsetof(InterruptFrame, cs) == 8, "cs offset must be 8");
  static_assert(offsetof(InterruptFrame, rflags) == 16, "rflags offset must be 16");
  static_assert(offsetof(InterruptFrame, rsp) == 24, "rsp offset must be 24");
  static_assert(offsetof(InterruptFrame, ss) == 32, "ss offset must be 32");

  enum class DescriptorType : uint8_t {
    kUpper8Bytes    = 0,
    kLDT            = 2,
    kTSSAvailable   = 9,
    kTSSBusy        = 11,
    kCallGate       = 12,
    kInterruptGate  = 14,
    kTrapGate       = 15,
  };

  union InterruptDescriptorAttribute {
    uint16_t data;
    struct {
      uint16_t interrupt_stack_table : 3;
      uint16_t : 5;
      DescriptorType type : 4;
      uint16_t : 1;
      uint16_t descriptor_privilege_level : 2;
      uint16_t present : 1;
    } __attribute__((packed)) bits;
  };

  struct __attribute__((packed)) InterruptDescriptor {
    uint16_t offset_low;
    uint16_t segment_selector;
    InterruptDescriptorAttribute attr;
    uint16_t offset_middle;
    uint32_t offset_high;
    uint32_t reserved;

    void SetOffset(uint64_t offset) {
      offset_low = offset & 0xffffu;
      offset_middle = (offset >> 16) & 0xffffu;
      offset_high = (offset >> 32) & 0xffffffffu;
    }

    [[nodiscard]] uint64_t GetOffset() const {
      return static_cast<uint64_t>(offset_low) |
             (static_cast<uint64_t>(offset_middle) << 16) |
             (static_cast<uint64_t>(offset_high) << 32);
    }
  };

  static_assert(sizeof(InterruptDescriptor) == 16, "IDT descriptor must be 16 bytes");

  namespace InterruptVector {
    constexpr uint8_t kXHCI = 0x40; // Vector 64
  }

  // Local APIC Constants
  constexpr uintptr_t kLocalApicBase = 0xFEE00000;
  constexpr uintptr_t kLocalApicIdOffset = 0x20;
  constexpr uintptr_t kLocalApicEoiOffset = 0xB0;

  constexpr uint32_t FormatMSIAddress(uint8_t apic_id) {
    return 0xFEE00000u | (static_cast<uint32_t>(apic_id) << 12);
  }

  constexpr uint16_t FormatMSIData(uint8_t vector, uint8_t delivery_mode, bool level_assert) {
    return (static_cast<uint16_t>(delivery_mode & 0x07) << 8) |
           static_cast<uint16_t>(vector) |
           (level_assert ? 0xC000u : 0x0000u);
  }

  // xHCI Transfer Event TRB
  struct __attribute__((packed)) TransferEventTRB {
    uint64_t trb_pointer{0};
    uint32_t status{0};
    uint32_t control{0};

    int8_t displacement_x{0};
    int8_t displacement_y{0};
  };

  enum class PipelineStage {
    kMouseMovement,
    kXhciEventRing,
    kPciMsiWrite,
    kLapicReceive,
    kCpuInterruptAccept,
    kHardwareStackPush,
    kIdtLookup,
    kHandlerPrologue,
    kHandlerProcessEvent,
    kLapicNotifyEoi,
    kHandlerEpilogue,
    kCpuIretq,
  };

  struct PipelineLogEntry {
    PipelineStage stage;
    std::string domain;
    std::string component;
    std::string description;
    uint64_t rip;
    uint64_t rsp;
    uint64_t rflags;
  };

  // ============================================================================
  // 2. End-to-End Interrupt Pipeline Simulator
  // ============================================================================

  class InterruptFlowSimulator {
  public:
    InterruptFlowSimulator(uint8_t bsp_apic_id = 0, uint8_t vector = InterruptVector::kXHCI)
      : bsp_apic_id_{bsp_apic_id}, vector_{vector} {
      SetupIDT();
    }

    void SetupIDT() {
      InterruptDescriptorAttribute attr{};
      attr.bits.interrupt_stack_table = 0;
      attr.bits.type = DescriptorType::kInterruptGate;
      attr.bits.descriptor_privilege_level = 0;
      attr.bits.present = 1;

      idt_[vector_].attr = attr;
      idt_[vector_].segment_selector = 0x08; // Kernel Code Segment
      idt_[vector_].SetOffset(handler_address_);
    }

    struct SimulationResult {
      std::vector<PipelineLogEntry> logs;
      int final_mouse_x;
      int final_mouse_y;
      bool eoi_written;
      bool if_flag_restored;
      bool rsp_restored;
      bool rip_restored;
    };

    SimulationResult TriggerMouseMovement(int8_t dx, int8_t dy) {
      std::vector<PipelineLogEntry> logs;
      lapic_eoi_written_ = false;

      // 1. Mouse Hardware Event
      logs.push_back({
        PipelineStage::kMouseMovement, "Hardware", "Mouse / USB Controller",
        "Mouse motion sensor detected displacement (" + std::to_string(dx) + ", " + std::to_string(dy) + ")",
        rip_, rsp_, rflags_
      });

      // 2. xHCI Transfer Event TRB
      TransferEventTRB trb{};
      trb.displacement_x = dx;
      trb.displacement_y = dy;
      event_ring_.push_back(trb);
      logs.push_back({
        PipelineStage::kXhciEventRing, "Hardware", "xHCI Controller",
        "xHCI wrote Transfer Event TRB into Primary Event Ring and toggled Cycle Bit",
        rip_, rsp_, rflags_
      });

      // 3. PCIe MSI Memory Write Transaction
      uint32_t msi_addr = FormatMSIAddress(bsp_apic_id_);
      uint16_t msi_data = FormatMSIData(vector_, 0, true);
      logs.push_back({
        PipelineStage::kPciMsiWrite, "Hardware", "PCIe Bus / System Fabric",
        "xHCI posted MSI Memory Write to 0x" + ToHex(msi_addr) + " with data 0x" + ToHex(msi_data),
        rip_, rsp_, rflags_
      });

      // 4. Local APIC Receive & IRR Set
      lapic_irr_ |= (1ull << vector_);
      logs.push_back({
        PipelineStage::kLapicReceive, "Hardware", "Local APIC",
        "Local APIC matched APIC ID " + std::to_string(bsp_apic_id_) + ", set IRR bit " + std::to_string(vector_) + ", asserted INTR line",
        rip_, rsp_, rflags_
      });

      // 5. CPU Interrupt Accept & ISR Set & IF=0
      assert(rflags_ & 0x200); // Verify IF == 1
      lapic_irr_ &= ~(1ull << vector_);
      lapic_isr_ |= (1ull << vector_);
      uint64_t old_rflags = rflags_;
      rflags_ &= ~0x200; // Hardware clears IF (CLI mode)
      logs.push_back({
        PipelineStage::kCpuInterruptAccept, "Hardware", "CPU Execution Core",
        "CPU reached instruction boundary (at HLT), acknowledged interrupt, set ISR bit, cleared IF flag",
        rip_, rsp_, rflags_
      });

      // 6. CPU Hardware Stack Push
      uint64_t old_rsp = rsp_;
      uint64_t aligned_rsp = old_rsp & ~0xFull;
      InterruptFrame frame{
        .rip = rip_,
        .cs = cs_,
        .rflags = old_rflags,
        .rsp = old_rsp,
        .ss = 0,
      };
      stack_memory_[aligned_rsp - 8] = frame.ss;
      stack_memory_[aligned_rsp - 16] = frame.rsp;
      stack_memory_[aligned_rsp - 24] = frame.rflags;
      stack_memory_[aligned_rsp - 32] = frame.cs;
      stack_memory_[aligned_rsp - 40] = frame.rip;
      rsp_ = aligned_rsp - 40;
      logs.push_back({
        PipelineStage::kHardwareStackPush, "Hardware", "CPU Interrupt Logic",
        "CPU aligned stack and pushed InterruptFrame (SS, RSP, RFLAGS, CS, RIP). New RSP=0x" + ToHex(rsp_),
        rip_, rsp_, rflags_
      });

      // 7. IDTR & IDT Gate Lookup
      const auto& desc = idt_[vector_];
      assert(desc.attr.bits.present == 1);
      assert(desc.attr.bits.type == DescriptorType::kInterruptGate);
      rip_ = desc.GetOffset();
      cs_ = desc.segment_selector;
      logs.push_back({
        PipelineStage::kIdtLookup, "Hardware", "CPU & IDT",
        "CPU loaded Interrupt Gate from idt[" + std::to_string(vector_) + "], updated RIP to 0x" + ToHex(rip_),
        rip_, rsp_, rflags_
      });

      // 8. Handler Prologue (__attribute__((interrupt)))
      std::vector<std::string> scratch_regs = {"RAX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "R10", "R11"};
      for (size_t i = 0; i < scratch_regs.size(); ++i) {
        rsp_ -= 8;
        stack_memory_[rsp_] = 0xDEAD0000u + i;
      }
      logs.push_back({
        PipelineStage::kHandlerPrologue, "Kernel Software", "IntHandlerXHCI (Prologue)",
        "Compiler pushed scratch registers (RAX..R11) onto stack. RSP=0x" + ToHex(rsp_),
        rip_, rsp_, rflags_
      });

      // 9. ProcessEvent & Mouse Move
      int processed = 0;
      while (!event_ring_.empty()) {
        auto ev = event_ring_.front();
        event_ring_.pop_front();
        mouse_x_ = std::clamp(mouse_x_ + ev.displacement_x, 0, 800 - 1);
        mouse_y_ = std::clamp(mouse_y_ + ev.displacement_y, 0, 600 - 1);
        processed++;
      }
      logs.push_back({
        PipelineStage::kHandlerProcessEvent, "Kernel Software", "IntHandlerXHCI & MouseObserver",
        "Drained Event Ring (" + std::to_string(processed) + " events). Updated cursor to (" + std::to_string(mouse_x_) + ", " + std::to_string(mouse_y_) + ")",
        rip_ + 0x40, rsp_, rflags_
      });

      // 10. LAPIC EOI Write & ISR Clear
      lapic_eoi_written_ = true;
      lapic_isr_ &= ~(1ull << vector_);
      logs.push_back({
        PipelineStage::kLapicNotifyEoi, "Kernel Software / Hardware Handshake", "NotifyEndOfInterrupt()",
        "Wrote 0 to Local APIC EOI register (0xFEE000B0). Cleared ISR bit " + std::to_string(vector_),
        rip_ + 0x80, rsp_, rflags_
      });

      // 11. Handler Epilogue (Pop GPRs)
      for (size_t i = 0; i < scratch_regs.size(); ++i) {
        rsp_ += 8;
      }
      logs.push_back({
        PipelineStage::kHandlerEpilogue, "Kernel Software", "IntHandlerXHCI (Epilogue)",
        "Compiler popped scratch registers (R11..RAX). RSP=0x" + ToHex(rsp_),
        rip_ + 0x90, rsp_, rflags_
      });

      // 12. IRETQ & CPU Stack Restore
      uint64_t ret_rip = stack_memory_[rsp_];
      uint64_t ret_cs = stack_memory_[rsp_ + 8];
      uint64_t ret_rflags = stack_memory_[rsp_ + 16];
      uint64_t ret_rsp = stack_memory_[rsp_ + 24];
      // uint64_t ret_ss = stack_memory_[rsp_ + 32];

      rip_ = ret_rip;
      cs_ = static_cast<uint16_t>(ret_cs);
      rflags_ = ret_rflags;
      rsp_ = ret_rsp;

      logs.push_back({
        PipelineStage::kCpuIretq, "Hardware", "CPU Execution Core (IRETQ)",
        "Executed IRETQ. Restored RIP=0x" + ToHex(rip_) + ", RFLAGS (IF=1!), RSP=0x" + ToHex(rsp_) + ". Resumed KernelMain HLT",
        rip_, rsp_, rflags_
      });

      SimulationResult result;
      result.logs = std::move(logs);
      result.final_mouse_x = mouse_x_;
      result.final_mouse_y = mouse_y_;
      result.eoi_written = lapic_eoi_written_;
      result.if_flag_restored = (rflags_ & 0x200) != 0;
      result.rsp_restored = (rsp_ == initial_rsp_);
      result.rip_restored = (rip_ == initial_rip_);

      return result;
    }

    [[nodiscard]] int GetMouseX() const { return mouse_x_; }
    [[nodiscard]] int GetMouseY() const { return mouse_y_; }
    [[nodiscard]] uint64_t GetRFLAGS() const { return rflags_; }
    [[nodiscard]] uint64_t GetRSP() const { return rsp_; }
    [[nodiscard]] uint64_t GetRIP() const { return rip_; }

  private:
    static std::string ToHex(uint64_t val) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%lX", val);
      return std::string(buf);
    }

    uint8_t bsp_apic_id_{0};
    uint8_t vector_{InterruptVector::kXHCI};
    int mouse_x_{300};
    int mouse_y_{200};

    uint64_t initial_rip_{0x102030};
    uint64_t initial_rsp_{0x7FFFFFFF0};
    uint64_t rip_{0x102030};
    uint64_t rsp_{0x7FFFFFFF0};
    uint16_t cs_{0x08};
    uint64_t rflags_{0x202}; // IF=1

    uint64_t lapic_irr_{0};
    uint64_t lapic_isr_{0};
    bool lapic_eoi_written_{false};

    uint64_t handler_address_{0x101500};
    std::array<InterruptDescriptor, 256> idt_{};
    std::deque<TransferEventTRB> event_ring_{};
    std::map<uint64_t, uint64_t> stack_memory_{};
  };

} // namespace os

// ============================================================================
// 3. Test Cases for Section 7.6
// ============================================================================

void Test1_InterruptFrameLayout() {
  std::cout << "[Test 1] Verifying InterruptFrame layout and offsets..." << std::endl;
  assert(sizeof(os::InterruptFrame) == 40);
  assert(offsetof(os::InterruptFrame, rip) == 0);
  assert(offsetof(os::InterruptFrame, cs) == 8);
  assert(offsetof(os::InterruptFrame, rflags) == 16);
  assert(offsetof(os::InterruptFrame, rsp) == 24);
  assert(offsetof(os::InterruptFrame, ss) == 32);
  std::cout << "  -> PASS: InterruptFrame is exactly 40 bytes with correct System V ABI alignment.\n";
}

void Test2_LocalApicOffsetsAndMSI() {
  std::cout << "[Test 2] Verifying Local APIC MMIO and MSI Address/Data calculations..." << std::endl;
  assert(os::kLocalApicBase == 0xFEE00000);
  assert(os::kLocalApicIdOffset == 0x20);
  assert(os::kLocalApicEoiOffset == 0xB0);

  uint32_t msi_addr_bsp0 = os::FormatMSIAddress(0);
  assert(msi_addr_bsp0 == 0xFEE00000);

  uint32_t msi_addr_apic3 = os::FormatMSIAddress(3);
  assert(msi_addr_apic3 == 0xFEE03000);

  uint16_t msi_data = os::FormatMSIData(os::InterruptVector::kXHCI, 0, true);
  // (0 << 8) | 0x40 | 0xC000 = 0xC040
  assert(msi_data == 0xC040);

  std::cout << "  -> PASS: Local APIC address 0x" << std::hex << msi_addr_apic3
            << ", data 0x" << msi_data << std::dec << " correctly formatted.\n";
}

void Test3_IDTDescriptorConstruction() {
  std::cout << "[Test 3] Verifying 16-byte IDT Interrupt Gate descriptor..." << std::endl;
  os::InterruptDescriptor desc{};
  uint64_t handler_addr = 0x101500;
  desc.SetOffset(handler_addr);

  assert(desc.GetOffset() == handler_addr);
  assert(desc.offset_low == 0x1500);
  assert(desc.offset_middle == 0x0010);
  assert(desc.offset_high == 0x0000);

  os::InterruptDescriptorAttribute attr{};
  attr.bits.type = os::DescriptorType::kInterruptGate;
  attr.bits.descriptor_privilege_level = 0;
  attr.bits.present = 1;
  desc.attr = attr;

  assert(desc.attr.bits.type == os::DescriptorType::kInterruptGate);
  assert(desc.attr.bits.present == 1);
  std::cout << "  -> PASS: IDT descriptor correctly encodes 64-bit target address 0x"
            << std::hex << desc.GetOffset() << std::dec << ".\n";
}

void Test4_SingleInterruptPipelineFlow() {
  std::cout << "[Test 4] Simulating complete 12-stage interrupt pipeline for single mouse event..." << std::endl;
  os::InterruptFlowSimulator sim(0, os::InterruptVector::kXHCI);

  auto res = sim.TriggerMouseMovement(15, -10);
  assert(res.logs.size() == 12);
  assert(res.final_mouse_x == 315);
  assert(res.final_mouse_y == 190);
  assert(res.eoi_written == true);
  assert(res.if_flag_restored == true);
  assert(res.rsp_restored == true);
  assert(res.rip_restored == true);

  // Check stage order
  assert(res.logs[0].stage == os::PipelineStage::kMouseMovement);
  assert(res.logs[2].stage == os::PipelineStage::kPciMsiWrite);
  assert(res.logs[3].stage == os::PipelineStage::kLapicReceive);
  assert(res.logs[4].stage == os::PipelineStage::kCpuInterruptAccept);
  assert(res.logs[5].stage == os::PipelineStage::kHardwareStackPush);
  assert(res.logs[6].stage == os::PipelineStage::kIdtLookup);
  assert(res.logs[7].stage == os::PipelineStage::kHandlerPrologue);
  assert(res.logs[8].stage == os::PipelineStage::kHandlerProcessEvent);
  assert(res.logs[9].stage == os::PipelineStage::kLapicNotifyEoi);
  assert(res.logs[10].stage == os::PipelineStage::kHandlerEpilogue);
  assert(res.logs[11].stage == os::PipelineStage::kCpuIretq);

  std::cout << "  -> PASS: All 12 stages recorded with correct transitions and state verification.\n";
}

void Test5_StackAlignmentInvariant() {
  std::cout << "[Test 5] Verifying 16-byte stack alignment during interrupt handling..." << std::endl;
  os::InterruptFlowSimulator sim(0, os::InterruptVector::kXHCI);

  // Initial RSP is 0x7FFFFFFF0 (ends in 0, 16-byte aligned)
  assert((sim.GetRSP() % 16) == 0);

  auto res = sim.TriggerMouseMovement(5, 5);
  // Post hardware push: RSP should be aligned - 40
  // 40 % 16 == 8, so RSP % 16 == 8 (which is standard entry condition for x86-64 function calls!)
  auto hw_rsp = res.logs[5].rsp;
  assert((hw_rsp % 16) == 8);

  // Post compiler push: 9 registers * 8 bytes = 72 bytes. 8 - (72 % 16) = 8 - 8 = 0.
  // Scratch registers push aligns stack back to 16 bytes for calling C++ ProcessEvent!
  auto prologue_rsp = res.logs[7].rsp;
  assert((prologue_rsp % 16) == 0);

  // Post IRETQ: RSP is back to initial
  assert(sim.GetRSP() == 0x7FFFFFFF0);
  std::cout << "  -> PASS: 16-byte stack alignment invariants strictly preserved across stages.\n";
}

void Test6_CliStiReentrancySafety() {
  std::cout << "[Test 6] Verifying CLI/STI reentrancy prevention and IF flag restoration..." << std::endl;
  os::InterruptFlowSimulator sim(0, os::InterruptVector::kXHCI);

  auto res = sim.TriggerMouseMovement(-20, 30);

  // Stage 4: IF is still 1
  assert((res.logs[3].rflags & 0x200) != 0);

  // Stage 5: Hardware clears IF
  assert((res.logs[4].rflags & 0x200) == 0);

  // Stage 8 (Inside handler): IF must remain 0
  assert((res.logs[7].rflags & 0x200) == 0);

  // Stage 11 (End of handler): IF must remain 0
  assert((res.logs[10].rflags & 0x200) == 0);

  // Stage 12 (Post-IRETQ): IF must be restored to 1 by pop of RFLAGS
  assert((res.logs[11].rflags & 0x200) != 0);
  assert((sim.GetRFLAGS() & 0x200) != 0);

  std::cout << "  -> PASS: CPU hardware correctly cleared IF during handler and restored upon IRETQ.\n";
}

void Test7_ConsecutiveMouseEvents() {
  std::cout << "[Test 7] Verifying multiple consecutive mouse movement events..." << std::endl;
  os::InterruptFlowSimulator sim(0, os::InterruptVector::kXHCI);

  sim.TriggerMouseMovement(10, 10);
  assert(sim.GetMouseX() == 310);
  assert(sim.GetMouseY() == 210);

  sim.TriggerMouseMovement(-50, -30);
  assert(sim.GetMouseX() == 260);
  assert(sim.GetMouseY() == 180);

  sim.TriggerMouseMovement(100, 100);
  assert(sim.GetMouseX() == 360);
  assert(sim.GetMouseY() == 280);

  // Boundary clamping test: repeat movement to reach screen edges
  for (int i = 0; i < 10; ++i) {
    sim.TriggerMouseMovement(100, 100);
  }
  assert(sim.GetMouseX() == 799);
  assert(sim.GetMouseY() == 599);

  std::cout << "  -> PASS: Consecutive events and screen clamping verified successfully.\n";
}

int main() {
  std::cout << "=================================================================\n";
  std::cout << " MikanOS Section 7.6: Interrupt Flow & Pipeline Verification\n";
  std::cout << "=================================================================\n";

  Test1_InterruptFrameLayout();
  Test2_LocalApicOffsetsAndMSI();
  Test3_IDTDescriptorConstruction();
  Test4_SingleInterruptPipelineFlow();
  Test5_StackAlignmentInvariant();
  Test6_CliStiReentrancySafety();
  Test7_ConsecutiveMouseEvents();

  std::cout << "\n=================================================================\n";
  std::cout << " ALL 7 TESTS PASSED SUCCESSFULLY!\n";
  std::cout << "=================================================================\n";

  return 0;
}
