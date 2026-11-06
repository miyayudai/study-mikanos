/**
 * @file gdt_segmentation_sim.cpp
 * @brief Chapter 8.5 Global Descriptor Table (GDT) & x86-64 Segmentation Simulator
 *
 * MikanOS Chapter 8.5: Setting Up Segmentation (osbook_day08b/c)
 * Demonstrates:
 *  1. Role of GDT in x86-64 Long Mode (flat memory model, L-bit 64-bit code indicator, DPL privilege separation).
 *  2. Bit-level layout of 64-bit SegmentDescriptor (Base, Limit, Type, S, DPL, P, L, D/B, G).
 *  3. GDT configuration with Null Descriptor, Kernel Code Segment (Ring 0), and User Code Segment (Ring 3).
 *  4. 16-bit Segment Selector structure (Index, Table Indicator, Requested Privilege Level).
 *  5. 10-byte GDTR register pseudo-descriptor (16-bit Limit + 64-bit Base Address) and LGDT instruction semantics.
 *  6. Safe CS register reload via iretq stack frame simulation.
 *  7. Resetting data segment registers (DS, ES, SS, FS, GS) to NULL selector (0).
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <array>
#include <cassert>
#include <cstring>
#include <sstream>

namespace mikanos {

/**
 * @brief x86 Descriptor Types (shared between segments and gates)
 * Using uint8_t underlying type prevents signed 4-bit sign-extension bugs in bitfields.
 */
enum class DescriptorType : uint8_t {
    // System segment & gate descriptor types
    kUpper8Bytes   = 0,
    kLDT           = 2,
    kTSSAvailable  = 9,
    kTSSBusy       = 11,
    kCallGate      = 12,
    kInterruptGate = 14,
    kTrapGate      = 15,

    // Code & data segment types
    kExecuteRead   = 10,
    kReadWrite     = 2,
};

/**
 * @brief 64-bit Segment Descriptor matching Intel SDM & MikanOS segment.hpp
 */
union SegmentDescriptor {
    uint64_t data;
    struct {
        uint64_t limit_low : 16;
        uint64_t base_low : 16;
        uint64_t base_middle : 8;
        DescriptorType type : 4;
        uint64_t system_segment : 1;           // 1: code/data, 0: system (TSS, LDT, Gate)
        uint64_t descriptor_privilege_level : 2; // DPL: 0 = Kernel, 3 = User
        uint64_t present : 1;                   // 1: valid in memory
        uint64_t limit_high : 4;
        uint64_t available : 1;                 // AVL (software use)
        uint64_t long_mode : 1;                 // L-bit: 1 = 64-bit code segment, 0 = compatibility
        uint64_t default_operation_size : 1;    // D/B: Must be 0 when long_mode == 1
        uint64_t granularity : 1;               // G: 1 = 4KiB page units, 0 = 1 byte units
        uint64_t base_high : 8;
    } __attribute__((packed)) bits;
} __attribute__((packed));

static_assert(sizeof(SegmentDescriptor) == 8, "SegmentDescriptor must be exactly 8 bytes (64 bits)");

/**
 * @brief 16-bit Segment Selector representation
 */
union SegmentSelector {
    uint16_t data;
    struct {
        uint16_t rpl : 2;  // Requested Privilege Level (0: Kernel, 3: User)
        uint16_t ti : 1;   // Table Indicator (0: GDT, 1: LDT)
        uint16_t index : 13; // Index in GDT/LDT (0..8191)
    } __attribute__((packed)) bits;
} __attribute__((packed));

static_assert(sizeof(SegmentSelector) == 2, "SegmentSelector must be exactly 2 bytes (16 bits)");

/**
 * @brief 10-byte GDTR Pseudo-Descriptor for LGDT
 */
struct __attribute__((packed)) GdtrPointer {
    uint16_t limit;
    uint64_t base;
};

static_assert(sizeof(GdtrPointer) == 10, "GdtrPointer must be exactly 10 bytes");

/**
 * @brief Configure a code segment descriptor for x86-64 Long Mode
 */
void SetCodeSegment(SegmentDescriptor& desc,
                    DescriptorType type,
                    unsigned int descriptor_privilege_level,
                    uint32_t base,
                    uint32_t limit) {
    desc.data = 0;

    desc.bits.base_low = base & 0xffffu;
    desc.bits.base_middle = (base >> 16) & 0xffu;
    desc.bits.base_high = (base >> 24) & 0xffu;

    desc.bits.limit_low = limit & 0xffffu;
    desc.bits.limit_high = (limit >> 16) & 0xfu;

    desc.bits.type = type;
    desc.bits.system_segment = 1; // 1: code & data segment
    desc.bits.descriptor_privilege_level = descriptor_privilege_level;
    desc.bits.present = 1;
    desc.bits.available = 0;
    desc.bits.long_mode = 1;                // 64-bit code segment
    desc.bits.default_operation_size = 0;   // Must be 0 when long_mode == 1
    desc.bits.granularity = 1;              // 4KiB page granularity
}

/**
 * @brief Virtual CPU State simulating segment registers and control flow
 */
struct VirtualCpuState {
    uint16_t cs{0};
    uint16_t ds{0};
    uint16_t es{0};
    uint16_t ss{0};
    uint16_t fs{0};
    uint16_t gs{0};
    uint64_t rip{0};
    uint64_t rsp{0};
    uint64_t rbp{0};
    uint64_t rflags{0x202}; // IF (Interrupt Enable) flag set by default
    GdtrPointer gdtr{0, 0};
    bool in_long_mode{true};
    uint8_t current_privilege_level{0}; // CPL: 0 or 3
};

/**
 * @brief Simulator for Global Descriptor Table & Segmentation
 */
class GdtSegmentationSimulator {
public:
    static constexpr size_t kGdtEntryCount = 3;
    std::array<SegmentDescriptor, kGdtEntryCount> gdt;
    VirtualCpuState cpu;

    GdtSegmentationSimulator() {
        // Initialize GDT table and CPU
        gdt.fill(SegmentDescriptor{0});
        cpu.rsp = 0x7FFF0000;
        cpu.rbp = 0x7FFF0000;
        cpu.rip = 0x100000;
    }

    /**
     * @brief Build MikanOS standard Chapter 8 GDT:
     *  Index 0: Null Descriptor
     *  Index 1: Kernel Code Segment (DPL = 0)
     *  Index 2: Application Code Segment (DPL = 3)
     */
    void setup_segments() {
        gdt[0].data = 0; // Null descriptor
        SetCodeSegment(gdt[1], DescriptorType::kExecuteRead, 0, 0, 0xfffff);
        SetCodeSegment(gdt[2], DescriptorType::kExecuteRead, 3, 0, 0xfffff);

        // Load into GDTR
        cpu.gdtr.limit = static_cast<uint16_t>(sizeof(gdt) - 1);
        cpu.gdtr.base = reinterpret_cast<uint64_t>(&gdt[0]);
    }

    /**
     * @brief Helper to construct a segment selector
     */
    static constexpr uint16_t make_selector(uint16_t index, uint16_t rpl, uint16_t ti = 0) {
        return static_cast<uint16_t>((index << 3) | ((ti & 1) << 2) | (rpl & 3));
    }

    /**
     * @brief Simulate SetDSAll(uint16_t value)
     */
    void set_ds_all(uint16_t value) {
        cpu.ds = value;
        cpu.es = value;
        cpu.ss = value;
        cpu.fs = value;
        cpu.gs = value;
    }

    /**
     * @brief Simulate CS reload via iretq stack frame mechanism:
     * In asmfunc.asm SetCS:
     *   xor eax, eax; mov ax, ss
     *   push rax    ; SS
     *   push rbp    ; RSP
     *   pushfq      ; RFLAGS
     *   push rdi    ; new CS
     *   push .next  ; new RIP
     *   iretq
     */
    bool simulate_set_cs_via_iretq(uint16_t target_cs, uint64_t next_rip) {
        // Step 1: Push simulated stack frame
        struct IretqStackFrame {
            uint64_t rip;
            uint64_t cs;
            uint64_t rflags;
            uint64_t rsp;
            uint64_t ss;
        } frame;

        frame.ss = cpu.ss;
        frame.rsp = cpu.rsp;
        frame.rflags = cpu.rflags;
        frame.cs = target_cs;
        frame.rip = next_rip;

        // Step 2: Validate selector against GDT
        SegmentSelector sel{.data = target_cs};
        if (sel.bits.ti != 0) {
            std::cerr << "Error: LDT selectors not supported\n";
            return false;
        }
        if (sel.bits.index >= kGdtEntryCount) {
            std::cerr << "Error: GDT index out of bounds (" << sel.bits.index << " >= " << kGdtEntryCount << ")\n";
            return false;
        }
        const SegmentDescriptor& desc = gdt[sel.bits.index];
        if (!desc.bits.present) {
            std::cerr << "Error: Segment descriptor not present (#NP fault)\n";
            return false;
        }
        if (desc.bits.system_segment != 1) {
            std::cerr << "Error: Cannot load system segment into CS (#GP fault)\n";
            return false;
        }
        if (!desc.bits.long_mode) {
            std::cerr << "Error: Not a 64-bit long mode code segment\n";
            return false;
        }

        // Step 3: CPU executes iretq, popping the frame atomically
        cpu.rip = frame.rip;
        cpu.cs = static_cast<uint16_t>(frame.cs);
        cpu.rflags = frame.rflags;
        cpu.rsp = frame.rsp;
        cpu.ss = static_cast<uint16_t>(frame.ss);
        cpu.current_privilege_level = static_cast<uint8_t>(desc.bits.descriptor_privilege_level);

        return true;
    }

    /**
     * @brief Extract decoded properties of a descriptor
     */
    static std::string dump_descriptor(const SegmentDescriptor& desc, size_t index) {
        std::ostringstream oss;
        uint32_t base = (desc.bits.base_high << 24) | (desc.bits.base_middle << 16) | desc.bits.base_low;
        uint32_t limit = (desc.bits.limit_high << 16) | desc.bits.limit_low;
        uint64_t effective_limit = desc.bits.granularity ? (static_cast<uint64_t>(limit) + 1) * 4096 - 1 : limit;

        oss << "GDT[" << index << "]: 0x" << std::hex << std::setw(16) << std::setfill('0') << desc.data << std::dec << "\n";
        oss << "  Base: 0x" << std::hex << base << ", Raw Limit: 0x" << limit
            << ", Effective Limit: 0x" << effective_limit << std::dec << "\n";
        oss << "  Type: " << static_cast<int>(desc.bits.type) << " (kExecuteRead=" << (desc.bits.type == DescriptorType::kExecuteRead ? "YES" : "NO") << ")\n";
        oss << "  S (SystemSegment): " << desc.bits.system_segment << " (1=Code/Data)\n";
        oss << "  DPL (Privilege): " << desc.bits.descriptor_privilege_level
            << " (" << (desc.bits.descriptor_privilege_level == 0 ? "Kernel Ring 0" : "User Ring 3") << ")\n";
        oss << "  P (Present): " << desc.bits.present << "\n";
        oss << "  L (LongMode 64-bit): " << desc.bits.long_mode << "\n";
        oss << "  D/B (DefaultSize): " << desc.bits.default_operation_size << " (must be 0 for L=1)\n";
        oss << "  G (Granularity 4KiB): " << desc.bits.granularity << "\n";
        return oss.str();
    }
};

} // namespace mikanos

int main(int argc, char* argv[]) {
    bool run_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            run_tests = true;
        }
    }

    std::cout << "=================================================================\n";
    std::cout << " MikanOS Chapter 8.5: GDT & x86-64 Segmentation Simulator\n";
    std::cout << "=================================================================\n\n";

    mikanos::GdtSegmentationSimulator sim;
    sim.setup_segments();

    std::cout << "[Step 1] Initializing GDT (3 Descriptors):\n";
    for (size_t i = 0; i < mikanos::GdtSegmentationSimulator::kGdtEntryCount; ++i) {
        std::cout << mikanos::GdtSegmentationSimulator::dump_descriptor(sim.gdt[i], i) << "\n";
    }

    std::cout << "[Step 2] GDTR Loaded:\n";
    std::cout << "  Limit: " << sim.cpu.gdtr.limit << " bytes (sizeof(GDT) - 1)\n";
    std::cout << "  Base Address: 0x" << std::hex << sim.cpu.gdtr.base << std::dec << "\n\n";

    std::cout << "[Step 3] Initializing Data Segment Registers (SetDSAll(0)):\n";
    sim.set_ds_all(0);
    std::cout << "  DS=" << sim.cpu.ds << " ES=" << sim.cpu.es << " SS=" << sim.cpu.ss
              << " FS=" << sim.cpu.fs << " GS=" << sim.cpu.gs << " (All set to NULL selector)\n\n";

    std::cout << "[Step 4] Switching Code Segment to Kernel CS via iretq:\n";
    uint16_t kernel_cs = mikanos::GdtSegmentationSimulator::make_selector(1, 0); // Index 1, RPL 0
    std::cout << "  Target CS Selector: 0x" << std::hex << kernel_cs << std::dec << " (Index=1, TI=0, RPL=0)\n";

    uint64_t target_rip = 0x101500;
    bool success = sim.simulate_set_cs_via_iretq(kernel_cs, target_rip);
    if (success) {
        std::cout << "  iretq Switch Successful!\n";
        std::cout << "  Current CS=" << sim.cpu.cs << ", RIP=0x" << std::hex << sim.cpu.rip << std::dec
                  << ", CPL=" << static_cast<int>(sim.cpu.current_privilege_level) << " (Ring 0)\n\n";
    }

    if (run_tests) {
        std::cout << "[Running Verification Unit Tests...]\n";
        // Test 1: Size and layout invariants
        assert(sizeof(mikanos::SegmentDescriptor) == 8);
        assert(sizeof(mikanos::SegmentSelector) == 2);
        assert(sizeof(mikanos::GdtrPointer) == 10);

        // Test 2: GDT[0] must be completely zeroed Null Descriptor
        assert(sim.gdt[0].data == 0);

        // Test 3: GDT[1] Kernel Code Segment attributes
        const auto& k_desc = sim.gdt[1];
        assert(k_desc.bits.system_segment == 1);
        assert(k_desc.bits.descriptor_privilege_level == 0);
        assert(k_desc.bits.present == 1);
        assert(k_desc.bits.long_mode == 1);
        assert(k_desc.bits.default_operation_size == 0);
        assert(k_desc.bits.granularity == 1);
        assert(k_desc.bits.type == mikanos::DescriptorType::kExecuteRead);

        // Test 4: GDT[2] User Code Segment attributes
        const auto& u_desc = sim.gdt[2];
        assert(u_desc.bits.system_segment == 1);
        assert(u_desc.bits.descriptor_privilege_level == 3);
        assert(u_desc.bits.present == 1);
        assert(u_desc.bits.long_mode == 1);
        assert(u_desc.bits.default_operation_size == 0);
        assert(u_desc.bits.granularity == 1);

        // Test 5: Selector bit packing
        uint16_t user_cs = mikanos::GdtSegmentationSimulator::make_selector(2, 3); // Index 2, RPL 3
        mikanos::SegmentSelector user_sel{.data = user_cs};
        assert(user_sel.bits.index == 2);
        assert(user_sel.bits.rpl == 3);
        assert(user_sel.bits.ti == 0);
        assert(user_cs == 0x13); // (2 << 3) | 3 = 19 = 0x13

        // Test 6: iretq switch to user CS
        bool u_switch = sim.simulate_set_cs_via_iretq(user_cs, 0x200000);
        assert(u_switch);
        assert(sim.cpu.cs == user_cs);
        assert(sim.cpu.current_privilege_level == 3);
        assert(sim.cpu.rip == 0x200000);

        // Test 7: Out-of-bounds selector rejection
        uint16_t invalid_cs = mikanos::GdtSegmentationSimulator::make_selector(99, 0);
        bool inv_switch = sim.simulate_set_cs_via_iretq(invalid_cs, 0x300000);
        assert(!inv_switch);

        std::cout << "All C++20 Verification Unit Tests Passed Successfully!\n";
    }

    return 0;
}
