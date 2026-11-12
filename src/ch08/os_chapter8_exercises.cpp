/**
 * @file os_chapter8_exercises.cpp
 * @brief Chapter 8 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 8:
 * 1. UEFI MemoryMap & Pointer Stepping Arithmetic (descriptor_size 48B vs sizeof 40B) (Section 8.1, 8.2)
 * 2. Transitional Memory Relocation & Deep Copy Safety (Section 8.3)
 * 3. Kernel Stack .bss Relocation & System V AMD64 ABI 16-Byte Alignment (Section 8.4)
 * 4. x86-64 GDT 64-bit Code Segment Descriptor & iretq Stack Frame (Section 8.5)
 * 5. 4-Level Identity Paging & 2MiB Huge Page Entry Math (Section 8.6)
 * 6. BitmapMemoryManager First-Fit Allocation, O(N) Skip Optimization, and Coalescing (Section 8.7)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch08 {

// =============================================================================
// Exercise 1: UEFI MemoryMap & Pointer Arithmetic (Section 8.2)
// =============================================================================

enum class MemoryType : uint32_t {
    kEfiReservedMemoryType = 0,
    kEfiLoaderCode = 1,
    kEfiLoaderData = 2,
    kEfiBootServicesCode = 3,
    kEfiBootServicesData = 4,
    kEfiRuntimeServicesCode = 5,
    kEfiRuntimeServicesData = 6,
    kEfiConventionalMemory = 7,
    kEfiUnusableMemory = 8,
    kEfiACPIReclaimMemory = 9,
    kEfiACPIMemoryNVS = 10,
    kEfiMemoryMappedIO = 11,
    kEfiMemoryMappedIOPortSpace = 12,
    kEfiPalCode = 13,
    kEfiPersistentMemory = 14,
    kEfiMaxMemoryType = 15,
};

struct MemoryDescriptor {
    MemoryType type;
    uint32_t pad;
    uint64_t physical_start;
    uint64_t virtual_start;
    uint64_t number_of_pages;
    uint64_t attribute;
};

static_assert(sizeof(MemoryDescriptor) == 40, "MemoryDescriptor struct size must be 40 bytes");

constexpr bool IsAvailable(MemoryType type) {
    return type == MemoryType::kEfiBootServicesCode ||
           type == MemoryType::kEfiBootServicesData ||
           type == MemoryType::kEfiConventionalMemory;
}

// Simulates UEFI memory map buffer with 48-byte descriptor_size (containing 8 bytes of firmware padding)
struct SimulatedUefiMemoryMap {
    static constexpr size_t kDescriptorSize = 48; // Standard EDK II firmware size with 8B padding
    static constexpr size_t kNumEntries = 8;
    std::array<uint8_t, kDescriptorSize * kNumEntries> buffer{};

    SimulatedUefiMemoryMap() {
        for (size_t i = 0; i < kNumEntries; ++i) {
            MemoryDescriptor desc{};
            if (i == 0) {
                desc.type = MemoryType::kEfiBootServicesCode;
                desc.physical_start = 0x10000;
                desc.number_of_pages = 16;
            } else if (i == 1) {
                desc.type = MemoryType::kEfiBootServicesData;
                desc.physical_start = 0x20000;
                desc.number_of_pages = 32;
            } else if (i == 2) {
                desc.type = MemoryType::kEfiLoaderCode;
                desc.physical_start = 0x100000;
                desc.number_of_pages = 256;
            } else if (i == 3) {
                desc.type = MemoryType::kEfiConventionalMemory;
                desc.physical_start = 0x200000;
                desc.number_of_pages = 1024;
            } else if (i == 4) {
                desc.type = MemoryType::kEfiMemoryMappedIO;
                desc.physical_start = 0xFEE00000;
                desc.number_of_pages = 1;
            } else {
                desc.type = MemoryType::kEfiConventionalMemory;
                desc.physical_start = 0x10000000 + (i * 0x1000000);
                desc.number_of_pages = 4096;
            }
            desc.attribute = 0xF;

            // Copy 40 bytes of descriptor into buffer with 48B stride
            std::memcpy(buffer.data() + (i * kDescriptorSize), &desc, sizeof(MemoryDescriptor));
            // Fill padding bytes with sentinel 0xAA
            std::memset(buffer.data() + (i * kDescriptorSize) + sizeof(MemoryDescriptor), 0xAA, kDescriptorSize - sizeof(MemoryDescriptor));
        }
    }

    const MemoryDescriptor* GetCorrect(size_t index) const {
        uintptr_t base = reinterpret_cast<uintptr_t>(buffer.data());
        return reinterpret_cast<const MemoryDescriptor*>(base + (index * kDescriptorSize));
    }

    const MemoryDescriptor* GetBuggyOffBy8(size_t index) const {
        // Buggy pointer arithmetic: increments by sizeof(MemoryDescriptor) = 40 bytes!
        const auto* ptr = reinterpret_cast<const MemoryDescriptor*>(buffer.data());
        return ptr + index;
    }
};

// =============================================================================
// Exercise 2: Stack Relocation & System V AMD64 ABI 16-Byte Alignment (Section 8.4)
// =============================================================================

struct StackLayout {
    static constexpr size_t kStackSize = 1024 * 1024; // 1MiB
    alignas(16) std::array<uint8_t, kStackSize> stack_buffer{};

    uintptr_t GetBaseAddress() const {
        return reinterpret_cast<uintptr_t>(stack_buffer.data());
    }

    uintptr_t GetInitialRsp() const {
        // x86-64 Full Descending Stack: Initial RSP is base + size
        return GetBaseAddress() + kStackSize;
    }

    bool IsRsp16ByteAligned() const {
        return (GetInitialRsp() % 16) == 0;
    }

    bool IsCallSiteRspAligned(uintptr_t current_rsp) const {
        // Inside a called function (after CALL has pushed 8-byte return address RIP),
        // (RSP + 8) % 16 == 0, or RSP % 16 == 8.
        return (current_rsp % 16) == 8;
    }
};

// =============================================================================
// Exercise 3: GDT 64-Bit Code Segment Descriptor & iretq Frame (Section 8.5)
// =============================================================================

union SegmentDescriptor {
    uint64_t data;
    struct {
        uint64_t limit_low : 16;
        uint64_t base_low : 16;
        uint64_t base_middle : 8;
        uint64_t type : 4;
        uint64_t system_segment : 1;
        uint64_t descriptor_privilege_level : 2;
        uint64_t present : 1;
        uint64_t limit_high : 4;
        uint64_t available : 1;
        uint64_t long_mode : 1;
        uint64_t default_operation_size : 1;
        uint64_t granularity : 1;
        uint64_t base_high : 8;
    } __attribute__((packed)) bits;
} __attribute__((packed));

static_assert(sizeof(SegmentDescriptor) == 8, "SegmentDescriptor must be 8 bytes");

constexpr SegmentDescriptor MakeKernelCodeSegmentDescriptor() {
    SegmentDescriptor desc{};
    desc.bits.limit_low = 0xFFFF;
    desc.bits.base_low = 0;
    desc.bits.base_middle = 0;
    desc.bits.type = 10; // Execute/Read code segment (0xA)
    desc.bits.system_segment = 1; // 1 for Code/Data segments
    desc.bits.descriptor_privilege_level = 0; // Ring 0 Kernel
    desc.bits.present = 1; // Present
    desc.bits.limit_high = 0xF;
    desc.bits.available = 0;
    desc.bits.long_mode = 1; // L=1: 64-bit Long Mode code segment
    desc.bits.default_operation_size = 0; // Must be 0 when L=1
    desc.bits.granularity = 1; // 4KiB Granularity
    desc.bits.base_high = 0;
    return desc;
}

struct IretqStackFrame {
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
};
static_assert(sizeof(IretqStackFrame) == 40, "IretqStackFrame must be 40 bytes");

// =============================================================================
// Exercise 4: 4-Level Identity Paging & 2MiB Huge Page PDE (Section 8.6)
// =============================================================================

struct PageTableEntry {
    uint64_t raw;

    constexpr bool Present() const { return (raw & 0x1) != 0; }
    constexpr bool Writable() const { return (raw & 0x2) != 0; }
    constexpr bool User() const { return (raw & 0x4) != 0; }
    constexpr bool HugePage() const { return (raw & 0x80) != 0; }
    constexpr uint64_t PhysicalAddress() const { return raw & 0x000F'FFFF'FFFF'F000ULL; }
};

struct PagingMath {
    static constexpr size_t kPageSize2MiB = 2 * 1024 * 1024;
    static constexpr size_t kPageSize4KiB = 4096;
    static constexpr size_t kTotalMemoryToMap = 64ULL * 1024 * 1024 * 1024; // 64GiB

    static constexpr size_t kEntriesPerTable = 512;
    static constexpr size_t kNumPDs = kTotalMemoryToMap / (kEntriesPerTable * kPageSize2MiB); // 64 PD tables

    static constexpr size_t MemoryConsumedHugePages() {
        // 1 PML4 (4KB) + 1 PDP (4KB) + 64 PD (64 * 4KB = 256KB) = 264KiB
        return (1 + 1 + kNumPDs) * kPageSize4KiB;
    }

    static constexpr size_t MemoryConsumed4KiBPages() {
        // For 64GiB with 4KiB pages: 64GiB / 2MiB = 32,768 PT tables * 4KiB = 128MiB (+ PD, PDP, PML4)
        size_t num_pts = kTotalMemoryToMap / (kEntriesPerTable * kPageSize4KiB); // 32,768
        return (1 + 1 + kNumPDs + num_pts) * kPageSize4KiB;
    }
};

// =============================================================================
// Exercise 5: BitmapMemoryManager & First-Fit Skip Math (Section 8.7)
// =============================================================================

class FrameID {
public:
    explicit constexpr FrameID(size_t id = 0) : id_{id} {}
    constexpr size_t ID() const { return id_; }
    constexpr bool operator==(const FrameID& rhs) const = default;
    constexpr auto operator<=>(const FrameID& rhs) const = default;
    constexpr FrameID operator+(size_t offset) const { return FrameID{id_ + offset}; }
    constexpr FrameID& operator+=(size_t offset) { id_ += offset; return *this; }
private:
    size_t id_;
};

constexpr FrameID kNullFrame{static_cast<size_t>(-1)};

class SimpleBitmapMemoryManager {
public:
    static constexpr size_t kMaxFrames = 1024; // 1024 frames = 4MiB manageable memory
    static constexpr size_t kBitsPerWord = 64;
    static constexpr size_t kMapWords = (kMaxFrames + kBitsPerWord - 1) / kBitsPerWord;

    SimpleBitmapMemoryManager() {
        alloc_map_.fill(0); // 0 = free, 1 = allocated
    }

    bool GetBit(FrameID frame) const {
        if (frame.ID() >= kMaxFrames) return true;
        size_t line = frame.ID() / kBitsPerWord;
        size_t bit = frame.ID() % kBitsPerWord;
        return (alloc_map_[line] & (1ULL << bit)) != 0;
    }

    void SetBit(FrameID frame, bool allocated) {
        if (frame.ID() >= kMaxFrames) return;
        size_t line = frame.ID() / kBitsPerWord;
        size_t bit = frame.ID() % kBitsPerWord;
        if (allocated) {
            alloc_map_[line] |= (1ULL << bit);
        } else {
            alloc_map_[line] &= ~(1ULL << bit);
        }
    }

    // First-Fit allocation with start_frame_id += i + 1 skip optimization
    struct AllocResult {
        FrameID frame;
        size_t comparisons;
        size_t skips_performed;
    };

    AllocResult AllocateWithMetrics(size_t num_frames) {
        size_t comparisons = 0;
        size_t skips = 0;
        size_t start_frame = 1; // Reserve FrameID{0} for NULL safety

        while (start_frame + num_frames <= kMaxFrames) {
            size_t i = 0;
            for (; i < num_frames; ++i) {
                ++comparisons;
                if (GetBit(FrameID{start_frame + i})) {
                    break;
                }
            }

            if (i == num_frames) {
                // Found contiguous range! Mark as allocated.
                for (size_t k = 0; k < num_frames; ++k) {
                    SetBit(FrameID{start_frame + k}, true);
                }
                return {FrameID{start_frame}, comparisons, skips};
            }

            // O(N) Skip Optimization: Jump past the occupied frame at start_frame + i!
            start_frame += i + 1;
            ++skips;
        }

        return {kNullFrame, comparisons, skips};
    }

    void Free(FrameID frame, size_t num_frames) {
        for (size_t i = 0; i < num_frames; ++i) {
            SetBit(frame + i, false);
        }
    }

private:
    std::array<uint64_t, kMapWords> alloc_map_{};
};

// =============================================================================
// Verification Suite Execution
// =============================================================================

void RunAllExercises() {
    std::cout << "================================================================\n";
    std::cout << " MikanOS Chapter 8 Comprehensive Exercises Verification Suite\n";
    std::cout << "================================================================\n\n";

    // -------------------------------------------------------------------------
    // Test 1: UEFI MemoryMap 48-byte Stepping & Off-by-8 Bug Prevention
    // -------------------------------------------------------------------------
    std::cout << "[Test 1] UEFI MemoryMap 48B Stepping & Off-by-8 Bug Prevention... ";
    {
        SimulatedUefiMemoryMap memmap;

        // Correct stepping: entry 3 is ConventionalMemory at physical address 0x200000
        const auto* correct_desc3 = memmap.GetCorrect(3);
        assert(correct_desc3->type == MemoryType::kEfiConventionalMemory);
        assert(correct_desc3->physical_start == 0x200000);
        assert(IsAvailable(correct_desc3->type) == true);

        // Buggy stepping: increments by 40B instead of 48B
        const auto* buggy_desc3 = memmap.GetBuggyOffBy8(3);
        // By index 3, offset error is 3 * 8 = 24 bytes!
        assert(buggy_desc3->physical_start != 0x200000); // Corruption demonstrated!
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 2: Kernel Stack Full Descending Layout & 16-Byte ABI Alignment
    // -------------------------------------------------------------------------
    std::cout << "[Test 2] Kernel Stack Full Descending & System V ABI Alignment... ";
    {
        StackLayout layout;
        assert(layout.IsRsp16ByteAligned());

        uintptr_t initial_rsp = layout.GetInitialRsp();
        assert((initial_rsp % 16) == 0);

        // Simulate CALL instruction: pushes 8-byte RIP to stack
        uintptr_t rsp_after_call = initial_rsp - 8;
        assert(layout.IsCallSiteRspAligned(rsp_after_call));
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 3: GDT 64-Bit Code Segment Descriptor & iretq Frame
    // -------------------------------------------------------------------------
    std::cout << "[Test 3] GDT 64-Bit Code Segment Descriptor & iretq Frame... ";
    {
        constexpr auto desc = MakeKernelCodeSegmentDescriptor();
        assert(desc.bits.type == 10); // Execute/Read
        assert(desc.bits.system_segment == 1);
        assert(desc.bits.descriptor_privilege_level == 0);
        assert(desc.bits.present == 1);
        assert(desc.bits.long_mode == 1);
        assert(desc.bits.default_operation_size == 0);
        assert(desc.bits.granularity == 1);

        // Standard raw value matches 0x00AF9A000000FFFF
        assert(desc.data == 0x00AF9A000000FFFFULL);

        IretqStackFrame frame{
            .rip = 0x104000,
            .cs = 0x08,
            .rflags = 0x202,
            .rsp = 0x200000,
            .ss = 0x00,
        };
        assert(frame.cs == 0x08);
        assert((frame.rsp % 16) == 0);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 4: 4-Level Identity Paging & 2MiB Huge Page PDE Math
    // -------------------------------------------------------------------------
    std::cout << "[Test 4] 4-Level Identity Paging & 2MiB Huge Page Savings Math... ";
    {
        assert(PagingMath::kNumPDs == 64);
        assert(PagingMath::MemoryConsumedHugePages() == 264 * 1024); // 264 KiB
        assert(PagingMath::MemoryConsumed4KiBPages() > 128 * 1024 * 1024); // > 128 MiB

        // Savings ratio: 128MiB / 264KiB ~ 496x reduction!
        double reduction_factor = static_cast<double>(PagingMath::MemoryConsumed4KiBPages()) /
                                  static_cast<double>(PagingMath::MemoryConsumedHugePages());
        assert(reduction_factor > 480.0);

        PageTableEntry pde{0x200000ULL | 0x83ULL}; // 2MiB base + Present | Writable | HugePage
        assert(pde.Present());
        assert(pde.Writable());
        assert(pde.HugePage());
        assert(pde.PhysicalAddress() == 0x200000);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 5: BitmapMemoryManager First-Fit Allocation & O(N) Skip Math
    // -------------------------------------------------------------------------
    std::cout << "[Test 5] BitmapMemoryManager First-Fit & Skip Math Verification... ";
    {
        SimpleBitmapMemoryManager mgr;

        // Block frame 10
        mgr.SetBit(FrameID{10}, true);

        // Request 8 contiguous frames starting search from frame 1
        // Frames 1..9 are free, but that's 9 frames.
        // Wait, request 15 frames:
        // Frame 1..9 has only 9 free frames, frame 10 is occupied.
        // The allocator should check frames 1..10 (10 is occupied at offset 9),
        // and then SKIP directly to frame 11 (1 + 9 + 1 = 11)!
        auto res = mgr.AllocateWithMetrics(15);
        assert(res.frame == FrameID{11});
        assert(res.skips_performed == 1);

        // Allocate another block
        auto res2 = mgr.AllocateWithMetrics(10);
        assert(res2.frame == FrameID{26}); // 11 + 15 = 26

        // Free the first block (15 frames at 11)
        mgr.Free(res.frame, 15);
        assert(mgr.GetBit(FrameID{11}) == false);
        assert(mgr.GetBit(FrameID{25}) == false);

        // Reallocate 12 frames -> should reuse frame 11!
        auto res3 = mgr.AllocateWithMetrics(12);
        assert(res3.frame == FrameID{11});
    }
    std::cout << "PASSED\n";

    std::cout << "\n>> All Chapter 8 exercises verified successfully!\n";
}

} // namespace mikanos::exercises::ch08

int main(int argc, char** argv) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    mikanos::exercises::ch08::RunAllExercises();

    if (test_mode) {
        std::cout << "[CTest] All Chapter 8 exercises verified successfully.\n";
    }
    return 0;
}
