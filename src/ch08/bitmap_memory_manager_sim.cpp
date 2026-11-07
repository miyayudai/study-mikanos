/**
 * @file bitmap_memory_manager_sim.cpp
 * @brief Chapter 8.7 BitmapMemoryManager & Physical Frame Allocation Simulator
 *
 * MikanOS Chapter 8.7: Challenging Memory Management (osbook_day08c)
 * Demonstrates:
 *  1. User-defined literals (_KiB, _MiB, _GiB) and strong type FrameID / kNullFrame.
 *  2. Exact MikanOS BitmapMemoryManager (128GiB max capacity = 33,554,432 frames = 4MiB bitmap).
 *  3. Why placement new on static .bss buffer (memory_manager_buf) is mandatory on a 1MiB stack.
 *  4. Bit-addressing math (line_index = id / 64, bit_index = id % 64) and safe 64-bit shift casting.
 *  5. First-Fit contiguous frame search (Allocate) with O(N) skip optimization (start_frame_id += i + 1).
 *  6. Frame release (Free), External Fragmentation mechanics, and SetMemoryRange bounds.
 *  7. UEFI MemoryMap scanning initialization (hole protection, EfiLoaderData kernel protection, FrameID{1} nullptr guard).
 *  8. Full reproduction of all 7 official osbook_day08c unit tests (kernel/test/test_memory_manager.cpp) + edge cases.
 */

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace mikanos {

// ============================================================================
// 1. Error & WithError<T> (Reproducing kernel/error.hpp)
// ============================================================================
class Error {
public:
    enum Code {
        kSuccess,
        kFull,
        kEmpty,
        kNoEnoughMemory,
        kIndexOutOfRange,
        kLastOfCode,
    };

private:
    static constexpr std::array<const char*, 5> code_names_{
        "kSuccess",
        "kFull",
        "kEmpty",
        "kNoEnoughMemory",
        "kIndexOutOfRange",
    };
    static_assert(Error::Code::kLastOfCode == code_names_.size());

public:
    Error(Code code, const char* file, int line)
        : code_{code}, line_{line}, file_{file} {}

    Code Cause() const { return code_; }
    explicit operator bool() const { return code_ != kSuccess; }
    const char* Name() const { return code_names_[static_cast<size_t>(code_)]; }
    const char* File() const { return file_; }
    int Line() const { return line_; }

private:
    Code code_;
    int line_;
    const char* file_;
};

#define MAKE_ERROR(code) ::mikanos::Error((code), __FILE__, __LINE__)

template <class T>
struct WithError {
    T value;
    Error error;
};

// ============================================================================
// 2. User-Defined Literals & FrameID (Reproducing kernel/memory_manager.hpp)
// ============================================================================
constexpr unsigned long long operator""_KiB(unsigned long long kib) {
    return kib * 1024ULL;
}

constexpr unsigned long long operator""_MiB(unsigned long long mib) {
    return mib * 1024_KiB;
}

constexpr unsigned long long operator""_GiB(unsigned long long gib) {
    return gib * 1024_MiB;
}

/** @brief Physical page frame size in bytes (4 KiB = 4,096 bytes) */
inline constexpr size_t kBytesPerFrame{4_KiB};

/** @brief Strongly-typed wrapper around a physical 4KiB page frame index */
class FrameID {
public:
    explicit constexpr FrameID(size_t id) : id_{id} {}
    constexpr size_t ID() const { return id_; }
    void* Frame() const { return reinterpret_cast<void*>(id_ * kBytesPerFrame); }
    constexpr uintptr_t PhysicalAddress() const { return static_cast<uintptr_t>(id_) * kBytesPerFrame; }

    friend constexpr bool operator==(FrameID lhs, FrameID rhs) { return lhs.id_ == rhs.id_; }
    friend constexpr bool operator!=(FrameID lhs, FrameID rhs) { return lhs.id_ != rhs.id_; }

private:
    size_t id_;
};

inline constexpr FrameID kNullFrame{std::numeric_limits<size_t>::max()};

// ============================================================================
// 3. BitmapMemoryManager (Exact reproduction of kernel/memory_manager.hpp/.cpp)
// ============================================================================
class BitmapMemoryManager {
public:
    /** @brief Maximum physical memory capacity managed by this class (128 GiB) */
    static constexpr auto kMaxPhysicalMemoryBytes{128_GiB};
    /** @brief Total number of 4KiB frames required to cover 128 GiB (33,554,432) */
    static constexpr size_t kFrameCount{kMaxPhysicalMemoryBytes / kBytesPerFrame};

    /** @brief Element type of the underlying bitmap array (64-bit on x86-64 LP64) */
    using MapLineType = uint64_t;
    /** @brief Number of bits (frames) stored per MapLineType element (64) */
    static constexpr size_t kBitsPerMapLine{8 * sizeof(MapLineType)};
    /** @brief Number of elements in alloc_map_ (524,288 lines = 4 MiB) */
    static constexpr size_t kMapLineCount{kFrameCount / kBitsPerMapLine};

    BitmapMemoryManager()
        : alloc_map_{}, range_begin_{FrameID{0}}, range_end_{FrameID{kFrameCount}} {}

    /** @brief First-Fit contiguous frame allocation with O(N) jump skip */
    WithError<FrameID> Allocate(size_t num_frames) {
        size_t start_frame_id = range_begin_.ID();
        while (true) {
            size_t i = 0;
            for (; i < num_frames; ++i) {
                if (start_frame_id + i >= range_end_.ID()) {
                    return {kNullFrame, MAKE_ERROR(Error::kNoEnoughMemory)};
                }
                if (GetBit(FrameID{start_frame_id + i})) {
                    // Frame at "start_frame_id + i" is already allocated
                    break;
                }
            }
            if (i == num_frames) {
                // Found num_frames contiguous free frames!
                MarkAllocated(FrameID{start_frame_id}, num_frames);
                return {
                    FrameID{start_frame_id},
                    MAKE_ERROR(Error::kSuccess),
                };
            }
            // Jump directly past the occupied frame (O(N) skip optimization)
            start_frame_id += i + 1;
        }
    }

    /** @brief Instrumented First-Fit allocation comparing O(N) jump vs naive +1 step */
    struct AllocationMetrics {
        FrameID frame{kNullFrame};
        Error::Code error_code{Error::kSuccess};
        size_t bit_checks_jump{0};
        size_t bit_checks_naive{0};
    };

    AllocationMetrics AllocateWithMetrics(size_t num_frames) {
        // First, count naive (+1 step) bit checks without mutating state
        size_t naive_checks = 0;
        {
            size_t s = range_begin_.ID();
            while (true) {
                size_t i = 0;
                bool out_of_range = false;
                for (; i < num_frames; ++i) {
                    if (s + i >= range_end_.ID()) {
                        out_of_range = true;
                        break;
                    }
                    ++naive_checks;
                    if (GetBit(FrameID{s + i})) {
                        break;
                    }
                }
                if (out_of_range || i == num_frames) {
                    break;
                }
                s += 1; // Naive single-step increment
            }
        }

        // Now perform actual MikanOS O(N) jump search and count checks
        size_t jump_checks = 0;
        size_t start_frame_id = range_begin_.ID();
        while (true) {
            size_t i = 0;
            for (; i < num_frames; ++i) {
                if (start_frame_id + i >= range_end_.ID()) {
                    return {kNullFrame, Error::kNoEnoughMemory, jump_checks, naive_checks};
                }
                ++jump_checks;
                if (GetBit(FrameID{start_frame_id + i})) {
                    break;
                }
            }
            if (i == num_frames) {
                MarkAllocated(FrameID{start_frame_id}, num_frames);
                return {FrameID{start_frame_id}, Error::kSuccess, jump_checks, naive_checks};
            }
            start_frame_id += i + 1;
        }
    }

    Error Free(FrameID start_frame, size_t num_frames) {
        for (size_t i = 0; i < num_frames; ++i) {
            SetBit(FrameID{start_frame.ID() + i}, false);
        }
        return MAKE_ERROR(Error::kSuccess);
    }

    void MarkAllocated(FrameID start_frame, size_t num_frames) {
        for (size_t i = 0; i < num_frames; ++i) {
            SetBit(FrameID{start_frame.ID() + i}, true);
        }
    }

    void SetMemoryRange(FrameID range_begin, FrameID range_end) {
        range_begin_ = range_begin;
        range_end_ = range_end;
    }

    FrameID RangeBegin() const { return range_begin_; }
    FrameID RangeEnd() const { return range_end_; }
    bool InspectBit(FrameID frame) const { return GetBit(frame); }
    MapLineType InspectLine(size_t line_index) const { return alloc_map_.at(line_index); }

    size_t CountFreeFramesInRange() const {
        size_t free_count = 0;
        for (size_t id = range_begin_.ID(); id < range_end_.ID(); ++id) {
            if (!GetBit(FrameID{id})) {
                ++free_count;
            }
        }
        return free_count;
    }

private:
    std::array<MapLineType, kMapLineCount> alloc_map_;
    FrameID range_begin_;
    FrameID range_end_;

    bool GetBit(FrameID frame) const {
        auto line_index = frame.ID() / kBitsPerMapLine;
        auto bit_index = frame.ID() % kBitsPerMapLine;
        return (alloc_map_[line_index] & (static_cast<MapLineType>(1) << bit_index)) != 0;
    }

    void SetBit(FrameID frame, bool allocated) {
        auto line_index = frame.ID() / kBitsPerMapLine;
        auto bit_index = frame.ID() % kBitsPerMapLine;

        if (allocated) {
            alloc_map_[line_index] |= (static_cast<MapLineType>(1) << bit_index);
        } else {
            alloc_map_[line_index] &= ~(static_cast<MapLineType>(1) << bit_index);
        }
    }
};

// ============================================================================
// 4. UEFI MemoryMap Integration (Reproducing kernel/memory_map.hpp & main.cpp)
// ============================================================================
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

inline constexpr bool IsAvailable(MemoryType memory_type) {
    return memory_type == MemoryType::kEfiBootServicesCode ||
           memory_type == MemoryType::kEfiBootServicesData ||
           memory_type == MemoryType::kEfiConventionalMemory;
}

inline constexpr size_t kUEFIPageSize = 4096;

struct MemoryDescriptor {
    uint32_t type;
    uint32_t pad;
    uintptr_t physical_start;
    uintptr_t virtual_start;
    uint64_t number_of_pages;
    uint64_t attribute;
};

struct MemoryMap {
    const void* buffer;
    uint64_t buffer_size;
    uint64_t map_size;
    uint64_t map_key;
    uint64_t descriptor_size;
    uint32_t descriptor_version;
};

struct InitSummary {
    uintptr_t available_end{0};
    size_t gap_sweeps_triggered{0};
    size_t unlisted_holes_detected{0};
    size_t unlisted_hole_frames{0};
    size_t non_available_frames_marked{0};
    size_t free_frames_in_range{0};
};

/**
 * @brief Exact reproduction of KernelMainNewStack's memory_manager initialization loop
 */
InitSummary InitializeBitmapFromUEFIMemoryMap(BitmapMemoryManager& memory_manager, const MemoryMap& memory_map) {
    InitSummary summary{};
    const auto memory_map_base = reinterpret_cast<uintptr_t>(memory_map.buffer);
    uintptr_t available_end = 0;
    uintptr_t prev_physical_end = 0;

    for (uintptr_t iter = memory_map_base;
         iter < memory_map_base + memory_map.map_size;
         iter += memory_map.descriptor_size) {
        const auto* desc = reinterpret_cast<const MemoryDescriptor*>(iter);
        if (prev_physical_end < desc->physical_start) {
            ++summary.unlisted_holes_detected;
            summary.unlisted_hole_frames += (desc->physical_start - prev_physical_end) / kBytesPerFrame;
        }
        if (available_end < desc->physical_start) {
            memory_manager.MarkAllocated(
                FrameID{available_end / kBytesPerFrame},
                (desc->physical_start - available_end) / kBytesPerFrame);
            ++summary.gap_sweeps_triggered;
        }

        const auto physical_end =
            desc->physical_start + desc->number_of_pages * kUEFIPageSize;
        prev_physical_end = physical_end;
        if (IsAvailable(static_cast<MemoryType>(desc->type))) {
            available_end = physical_end;
        } else {
            size_t reserved_frames = desc->number_of_pages * kUEFIPageSize / kBytesPerFrame;
            memory_manager.MarkAllocated(
                FrameID{desc->physical_start / kBytesPerFrame},
                reserved_frames);
            summary.non_available_frames_marked += reserved_frames;
        }
    }

    memory_manager.SetMemoryRange(FrameID{1}, FrameID{available_end / kBytesPerFrame});
    summary.available_end = available_end;
    summary.free_frames_in_range = memory_manager.CountFreeFramesInRange();
    return summary;
}

// Static .bss placement buffer replicating kernel/main.cpp
alignas(BitmapMemoryManager) char g_memory_manager_buf[sizeof(BitmapMemoryManager)];

} // namespace mikanos

namespace {

std::vector<uint8_t> BuildSampleUEFIMemoryMapBuffer(size_t descriptor_stride = 48) {
    using mikanos::MemoryDescriptor;
    using mikanos::MemoryType;

    // Realistic QEMU OVMF miniature memory map (total available_end = 16 MiB = 4,096 frames)
    // Includes a physical memory hole at [0x000A0000, 0x00100000) (Legacy VGA/BIOS ROM hole: 96 frames)
    // and Kernel ELF at [0x00100000, 0x00210000) (EfiLoaderData: 272 frames)
    const std::vector<MemoryDescriptor> descs = {
        // 0x00000000 - 0x00001000 (1 page = Frame 0): EfiBootServicesCode (will be excluded by FrameID{1}!)
        {static_cast<uint32_t>(MemoryType::kEfiBootServicesCode), 0, 0x00000000, 0, 1, 0xF},
        // 0x00001000 - 0x000A0000 (159 pages = Frames 1..159): EfiConventionalMemory (636 KiB free)
        {static_cast<uint32_t>(MemoryType::kEfiConventionalMemory), 0, 0x00001000, 0, 159, 0xF},
        // [GAP / HOLE]: 0x000A0000 - 0x00100000 (96 pages = Frames 160..255) -> Unlisted VGA/ROM Hole!
        // 0x00100000 - 0x00210000 (272 pages = Frames 256..527): EfiLoaderData (MikanOS Kernel .text/.data/.bss!)
        {static_cast<uint32_t>(MemoryType::kEfiLoaderData), 0, 0x00100000, 0, 272, 0xF},
        // 0x00210000 - 0x00800000 (1,520 pages = Frames 528..2047): EfiConventionalMemory
        {static_cast<uint32_t>(MemoryType::kEfiConventionalMemory), 0, 0x00210000, 0, 1520, 0xF},
        // 0x00800000 - 0x00820000 (32 pages = Frames 2048..2079): EfiRuntimeServicesData (UEFI Runtime -> Keep!)
        {static_cast<uint32_t>(MemoryType::kEfiRuntimeServicesData), 0, 0x00800000, 0, 32, 0x800000000000000FULL},
        // 0x00820000 - 0x01000000 (2,016 pages = Frames 2080..4095): EfiBootServicesData (Reclaimed after ExitBootServices!)
        {static_cast<uint32_t>(MemoryType::kEfiBootServicesData), 0, 0x00820000, 0, 2016, 0xF},
        // [GAP / PCI MMIO HOLE]: 0x01000000 - 0x01400000 (1,024 pages = Frames 4096..5119)
        // 0x01400000 - 0x01410000 (16 pages = Frames 5120..5135): EfiMemoryMappedIO (Trailing MMIO beyond RAM!)
        {static_cast<uint32_t>(MemoryType::kEfiMemoryMappedIO), 0, 0x01400000, 0, 16, 0x1},
    };

    std::vector<uint8_t> raw(descs.size() * descriptor_stride, 0);
    for (size_t i = 0; i < descs.size(); ++i) {
        auto* dst = reinterpret_cast<MemoryDescriptor*>(raw.data() + i * descriptor_stride);
        *dst = descs[i];
    }
    return raw;
}

void RunAllUnitTests() {
    using namespace mikanos;

    // ------------------------------------------------------------------------
    // Test 0: Compile-time Structural & Memory Footprint Invariants
    // ------------------------------------------------------------------------
    static_assert(kBytesPerFrame == 4096);
    static_assert(BitmapMemoryManager::kMaxPhysicalMemoryBytes == 137438953472ULL);
    static_assert(BitmapMemoryManager::kFrameCount == 33554432ULL);
    static_assert(BitmapMemoryManager::kBitsPerMapLine == 64);
    static_assert(BitmapMemoryManager::kMapLineCount == 524288ULL);
    // 524,288 * 8B (4,194,304 B = 4 MiB) + 2 * 8B (FrameID) = 4,194,320 bytes!
    static_assert(sizeof(BitmapMemoryManager) == 4194320ULL);
    // Exceeds 1 MiB kernel_main_stack by > 3 MiB -> Placement new in .bss is mandatory!
    static_assert(sizeof(BitmapMemoryManager) > 1_MiB);

    // ------------------------------------------------------------------------
    // Official MikanOS CppUTest Suite (kernel/test/test_memory_manager.cpp)
    // ------------------------------------------------------------------------
    // 1. TEST(MemoryManager, Allocate)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        const auto frame1 = mgr->Allocate(3);
        const auto frame2 = mgr->Allocate(1);

        assert(!frame1.error);
        assert(!frame2.error);
        assert(frame1.value.ID() == 0);
        assert(3 <= frame2.value.ID());
        assert(frame2.value.ID() == 3);
    }

    // 2. TEST(MemoryManager, AllocateMultipleLine)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        const auto frame1 = mgr->Allocate(BitmapMemoryManager::kBitsPerMapLine + 3); // 67 frames
        const auto frame2 = mgr->Allocate(1);

        assert(!frame1.error);
        assert(!frame2.error);
        assert(frame1.value.ID() == 0);
        assert(BitmapMemoryManager::kBitsPerMapLine + 3 <= frame2.value.ID());
        assert(frame2.value.ID() == 67);
        // Line 0 has all 64 bits set (~0ULL); Line 1 has bits 0,1,2 (frame1: 64..66) + bit 3 (frame2: 67) -> 0b1111ULL
        assert(mgr->InspectLine(0) == ~uint64_t{0});
        assert(mgr->InspectLine(1) == 0b1111ULL);
    }

    // 3. TEST(MemoryManager, AllocateNoEnoughMemory)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        const auto frame1 = mgr->Allocate(BitmapMemoryManager::kFrameCount + 1);

        assert(static_cast<bool>(frame1.error));
        assert(frame1.error.Cause() == Error::kNoEnoughMemory);
        assert(frame1.value.ID() == kNullFrame.ID());
    }

    // 4. TEST(MemoryManager, Free)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        const auto frame1 = mgr->Allocate(3);
        const auto err = mgr->Free(frame1.value, 3);
        const auto frame2 = mgr->Allocate(2);

        assert(!err);
        assert(frame1.value.ID() == 0);
        assert(frame2.value.ID() == 0);
    }

    // 5. TEST(MemoryManager, FreeDifferentFrames)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        const auto frame1 = mgr->Allocate(3); // [0, 1, 2] allocated
        mgr->Free(frame1.value, 1);           // [0] freed, [1, 2] remain allocated
        const auto frame2 = mgr->Allocate(2); // Needs 2 contiguous -> skips [0] & [1], jumps to 2? [2] busy -> lands at 3!

        assert(frame1.value.ID() == 0);
        assert(3 <= frame2.value.ID());
        assert(frame2.value.ID() == 3);
    }

    // 6. TEST(MemoryManager, MarkAllocated)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        mgr->MarkAllocated(FrameID{61}, 3); // Frames 61, 62, 63 occupied
        const auto frame1 = mgr->Allocate(64);

        assert(!frame1.error);
        assert(frame1.value.ID() == 64);
    }

    // 7. TEST(MemoryManager, SetMemoryRange)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        const auto frame1 = mgr->Allocate(1);
        mgr->SetMemoryRange(FrameID{10}, FrameID{64});
        const auto frame2 = mgr->Allocate(1);

        assert(frame1.value.ID() == 0);
        assert(frame2.value.ID() == 10);
    }

    // ------------------------------------------------------------------------
    // Extended Edge-Case & Mathematical Verification Tests
    // ------------------------------------------------------------------------
    // Test 8: High bit-index (>= 32 and == 63) safety against 32-bit signed shift UB
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        mgr->SetMemoryRange(FrameID{1}, FrameID{256});
        mgr->MarkAllocated(FrameID{31}, 1);
        mgr->MarkAllocated(FrameID{32}, 1);
        mgr->MarkAllocated(FrameID{63}, 1);
        assert(mgr->InspectBit(FrameID{31}));
        assert(mgr->InspectBit(FrameID{32}));
        assert(mgr->InspectBit(FrameID{63}));
        assert(!mgr->InspectBit(FrameID{30}));
        assert(!mgr->InspectBit(FrameID{33}));
        assert(!mgr->InspectBit(FrameID{62}));
        assert(mgr->InspectLine(0) == ((uint64_t{1} << 31) | (uint64_t{1} << 32) | (uint64_t{1} << 63)));
    }

    // Test 9: O(N) Skip Optimization Proof (start_frame_id += i + 1 vs naive +1)
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        mgr->SetMemoryRange(FrameID{0}, FrameID{128});
        // Place blockers at frames 7, 15, 23, 31 (every 8th frame has 7 free followed by 1 busy)
        mgr->MarkAllocated(FrameID{7}, 1);
        mgr->MarkAllocated(FrameID{15}, 1);
        mgr->MarkAllocated(FrameID{23}, 1);
        mgr->MarkAllocated(FrameID{31}, 1);

        // Request 8 contiguous frames -> must land at Frame 32!
        auto metrics = mgr->AllocateWithMetrics(8);
        assert(metrics.error_code == Error::kSuccess);
        assert(metrics.frame.ID() == 32);
        // Jump checks: [0..7](8) + [8..15](8) + [16..23](8) + [24..31](8) + [32..39](8) = 40 bit checks (each bit checked ONCE!)
        assert(metrics.bit_checks_jump == 40);
        // Naive checks: for each of 4 blocks of 8 positions, checks 8+7+6+5+4+3+2+1 = 36 -> 4 * 36 + 8 = 152 checks!
        assert(metrics.bit_checks_naive == 152);
        assert(metrics.bit_checks_jump < metrics.bit_checks_naive);
    }

    // Test 10: External Fragmentation Verification
    {
        auto mgr = std::make_unique<BitmapMemoryManager>();
        mgr->SetMemoryRange(FrameID{0}, FrameID{12});
        // Occupy every 3rd frame: 2, 5, 8, 11 -> Total free frames = 8 (2+2+2+2), max contiguous = 2
        mgr->MarkAllocated(FrameID{2}, 1);
        mgr->MarkAllocated(FrameID{5}, 1);
        mgr->MarkAllocated(FrameID{8}, 1);
        mgr->MarkAllocated(FrameID{11}, 1);
        assert(mgr->CountFreeFramesInRange() == 8);

        // Requesting 3 contiguous frames fails despite 8 free frames!
        auto fail_res = mgr->Allocate(3);
        assert(fail_res.error.Cause() == Error::kNoEnoughMemory);
        assert(fail_res.value == kNullFrame);

        // Freeing Frame 5 merges [3,4] + [5] + [6,7] into a 5-frame contiguous block [3..7]!
        mgr->Free(FrameID{5}, 1);
        auto ok_res = mgr->Allocate(5);
        assert(!ok_res.error);
        assert(ok_res.value.ID() == 3);
    }

    // Test 11: Placement new on static .bss buffer + UEFI MemoryMap Initialization
    {
        auto* mgr = new (g_memory_manager_buf) BitmapMemoryManager;
        auto raw_map = BuildSampleUEFIMemoryMapBuffer(48);
        MemoryMap memmap{
            raw_map.data(),
            raw_map.size(),
            raw_map.size(),
            0x1234,
            48,
            1,
        };

        auto summary = InitializeBitmapFromUEFIMemoryMap(*mgr, memmap);

        // Trailing MMIO at 0x01400000 must NOT advance available_end beyond 0x01000000 (16 MiB = Frame 4096)
        assert(summary.available_end == 0x01000000ULL);
        assert(mgr->RangeBegin().ID() == 1);
        assert(mgr->RangeEnd().ID() == 4096);

        // Two unlisted gaps detected: VGA hole [160..255] (96) + PCI hole [4096..5119] (1024)
        assert(summary.unlisted_holes_detected == 2);
        assert(summary.unlisted_hole_frames == 96 + 1024);
        // available_end < desc->physical_start sweeps idempotently 4 times (including across non-available regions!)
        assert(summary.gap_sweeps_triggered == 4);

        // Frame 0 excluded by range_begin_ = FrameID{1}: first allocation lands at Frame 1 (0x1000 != nullptr!)
        auto first_alloc = mgr->Allocate(1);
        assert(!first_alloc.error);
        assert(first_alloc.value.ID() == 1);
        assert(first_alloc.value.PhysicalAddress() == 0x1000ULL);
        assert(first_alloc.value.Frame() != nullptr);

        // Requesting 200 contiguous frames cannot fit in Frames [2..159] (158 free),
        // skips VGA hole [160..255] AND Kernel EfiLoaderData [256..527], landing safely at Frame 528 (0x210000)!
        auto large_alloc = mgr->Allocate(200);
        assert(!large_alloc.error);
        assert(large_alloc.value.ID() == 528);
        assert(large_alloc.value.PhysicalAddress() == 0x00210000ULL);

        std::destroy_at(mgr);
    }
}

} // namespace

int main(int argc, char** argv) {
    bool run_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--test") {
            run_tests = true;
        }
    }

    if (run_tests) {
        RunAllUnitTests();
        std::cout << "All C++20 BitmapMemoryManager Verification Unit Tests Passed Successfully!\n";
        return 0;
    }

    using namespace mikanos;
    std::cout << "======================================================================\n";
    std::cout << "  MikanOS Chapter 8.7: BitmapMemoryManager Simulator (osbook_day08c)\n";
    std::cout << "======================================================================\n";
    std::cout << "1. Structural & Memory Footprint Specifications:\n";
    std::cout << "   kBytesPerFrame          : " << kBytesPerFrame << " B (4 KiB)\n";
    std::cout << "   kMaxPhysicalMemoryBytes : " << (BitmapMemoryManager::kMaxPhysicalMemoryBytes / 1_GiB) << " GiB\n";
    std::cout << "   kFrameCount             : " << BitmapMemoryManager::kFrameCount << " frames\n";
    std::cout << "   kBitsPerMapLine         : " << BitmapMemoryManager::kBitsPerMapLine << " bits (uint64_t)\n";
    std::cout << "   alloc_map_ lines        : " << BitmapMemoryManager::kMapLineCount << " lines\n";
    std::cout << "   sizeof(BitmapMemoryMgr) : " << sizeof(BitmapMemoryManager) << " bytes (~4.00 MiB)\n\n";

    auto* mgr = new (g_memory_manager_buf) BitmapMemoryManager;
    auto raw_map = BuildSampleUEFIMemoryMapBuffer(48);
    MemoryMap memmap{raw_map.data(), raw_map.size(), raw_map.size(), 0x1234, 48, 1};
    auto summary = InitializeBitmapFromUEFIMemoryMap(*mgr, memmap);

    std::cout << "2. UEFI MemoryMap Initialization Summary:\n";
    std::cout << "   Managed Range           : [Frame " << mgr->RangeBegin().ID()
              << ", Frame " << mgr->RangeEnd().ID() << ") -> [0x"
              << std::hex << mgr->RangeBegin().PhysicalAddress() << ", 0x"
              << mgr->RangeEnd().PhysicalAddress() << std::dec << ")\n";
    std::cout << "   Memory Holes Protected  : " << summary.unlisted_holes_detected
              << " holes (" << summary.unlisted_hole_frames << " frames, "
              << summary.gap_sweeps_triggered << " idempotent sweeps)\n";
    std::cout << "   Free Frames Available   : " << summary.free_frames_in_range
              << " frames (" << (summary.free_frames_in_range * 4) << " KiB)\n\n";

    auto a1 = mgr->Allocate(4);
    auto a2 = mgr->Allocate(200);
    std::cout << "3. Sample Contiguous Frame Allocations:\n";
    std::cout << "   Allocate(4 frames)      -> FrameID{" << a1.value.ID()
              << "} (Phys: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << a1.value.PhysicalAddress() << std::dec << ")\n";
    std::cout << "   Allocate(200 frames)    -> FrameID{" << a2.value.ID()
              << "} (Phys: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << a2.value.PhysicalAddress() << std::dec
              << ") [Safely skipped VGA Hole & Kernel ELF!]\n";

    std::destroy_at(mgr);
    return 0;
}
