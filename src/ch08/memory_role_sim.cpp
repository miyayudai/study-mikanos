/**
 * @file memory_role_sim.cpp
 * @brief Chapter 8.1: Role of OS Memory Management & Address Translation Sim (C++20)
 *
 * Demonstrates:
 * 1. 4 KiB Physical Page Frame arithmetic and boundary alignments
 * 2. x86-64 4-Level Paging 48-bit canonical virtual address decomposition (9+9+9+9+12)
 * 3. Physical Frame Bitmap allocator simulation (First-Fit, O(1) deallocation)
 * 4. PMM overhead calculations across physical RAM capacities
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace mikanos::study {

constexpr size_t kPageSize = 4096;        // 4 KiB
constexpr uintptr_t kPageMask = 0xFFFULL;  // Lower 12 bits

struct VirtualAddressParts {
    uintptr_t raw;
    bool is_canonical;
    uint16_t pml4_index;
    uint16_t pdp_index;
    uint16_t pd_index;
    uint16_t pt_index;
    uint16_t offset;
};

class MemoryAddressMath {
public:
    static constexpr size_t PhysToFrame(uintptr_t phys_addr) noexcept {
        return phys_addr / kPageSize;
    }

    static constexpr uintptr_t FrameToPhys(size_t frame_index) noexcept {
        return static_cast<uintptr_t>(frame_index) * kPageSize;
    }

    static constexpr bool IsPageAligned(uintptr_t addr) noexcept {
        return (addr & kPageMask) == 0;
    }

    static constexpr uintptr_t AlignDown(uintptr_t addr) noexcept {
        return addr & ~kPageMask;
    }

    static constexpr uintptr_t AlignUp(uintptr_t addr) noexcept {
        return (addr + kPageMask) & ~kPageMask;
    }

    static constexpr bool IsCanonical(uintptr_t vaddr) noexcept {
        // In 48-bit canonical addressing, bits 48..63 must equal bit 47
        uintptr_t bit47 = (vaddr >> 47) & 1ULL;
        uintptr_t upper_bits = (vaddr >> 48) & 0xFFFFULL;
        uintptr_t expected_upper = bit47 ? 0xFFFFULL : 0x0000ULL;
        return upper_bits == expected_upper;
    }

    static VirtualAddressParts DecomposeVirtualAddress(uintptr_t vaddr) noexcept {
        return VirtualAddressParts{
            .raw = vaddr,
            .is_canonical = IsCanonical(vaddr),
            .pml4_index = static_cast<uint16_t>((vaddr >> 39) & 0x1FFULL),
            .pdp_index = static_cast<uint16_t>((vaddr >> 30) & 0x1FFULL),
            .pd_index = static_cast<uint16_t>((vaddr >> 21) & 0x1FFULL),
            .pt_index = static_cast<uint16_t>((vaddr >> 12) & 0x1FFULL),
            .offset = static_cast<uint16_t>(vaddr & kPageMask),
        };
    }
};

class SimpleBitmapMemoryManager {
public:
    static constexpr size_t kBitsPerWord = 64;

    explicit SimpleBitmapMemoryManager(size_t total_frames)
        : total_frames_(total_frames),
          bitmap_((total_frames + kBitsPerWord - 1) / kBitsPerWord, 0ULL) {}

    bool GetBit(size_t frame_index) const {
        if (frame_index >= total_frames_) return true; // Treat out-of-range as allocated
        size_t word_idx = frame_index / kBitsPerWord;
        size_t bit_idx = frame_index % kBitsPerWord;
        return (bitmap_[word_idx] & (1ULL << bit_idx)) != 0;
    }

    void SetBit(size_t frame_index, bool allocated) {
        if (frame_index >= total_frames_) return;
        size_t word_idx = frame_index / kBitsPerWord;
        size_t bit_idx = frame_index % kBitsPerWord;
        if (allocated) {
            bitmap_[word_idx] |= (1ULL << bit_idx);
        } else {
            bitmap_[word_idx] &= ~(1ULL << bit_idx);
        }
    }

    size_t Allocate(size_t num_frames) {
        if (num_frames == 0) return kNullFrame;

        // First-Fit search
        size_t consecutive = 0;
        size_t candidate_start = 0;

        for (size_t i = 0; i < total_frames_; ++i) {
            if (!GetBit(i)) {
                if (consecutive == 0) {
                    candidate_start = i;
                }
                ++consecutive;
                if (consecutive == num_frames) {
                    // Mark all as allocated
                    for (size_t j = 0; j < num_frames; ++j) {
                        SetBit(candidate_start + j, true);
                    }
                    return candidate_start;
                }
            } else {
                consecutive = 0;
            }
        }
        return kNullFrame; // Out of memory
    }

    void Free(size_t start_frame, size_t num_frames) {
        for (size_t j = 0; j < num_frames; ++j) {
            SetBit(start_frame + j, false);
        }
    }

    size_t TotalFrames() const noexcept { return total_frames_; }
    size_t BitmapByteSize() const noexcept { return bitmap_.size() * sizeof(uint64_t); }

    static constexpr size_t kNullFrame = static_cast<size_t>(-1);

private:
    size_t total_frames_;
    std::vector<uint64_t> bitmap_;
};

} // namespace mikanos::study

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    using namespace mikanos::study;

    std::cout << "=== MikanOS Chapter 8.1: Role of OS Memory Management (C++20 Sim) ===\n";

    // 1. Physical Page Frame Address Mathematics Verification
    assert(MemoryAddressMath::PhysToFrame(0x0000) == 0);
    assert(MemoryAddressMath::PhysToFrame(0x1000) == 1);
    assert(MemoryAddressMath::PhysToFrame(0x1FFF) == 1);
    assert(MemoryAddressMath::PhysToFrame(0x2000) == 2);

    assert(MemoryAddressMath::FrameToPhys(0) == 0x0000);
    assert(MemoryAddressMath::FrameToPhys(1) == 0x1000);
    assert(MemoryAddressMath::FrameToPhys(256) == 0x100000); // 1 MiB

    assert(MemoryAddressMath::IsPageAligned(0x0000) == true);
    assert(MemoryAddressMath::IsPageAligned(0x1000) == true);
    assert(MemoryAddressMath::IsPageAligned(0x1008) == false);

    assert(MemoryAddressMath::AlignDown(0x1FEE) == 0x1000);
    assert(MemoryAddressMath::AlignUp(0x1001) == 0x2000);
    assert(MemoryAddressMath::AlignUp(0x2000) == 0x2000);

    std::cout << "[PASS] 4 KiB Page Frame arithmetic verified.\n";

    // 2. x86-64 48-bit Canonical Virtual Address Decomposition
    uintptr_t user_vaddr = 0x00007FFFFFFF1234ULL;
    auto user_parts = MemoryAddressMath::DecomposeVirtualAddress(user_vaddr);
    assert(user_parts.is_canonical == true);
    assert(user_parts.offset == 0x234);

    uintptr_t kernel_vaddr = 0xFFFF800012345678ULL;
    auto kernel_parts = MemoryAddressMath::DecomposeVirtualAddress(kernel_vaddr);
    assert(kernel_parts.is_canonical == true);
    assert(kernel_parts.pml4_index == 256); // Bit 47 is 1 -> index 256
    assert(kernel_parts.pdp_index == 0);
    assert(kernel_parts.pd_index == 145);
    assert(kernel_parts.pt_index == 325);
    assert(kernel_parts.offset == 0x678);

    uintptr_t non_canonical = 0x0000800000000000ULL;
    assert(MemoryAddressMath::IsCanonical(non_canonical) == false);

    std::cout << "[PASS] Canonical 48-bit virtual address decomposition verified.\n";

    // 3. Physical Frame Bitmap Allocator Simulation
    constexpr size_t kTestFrames = 512; // 2 MiB managed pool
    SimpleBitmapMemoryManager mm(kTestFrames);
    assert(mm.BitmapByteSize() == (512 / 64) * 8); // 64 bytes

    size_t f1 = mm.Allocate(4); // Allocate 4 frames (16 KiB)
    assert(f1 == 0);
    assert(mm.GetBit(0) == true && mm.GetBit(3) == true && mm.GetBit(4) == false);

    size_t f2 = mm.Allocate(2); // Allocate 2 frames
    assert(f2 == 4);

    mm.Free(f1, 4); // Free first 4 frames
    assert(mm.GetBit(0) == false && mm.GetBit(3) == false);
    assert(mm.GetBit(4) == true); // f2 still allocated

    size_t f3 = mm.Allocate(3); // First-fit reuses slot 0..2
    assert(f3 == 0);

    std::cout << "[PASS] Bitmap Memory Manager allocation and free verified.\n";

    // 4. Memory Footprint Calculation (4 GiB RAM -> 128 KiB Bitmap)
    constexpr size_t kRam4GiB = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr size_t kFrames4GiB = kRam4GiB / kPageSize; // 1,048,576 frames
    constexpr size_t kBitmapBytes4GiB = kFrames4GiB / 8; // 131,072 bytes = 128 KiB
    double overhead_pct = (static_cast<double>(kBitmapBytes4GiB) / kRam4GiB) * 100.0;

    std::cout << "4 GiB RAM Frames: " << kFrames4GiB << " frames\n";
    std::cout << "Bitmap size: " << kBitmapBytes4GiB / 1024 << " KiB\n";
    std::cout << "Overhead: " << std::fixed << std::setprecision(6) << overhead_pct << " %\n";
    assert(kBitmapBytes4GiB == 131072);

    std::cout << "✅ All Chapter 8.1 C++20 memory role verifications passed successfully!\n";
    return 0;
}
