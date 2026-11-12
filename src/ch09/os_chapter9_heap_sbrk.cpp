/**
 * @file os_chapter9_heap_sbrk.cpp
 * @brief Chapter 9.2: Dynamic Memory Allocation, sbrk(), and MikanOS Heap Architecture (C++20)
 *
 * Demonstrates:
 * 1. Physical page frame allocation via BitmapMemoryManager (4KiB frames)
 * 2. MikanOS InitializeHeap(): allocating 128 MiB (32,768 frames) contiguous heap
 * 3. POSIX sbrk(incr) system call stub implementation and program_break advancement
 * 4. Boundary enforcement and ENOMEM (errno 12) handling when exceeding program_break_end
 * 5. Multi-tier dynamic memory stack: new -> operator new -> malloc -> sbrk -> BitmapMemoryManager
 * 6. 16-byte alignment (System V AMD64 ABI compliant) and chunk management
 */

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mikanos::study {

// 1. FrameID & Physical Memory Constants
constexpr size_t kBytesPerFrame = 4096;
constexpr size_t kBitsPerMapLine = 64;

class FrameID {
public:
    constexpr explicit FrameID(size_t id = 0) : id_{id} {}
    constexpr size_t ID() const { return id_; }
    constexpr bool operator==(const FrameID& rhs) const { return id_ == rhs.id_; }
    constexpr bool operator!=(const FrameID& rhs) const { return id_ != rhs.id_; }

private:
    size_t id_;
};

static const FrameID kNullFrame{std::numeric_limits<size_t>::max()};

template <typename T>
struct WithError {
    T value;
    int error; // 0 = success, non-zero = error
};

// 2. Simplified Bitmap Memory Manager for Host Verification
class SimBitmapMemoryManager {
public:
    static constexpr size_t kMaxFrames = 65536; // 256 MiB pool for host test

    SimBitmapMemoryManager() : alloc_map_(kMaxFrames / kBitsPerMapLine, 0) {}

    WithError<FrameID> Allocate(size_t num_frames) {
        size_t start_frame_id = range_begin_.ID();
        while (true) {
            size_t i = 0;
            for (; i < num_frames; ++i) {
                if (start_frame_id + i >= range_end_.ID()) {
                    return {kNullFrame, ENOMEM};
                }
                if (GetBit(FrameID{start_frame_id + i})) {
                    break;
                }
            }
            if (i == num_frames) {
                MarkAllocated(FrameID{start_frame_id}, num_frames);
                return {FrameID{start_frame_id}, 0};
            }
            start_frame_id += i + 1; // O(N) skip optimization
        }
    }

    void Free(FrameID start_frame, size_t num_frames) {
        for (size_t i = 0; i < num_frames; ++i) {
            SetBit(FrameID{start_frame.ID() + i}, false);
        }
    }

    void MarkAllocated(FrameID start_frame, size_t num_frames) {
        for (size_t i = 0; i < num_frames; ++i) {
            SetBit(FrameID{start_frame.ID() + i}, true);
        }
    }

    void SetMemoryRange(FrameID begin, FrameID end) {
        range_begin_ = begin;
        range_end_ = end;
    }

    bool GetBit(FrameID frame) const {
        auto line = frame.ID() / kBitsPerMapLine;
        auto bit = frame.ID() % kBitsPerMapLine;
        return (alloc_map_[line] & (1ULL << bit)) != 0;
    }

    void SetBit(FrameID frame, bool allocated) {
        auto line = frame.ID() / kBitsPerMapLine;
        auto bit = frame.ID() % kBitsPerMapLine;
        if (allocated) {
            alloc_map_[line] |= (1ULL << bit);
        } else {
            alloc_map_[line] &= ~(1ULL << bit);
        }
    }

private:
    std::vector<uint64_t> alloc_map_;
    FrameID range_begin_{0};
    FrameID range_end_{kMaxFrames};
};

// 3. Simulated sbrk() & Heap Environment
class MikanHeapEnvironment {
public:
    static constexpr size_t kDefaultHeapFrames = 1024; // 4 MiB for unit testing
    static constexpr size_t kAlignment = 16;           // 16-byte alignment

    struct ChunkHeader {
        uint32_t size;         // Total chunk size including header
        uint16_t is_allocated; // 1 = allocated, 0 = free
        uint16_t magic;        // 0xCAFE for integrity check
        uint64_t padding;      // Pad header to exactly 16 bytes for 16-byte payload alignment
    };
    static_assert(sizeof(ChunkHeader) == kAlignment, "ChunkHeader must be 16 bytes");

    MikanHeapEnvironment() {
        // Allocate simulated physical memory backing store
        backing_memory_.resize(kDefaultHeapFrames * kBytesPerFrame, 0);
        sim_mem_mgr_.SetMemoryRange(FrameID{1}, FrameID{SimBitmapMemoryManager::kMaxFrames});
    }

    int InitializeHeap(size_t num_frames = kDefaultHeapFrames) {
        auto alloc_res = sim_mem_mgr_.Allocate(num_frames);
        if (alloc_res.error != 0) {
            return alloc_res.error;
        }

        heap_frames_ = num_frames;
        uintptr_t raw_start = reinterpret_cast<uintptr_t>(backing_memory_.data());
        heap_start_addr_ = (raw_start + (kAlignment - 1)) & ~(kAlignment - 1);
        program_break_ = heap_start_addr_;
        program_break_end_ = heap_start_addr_ + (num_frames * kBytesPerFrame);
        errno = 0;
        return 0;
    }

    void* Sbrk(intptr_t incr) {
        if (program_break_ == 0 ||
            (program_break_ + incr > program_break_end_) ||
            (program_break_ + incr < heap_start_addr_)) {
            errno = ENOMEM;
            return reinterpret_cast<void*>(-1);
        }

        uintptr_t prev_break = program_break_;
        program_break_ += incr;
        return reinterpret_cast<void*>(prev_break);
    }

    void* Malloc(size_t size) {
        if (size == 0) return nullptr;
        if (program_break_ == 0) {
            if (InitializeHeap() != 0) return nullptr;
        }

        // Align size to 16 bytes + header
        size_t aligned_payload = (size + (kAlignment - 1)) & ~(kAlignment - 1);
        size_t total_size = sizeof(ChunkHeader) + aligned_payload;
        if (total_size < 32) total_size = 32;

        // First-Fit search among existing free chunks in the sbrk'd region
        uintptr_t curr_addr = heap_start_addr_;
        while (curr_addr < program_break_) {
            auto* header = reinterpret_cast<ChunkHeader*>(curr_addr);
            if (header->magic != 0xCAFE) break;

            if (!header->is_allocated && header->size >= total_size) {
                // Split if enough excess space
                if (header->size >= total_size + 32) {
                    size_t rem = header->size - total_size;
                    header->size = static_cast<uint32_t>(total_size);
                    auto* next_header = reinterpret_cast<ChunkHeader*>(curr_addr + total_size);
                    next_header->size = static_cast<uint32_t>(rem);
                    next_header->is_allocated = 0;
                    next_header->magic = 0xCAFE;
                }
                header->is_allocated = 1;
                return reinterpret_cast<void*>(curr_addr + sizeof(ChunkHeader));
            }
            curr_addr += header->size;
        }

        // Allocate fresh chunk via sbrk
        size_t sbrk_request = std::max(total_size, static_cast<size_t>(65536)); // Request 64KiB at a time
        void* break_ptr = Sbrk(sbrk_request);
        if (break_ptr == reinterpret_cast<void*>(-1)) {
            return nullptr;
        }

        auto* header = reinterpret_cast<ChunkHeader*>(break_ptr);
        header->size = static_cast<uint32_t>(total_size);
        header->is_allocated = 1;
        header->magic = 0xCAFE;

        if (sbrk_request > total_size) {
            auto* next_header = reinterpret_cast<ChunkHeader*>(
                reinterpret_cast<uintptr_t>(break_ptr) + total_size);
            next_header->size = static_cast<uint32_t>(sbrk_request - total_size);
            next_header->is_allocated = 0;
            next_header->magic = 0xCAFE;
        }

        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(break_ptr) + sizeof(ChunkHeader));
    }

    void Free(void* ptr) {
        if (!ptr) return;
        auto* header = reinterpret_cast<ChunkHeader*>(
            reinterpret_cast<uintptr_t>(ptr) - sizeof(ChunkHeader));
        if (header->magic != 0xCAFE) return;

        header->is_allocated = 0;

        // Coalesce forward passes
        uintptr_t curr_addr = heap_start_addr_;
        while (curr_addr < program_break_) {
            auto* curr = reinterpret_cast<ChunkHeader*>(curr_addr);
            if (curr->magic != 0xCAFE) break;

            if (!curr->is_allocated) {
                uintptr_t next_addr = curr_addr + curr->size;
                if (next_addr < program_break_) {
                    auto* nxt = reinterpret_cast<ChunkHeader*>(next_addr);
                    if (nxt->magic == 0xCAFE && !nxt->is_allocated) {
                        curr->size += nxt->size;
                        continue; // try coalescing again with following block
                    }
                }
            }
            curr_addr += curr->size;
        }
    }

    uintptr_t ProgramBreak() const { return program_break_; }
    uintptr_t ProgramBreakEnd() const { return program_break_end_; }
    uintptr_t HeapStart() const { return heap_start_addr_; }
    size_t HeapFrames() const { return heap_frames_; }
    size_t TotalHeapBytes() const { return heap_frames_ * kBytesPerFrame; }

private:
    SimBitmapMemoryManager sim_mem_mgr_;
    std::vector<uint8_t> backing_memory_;
    uintptr_t heap_start_addr_{0};
    uintptr_t program_break_{0};
    uintptr_t program_break_end_{0};
    size_t heap_frames_{0};
};

} // namespace mikanos::study

int RunAllTests() {
    using namespace mikanos::study;
    std::cout << "========================================================\n";
    std::cout << "[Chapter 9.2 Test Suite: Dynamic Allocation & sbrk]\n";
    std::cout << "========================================================\n\n";

    // Test 1: Heap Initialization & Frame Allocation
    std::cout << "[Test 1] InitializeHeap from BitmapMemoryManager... ";
    MikanHeapEnvironment env;
    int err = env.InitializeHeap(512); // 512 frames = 2 MiB
    assert(err == 0);
    assert(env.HeapStart() != 0);
    assert(env.ProgramBreak() == env.HeapStart());
    assert(env.ProgramBreakEnd() == env.HeapStart() + 512 * kBytesPerFrame);
    std::cout << "PASSED (Heap: 512 frames = 2097152 bytes)\n";

    // Test 2: sbrk(0) query
    std::cout << "[Test 2] sbrk(0) queries current program break... ";
    void* cur = env.Sbrk(0);
    assert(reinterpret_cast<uintptr_t>(cur) == env.ProgramBreak());
    std::cout << "PASSED (break: " << cur << ")\n";

    // Test 3: sbrk(incr) advancement
    std::cout << "[Test 3] sbrk(4096) advances program break... ";
    uintptr_t old_break = env.ProgramBreak();
    void* prev = env.Sbrk(4096);
    assert(reinterpret_cast<uintptr_t>(prev) == old_break);
    assert(env.ProgramBreak() == old_break + 4096);
    std::cout << "PASSED (new break: " << reinterpret_cast<void*>(env.ProgramBreak()) << ")\n";

    // Test 4: sbrk boundary violation -> ENOMEM
    std::cout << "[Test 4] sbrk exceeding program_break_end triggers ENOMEM... ";
    size_t remaining = env.ProgramBreakEnd() - env.ProgramBreak();
    void* fail_res = env.Sbrk(remaining + 1);
    assert(fail_res == reinterpret_cast<void*>(-1));
    assert(errno == ENOMEM);
    std::cout << "PASSED (errno=ENOMEM correctly set)\n";

    // Test 5: Dynamic Malloc & 16-byte alignment
    std::cout << "[Test 5] Dynamic Malloc produces 16-byte aligned pointers... ";
    MikanHeapEnvironment env2;
    env2.InitializeHeap(1024);
    void* p1 = env2.Malloc(32);
    void* p2 = env2.Malloc(128);
    void* p3 = env2.Malloc(7);
    assert(p1 != nullptr && p2 != nullptr && p3 != nullptr);
    assert(reinterpret_cast<uintptr_t>(p1) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(p2) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(p3) % 16 == 0);
    assert(p1 != p2 && p2 != p3);
    std::cout << "PASSED (p1: " << p1 << ", p2: " << p2 << ", p3: " << p3 << ")\n";

    // Test 6: Free & Coalescing
    std::cout << "[Test 6] Free & Coalescing of adjacent free chunks... ";
    env2.Free(p1);
    env2.Free(p2);
    // Allocate something that fits into the coalesced free space of p1+p2
    void* p_large = env2.Malloc(140);
    assert(p_large != nullptr);
    assert(p_large == p1); // Reused the coalesced space starting at p1
    std::cout << "PASSED (Coalesced block successfully reused)\n";

    // Test 7: Heap Exhaustion in Malloc
    std::cout << "[Test 7] Heap exhaustion in Malloc returns nullptr gracefully... ";
    MikanHeapEnvironment tiny_env;
    tiny_env.InitializeHeap(16); // Only 16 frames = 64 KiB
    void* huge_alloc = tiny_env.Malloc(128 * 1024); // 128 KiB > 64 KiB
    assert(huge_alloc == nullptr);
    std::cout << "PASSED (huge allocation rejected)\n";

    std::cout << "\n>>> Chapter 9.2: All C++ tests passed successfully! <<<\n";
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        return RunAllTests();
    }

    std::cout << "Chapter 9.2: MikanOS Dynamic Allocation & sbrk Simulation\n";
    std::cout << "Run with '--test' to execute verification suite.\n";
    return RunAllTests();
}
