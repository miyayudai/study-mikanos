/**
 * @file os_chapter4_exercises.cpp
 * @brief Chapter 4 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 4:
 * 1. Makefile Dependency Resolution & Automatic Variables ($@, $<, $^)
 * 2. FrameBufferConfig VRAM Coordinate Mapping & Scanline Stride Math
 * 3. PixelWriter Abstract Base Class & Placement new in Freestanding Environment
 * 4. Itanium C++ ABI Virtual Method Table (vtable), _vptr & __cxa_pure_virtual
 * 5. ELF64 PT_LOAD Segments & .bss Section Zero-Clearing Math
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch04 {

// =============================================================================
// Exercise 1: Makefile Dependency Graph & Automatic Variables
// =============================================================================

struct MakeTarget {
    std::string name;
    std::vector<std::string> prerequisites;
    std::string recipe;

    std::string expand_automatic_variables() const {
        std::string result = recipe;
        // Replace $@ with target name
        size_t pos = 0;
        while ((pos = result.find("$@")) != std::string::npos) {
            result.replace(pos, 2, name);
        }
        // Replace $< with first prerequisite
        while ((pos = result.find("$<")) != std::string::npos) {
            std::string first = prerequisites.empty() ? "" : prerequisites.front();
            result.replace(pos, 2, first);
        }
        // Replace $^ with all prerequisites joined with space
        while ((pos = result.find("$^")) != std::string::npos) {
            std::string all_prereqs;
            for (size_t i = 0; i < prerequisites.size(); ++i) {
                if (i > 0) all_prereqs += " ";
                all_prereqs += prerequisites[i];
            }
            result.replace(pos, 2, all_prereqs);
        }
        return result;
    }
};

// =============================================================================
// Exercise 2: FrameBufferConfig & 2D Coordinate Mapping Math
// =============================================================================

enum class PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};
};

struct FrameBufferConfig {
    uint8_t* frame_buffer{nullptr};
    uint32_t pixels_per_scan_line{0};
    uint32_t horizontal_resolution{0};
    uint32_t vertical_resolution{0};
    PixelFormat pixel_format{PixelFormat::kPixelRGBResv8BitPerColor};

    [[nodiscard]] size_t calculate_byte_offset(int x, int y) const noexcept {
        return 4 * (static_cast<size_t>(y) * pixels_per_scan_line + static_cast<size_t>(x));
    }

    [[nodiscard]] size_t total_buffer_bytes() const noexcept {
        return 4 * static_cast<size_t>(pixels_per_scan_line) * vertical_resolution;
    }
};

// =============================================================================
// Exercise 3: PixelWriter OOP & Placement new
// =============================================================================

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;

    [[nodiscard]] const FrameBufferConfig& GetConfig() const noexcept {
        return config_;
    }

protected:
    explicit PixelWriter(const FrameBufferConfig& config) : config_{config} {}
    FrameBufferConfig config_;
};

class RGBResv8BitPerColorPixelWriter : public PixelWriter {
public:
    explicit RGBResv8BitPerColorPixelWriter(const FrameBufferConfig& config)
        : PixelWriter{config} {}

    void Write(int x, int y, const PixelColor& c) override {
        size_t offset = config_.calculate_byte_offset(x, y);
        uint8_t* p = &config_.frame_buffer[offset];
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0; // Reserved
    }
};

class BGRResv8BitPerColorPixelWriter : public PixelWriter {
public:
    explicit BGRResv8BitPerColorPixelWriter(const FrameBufferConfig& config)
        : PixelWriter{config} {}

    void Write(int x, int y, const PixelColor& c) override {
        size_t offset = config_.calculate_byte_offset(x, y);
        uint8_t* p = &config_.frame_buffer[offset];
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0; // Reserved
    }
};

// =============================================================================
// Exercise 4: Itanium C++ ABI vtable & _vptr Layout Simulation
// =============================================================================

struct PolymorphicObjectLayout {
    uintptr_t vptr; // Offset 0: pointer to virtual table
    uint8_t member_data[8];
};

struct MockVTable {
    uintptr_t typeinfo_ptr;
    void (*write_fn)(void* instance, int x, int y, const PixelColor& c);
    void (*dtor_fn)(void* instance);
};

// =============================================================================
// Exercise 5: ELF64 PT_LOAD & .bss Zero-Clear Simulator
// =============================================================================

struct MockElf64Phdr {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
};

constexpr uint32_t PT_LOAD = 1;

struct SegmentLoadResult {
    uint64_t load_addr;
    uint64_t copy_bytes;
    uint64_t bss_zero_start;
    uint64_t bss_zero_bytes;
};

SegmentLoadResult simulate_load_segment(const MockElf64Phdr& phdr) {
    assert(phdr.p_type == PT_LOAD);
    SegmentLoadResult res{};
    res.load_addr = phdr.p_vaddr;
    res.copy_bytes = phdr.p_filesz;
    res.bss_zero_start = phdr.p_vaddr + phdr.p_filesz;
    res.bss_zero_bytes = (phdr.p_memsz > phdr.p_filesz) ? (phdr.p_memsz - phdr.p_filesz) : 0;
    return res;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_all_verifications() {
    std::cout << "====================================================================\n";
    std::cout << " [Study] MikanOS Chapter 4 Comprehensive Exercises Verification (C++20)\n";
    std::cout << "====================================================================\n\n";

    // 1. Makefile automatic variables verification
    {
        MakeTarget rule{
            .name = "kernel.elf",
            .prerequisites = {"main.o", "graphics.o"},
            .recipe = "ld.lld -o $@ $^"
        };
        std::string expanded = rule.expand_automatic_variables();
        assert(expanded == "ld.lld -o kernel.elf main.o graphics.o");
        std::cout << "[PASS] Exercise 1: Makefile variable expansion: " << expanded << "\n";
    }

    // 2. FrameBufferConfig 2D offset calculation
    {
        std::vector<uint8_t> vram(832 * 600 * 4, 0);
        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = 832,
            .horizontal_resolution = 800,
            .vertical_resolution = 600,
            .pixel_format = PixelFormat::kPixelRGBResv8BitPerColor,
        };

        size_t offset = config.calculate_byte_offset(100, 50);
        size_t expected_offset = 4 * (50 * 832 + 100);
        assert(offset == expected_offset);
        assert(offset == 166800);
        std::cout << "[PASS] Exercise 2: FrameBuffer coordinate math: (100, 50) -> offset "
                  << offset << " bytes (ppsl=832)\n";
    }

    // 3. PixelWriter OOP & Placement new
    {
        std::vector<uint8_t> vram(64 * 64 * 4, 0);
        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = 64,
            .horizontal_resolution = 64,
            .vertical_resolution = 64,
            .pixel_format = PixelFormat::kPixelRGBResv8BitPerColor,
        };

        alignas(alignof(RGBResv8BitPerColorPixelWriter))
            uint8_t writer_buf[sizeof(RGBResv8BitPerColorPixelWriter)];

        // Placement new in static buffer
        auto* writer = new (writer_buf) RGBResv8BitPerColorPixelWriter(config);
        PixelColor orange{255, 128, 0};
        writer->Write(10, 20, orange);

        size_t target_offset = config.calculate_byte_offset(10, 20);
        assert(vram[target_offset + 0] == 255); // R
        assert(vram[target_offset + 1] == 128); // G
        assert(vram[target_offset + 2] == 0);   // B
        assert(vram[target_offset + 3] == 0);   // Reserved

        writer->~RGBResv8BitPerColorPixelWriter();
        std::cout << "[PASS] Exercise 3: Placement new & RGB PixelWriter verified successfully\n";
    }

    // 4. BGR PixelWriter format verification
    {
        std::vector<uint8_t> vram(64 * 64 * 4, 0);
        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = 64,
            .horizontal_resolution = 64,
            .vertical_resolution = 64,
            .pixel_format = PixelFormat::kPixelBGRResv8BitPerColor,
        };

        alignas(alignof(BGRResv8BitPerColorPixelWriter))
            uint8_t writer_buf[sizeof(BGRResv8BitPerColorPixelWriter)];

        auto* writer = new (writer_buf) BGRResv8BitPerColorPixelWriter(config);
        PixelColor sky_blue{10, 20, 240};
        writer->Write(5, 5, sky_blue);

        size_t target_offset = config.calculate_byte_offset(5, 5);
        assert(vram[target_offset + 0] == 240); // B
        assert(vram[target_offset + 1] == 20);  // G
        assert(vram[target_offset + 2] == 10);  // R

        writer->~BGRResv8BitPerColorPixelWriter();
        std::cout << "[PASS] Exercise 4: BGR PixelWriter byte ordering [B, G, R, 0] verified\n";
    }

    // 5. Itanium C++ ABI vtable _vptr layout
    {
        std::vector<uint8_t> vram(16 * 16 * 4, 0);
        FrameBufferConfig config{
            .frame_buffer = vram.data(),
            .pixels_per_scan_line = 16,
            .horizontal_resolution = 16,
            .vertical_resolution = 16,
        };
        RGBResv8BitPerColorPixelWriter writer(config);
        // The first 8 bytes of writer instance MUST be the vptr
        uintptr_t* raw_vptr = reinterpret_cast<uintptr_t*>(&writer);
        assert(*raw_vptr != 0); // vptr points to valid vtable

        std::cout << "[PASS] Exercise 5: Itanium C++ ABI _vptr non-null: 0x"
                  << std::hex << *raw_vptr << std::dec << "\n";
    }

    // 6. ELF PT_LOAD & .bss zero clearing
    {
        MockElf64Phdr phdr{
            .p_type = PT_LOAD,
            .p_flags = 7,
            .p_offset = 0x1000,
            .p_vaddr = 0x100000,
            .p_paddr = 0x100000,
            .p_filesz = 0x2400,
            .p_memsz = 0x3000,
            .p_align = 0x1000,
        };

        auto res = simulate_load_segment(phdr);
        assert(res.load_addr == 0x100000);
        assert(res.copy_bytes == 0x2400);
        assert(res.bss_zero_start == 0x102400);
        assert(res.bss_zero_bytes == 0xC00); // 0x3000 - 0x2400 = 0xC00 (3072 bytes)

        std::cout << "[PASS] Exercise 6: ELF PT_LOAD .bss zero-clear range: [0x"
                  << std::hex << res.bss_zero_start << ", 0x"
                  << (res.bss_zero_start + res.bss_zero_bytes) << ") size="
                  << std::dec << res.bss_zero_bytes << " bytes\n";
    }

    std::cout << "\n====================================================================\n";
    std::cout << " All Chapter 4 Exercises C++20 Verifications PASSED!\n";
    std::cout << "====================================================================\n";
}

} // namespace mikanos::exercises::ch04

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    mikanos::exercises::ch04::run_all_verifications();

    if (test_mode) {
        std::cout << "Executed in --test mode.\n";
    }
    return 0;
}
