/**
 * @file pixel_writer_oop_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 4.3 (Refactoring with C++ OOP & Placement New)
 *
 * Covers:
 * 1. PixelWriter abstract base class: FrameBufferConfig reference member, virtual destructor, pure virtual Write()
 * 2. Derived classes: RGBResv8BitPerColorPixelWriter & BGRResv8BitPerColorPixelWriter with inheriting constructors
 * 3. Placement new in freestanding environment: custom operator new(size_t, void*) and dummy operator delete
 * 4. Static memory allocation: pixel_writer_buf sizing and 64-bit alignment verification
 * 5. Polymorphic rendering: drawing background and rectangle via PixelWriter* base pointer
 * 6. Column 4.2 principles: Compile Errors are your Friends (abstract class check, virtual destructor, type safety)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace mikan::ch04_oop {

// ============================================================================
// 1. MikanOS Day04c Core Types & PixelWriter Class Hierarchy
// ============================================================================

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    enum PixelFormat pixel_format;
};

// Compile-time verification of FrameBufferConfig
static_assert(sizeof(FrameBufferConfig) == 24, "FrameBufferConfig size must be exactly 24 bytes in 64-bit ABI");
static_assert(alignof(FrameBufferConfig) == 8, "FrameBufferConfig alignment must be 8 bytes");

struct PixelColor {
    uint8_t r, g, b;
};

/**
 * @brief Abstract base class for pixel rendering (matching MikanOS PixelWriter).
 */
class PixelWriter {
public:
    explicit PixelWriter(const FrameBufferConfig& config) : config_{config} {}
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;

    [[nodiscard]] const FrameBufferConfig& GetConfig() const noexcept { return config_; }

protected:
    uint8_t* PixelAt(int x, int y) {
        return config_.frame_buffer + 4 * (config_.pixels_per_scan_line * y + x);
    }

private:
    const FrameBufferConfig& config_;
};

/**
 * @brief PixelWriter for RGB pixel format (kPixelRGBResv8BitPerColor).
 */
class RGBResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;

    void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0;
    }
};

/**
 * @brief PixelWriter for BGR pixel format (kPixelBGRResv8BitPerColor).
 */
class BGRResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;

    void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0;
    }
};

// ============================================================================
// 2. Compile-Time Memory Layout & Object Size Verifications
// ============================================================================

// PixelWriter has a 64-bit vptr (8 bytes) + 64-bit reference pointer (8 bytes) = 16 bytes
static_assert(sizeof(PixelWriter) == 16, "PixelWriter size must be 16 bytes (8B vptr + 8B reference)");
static_assert(alignof(PixelWriter) == 8, "PixelWriter alignment must be 8 bytes");

static_assert(sizeof(RGBResv8BitPerColorPixelWriter) == 16, "RGB Writer size must be 16 bytes");
static_assert(sizeof(BGRResv8BitPerColorPixelWriter) == 16, "BGR Writer size must be 16 bytes");
static_assert(alignof(RGBResv8BitPerColorPixelWriter) == 8, "RGB Writer alignment must be 8 bytes");
static_assert(alignof(BGRResv8BitPerColorPixelWriter) == 8, "BGR Writer alignment must be 8 bytes");

// Type traits verification (Column 4.2 compile safety)
static_assert(std::is_abstract_v<PixelWriter>, "PixelWriter must be an abstract class");
static_assert(!std::is_abstract_v<RGBResv8BitPerColorPixelWriter>, "RGB Writer must be concrete");
static_assert(!std::is_abstract_v<BGRResv8BitPerColorPixelWriter>, "BGR Writer must be concrete");
static_assert(std::has_virtual_destructor_v<PixelWriter>, "PixelWriter must have a virtual destructor");
static_assert(std::is_base_of_v<PixelWriter, RGBResv8BitPerColorPixelWriter>, "RGB Writer must inherit from PixelWriter");
static_assert(std::is_base_of_v<PixelWriter, BGRResv8BitPerColorPixelWriter>, "BGR Writer must inherit from PixelWriter");

// ============================================================================
// 3. Placement New in Bare-Metal Freestanding Environment
// ============================================================================

// Static storage buffer matching MikanOS kernel/main.cpp
alignas(void*) char pixel_writer_buf[sizeof(RGBResv8BitPerColorPixelWriter)];
PixelWriter* pixel_writer = nullptr;

// Custom placement operator new and delete implementation
inline void* CustomPlacementNew(size_t /*size*/, void* buf) noexcept {
    return buf;
}

inline void CustomPlacementDelete(void* /*obj*/) noexcept {
    // No-op for static memory
}

// ============================================================================
// 4. Verification Test Cases
// ============================================================================

void TestPlacementNewConstruction() {
    std::cout << "[Test 1] Testing Placement New Construction on Static Buffer... ";

    std::vector<uint8_t> dummy_vram(800 * 600 * 4, 0);
    FrameBufferConfig config{
        .frame_buffer = dummy_vram.data(),
        .pixels_per_scan_line = 800,
        .horizontal_resolution = 800,
        .vertical_resolution = 600,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    // Construct RGBResv8BitPerColorPixelWriter in pixel_writer_buf
    void* memory_slot = CustomPlacementNew(sizeof(RGBResv8BitPerColorPixelWriter), pixel_writer_buf);
    assert(memory_slot == static_cast<void*>(pixel_writer_buf));

    pixel_writer = new (memory_slot) RGBResv8BitPerColorPixelWriter{config};
    assert(pixel_writer != nullptr);
    assert(static_cast<void*>(pixel_writer) == static_cast<void*>(pixel_writer_buf));
    assert(&pixel_writer->GetConfig() == &config);

    // Write a pixel and verify
    pixel_writer->Write(10, 20, PixelColor{0x12, 0x34, 0x56});
    size_t offset = 4 * (800 * 20 + 10);
    assert(dummy_vram[offset + 0] == 0x12);
    assert(dummy_vram[offset + 1] == 0x34);
    assert(dummy_vram[offset + 2] == 0x56);
    assert(dummy_vram[offset + 3] == 0x00);

    // Call destructor manually (matching placement new lifecycle)
    pixel_writer->~PixelWriter();
    CustomPlacementDelete(pixel_writer);
    pixel_writer = nullptr;

    std::cout << "PASSED" << std::endl;
}

void TestRGBAndBGRPixelWriters() {
    std::cout << "[Test 2] Testing RGB vs BGR Polymorphic Pixel Writers... ";

    std::vector<uint8_t> vram(100 * 100 * 4, 0);
    FrameBufferConfig rgb_config{
        .frame_buffer = vram.data(),
        .pixels_per_scan_line = 100,
        .horizontal_resolution = 100,
        .vertical_resolution = 100,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    RGBResv8BitPerColorPixelWriter rgb_writer{rgb_config};
    PixelWriter* writer_ptr = &rgb_writer;

    // Test RGB channel placement: [R, G, B, 0]
    PixelColor red{255, 0, 0};
    writer_ptr->Write(5, 5, red);
    size_t off_rgb = 4 * (100 * 5 + 5);
    assert(vram[off_rgb + 0] == 255);
    assert(vram[off_rgb + 1] == 0);
    assert(vram[off_rgb + 2] == 0);

    // Reconstruct with BGR writer
    FrameBufferConfig bgr_config = rgb_config;
    bgr_config.pixel_format = kPixelBGRResv8BitPerColor;
    BGRResv8BitPerColorPixelWriter bgr_writer{bgr_config};
    writer_ptr = &bgr_writer;

    // Test BGR channel placement: [B, G, R, 0]
    writer_ptr->Write(15, 15, red);
    size_t off_bgr = 4 * (100 * 15 + 15);
    assert(vram[off_bgr + 0] == 0);    // Blue
    assert(vram[off_bgr + 1] == 0);    // Green
    assert(vram[off_bgr + 2] == 255);  // Red

    std::cout << "PASSED" << std::endl;
}

void TestDay04cDrawingSimulation() {
    std::cout << "[Test 3] Testing Day04c Drawing (White Background & Green Rectangle)... ";

    const uint32_t width = 800;
    const uint32_t height = 600;
    std::vector<uint8_t> vram(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = vram.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    // Instantiate via placement new
    pixel_writer = new (pixel_writer_buf) RGBResv8BitPerColorPixelWriter{config};

    // 1. Fill entire screen with white (255, 255, 255)
    for (uint32_t x = 0; x < config.horizontal_resolution; ++x) {
        for (uint32_t y = 0; y < config.vertical_resolution; ++y) {
            pixel_writer->Write(static_cast<int>(x), static_cast<int>(y), {255, 255, 255});
        }
    }

    // 2. Draw green rectangle (200x100) at (100, 100) with color (0, 255, 0)
    for (int x = 0; x < 200; ++x) {
        for (int y = 0; y < 100; ++y) {
            pixel_writer->Write(100 + x, 100 + y, {0, 255, 0});
        }
    }

    // Verify background pixel (0, 0) is white
    size_t off_bg = 4 * (width * 0 + 0);
    assert(vram[off_bg + 0] == 255 && vram[off_bg + 1] == 255 && vram[off_bg + 2] == 255);

    // Verify inside rectangle (150, 150) is green
    size_t off_rect = 4 * (width * 150 + 150);
    assert(vram[off_rect + 0] == 0 && vram[off_rect + 1] == 255 && vram[off_rect + 2] == 0);

    // Verify outside rectangle (350, 150) is still white
    size_t off_out = 4 * (width * 150 + 350);
    assert(vram[off_out + 0] == 255 && vram[off_out + 1] == 255 && vram[off_out + 2] == 255);

    pixel_writer->~PixelWriter();
    pixel_writer = nullptr;

    std::cout << "PASSED" << std::endl;
}

void TestColumn42CompileTimeSafety() {
    std::cout << "[Test 4] Testing Column 4.2 Compile-Time Safety Principles... ";

    // Verify that PixelWriter cannot be directly instantiated (pure virtual function Write)
    static_assert(std::is_abstract_v<PixelWriter>, "Column 4.2: Abstract class cannot be instantiated");

    // Verify that deleting through base pointer is safe (virtual destructor)
    static_assert(std::has_virtual_destructor_v<PixelWriter>, "Column 4.2: Virtual destructor is required for base class");

    // Verify derived class sizes are identical (no slicing issues with uniform buffer)
    static_assert(sizeof(RGBResv8BitPerColorPixelWriter) == sizeof(BGRResv8BitPerColorPixelWriter));
    static_assert(sizeof(pixel_writer_buf) >= sizeof(RGBResv8BitPerColorPixelWriter));

    std::cout << "PASSED" << std::endl;
}

void BenchmarkBranchVsVtable() {
    std::cout << "[Test 5] Benchmarking If-Else Branching vs vtable Polymorphism... " << std::flush;

    const uint32_t width = 800;
    const uint32_t height = 600;
    std::vector<uint8_t> vram(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = vram.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    // If-Else approach (day04b style)
    auto start_branch = std::chrono::high_resolution_clock::now();
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t* p = &config.frame_buffer[4 * (config.pixels_per_scan_line * y + x)];
            if (config.pixel_format == kPixelRGBResv8BitPerColor) {
                p[0] = 255;
                p[1] = 255;
                p[2] = 255;
            } else if (config.pixel_format == kPixelBGRResv8BitPerColor) {
                p[0] = 255;
                p[1] = 255;
                p[2] = 255;
            }
        }
    }
    auto end_branch = std::chrono::high_resolution_clock::now();
    double branch_ms = std::chrono::duration<double, std::milli>(end_branch - start_branch).count();

    // Vtable polymorphism approach (day04c style)
    RGBResv8BitPerColorPixelWriter writer{config};
    PixelWriter* pwriter = &writer;
    auto start_vtable = std::chrono::high_resolution_clock::now();
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            pwriter->Write(static_cast<int>(x), static_cast<int>(y), {255, 255, 255});
        }
    }
    auto end_vtable = std::chrono::high_resolution_clock::now();
    double vtable_ms = std::chrono::duration<double, std::milli>(end_vtable - start_vtable).count();

    std::cout << "PASSED (Branch: " << std::fixed << std::setprecision(2) << branch_ms
              << " ms, vtable: " << vtable_ms << " ms for 480k pixels)" << std::endl;
}

}  // namespace mikan::ch04_oop

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " MikanOS Chapter 4.3: Refactoring with C++ OOP & Placement New   " << std::endl;
    std::cout << "=================================================================" << std::endl;

    mikan::ch04_oop::TestPlacementNewConstruction();
    mikan::ch04_oop::TestRGBAndBGRPixelWriters();
    mikan::ch04_oop::TestDay04cDrawingSimulation();
    mikan::ch04_oop::TestColumn42CompileTimeSafety();
    mikan::ch04_oop::BenchmarkBranchVsVtable();

    std::cout << "=================================================================" << std::endl;
    std::cout << " All Section 4.3 C++20 Verification Tests Passed Successfully!   " << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
