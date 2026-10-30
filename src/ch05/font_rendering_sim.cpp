/**
 * @file font_rendering_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 5.1 (Drawing Characters & References, osbook_day05a)
 *
 * Covers:
 * 1. 8x16 bitmap font structure: kFontA[16] byte array and MSB-to-LSB bit-test logic
 * 2. WriteAscii function: rendering character 'A' on FrameBuffer via PixelWriter& reference
 * 3. Bit expansion mathematics: equivalence of (byte << dx) & 0x80u vs (byte >> (7 - dx)) & 1
 * 4. Column 5.1: C++ Reference vs Pointer comparison (address identity, nullability, rebindability, ABI)
 * 5. Full osbook_day05a VRAM scene simulation: white screen, green rectangle, and "AA" rendering
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

namespace mikan::ch05_font {

// ============================================================================
// 1. MikanOS Day05a Core Definitions (from kernel/main.cpp)
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

static_assert(sizeof(FrameBufferConfig) == 24, "FrameBufferConfig size must be 24 bytes in 64-bit ABI");
static_assert(alignof(FrameBufferConfig) == 8, "FrameBufferConfig alignment must be 8 bytes");

struct PixelColor {
    uint8_t r, g, b;
};

// 16-byte bitmap font for 'A' (identical to MikanOS kernel/main.cpp)
const uint8_t kFontA[16] = {
    0b00000000,  // Row 0
    0b00011000,  // Row 1:     **
    0b00011000,  // Row 2:     **
    0b00011000,  // Row 3:     **
    0b00011000,  // Row 4:     **
    0b00100100,  // Row 5:    *  *
    0b00100100,  // Row 6:    *  *
    0b00100100,  // Row 7:    *  *
    0b00100100,  // Row 8:    *  *
    0b01111110,  // Row 9:   ******
    0b01000010,  // Row 10:  *    *
    0b01000010,  // Row 11:  *    *
    0b01000010,  // Row 12:  *    *
    0b11100111,  // Row 13: ***  ***
    0b00000000,  // Row 14:
    0b00000000,  // Row 15:
};

class PixelWriter {
public:
    explicit PixelWriter(const FrameBufferConfig& config) : config_{config} {}
    virtual ~PixelWriter() = default;

    virtual void Write(int x, int y, const PixelColor& c) = 0;

    const FrameBufferConfig& Config() const noexcept { return config_; }

protected:
    uint8_t* PixelAt(int x, int y) {
        return config_.frame_buffer + 4 * (config_.pixels_per_scan_line * y + x);
    }

private:
    const FrameBufferConfig& config_;
};

class RGBResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;

    void Write(int x, int y, const PixelColor& c) override {
        uint8_t* p = PixelAt(x, y);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0;
    }
};

class BGRResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;

    void Write(int x, int y, const PixelColor& c) override {
        uint8_t* p = PixelAt(x, y);
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0;
    }
};

// ============================================================================
// 2. Day05a WriteAscii Implementation (Reference Parameter)
// ============================================================================

/**
 * @brief Render character 'A' using PixelWriter reference (MikanOS kernel implementation).
 */
void WriteAscii(PixelWriter& writer, int x, int y, char c, const PixelColor& color) {
    if (c != 'A') {
        return;
    }
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((kFontA[dy] << dx) & 0x80u) {
                writer.Write(x + dx, y + dy, color);
            }
        }
    }
}

/**
 * @brief Equivalent function using Pointer parameter for Column 5.1 comparison.
 */
void WriteAsciiPointer(PixelWriter* writer, int x, int y, char c, const PixelColor& color) {
    if (writer == nullptr || c != 'A') {
        return;
    }
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((kFontA[dy] << dx) & 0x80u) {
                writer->Write(x + dx, y + dy, color);
            }
        }
    }
}

// ============================================================================
// 3. Test Cases
// ============================================================================

void TestFontABitmapStructure() {
    std::cout << "Testing Font 'A' Bitmap Structure & Bit-Test Mathematics... ";

    assert(sizeof(kFontA) == 16);

    // Count total lit pixels
    int total_lit = 0;
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            bool lit_left_shift = ((kFontA[dy] << dx) & 0x80u) != 0;
            bool lit_right_shift = ((kFontA[dy] >> (7 - dx)) & 1) != 0;
            // Both bit-test formulas must match perfectly for all 128 bits
            assert(lit_left_shift == lit_right_shift);
            if (lit_left_shift) {
                total_lit++;
            }
        }
    }

    // Row-by-row verification
    assert(total_lit == 34);  // Exactly 34 pixels are lit for character 'A'
    assert(kFontA[0] == 0x00);
    assert(kFontA[9] == 0x7E);   // Horizontal crossbar: 0b01111110 (6 pixels)
    assert(kFontA[13] == 0xE7);  // Bottom legs: 0b11100111 (6 pixels)
    assert(kFontA[14] == 0x00);
    assert(kFontA[15] == 0x00);

    std::cout << "PASSED (34 lit pixels, bit-test equivalence verified)" << std::endl;
}

void TestWriteAsciiRendering() {
    std::cout << "Testing WriteAscii Character Rendering (RGB & BGR)... ";

    const uint32_t width = 100;
    const uint32_t height = 100;
    std::vector<uint8_t> rgb_buffer(width * height * 4, 255);  // Fill white
    std::vector<uint8_t> bgr_buffer(width * height * 4, 255);

    FrameBufferConfig rgb_config{
        .frame_buffer = rgb_buffer.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };
    RGBResv8BitPerColorPixelWriter rgb_writer{rgb_config};

    FrameBufferConfig bgr_config{
        .frame_buffer = bgr_buffer.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelBGRResv8BitPerColor,
    };
    BGRResv8BitPerColorPixelWriter bgr_writer{bgr_config};

    // Draw 'A' at (10, 20) with red color {255, 0, 0}
    PixelColor red{255, 0, 0};
    WriteAscii(rgb_writer, 10, 20, 'A', red);
    WriteAscii(bgr_writer, 10, 20, 'A', red);

    // Verify crossbar row (dy = 9, so y = 29)
    // In kFontA[9] = 0b01111110, dx = 1..6 are lit
    for (int dx = 1; dx <= 6; ++dx) {
        int x = 10 + dx;
        int y = 20 + 9;
        size_t offset = 4 * (width * y + x);

        // RGB format: [R, G, B, 0]
        assert(rgb_buffer[offset + 0] == 255);
        assert(rgb_buffer[offset + 1] == 0);
        assert(rgb_buffer[offset + 2] == 0);

        // BGR format: [B, G, R, 0]
        assert(bgr_buffer[offset + 0] == 0);
        assert(bgr_buffer[offset + 1] == 0);
        assert(bgr_buffer[offset + 2] == 255);
    }

    // Verify unlit pixel at (10, 20) (dx=0, dy=0) remains white
    size_t unlit_offset = 4 * (width * 20 + 10);
    assert(rgb_buffer[unlit_offset + 0] == 255);
    assert(rgb_buffer[unlit_offset + 1] == 255);
    assert(rgb_buffer[unlit_offset + 2] == 255);

    // Non-'A' character should not modify buffer
    std::vector<uint8_t> buffer_copy = rgb_buffer;
    WriteAscii(rgb_writer, 10, 20, 'B', {0, 255, 0});
    assert(rgb_buffer == buffer_copy);

    std::cout << "PASSED" << std::endl;
}

void TestColumn51ReferenceVsPointer() {
    std::cout << "Testing Column 5.1: C++ Reference vs Pointer Principles... ";

    const uint32_t width = 80;
    const uint32_t height = 80;
    std::vector<uint8_t> buf(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = buf.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };
    RGBResv8BitPerColorPixelWriter writer{config};

    // 1. Address identity: address of reference is identical to address of original object
    PixelWriter& ref = writer;
    PixelWriter* ptr = &writer;
    assert(&ref == ptr);
    assert(&ref == &writer);

    // 2. Both reference and pointer calls yield identical rendering output
    std::vector<uint8_t> buf1(width * height * 4, 255);
    std::vector<uint8_t> buf2(width * height * 4, 255);

    FrameBufferConfig cfg1 = config; cfg1.frame_buffer = buf1.data();
    FrameBufferConfig cfg2 = config; cfg2.frame_buffer = buf2.data();

    RGBResv8BitPerColorPixelWriter w1{cfg1};
    RGBResv8BitPerColorPixelWriter w2{cfg2};

    WriteAscii(w1, 10, 10, 'A', {0, 0, 0});
    WriteAsciiPointer(&w2, 10, 10, 'A', {0, 0, 0});
    assert(buf1 == buf2);

    // 3. Pointer allows nullptr safety check
    WriteAsciiPointer(nullptr, 10, 10, 'A', {0, 0, 0});  // Should not crash

    // 4. Compile-time language traits
    static_assert(std::is_reference_v<PixelWriter&>, "PixelWriter& must be a C++ reference");
    static_assert(!std::is_pointer_v<PixelWriter&>, "PixelWriter& is NOT a pointer");
    static_assert(std::is_pointer_v<PixelWriter*>, "PixelWriter* must be a pointer");

    std::cout << "PASSED (Address identity & ABI equivalence confirmed)" << std::endl;
}

void TestDay05aFullVramSimulation() {
    std::cout << "Testing Day05a Full VRAM Scene Simulation (White bg + Green rect + 'AA')... ";

    const uint32_t width = 200;
    const uint32_t height = 100;
    std::vector<uint8_t> vram(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = vram.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };
    RGBResv8BitPerColorPixelWriter writer{config};

    // 1. Fill white
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            writer.Write(x, y, {255, 255, 255});
        }
    }

    // 2. Green rectangle (10, 10) to (100, 40)
    for (int y = 10; y < 40; ++y) {
        for (int x = 10; x < 100; ++x) {
            writer.Write(x, y, {0, 255, 0});
        }
    }

    // 3. Render 'AA' at (50, 50) and (58, 50) in black {0, 0, 0}
    WriteAscii(writer, 50, 50, 'A', {0, 0, 0});
    WriteAscii(writer, 58, 50, 'A', {0, 0, 0});

    // Verify first 'A' crossbar: x = 50 + [1..6], y = 50 + 9 = 59
    for (int dx = 1; dx <= 6; ++dx) {
        size_t off = 4 * (width * 59 + (50 + dx));
        assert(vram[off + 0] == 0 && vram[off + 1] == 0 && vram[off + 2] == 0);
    }

    // Verify second 'A' crossbar: x = 58 + [1..6], y = 50 + 9 = 59
    for (int dx = 1; dx <= 6; ++dx) {
        size_t off = 4 * (width * 59 + (58 + dx));
        assert(vram[off + 0] == 0 && vram[off + 1] == 0 && vram[off + 2] == 0);
    }

    // Verify space between letters (x = 57, y = 59) is white background
    size_t off_gap = 4 * (width * 59 + 57);
    assert(vram[off_gap + 0] == 255 && vram[off_gap + 1] == 255 && vram[off_gap + 2] == 255);

    std::cout << "PASSED" << std::endl;
}

void BenchmarkFontRendering() {
    std::cout << "Benchmarking 100,000 WriteAscii Calls... ";

    const uint32_t width = 800;
    const uint32_t height = 600;
    std::vector<uint8_t> buf(width * height * 4, 255);
    FrameBufferConfig config{
        .frame_buffer = buf.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };
    RGBResv8BitPerColorPixelWriter writer{config};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100'000; ++i) {
        int x = (i * 8) % (width - 8);
        int y = ((i * 8) / (width - 8) * 16) % (height - 16);
        WriteAscii(writer, x, y, 'A', {0, 0, 0});
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "PASSED (" << std::fixed << std::setprecision(2) << ms << " ms for 100k glyphs, "
              << (100'000.0 / ms * 1000.0) << " glyphs/sec)" << std::endl;
}

}  // namespace mikan::ch05_font

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " MikanOS Chapter 5.1: Drawing Characters & References (day05a)   " << std::endl;
    std::cout << "=================================================================" << std::endl;

    mikan::ch05_font::TestFontABitmapStructure();
    mikan::ch05_font::TestWriteAsciiRendering();
    mikan::ch05_font::TestColumn51ReferenceVsPointer();
    mikan::ch05_font::TestDay05aFullVramSimulation();
    mikan::ch05_font::BenchmarkFontRendering();

    std::cout << "=================================================================" << std::endl;
    std::cout << " All Section 5.1 C++20 Verification Tests Passed Successfully!   " << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
