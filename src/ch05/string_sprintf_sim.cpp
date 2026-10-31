/**
 * @file string_sprintf_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 5.4 (String Rendering and sprintf, osbook_day05d)
 *
 * Covers:
 * 1. String rendering algorithm (WriteString) with 8-dot fixed-width stride
 * 2. Newlib C standard library integration (-lc) and system call stub (sbrk)
 * 3. Variable arguments (<cstdarg>, va_list, va_start, va_end) and System V AMD64 ABI calling convention
 * 4. In-kernel formatted printing via sprintf() and verification on FrameBuffer
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mikan::ch05_sprintf {

// ============================================================================
// 1. FrameBuffer & PixelWriter Infrastructure
// ============================================================================

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct PixelColor {
    uint8_t r, g, b;

    bool operator==(const PixelColor& other) const {
        return r == other.r && g == other.g && b == other.b;
    }
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    enum PixelFormat pixel_format;
};

class PixelWriter {
public:
    PixelWriter(const FrameBufferConfig& config) : config_{config} {}
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;

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
    virtual void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
    }
};

// ============================================================================
// 2. Embedded Hankaku Font & Glyph Resolution
// ============================================================================

// Embedded minimal 8x16 font table for characters: ' ', '!', 'H', 'e', 'l', 'o', ',', 'w', 'r', 'd', '1', '+', '2', '=', '3'
// and fallback glyph for other characters.
static const uint8_t kGlyphA[16] = {
    0b00000000,
    0b00011000,
    0b00100100,
    0b01000010,
    0b01000010,
    0b01111110,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01000010,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
};

static const uint8_t kGlyphH[16] = {
    0b00000000,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01111110,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01000010,
    0b01000010,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
};

static const uint8_t kGlyph1[16] = {
    0b00000000,
    0b00001000,
    0b00011000,
    0b00101000,
    0b00001000,
    0b00001000,
    0b00001000,
    0b00001000,
    0b00111110,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
    0b00000000,
};

// Generates a mock 256-glyph 4KiB font table
class FontTable {
public:
    static FontTable& Instance() {
        static FontTable instance;
        return instance;
    }

    const uint8_t* GetFont(char c) const {
        unsigned char uc = static_cast<unsigned char>(c);
        return &table_[uc * 16];
    }

private:
    FontTable() {
        table_.fill(0);
        // Fill default pattern for printable ASCII
        for (int i = 0x20; i <= 0x7E; ++i) {
            uint8_t* p = &table_[i * 16];
            p[1] = 0b00111100;
            p[2] = 0b01000010;
            p[8] = 0b01000010;
            p[9] = 0b00111100;
        }
        // Specific override for 'H', 'A', '1'
        std::memcpy(&table_[static_cast<unsigned char>('H') * 16], kGlyphH, 16);
        std::memcpy(&table_[static_cast<unsigned char>('A') * 16], kGlyphA, 16);
        std::memcpy(&table_[static_cast<unsigned char>('1') * 16], kGlyph1, 16);
    }

    std::array<uint8_t, 256 * 16> table_;
};

const uint8_t* GetFont(char c) {
    return FontTable::Instance().GetFont(c);
}

// ============================================================================
// 3. MikanOS WriteAscii & WriteString Implementation
// ============================================================================

void WriteAscii(PixelWriter& writer, int x, int y, char c, const PixelColor& color) {
    const uint8_t* font = GetFont(c);
    if (font == nullptr) {
        return;
    }
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((font[dy] << dx) & 0x80) {
                writer.Write(x + dx, y + dy, color);
            }
        }
    }
}

void WriteString(PixelWriter& writer, int x, int y, const char* s, const PixelColor& color) {
    for (int i = 0; s[i] != '\0'; ++i) {
        WriteAscii(writer, x + 8 * i, y, s[i], color);
    }
}

// ============================================================================
// 4. Newlib Support Stub Simulation (newlib_support.c)
// ============================================================================

extern "C" {
    // Simulates bare-metal sbrk stub from osbook_day05d
    void* fake_sbrk(intptr_t incr) {
        (void)incr;
        // In day05d, MikanOS has no dynamic heap allocator yet.
        // It returns NULL to signal out of memory, satisfying the linker without crashing.
        return nullptr;
    }
}

// ============================================================================
// 5. System V AMD64 ABI Variable Arguments & Formatted Output
// ============================================================================

int KernelSprintf(char* buf, const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    int ret = vsnprintf(buf, 128, format, ap);
    va_end(ap);
    return ret;
}

} // namespace mikan::ch05_sprintf

// ============================================================================
// Main Verification Runner
// ============================================================================

int main() {
    using namespace mikan::ch05_sprintf;

    std::cout << "====================================================================\n";
    std::cout << " MikanOS Chapter 5.4 C++20 Simulation: String Rendering and sprintf \n";
    std::cout << "====================================================================\n\n";

    // 1. FrameBuffer Allocation
    constexpr uint32_t kWidth = 400;
    constexpr uint32_t kHeight = 120;
    constexpr uint32_t kStride = 400;
    std::vector<uint8_t> vram(kStride * kHeight * 4, 0xFF); // Background white

    FrameBufferConfig fb_config{
        .frame_buffer = vram.data(),
        .pixels_per_scan_line = kStride,
        .horizontal_resolution = kWidth,
        .vertical_resolution = kHeight,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };
    RGBResv8BitPerColorPixelWriter writer(fb_config);

    // 2. Test WriteString ("Hello, world!" in blue)
    std::cout << "[Step 1] Verifying WriteString(\"Hello, world!\", blue) at (0, 66)...\n";
    PixelColor blue{0, 0, 255};
    WriteString(writer, 0, 66, "Hello, world!", blue);

    // Verify 'H' at x=0, y=66 (from kGlyphH: row 1 has bits at dx=1 and dx=6)
    // row 1, dx=1 -> (1, 67)
    size_t offset_h1 = 4 * (67 * kStride + 1);
    assert(vram[offset_h1 + 0] == 0 && vram[offset_h1 + 1] == 0 && vram[offset_h1 + 2] == 255);
    // row 1, dx=6 -> (6, 67)
    size_t offset_h6 = 4 * (67 * kStride + 6);
    assert(vram[offset_h6 + 0] == 0 && vram[offset_h6 + 1] == 0 && vram[offset_h6 + 2] == 255);
    std::cout << "  -> WriteString coordinates and pixel offsets verified successfully!\n\n";

    // 3. Test sprintf formatting and Newlib stub
    std::cout << "[Step 2] Verifying sprintf formatting and sbrk stub...\n";
    assert(fake_sbrk(1024) == nullptr);
    std::cout << "  -> fake_sbrk stub returns NULL correctly without throwing.\n";

    char buf[128];
    int len = KernelSprintf(buf, "1 + 2 = %d", 1 + 2);
    assert(len == 9);
    assert(std::strcmp(buf, "1 + 2 = 3") == 0);
    std::cout << "  -> KernelSprintf output: \"" << buf << "\" (length: " << len << ")\n";

    // 4. Render formatted string at y=82 in black
    std::cout << "[Step 3] Verifying WriteString(buf, black) at (0, 82)...\n";
    PixelColor black{0, 0, 0};
    WriteString(writer, 0, 82, buf, black);

    // Verify '1' at x=0, y=82 (from kGlyph1: row 8 has bits 0b00111110 -> dx=2..6)
    size_t offset_1 = 4 * (90 * kStride + 4);
    assert(vram[offset_1 + 0] == 0 && vram[offset_1 + 1] == 0 && vram[offset_1 + 2] == 0);
    std::cout << "  -> Formatted string correctly rendered on simulated VRAM.\n\n";

    // 5. Test hexadecimal and string format specifiers
    char hex_buf[64];
    KernelSprintf(hex_buf, "VRAM @ 0x%08x, Str: %s", 0x100000, "OK");
    std::cout << "[Step 4] Advanced sprintf formatting: \"" << hex_buf << "\"\n";
    assert(std::strstr(hex_buf, "0x00100000") != nullptr);
    assert(std::strstr(hex_buf, "Str: OK") != nullptr);

    std::cout << "\n====================================================================\n";
    std::cout << " [ALL TESTS PASSED] Chapter 5.4 String Rendering and sprintf verified!\n";
    std::cout << "====================================================================\n";

    return 0;
}
