/**
 * @file console_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 5.5 (Console Class, osbook_day05e)
 *
 * Covers:
 * 1. Console class architecture (80 columns x 25 rows text buffer)
 * 2. Cursor coordinate management (cursor_row_, cursor_column_) and newline handling
 * 3. Screen rectangle blanking and 1-line scroll algorithm (memcpy & WriteString redraw)
 * 4. Boundary protection (column limit kColumns - 1 clipping)
 * 5. Improved Newlib sbrk system call stub returning (caddr_t)-1 with errno = ENOMEM
 * 6. Verification of the 27-line printing experiment (line 0..26)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mikan::ch05_console {

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

    const FrameBufferConfig& GetConfig() const { return config_; }

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
// 2. Embedded Minimal Font & String Rendering
// ============================================================================

// Minimal glyph generator for ASCII rendering verification
inline void GenerateGlyphPattern(char c, uint8_t out_glyph[16]) {
    std::memset(out_glyph, 0, 16);
    if (c >= '0' && c <= '9') {
        // Simple digit pattern: box with digit-specific crossbar
        out_glyph[2]  = 0b01111110;
        out_glyph[3]  = 0b01000010;
        out_glyph[4]  = 0b01000010;
        out_glyph[8]  = 0b01111110;
        out_glyph[12] = 0b01000010;
        out_glyph[13] = 0b01000010;
        out_glyph[14] = 0b01111110;
    } else if (c >= 'a' && c <= 'z') {
        // Lowercase pattern
        out_glyph[6]  = 0b00111100;
        out_glyph[7]  = 0b01000010;
        out_glyph[8]  = 0b01111110;
        out_glyph[9]  = 0b01000010;
        out_glyph[14] = 0b01000010;
    } else if (c == ' ') {
        // Space: blank
    } else {
        // Generic fallback symbol
        out_glyph[2]  = 0b01111110;
        out_glyph[8]  = 0b00011000;
        out_glyph[14] = 0b01111110;
    }
}

inline void WriteAscii(PixelWriter& writer, int x, int y, char c, const PixelColor& color) {
    uint8_t glyph[16];
    GenerateGlyphPattern(c, glyph);
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((glyph[dy] << dx) & 0x80u) {
                writer.Write(x + dx, y + dy, color);
            }
        }
    }
}

inline void WriteString(PixelWriter& writer, int x, int y, const char* s, const PixelColor& color) {
    for (int i = 0; s[i] != '\0'; ++i) {
        WriteAscii(writer, x + 8 * i, y, s[i], color);
    }
}

// ============================================================================
// 3. Console Class Implementation (Faithful to MikanOS day05e)
// ============================================================================

class Console {
public:
    static const int kRows = 25, kColumns = 80;

    Console(PixelWriter& writer,
            const PixelColor& fg_color, const PixelColor& bg_color)
        : writer_{writer}, fg_color_{fg_color}, bg_color_{bg_color},
          buffer_{}, cursor_row_{0}, cursor_column_{0}, scroll_count_{0} {
        for (int r = 0; r < kRows; ++r) {
            std::memset(buffer_[r], 0, sizeof(buffer_[r]));
        }
    }

    void PutString(const char* s) {
        while (*s) {
            if (*s == '\n') {
                Newline();
            } else if (cursor_column_ < kColumns - 1) {
                WriteAscii(writer_, 8 * cursor_column_, 16 * cursor_row_, *s, fg_color_);
                buffer_[cursor_row_][cursor_column_] = *s;
                ++cursor_column_;
            }
            ++s;
        }
    }

    // Inspection getters for verification
    int GetCursorRow() const { return cursor_row_; }
    int GetCursorColumn() const { return cursor_column_; }
    int GetScrollCount() const { return scroll_count_; }
    const char* GetLine(int row) const {
        if (row >= 0 && row < kRows) {
            return buffer_[row];
        }
        return "";
    }

private:
    void Newline() {
        cursor_column_ = 0;
        if (cursor_row_ < kRows - 1) {
            ++cursor_row_;
        } else {
            ++scroll_count_;
            // 1. Clear entire console region in VRAM
            for (int y = 0; y < 16 * kRows; ++y) {
                for (int x = 0; x < 8 * kColumns; ++x) {
                    writer_.Write(x, y, bg_color_);
                }
            }
            // 2. Shift buffer lines up and redraw
            for (int row = 0; row < kRows - 1; ++row) {
                std::memcpy(buffer_[row], buffer_[row + 1], kColumns + 1);
                WriteString(writer_, 0, 16 * row, buffer_[row], fg_color_);
            }
            // 3. Clear bottom row buffer
            std::memset(buffer_[kRows - 1], 0, kColumns + 1);
        }
    }

    PixelWriter& writer_;
    const PixelColor fg_color_, bg_color_;
    char buffer_[kRows][kColumns + 1];
    int cursor_row_, cursor_column_;
    int scroll_count_;
};

// ============================================================================
// 4. Improved Newlib sbrk Stub Verification (day05e)
// ============================================================================

using caddr_t = char*;

inline caddr_t day05e_sbrk([[maybe_unused]] int incr) {
    errno = ENOMEM;
    return reinterpret_cast<caddr_t>(-1);
}

} // namespace mikan::ch05_console

// ============================================================================
// 5. Main Execution and Comprehensive Verification
// ============================================================================

int main(int argc, char** argv) {
    using namespace mikan::ch05_console;

    bool json_output = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            json_output = true;
        }
    }

    std::cout << "====================================================================\n";
    std::cout << " MikanOS Chapter 5.5: Console Class & Scroll Simulation (osbook_day05e)\n";
    std::cout << "====================================================================\n\n";

    // 1. Allocate VRAM (640 x 400 pixels for 80x25 characters, 4 bytes per pixel)
    constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 400;
    constexpr uint32_t kStride = 640;
    std::vector<uint8_t> vram(kStride * kHeight * 4, 0xFF); // White background

    FrameBufferConfig fb_config{
        .frame_buffer = vram.data(),
        .pixels_per_scan_line = kStride,
        .horizontal_resolution = kWidth,
        .vertical_resolution = kHeight,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };
    RGBResv8BitPerColorPixelWriter writer(fb_config);

    PixelColor black{0, 0, 0};
    PixelColor white{255, 255, 255};
    Console console{writer, black, white};

    // 2. Verify Initial State
    std::cout << "[Step 1] Initial state check:\n";
    assert(console.GetCursorRow() == 0);
    assert(console.GetCursorColumn() == 0);
    assert(console.GetScrollCount() == 0);
    std::cout << "  -> Initial cursor (row=0, col=0), scroll_count=0 verified.\n\n";

    // 3. Print lines 0 to 23 (filling 24 rows, cursor reaches bottom row 24)
    std::cout << "[Step 2] Printing 24 lines (line 0 .. line 23)...\n";
    char buf[128];
    for (int i = 0; i < 24; ++i) {
        std::snprintf(buf, sizeof(buf), "line %d\n", i);
        console.PutString(buf);
    }
    assert(console.GetCursorRow() == 24);
    assert(console.GetCursorColumn() == 0);
    assert(console.GetScrollCount() == 0);
    assert(std::strcmp(console.GetLine(0), "line 0") == 0);
    assert(std::strcmp(console.GetLine(23), "line 23") == 0);
    assert(std::strlen(console.GetLine(24)) == 0);
    std::cout << "  -> 24 lines printed. Cursor reached bottom row 24 without scrolling.\n";
    std::cout << "     Top row [0]:  \"" << console.GetLine(0) << "\"\n";
    std::cout << "     Row [23]:     \"" << console.GetLine(23) << "\"\n";
    std::cout << "     cursor_row: " << console.GetCursorRow() << ", scroll_count: " << console.GetScrollCount() << "\n\n";

    // 4. Print line 24 (Triggers 1st Scroll upon '\\n')
    std::cout << "[Step 3] Printing line 24 (25th line, triggers 1st scroll upon '\\n')...\n";
    std::snprintf(buf, sizeof(buf), "line %d\n", 24);
    console.PutString(buf);
    assert(console.GetScrollCount() == 1);
    assert(console.GetCursorRow() == 24);
    assert(std::strcmp(console.GetLine(0), "line 1") == 0);
    assert(std::strcmp(console.GetLine(23), "line 24") == 0);
    assert(std::strlen(console.GetLine(24)) == 0);
    std::cout << "  -> 1st scroll occurred successfully:\n";
    std::cout << "     Top row [0] shifted to: \"" << console.GetLine(0) << "\" (line 0 scrolled out)\n";
    std::cout << "     Row [23] is now:        \"" << console.GetLine(23) << "\"\n";
    std::cout << "     scroll_count: " << console.GetScrollCount() << "\n\n";

    // 5. Print line 25 (Triggers 2nd Scroll)
    std::cout << "[Step 4] Printing line 25 (26th line, triggers 2nd scroll)...\n";
    std::snprintf(buf, sizeof(buf), "line %d\n", 25);
    console.PutString(buf);
    assert(console.GetScrollCount() == 2);
    assert(console.GetCursorRow() == 24);
    assert(std::strcmp(console.GetLine(0), "line 2") == 0);
    assert(std::strcmp(console.GetLine(23), "line 25") == 0);
    std::cout << "  -> 2nd scroll verified: Top row [0] is \"" << console.GetLine(0) << "\"\n\n";

    // 6. Print line 26 (Triggers 3rd Scroll - completes 27 lines from day05e main.cpp)
    std::cout << "[Step 5] Printing line 26 (27th line, completes loop of day05e main.cpp)...\n";
    std::snprintf(buf, sizeof(buf), "line %d\n", 26);
    console.PutString(buf);
    assert(console.GetScrollCount() == 3);
    assert(console.GetCursorRow() == 24);
    assert(std::strcmp(console.GetLine(0), "line 3") == 0);
    assert(std::strcmp(console.GetLine(23), "line 26") == 0);
    std::cout << "  -> 3rd scroll verified: Top row [0] is \"" << console.GetLine(0)
              << "\", Row [23] is \"" << console.GetLine(23) << "\"\n\n";

    // 7. Test Column Clipping Boundary (kColumns - 1 = 79 chars)
    std::cout << "[Step 6] Testing column limit clipping (80 columns)...\n";
    std::string long_str(100, 'x'); // 100 characters without newline
    console.PutString(long_str.c_str());
    assert(console.GetCursorColumn() == Console::kColumns - 1);
    assert(std::strlen(console.GetLine(24)) == Console::kColumns - 1);
    std::cout << "  -> Column properly clipped at " << console.GetCursorColumn()
              << " chars (kColumns - 1 = 79) to prevent buffer overflow.\n\n";

    // 8. Verify Improved sbrk Stub (day05e)
    std::cout << "[Step 7] Testing day05e sbrk POSIX-compliant stub...\n";
    errno = 0;
    caddr_t sbrk_res = day05e_sbrk(4096);
    assert(sbrk_res == reinterpret_cast<caddr_t>(-1));
    assert(errno == ENOMEM);
    std::cout << "  -> day05e_sbrk returned (caddr_t)-1 with errno = ENOMEM (" << ENOMEM << ") successfully.\n\n";

    if (json_output) {
        std::cout << "{\n";
        std::cout << "  \"kRows\": " << Console::kRows << ",\n";
        std::cout << "  \"kColumns\": " << Console::kColumns << ",\n";
        std::cout << "  \"final_cursor_row\": " << console.GetCursorRow() << ",\n";
        std::cout << "  \"final_cursor_col\": " << console.GetCursorColumn() << ",\n";
        std::cout << "  \"scroll_count\": " << console.GetScrollCount() << ",\n";
        std::cout << "  \"top_line\": \"" << console.GetLine(0) << "\",\n";
        std::cout << "  \"status\": \"SUCCESS\"\n";
        std::cout << "}\n";
    }

    std::cout << "====================================================================\n";
    std::cout << " [ALL TESTS PASSED] Chapter 5.5 Console Class successfully verified!\n";
    std::cout << "====================================================================\n";

    return 0;
}
