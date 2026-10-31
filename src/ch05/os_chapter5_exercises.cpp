/**
 * @file os_chapter5_exercises.cpp
 * @brief Chapter 5 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 5:
 * 1. 8x16 Bitmap Font Bit-Expansion Math & Glyph Rendering
 * 2. C++ Reference vs Pointer Semantics, Syntax & Safety
 * 3. objcopy Linker Symbol Resolution & Font Table Pointer Arithmetic
 * 4. Newlib sprintf, System V AMD64 Varargs ABI (va_list)
 * 5. Console Class Line Buffering, Newline Handling & Scroll Shift Logic
 * 6. printk Pipeline, Buffer Truncation Safety & Placement new
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
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

namespace mikanos::exercises::ch05 {

// =============================================================================
// Exercise 1: 8x16 Bitmap Font Bit-Expansion Math & Glyph Rendering
// =============================================================================

// Mock 8x16 font glyph for 'A' (ASCII 0x41)
inline constexpr std::array<uint8_t, 16> kMockFontA = {
    0x00, 0x18, 0x18, 0x24, 0x24, 0x42, 0x42, 0x7e,
    0x42, 0x42, 0x42, 0x42, 0x00, 0x00, 0x00, 0x00
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    bool operator==(const PixelColor& other) const = default;
};

class MockFrameBuffer {
public:
    MockFrameBuffer(int width, int height)
        : width_(width), height_(height), buffer_(width * height, PixelColor{0, 0, 0}) {}

    void WritePixel(int x, int y, PixelColor color) {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            buffer_[y * width_ + x] = color;
        }
    }

    PixelColor GetPixel(int x, int y) const {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            return buffer_[y * width_ + x];
        }
        return PixelColor{0, 0, 0};
    }

    int Width() const { return width_; }
    int Height() const { return height_; }

private:
    int width_;
    int height_;
    std::vector<PixelColor> buffer_;
};

void WriteAscii(MockFrameBuffer& fb, int x, int y, char c, const uint8_t* font, PixelColor color) {
    (void)c;
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            // MSB (bit 7) is on the leftmost column (dx = 0)
            if ((font[dy] << dx) & 0x80u) {
                fb.WritePixel(x + dx, y + dy, color);
            }
        }
    }
}

// =============================================================================
// Exercise 2: C++ Reference vs Pointer Semantics
// =============================================================================

struct MockKernelConfig {
    int log_level{1};
    bool enable_paging{true};
    uint32_t memory_limit_mb{1024};
};

// =============================================================================
// Exercise 3: objcopy Linker Symbols & Font Address Math
// =============================================================================

// Simulate objcopy output of hankaku font table (256 characters * 16 bytes = 4096 bytes)
inline uint8_t g_mock_hankaku_bin[256 * 16];

const uint8_t* const _binary_hankaku_bin_start = g_mock_hankaku_bin;
const uint8_t* const _binary_hankaku_bin_end = g_mock_hankaku_bin + sizeof(g_mock_hankaku_bin);
const size_t _binary_hankaku_bin_size = sizeof(g_mock_hankaku_bin);

const uint8_t* GetFont(char c) {
    auto index = static_cast<unsigned char>(c);
    auto font_addr = _binary_hankaku_bin_start + index * 16;
    if (font_addr >= _binary_hankaku_bin_end) {
        return nullptr;
    }
    return font_addr;
}

// =============================================================================
// Exercise 4: Newlib sprintf & Variable Arguments ABI
// =============================================================================

int FormatLog(char* buf, size_t buf_size, const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    int ret = vsnprintf(buf, buf_size, format, ap);
    va_end(ap);
    return ret;
}

// =============================================================================
// Exercise 5: Console Class Line Buffering, Newlines & Scroll Shift
// =============================================================================

class MockConsole {
public:
    static constexpr int kRows = 25;
    static constexpr int kColumns = 80;

    MockConsole() {
        Clear();
    }

    void PutChar(char c) {
        if (c == '\n') {
            NewLine();
        } else if (cursor_col_ < kColumns) {
            buffer_[cursor_row_][cursor_col_] = c;
            ++cursor_col_;
        } else {
            NewLine();
            buffer_[cursor_row_][cursor_col_] = c;
            ++cursor_col_;
        }
    }

    void PutString(const char* s) {
        while (*s) {
            PutChar(*s);
            ++s;
        }
    }

    void NewLine() {
        cursor_col_ = 0;
        if (cursor_row_ < kRows - 1) {
            ++cursor_row_;
        } else {
            Scroll();
        }
    }

    void Scroll() {
        // Shift lines 1..24 to 0..23
        for (int r = 0; r < kRows - 1; ++r) {
            std::memcpy(buffer_[r].data(), buffer_[r + 1].data(), kColumns + 1);
        }
        // Clear bottom line (row 24)
        std::memset(buffer_[kRows - 1].data(), 0, kColumns + 1);
        cursor_row_ = kRows - 1;
        cursor_col_ = 0;
        ++scroll_count_;
    }

    void Clear() {
        for (auto& row : buffer_) {
            row.fill(0);
        }
        cursor_row_ = 0;
        cursor_col_ = 0;
        scroll_count_ = 0;
    }

    int CursorRow() const { return cursor_row_; }
    int CursorCol() const { return cursor_col_; }
    int ScrollCount() const { return scroll_count_; }
    std::string_view GetRow(int r) const {
        return std::string_view(buffer_[r].data());
    }

private:
    std::array<std::array<char, kColumns + 1>, kRows> buffer_;
    int cursor_row_{0};
    int cursor_col_{0};
    int scroll_count_{0};
};

// =============================================================================
// Exercise 6: printk Pipeline & Buffer Truncation Safety
// =============================================================================

alignas(MockConsole) inline uint8_t g_console_buf[sizeof(MockConsole)];
inline MockConsole* g_console_ptr{nullptr};

int printk(const char* format, ...) {
    char s[128]; // Small buffer for testing truncation
    va_list ap;
    va_start(ap, format);
    int result = vsnprintf(s, sizeof(s), format, ap);
    va_end(ap);

    if (g_console_ptr != nullptr) {
        g_console_ptr->PutString(s);
    }
    return result;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_all_verifications() {
    std::cout << "====================================================================\n";
    std::cout << " MikanOS Chapter 5 Exercises: C++20 Verification Suite\n";
    std::cout << "====================================================================\n";

    // 1. Bitmap font bit-expansion & rendering
    {
        MockFrameBuffer fb(16, 16);
        PixelColor white{255, 255, 255};
        PixelColor black{0, 0, 0};

        WriteAscii(fb, 0, 0, 'A', kMockFontA.data(), white);

        // Verify row 1 (0x18 = 00011000): columns 3 and 4 should be white
        assert(fb.GetPixel(2, 1) == black);
        assert(fb.GetPixel(3, 1) == white);
        assert(fb.GetPixel(4, 1) == white);
        assert(fb.GetPixel(5, 1) == black);

        // Verify row 7 (0x7e = 01111110): columns 1 to 6 should be white
        assert(fb.GetPixel(0, 7) == black);
        for (int x = 1; x <= 6; ++x) {
            assert(fb.GetPixel(x, 7) == white);
        }
        assert(fb.GetPixel(7, 7) == black);

        std::cout << "[PASS] Exercise 1: 8x16 Bitmap Font MSB bit expansion & pixel rendering verified\n";
    }

    // 2. Reference vs Pointer semantics
    {
        MockKernelConfig config1{1, true, 1024};
        MockKernelConfig config2{2, false, 2048};

        MockKernelConfig& ref = config1;
        MockKernelConfig* ptr = &config1;

        assert(&ref == &config1);
        assert(ptr == &config1);

        // Modifying through reference alters original object
        ref.log_level = 3;
        assert(config1.log_level == 3);

        // Pointers can be reassigned (rebound), references cannot
        ptr = &config2;
        assert(ptr == &config2);
        assert(&ref == &config1); // ref still binds to config1

        std::cout << "[PASS] Exercise 2: C++ Reference vs Pointer semantics & non-rebindability verified\n";
    }

    // 3. objcopy symbols & font table indexing
    {
        // Populate mock font for 'A' (0x41)
        std::memcpy(&g_mock_hankaku_bin['A' * 16], kMockFontA.data(), 16);

        assert(_binary_hankaku_bin_size == 4096);
        assert(_binary_hankaku_bin_end - _binary_hankaku_bin_start == 4096);

        const uint8_t* font_A = GetFont('A');
        assert(font_A != nullptr);
        assert(font_A[1] == 0x18);
        assert(font_A[7] == 0x7e);

        const uint8_t* font_null = GetFont('\0');
        assert(font_null == _binary_hankaku_bin_start);

        std::cout << "[PASS] Exercise 3: objcopy symbols & font address arithmetic (base + c * 16) verified\n";
    }

    // 4. Newlib sprintf & varargs formatting
    {
        char buffer[128];
        int n = FormatLog(buffer, sizeof(buffer), "Kernel %s initialized at 0x%08x (mem=%dMB)",
                          "MikanOS", 0x100000, 512);
        assert(n > 0);
        std::string s(buffer);
        assert(s == "Kernel MikanOS initialized at 0x00100000 (mem=512MB)");

        std::cout << "[PASS] Exercise 4: System V AMD64 varargs ABI & sprintf formatting verified\n";
    }

    // 5. Console class buffering, newlines & scroll
    {
        MockConsole console;
        assert(console.CursorRow() == 0);
        assert(console.CursorCol() == 0);

        // Print 24 lines (0 to 23)
        for (int i = 0; i < 24; ++i) {
            char line[32];
            std::snprintf(line, sizeof(line), "Line %d\n", i);
            console.PutString(line);
        }
        assert(console.CursorRow() == 24);
        assert(console.CursorCol() == 0);
        assert(console.ScrollCount() == 0);

        // Print 25th line (causes 1st scroll)
        console.PutString("Line 24\n");
        assert(console.CursorRow() == 24);
        assert(console.CursorCol() == 0);
        assert(console.ScrollCount() == 1);
        // After scroll, row 0 should contain "Line 1"
        assert(console.GetRow(0) == "Line 1");
        // Row 23 should contain "Line 24"
        assert(console.GetRow(23) == "Line 24");
        // Row 24 should be empty
        assert(console.GetRow(24).empty());

        std::cout << "[PASS] Exercise 5: 25-row Console scrolling, line shift & row-24 clearing verified\n";
    }

    // 6. printk pipeline, buffer truncation & placement new
    {
        // Construct MockConsole using placement new in static buffer
        g_console_ptr = new (g_console_buf) MockConsole();
        assert(g_console_ptr != nullptr);

        int written = printk("Hello %s, CPU count=%d\n", "World", 4);
        assert(written > 0);
        assert(g_console_ptr->GetRow(0) == "Hello World, CPU count=4");

        // Test buffer truncation with a string longer than 128 bytes
        std::string long_str(200, 'X');
        int ret = printk("Long: %s\n", long_str.c_str());
        // vsnprintf returns number of characters that would have been written
        assert(ret >= 128); // truncation occurred
        // Ensure string wrapped cleanly across lines and cursor advanced
        assert(g_console_ptr->CursorRow() >= 1);

        // Perform 26 printk calls to verify scrolling through printk
        for (int i = 0; i < 26; ++i) {
            printk("log %d\n", i);
        }
        assert(g_console_ptr->ScrollCount() >= 1);

        std::cout << "[PASS] Exercise 6: printk pipeline, placement new & buffer truncation safety verified\n";
    }

    std::cout << "\n====================================================================\n";
    std::cout << " All Chapter 5 Exercises C++20 Verifications PASSED!\n";
    std::cout << "====================================================================\n";
}

} // namespace mikanos::exercises::ch05

int main(int argc, char* argv[]) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    mikanos::exercises::ch05::run_all_verifications();

    if (test_mode) {
        std::cout << "Executed in --test mode.\n";
    }
    return 0;
}
