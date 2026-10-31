/**
 * @file printk_sim.cpp
 * @brief C++20 Simulation and Verification of MikanOS Chapter 5.6 (printk, osbook_day05f)
 *
 * Covers:
 * 1. Global placement new for Console instance (console_buf + placement new)
 * 2. Kernel formatted print function printk() using variable arguments (<cstdarg>)
 * 3. System V AMD64 ABI variable arguments calling convention (va_list internals)
 * 4. Safe formatting with vsnprintf() preventing buffer overrun
 * 5. 27-line printing experiment (printk: 0..26) verifying 2-line auto-scroll
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace mikan::ch05_printk {

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
// 2. Minimal Glyph Generator & String Rendering
// ============================================================================

inline void GenerateGlyphPattern(char c, uint8_t out_glyph[16]) {
    std::memset(out_glyph, 0, 16);
    if (c >= '0' && c <= '9') {
        out_glyph[2]  = 0b01111110;
        out_glyph[3]  = 0b01000010;
        out_glyph[4]  = 0b01000010;
        out_glyph[8]  = 0b01111110;
        out_glyph[12] = 0b01000010;
        out_glyph[13] = 0b01000010;
        out_glyph[14] = 0b01111110;
    } else if (c >= 'a' && c <= 'z') {
        out_glyph[4]  = 0b00111100;
        out_glyph[5]  = 0b01000010;
        out_glyph[8]  = 0b01000010;
        out_glyph[12] = 0b01000010;
        out_glyph[13] = 0b01000010;
        out_glyph[14] = 0b00111100;
    } else if (c == ':') {
        out_glyph[5]  = 0b00011000;
        out_glyph[6]  = 0b00011000;
        out_glyph[10] = 0b00011000;
        out_glyph[11] = 0b00011000;
    } else if (c == ' ') {
        // Space: blank glyph
    } else {
        out_glyph[0]  = 0b11111111;
        out_glyph[15] = 0b11111111;
        for (int dy = 1; dy < 15; ++dy) {
            out_glyph[dy] = 0b10000001;
        }
    }
}

inline void WriteAscii(PixelWriter& writer, int x, int y, char c, const PixelColor& color) {
    uint8_t glyph[16];
    GenerateGlyphPattern(c, glyph);
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if ((glyph[dy] << dx) & 0x80) {
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
// 3. Console Class (from Chapter 5.5)
// ============================================================================

class Console {
public:
    static constexpr int kRows = 25;
    static constexpr int kColumns = 80;

    Console(PixelWriter& writer,
            const PixelColor& fg_color, const PixelColor& bg_color)
        : writer_{writer}, fg_color_{fg_color}, bg_color_{bg_color},
          buffer_{}, cursor_row_{0}, cursor_column_{0}, scroll_count_{0} {
    }

    void PutString(const char* s) {
        while (*s != '\0') {
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

    int GetCursorRow() const { return cursor_row_; }
    int GetCursorColumn() const { return cursor_column_; }
    int GetScrollCount() const { return scroll_count_; }

    const char* GetLine(int row) const {
        if (row < 0 || row >= kRows) return "";
        return buffer_[row];
    }

private:
    void Newline() {
        cursor_column_ = 0;
        if (cursor_row_ < kRows - 1) {
            ++cursor_row_;
        } else {
            // Screen blanking
            for (int y = 0; y < 16 * kRows; ++y) {
                for (int x = 0; x < 8 * kColumns; ++x) {
                    writer_.Write(x, y, bg_color_);
                }
            }
            // Row shift & redraw
            for (int row = 0; row < kRows - 1; ++row) {
                std::memcpy(buffer_[row], buffer_[row + 1], kColumns + 1);
                WriteString(writer_, 0, 16 * row, buffer_[row], fg_color_);
            }
            // Clear bottom row
            std::memset(buffer_[kRows - 1], 0, kColumns + 1);
            ++scroll_count_;
        }
    }

    PixelWriter& writer_;
    const PixelColor fg_color_, bg_color_;
    char buffer_[kRows][kColumns + 1];
    int cursor_row_, cursor_column_;
    int scroll_count_;
};

// ============================================================================
// 4. Global Console Instance & Placement New
// ============================================================================

alignas(alignof(Console)) char console_buf[sizeof(Console)];
Console* console = nullptr;

inline void InitializeConsole(PixelWriter& writer,
                              const PixelColor& fg_color = {0, 0, 0},
                              const PixelColor& bg_color = {255, 255, 255}) {
    console = new (console_buf) Console{writer, fg_color, bg_color};
}

// ============================================================================
// 5. Kernel Formatted Print: printk()
// ============================================================================

/**
 * @brief Kernel formatted print function.
 *
 * Implements MikanOS printk (osbook_day05f):
 * - Accepts variable arguments with va_list / va_start / va_end.
 * - Formats into a stack buffer using vsnprintf (safe version of vsprintf).
 * - Forwards formatted string to global console->PutString().
 *
 * @param format Format string.
 * @param ... Variable arguments.
 * @return Number of characters printed (excluding null terminator).
 */
int printk(const char* format, ...) {
    va_list ap;
    int result;
    char s[1024];

    va_start(ap, format);
    result = vsnprintf(s, sizeof(s), format, ap);
    va_end(ap);

    if (console != nullptr) {
        console->PutString(s);
    }
    return result;
}

// ============================================================================
// 6. System V AMD64 ABI Variable Arguments Simulator & Inspector
// ============================================================================

struct VaListInternalLayout {
    uint32_t gp_offset;         // offset to next available general purpose register in reg_save_area (0..48)
    uint32_t fp_offset;         // offset to next available floating point register in reg_save_area (48..304)
    uint64_t overflow_arg_area; // pointer to arguments passed on the stack
    uint64_t reg_save_area;     // pointer to the base of the register save area
};

struct AbiRegisterMapping {
    const char* reg_name;
    uint32_t offset;
    const char* role;
};

inline const std::vector<AbiRegisterMapping>& GetSystemVAbiGpRegisters() {
    static const std::vector<AbiRegisterMapping> kGpRegs = {
        {"RDI", 0,  "1st integer argument (format string)"},
        {"RSI", 8,  "2nd integer argument (1st vararg)"},
        {"RDX", 16, "3rd integer argument (2nd vararg)"},
        {"RCX", 24, "4th integer argument (3rd vararg)"},
        {"R8",  32, "5th integer argument (4th vararg)"},
        {"R9",  40, "6th integer argument (5th vararg)"},
    };
    return kGpRegs;
}

class SystemVVarargsInspector {
public:
    static constexpr size_t kGpRegCount = 6;
    static constexpr size_t kFpRegCount = 8;
    static constexpr size_t kGpSaveSize = 48; // 6 * 8 bytes
    static constexpr size_t kFpSaveSize = 128; // 8 * 16 bytes
    static constexpr size_t kTotalRegSaveAreaSize = kGpSaveSize + kFpSaveSize; // 176 bytes

    struct ArgRecord {
        std::string type_name;
        uint64_t value;
        std::string passed_via; // e.g. "RSI (offset 8)", "Stack (+0)"
    };

    static std::vector<ArgRecord> TraceCall(const std::vector<uint64_t>& integer_args) {
        std::vector<ArgRecord> trace;
        auto gp_regs = GetSystemVAbiGpRegisters();

        for (size_t i = 0; i < integer_args.size(); ++i) {
            ArgRecord rec;
            rec.value = integer_args[i];
            if (i == 0) {
                rec.type_name = "const char* format";
                rec.passed_via = "RDI (gp_offset 0)";
            } else if (i < kGpRegCount) {
                rec.type_name = "int / uint64_t";
                rec.passed_via = std::string(gp_regs[i].reg_name) + " (offset " + std::to_string(gp_regs[i].offset) + ")";
            } else {
                rec.type_name = "stack arg";
                size_t stack_off = (i - kGpRegCount) * 8;
                rec.passed_via = "Stack (overflow +" + std::to_string(stack_off) + ")";
            }
            trace.push_back(rec);
        }
        return trace;
    }
};

} // namespace mikan::ch05_printk

// ============================================================================
// 7. Standalone Verification Executable
// ============================================================================

int main(int argc, char* argv[]) {
    using namespace mikan::ch05_printk;

    bool json_output = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            json_output = true;
        }
    }

    // Allocate 640x400 RGBA frame buffer
    constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 400;
    std::vector<uint8_t> fb_memory(kWidth * kHeight * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = fb_memory.data(),
        .pixels_per_scan_line = kWidth,
        .horizontal_resolution = kWidth,
        .vertical_resolution = kHeight,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    RGBResv8BitPerColorPixelWriter writer{config};

    // 1. Verify global placement new
    InitializeConsole(writer, {0, 0, 0}, {255, 255, 255});
    assert(console != nullptr);
    assert(reinterpret_cast<void*>(console) == reinterpret_cast<void*>(console_buf));

    // 2. Perform the 27-line printk experiment from osbook_day05f
    // for (int i = 0; i < 27; ++i) { printk("printk: %d\n", i); }
    for (int i = 0; i < 27; ++i) {
        int chars = printk("printk: %d\n", i);
        assert(chars > 0);
    }

    // 3. Verification of console state
    // Total lines = 27, buffer rows = 25
    // - lines 0..23 fill rows 0..23 (cursor_row becomes 24)
    // - line 24 (25th line) triggers 1st scroll
    // - line 25 (26th line) triggers 2nd scroll
    // - line 26 (27th line) triggers 3rd scroll
    assert(console->GetScrollCount() == 3);
    assert(console->GetCursorRow() == 24);
    assert(console->GetCursorColumn() == 0);

    // Row 0 should be "printk: 3" (lines 0, 1, 2 were scrolled away)
    assert(std::string(console->GetLine(0)) == "printk: 3");
    // Row 23 should be "printk: 26" (shifted up by newline scroll)
    assert(std::string(console->GetLine(23)) == "printk: 26");
    // Row 24 should be empty waiting for next input
    assert(std::string(console->GetLine(24)) == "");

    // 4. Test multi-argument formatting
    printk("Hex: 0x%08x, Str: %s\n", 0x1234abcd, "MikanOS");
    assert(console->GetScrollCount() == 4);
    assert(std::string(console->GetLine(23)) == "Hex: 0x1234abcd, Str: MikanOS");
    assert(std::string(console->GetLine(24)) == "");

    // 5. System V ABI register mapping check
    auto trace = SystemVVarargsInspector::TraceCall({
        0x1000, 10, 20, 30, 40, 50, 60, 70
    });
    assert(trace.size() == 8);
    assert(trace[0].passed_via.find("RDI") != std::string::npos);
    assert(trace[1].passed_via.find("RSI") != std::string::npos);
    assert(trace[5].passed_via.find("R9") != std::string::npos);
    assert(trace[6].passed_via.find("Stack") != std::string::npos);

    if (json_output) {
        std::cout << "{\n"
                  << "  \"status\": \"success\",\n"
                  << "  \"chapter\": \"5.6\",\n"
                  << "  \"section\": \"printk()\",\n"
                  << "  \"scroll_count\": " << console->GetScrollCount() << ",\n"
                  << "  \"cursor_row\": " << console->GetCursorRow() << ",\n"
                  << "  \"cursor_column\": " << console->GetCursorColumn() << ",\n"
                  << "  \"line_0\": \"" << console->GetLine(0) << "\",\n"
                  << "  \"line_23\": \"" << console->GetLine(23) << "\",\n"
                  << "  \"placement_new_addr\": \"0x" << std::hex
                  << reinterpret_cast<uintptr_t>(console) << std::dec << "\",\n"
                  << "  \"console_buf_addr\": \"0x" << std::hex
                  << reinterpret_cast<uintptr_t>(console_buf) << std::dec << "\",\n"
                  << "  \"total_abi_trace_items\": " << trace.size() << "\n"
                  << "}\n";
    } else {
        std::cout << "========================================================\n"
                  << " Chapter 5.6: printk() Verification Suite\n"
                  << "========================================================\n"
                  << " Placement New:      PASSED (console == console_buf)\n"
                  << " 27-Line Experiment: PASSED (3 scrolls triggered)\n"
                  << " Row 0 (Oldest):     '" << console->GetLine(0) << "'\n"
                  << " Row 23 (Latest):    '" << console->GetLine(23) << "'\n"
                  << " Total Scrolls:      " << console->GetScrollCount() << "\n"
                  << " Varargs ABI Trace:  PASSED (6 GP regs + stack overflow)\n"
                  << "========================================================\n"
                  << " All C++20 assertions passed successfully!\n";
    }

    return 0;
}
