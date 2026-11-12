/**
 * @file os_chapter9_measuring_scroll_time.cpp
 * @brief Chapter 9.6: Measuring Console Scroll Time with Local APIC Timer (C++20)
 *
 * Demonstrates:
 * 1. Console architecture with 25 rows x 80 columns character grid (8x16 font, 640x400 px)
 * 2. MikanOS printk() instrumented with Local APIC Timer (osbook_day09d)
 * 3. Execution profile of Console::PutString() and Console::Newline()
 * 4. Microarchitectural bottleneck breakdown:
 *    - 256,000-pixel individual pixel clearing via nested loop
 *    - 24 rows x up to 80 chars font rendering per character (up to 1,920 chars)
 *    - Cache thrashing and pixel-by-pixel write overhead
 * 5. Quantitative performance comparison: Normal line output vs Scroll line output
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::study {

// =============================================================================
// 1. Basic Geometry and Color Definitions
// =============================================================================

struct Vector2D {
    int x{0};
    int y{0};
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    bool operator==(const PixelColor& o) const = default;
};

constexpr PixelColor kDesktopFGColor{255, 255, 255};
constexpr PixelColor kDesktopBGColor{45, 45, 45};

// =============================================================================
// 2. Simulated PixelWriter with Performance Counter Instrumentation
// =============================================================================

class InstrumentedPixelWriter {
public:
    InstrumentedPixelWriter(int width, int height)
        : width_{width}, height_{height}, buffer_(width * height, kDesktopBGColor) {}

    virtual ~InstrumentedPixelWriter() = default;

    virtual void Write(Vector2D pos, const PixelColor& c) {
        if (pos.x < 0 || pos.x >= width_ || pos.y < 0 || pos.y >= height_) {
            return;
        }
        buffer_[pos.y * width_ + pos.x] = c;
        ++total_pixel_writes_;
    }

    [[nodiscard]] int Width() const { return width_; }
    [[nodiscard]] int Height() const { return height_; }
    [[nodiscard]] uint64_t TotalPixelWrites() const { return total_pixel_writes_; }
    void ResetPixelCounter() { total_pixel_writes_ = 0; }

    [[nodiscard]] const PixelColor& At(int x, int y) const {
        return buffer_[y * width_ + x];
    }

private:
    int width_;
    int height_;
    std::vector<PixelColor> buffer_;
    uint64_t total_pixel_writes_{0};
};

// =============================================================================
// 3. Mock 8x16 Hankaku Font
// =============================================================================

// Simplified 8x16 font glyph (each char has representative dots)
inline bool IsFontPixelLit(char c, int dy, int dx) {
    if (c == ' ' || c == '\0') return false;
    // Dummy glyph pattern: draw borders / cross based on ascii code to simulate realistic font pixel count
    uint8_t hash = static_cast<uint8_t>(c) ^ static_cast<uint8_t>(dy * 7 + dx * 13);
    // Typical 8x16 character has ~25-35 lit pixels out of 128 (approx 20-30% density)
    return (hash % 100) < 25;
}

inline void WriteAscii(InstrumentedPixelWriter& writer, Vector2D pos, char c, const PixelColor& fg) {
    for (int dy = 0; dy < 16; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
            if (IsFontPixelLit(c, dy, dx)) {
                writer.Write(Vector2D{pos.x + dx, pos.y + dy}, fg);
            }
        }
    }
}

inline void WriteString(InstrumentedPixelWriter& writer, Vector2D pos, const char* s, const PixelColor& fg) {
    int col = 0;
    while (*s) {
        WriteAscii(writer, Vector2D{pos.x + 8 * col, pos.y}, *s, fg);
        ++col;
        ++s;
    }
}

// =============================================================================
// 4. Simulated Local APIC Timer (osbook_day09d instrumentation)
// =============================================================================

class SimulatedLAPICTimer {
public:
    static constexpr uint32_t kCountMax = 0xffffffffu;

    void Initialize() {
        is_running_ = false;
        calibrated_freq_hz_ = 1'000'000'000; // 1 GHz nominal LAPIC / CPU bus clock
    }

    void Start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = true;
    }

    [[nodiscard]] uint32_t Elapsed() const {
        if (!is_running_) return 0;
        auto now = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_).count();
        // Convert nanoseconds to LAPIC timer ticks (~1 tick per ns on modern x86)
        uint64_t ticks = static_cast<uint64_t>(ns);
        return static_cast<uint32_t>(std::min<uint64_t>(ticks, kCountMax));
    }

    void Stop() {
        is_running_ = false;
    }

private:
    bool is_running_{false};
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    uint64_t calibrated_freq_hz_{1'000'000'000};
};

// =============================================================================
// 5. Console Class Implementation (Matching osbook_day09d kernel/console.cpp)
// =============================================================================

struct LineMeasurement {
    std::string text;
    bool did_scroll{false};
    uint32_t elapsed_ticks{0};
    uint64_t pixel_writes{0};
    uint64_t bg_clear_writes{0};
    uint64_t font_redraw_writes{0};
};

class SimulatedConsole {
public:
    static constexpr int kRows = 25;
    static constexpr int kColumns = 80;
    static constexpr int kWidth = 8 * kColumns;   // 640 px
    static constexpr int kHeight = 16 * kRows;    // 400 px
    static constexpr int kTotalPixels = kWidth * kHeight; // 256,000 px

    SimulatedConsole(InstrumentedPixelWriter* writer,
                     PixelColor fg_color = kDesktopFGColor,
                     PixelColor bg_color = kDesktopBGColor)
        : writer_{writer}, fg_color_{fg_color}, bg_color_{bg_color} {
        std::memset(buffer_, 0, sizeof(buffer_));
    }

    void SetWriter(InstrumentedPixelWriter* writer) {
        writer_ = writer;
    }

    LineMeasurement PutStringInstrumented(const char* s, SimulatedLAPICTimer& timer) {
        LineMeasurement measurement;
        measurement.text = s;

        uint64_t initial_writes = writer_->TotalPixelWrites();
        last_newline_scrolled_ = false;
        last_bg_clear_writes_ = 0;
        last_font_redraw_writes_ = 0;

        timer.Start();

        // Core PutString logic from kernel/console.cpp
        while (*s) {
            if (*s == '\n') {
                Newline();
            } else if (cursor_column_ < kColumns - 1) {
                WriteAscii(*writer_, Vector2D{8 * cursor_column_, 16 * cursor_row_}, *s, fg_color_);
                buffer_[cursor_row_][cursor_column_] = *s;
                ++cursor_column_;
            }
            ++s;
        }

        measurement.elapsed_ticks = timer.Elapsed();
        timer.Stop();

        measurement.did_scroll = last_newline_scrolled_;
        measurement.pixel_writes = writer_->TotalPixelWrites() - initial_writes;
        measurement.bg_clear_writes = last_bg_clear_writes_;
        measurement.font_redraw_writes = last_font_redraw_writes_;

        return measurement;
    }

    [[nodiscard]] int CursorRow() const { return cursor_row_; }
    [[nodiscard]] int CursorColumn() const { return cursor_column_; }
    [[nodiscard]] const char* BufferRow(int row) const { return buffer_[row]; }

private:
    void Newline() {
        cursor_column_ = 0;
        if (cursor_row_ < kRows - 1) {
            ++cursor_row_;
        } else {
            // === SCROLL PROCESSING (osbook_day09d bottleneck) ===
            last_newline_scrolled_ = true;
            uint64_t start_writes = writer_->TotalPixelWrites();

            // 1. Nested loop background clearing: 256,000 pixel writes!
            for (int y = 0; y < 16 * kRows; ++y) {
                for (int x = 0; x < 8 * kColumns; ++x) {
                    writer_->Write(Vector2D{x, y}, bg_color_);
                }
            }
            last_bg_clear_writes_ = writer_->TotalPixelWrites() - start_writes;

            // 2. Text buffer shift and full character redraw: 24 rows of text!
            uint64_t redraw_start = writer_->TotalPixelWrites();
            for (int row = 0; row < kRows - 1; ++row) {
                std::memcpy(buffer_[row], buffer_[row + 1], kColumns + 1);
                WriteString(*writer_, Vector2D{0, 16 * row}, buffer_[row], fg_color_);
            }
            std::memset(buffer_[kRows - 1], 0, kColumns + 1);
            last_font_redraw_writes_ = writer_->TotalPixelWrites() - redraw_start;
        }
    }

    InstrumentedPixelWriter* writer_;
    PixelColor fg_color_;
    PixelColor bg_color_;
    char buffer_[kRows][kColumns + 1];
    int cursor_row_{0};
    int cursor_column_{0};

    bool last_newline_scrolled_{false};
    uint64_t last_bg_clear_writes_{0};
    uint64_t last_font_redraw_writes_{0};
};

// =============================================================================
// 6. Comprehensive Profiling Benchmark Suite
// =============================================================================

struct ScrollBenchmarkReport {
    std::vector<LineMeasurement> lines;
    uint32_t normal_line_avg_ticks{0};
    uint32_t scroll_line_avg_ticks{0};
    uint64_t normal_line_avg_writes{0};
    uint64_t scroll_line_avg_writes{0};
    double slowdown_ratio_time{0.0};
    double slowdown_ratio_writes{0.0};
};

inline ScrollBenchmarkReport RunScrollProfilingSuite(int line_count = 35) {
    InstrumentedPixelWriter writer(SimulatedConsole::kWidth, SimulatedConsole::kHeight);
    SimulatedConsole console(&writer);
    SimulatedLAPICTimer timer;
    timer.Initialize();

    ScrollBenchmarkReport report;
    std::vector<uint32_t> normal_ticks;
    std::vector<uint32_t> scroll_ticks;
    std::vector<uint64_t> normal_writes;
    std::vector<uint64_t> scroll_writes;

    for (int i = 0; i < line_count; ++i) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "MikanOS Kernel message line #%02d: testing LAPIC timer...\n", i + 1);
        auto m = console.PutStringInstrumented(buf, timer);
        report.lines.push_back(m);

        if (m.did_scroll) {
            scroll_ticks.push_back(m.elapsed_ticks);
            scroll_writes.push_back(m.pixel_writes);
        } else {
            normal_ticks.push_back(m.elapsed_ticks);
            normal_writes.push_back(m.pixel_writes);
        }
    }

    auto avg_u32 = [](const std::vector<uint32_t>& v) -> uint32_t {
        if (v.empty()) return 0;
        uint64_t sum = 0;
        for (auto x : v) sum += x;
        return static_cast<uint32_t>(sum / v.size());
    };

    auto avg_u64 = [](const std::vector<uint64_t>& v) -> uint64_t {
        if (v.empty()) return 0;
        uint64_t sum = 0;
        for (auto x : v) sum += x;
        return sum / v.size();
    };

    report.normal_line_avg_ticks = avg_u32(normal_ticks);
    report.scroll_line_avg_ticks = avg_u32(scroll_ticks);
    report.normal_line_avg_writes = avg_u64(normal_writes);
    report.scroll_line_avg_writes = avg_u64(scroll_writes);

    if (report.normal_line_avg_ticks > 0) {
        report.slowdown_ratio_time = static_cast<double>(report.scroll_line_avg_ticks) / report.normal_line_avg_ticks;
    }
    if (report.normal_line_avg_writes > 0) {
        report.slowdown_ratio_writes = static_cast<double>(report.scroll_line_avg_writes) / report.normal_line_avg_writes;
    }

    return report;
}

} // namespace mikanos::study

// =============================================================================
// Main Execution & Self-Verification
// =============================================================================

int main() {
    using namespace mikanos::study;

    std::cout << "==================================================================\n";
    std::cout << " MikanOS Chapter 9.6: Measuring Console Scroll Time (osbook_day09d)\n";
    std::cout << "==================================================================\n\n";

    auto report = RunScrollProfilingSuite(30);

    std::cout << std::left
              << std::setw(6)  << "Line"
              << std::setw(10) << "Type"
              << std::setw(16) << "LAPIC Ticks"
              << std::setw(16) << "Pixel Writes"
              << std::setw(16) << "BG Clear Writes"
              << std::setw(18) << "Font Redraw Writes"
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (size_t i = 0; i < report.lines.size(); ++i) {
        const auto& m = report.lines[i];
        std::cout << std::left
                  << std::setw(6)  << (i + 1)
                  << std::setw(10) << (m.did_scroll ? "SCROLL" : "NORMAL")
                  << std::setw(16) << m.elapsed_ticks
                  << std::setw(16) << m.pixel_writes
                  << std::setw(16) << m.bg_clear_writes
                  << std::setw(18) << m.font_redraw_writes
                  << "\n";
    }

    std::cout << std::string(82, '-') << "\n\n";
    std::cout << "=== Quantitative Bottleneck Analysis ===\n";
    std::cout << "Normal Line Avg Ticks  : " << report.normal_line_avg_ticks << " ticks\n";
    std::cout << "Scroll Line Avg Ticks  : " << report.scroll_line_avg_ticks << " ticks\n";
    std::cout << "Slowdown Factor (Time) : " << std::fixed << std::setprecision(2) << report.slowdown_ratio_time << "x\n\n";

    std::cout << "Normal Line Avg Writes : " << report.normal_line_avg_writes << " pixel writes\n";
    std::cout << "Scroll Line Avg Writes : " << report.scroll_line_avg_writes << " pixel writes\n";
    std::cout << "Slowdown Factor (Write): " << std::fixed << std::setprecision(2) << report.slowdown_ratio_writes << "x\n";
    std::cout << "Constant Background Fill: " << SimulatedConsole::kTotalPixels << " pixels (640x400)\n\n";

    // Self-tests (Assertions)
    assert(report.scroll_line_avg_writes > SimulatedConsole::kTotalPixels);
    assert(report.slowdown_ratio_writes > 100.0);
    assert(!report.lines.empty());
    assert(report.lines[24].did_scroll); // Line 25 triggers scroll (rows 0..24 are full)

    std::cout << ">> [PASS] All Chapter 9.6 C++20 self-verification assertions passed!\n";
    return 0;
}
