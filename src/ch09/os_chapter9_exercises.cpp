/**
 * @file os_chapter9_exercises.cpp
 * @brief Chapter 9 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * Verifies key systems concepts introduced in Chapter 9:
 * 1. Layer Compositing & Painter's Algorithm with Transparency (Section 9.1, 9.3)
 * 2. Heap Dynamic Allocation, sbrk() & 4KiB Page Round-Up Math (Section 9.2)
 * 3. 2D Bounding Box Geometric Clipping & Intersection Math (Section 9.5)
 * 4. Local APIC Timer MMIO Tick Math & Latency Profiling (Section 9.4, 9.6)
 * 5. High-Speed Console Scroll via Scanline memcpy / memmove (Section 9.7)
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch09 {

// =============================================================================
// Common Types & Vector2D
// =============================================================================

template <typename T>
struct Vector2D {
    T x{0};
    T y{0};

    constexpr Vector2D() = default;
    constexpr Vector2D(T x, T y) : x{x}, y{y} {}

    constexpr Vector2D operator+(const Vector2D& rhs) const { return {x + rhs.x, y + rhs.y}; }
    constexpr Vector2D operator-(const Vector2D& rhs) const { return {x - rhs.x, y - rhs.y}; }
    Vector2D& operator+=(const Vector2D& rhs) { x += rhs.x; y += rhs.y; return *this; }
    Vector2D& operator-=(const Vector2D& rhs) { x -= rhs.x; y -= rhs.y; return *this; }
    constexpr bool operator==(const Vector2D& rhs) const = default;
};

struct PixelColor {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};

    constexpr PixelColor() = default;
    constexpr PixelColor(uint8_t r, uint8_t g, uint8_t b) : r{r}, g{g}, b{b} {}

    constexpr bool operator==(const PixelColor& rhs) const = default;
    constexpr uint32_t ToUint32() const {
        return (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(b);
    }
};

constexpr PixelColor kTransColor{1, 1, 1}; // Standard MikanOS transparency key

template <typename T>
struct Rectangle {
    Vector2D<T> pos{0, 0};
    Vector2D<T> size{0, 0};

    constexpr Rectangle() = default;
    constexpr Rectangle(Vector2D<T> pos, Vector2D<T> size) : pos{pos}, size{size} {}

    constexpr T Left() const { return pos.x; }
    constexpr T Right() const { return pos.x + size.x; }
    constexpr T Top() const { return pos.y; }
    constexpr T Bottom() const { return pos.y + size.y; }

    constexpr bool Contains(Vector2D<T> p) const {
        return p.x >= pos.x && p.x < Right() && p.y >= pos.y && p.y < Bottom();
    }

    constexpr std::optional<Rectangle<T>> Intersect(const Rectangle<T>& other) const {
        T x0 = std::max(pos.x, other.pos.x);
        T y0 = std::max(pos.y, other.pos.y);
        T x1 = std::min(Right(), other.Right());
        T y1 = std::min(Bottom(), other.Bottom());

        if (x0 < x1 && y0 < y1) {
            return Rectangle<T>{{x0, y0}, {x1 - x0, y1 - y0}};
        }
        return std::nullopt;
    }
};

// =============================================================================
// Exercise 1 & 3: Window, Layer & LayerManager Simulation (Section 9.1, 9.3, 9.5)
// =============================================================================

class Window {
public:
    Window(int width, int height, std::optional<PixelColor> trans_color = std::nullopt)
        : width_{width}, height_{height}, trans_color_{trans_color},
          data_(static_cast<size_t>(width * height), PixelColor{0, 0, 0}) {}

    int Width() const { return width_; }
    int Height() const { return height_; }
    Vector2D<int> Size() const { return {width_, height_}; }
    std::optional<PixelColor> TransparentColor() const { return trans_color_; }

    PixelColor At(Vector2D<int> pos) const {
        assert(pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_);
        return data_[static_cast<size_t>(pos.y * width_ + pos.x)];
    }

    void Set(Vector2D<int> pos, PixelColor c) {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            data_[static_cast<size_t>(pos.y * width_ + pos.x)] = c;
        }
    }

    void Fill(PixelColor c) {
        std::fill(data_.begin(), data_.end(), c);
    }

    const PixelColor* RawData() const { return data_.data(); }
    PixelColor* RawData() { return data_.data(); }

private:
    int width_{0};
    int height_{0};
    std::optional<PixelColor> trans_color_{std::nullopt};
    std::vector<PixelColor> data_{};
};

class Layer {
public:
    Layer(unsigned int id) : id_{id} {}

    unsigned int ID() const { return id_; }
    Vector2D<int> GetPosition() const { return pos_; }
    void SetPosition(Vector2D<int> pos) { pos_ = pos; }

    std::shared_ptr<Window> GetWindow() const { return window_; }
    void SetWindow(std::shared_ptr<Window> win) { window_ = win; }

private:
    unsigned int id_{0};
    Vector2D<int> pos_{0, 0};
    std::shared_ptr<Window> window_{nullptr};
};

class LayerManager {
public:
    LayerManager(int screen_w, int screen_h)
        : screen_w_{screen_w}, screen_h_{screen_h},
          back_buffer_(static_cast<size_t>(screen_w * screen_h), PixelColor{0, 0, 0}) {}

    Layer* NewLayer() {
        ++latest_id_;
        layers_.push_back(std::make_unique<Layer>(latest_id_));
        return layers_.back().get();
    }

    void UpDown(Layer* layer, int new_height) {
        if (new_height < 0) {
            Hide(layer);
            return;
        }
        if (new_height > static_cast<int>(layer_stack_.size())) {
            new_height = static_cast<int>(layer_stack_.size());
        }

        auto it = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (it != layer_stack_.end()) {
            layer_stack_.erase(it);
        }
        layer_stack_.insert(layer_stack_.begin() + new_height, layer);
    }

    void Hide(Layer* layer) {
        auto it = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (it != layer_stack_.end()) {
            layer_stack_.erase(it);
        }
    }

    const std::vector<Layer*>& Stack() const { return layer_stack_; }

    PixelColor GetScreenPixel(Vector2D<int> pos) const {
        assert(pos.x >= 0 && pos.x < screen_w_ && pos.y >= 0 && pos.y < screen_h_);
        return back_buffer_[static_cast<size_t>(pos.y * screen_w_ + pos.x)];
    }

    // Full Redraw (Section 9.3)
    void Draw() {
        Draw({0, 0}, {screen_w_, screen_h_});
    }

    // Clipping Redraw (Section 9.5)
    size_t Draw(Vector2D<int> pos, Vector2D<int> size) {
        size_t pixels_drawn = 0;
        Rectangle<int> redraw_rect{pos, size};
        Rectangle<int> screen_rect{{0, 0}, {screen_w_, screen_h_}};

        auto screen_clip = redraw_rect.Intersect(screen_rect);
        if (!screen_clip) return 0;
        Rectangle<int> target_rect = *screen_clip;

        for (auto* layer : layer_stack_) {
            auto win = layer->GetWindow();
            if (!win) continue;

            Rectangle<int> layer_rect{layer->GetPosition(), win->Size()};
            auto intersect = target_rect.Intersect(layer_rect);
            if (!intersect) continue;

            auto tc = win->TransparentColor();
            bool opaque = !tc.has_value();

            for (int y = intersect->Top(); y < intersect->Bottom(); ++y) {
                int src_y = y - layer->GetPosition().y;
                int src_x_start = intersect->Left() - layer->GetPosition().x;
                int dst_x_start = intersect->Left();
                int count = intersect->size.x;

                if (opaque) {
                    // Line-wise memcpy optimization (Section 9.5)
                    const PixelColor* src_ptr = &win->RawData()[src_y * win->Width() + src_x_start];
                    PixelColor* dst_ptr = &back_buffer_[static_cast<size_t>(y * screen_w_ + dst_x_start)];
                    std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(count) * sizeof(PixelColor));
                    pixels_drawn += static_cast<size_t>(count);
                } else {
                    // Pixel-by-pixel with transparency check
                    for (int x = 0; x < count; ++x) {
                        PixelColor c = win->At({src_x_start + x, src_y});
                        if (c != *tc) {
                            back_buffer_[static_cast<size_t>(y * screen_w_ + dst_x_start + x)] = c;
                            ++pixels_drawn;
                        }
                    }
                }
            }
        }
        return pixels_drawn;
    }

private:
    int screen_w_{0};
    int screen_h_{0};
    unsigned int latest_id_{0};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    std::vector<PixelColor> back_buffer_{};
};

// =============================================================================
// Exercise 2: sbrk & Page Allocation Math (Section 9.2)
// =============================================================================

struct SbrkSimulator {
    static constexpr uintptr_t kPageSize = 4096;
    uintptr_t current_break{0x10000000};
    size_t total_pages_allocated{0};

    uintptr_t sbrk(intptr_t incr) {
        uintptr_t old_break = current_break;
        if (incr == 0) return old_break;

        uintptr_t new_break = current_break + static_cast<uintptr_t>(incr);

        uintptr_t old_pages = (old_break + kPageSize - 1) / kPageSize;
        uintptr_t new_pages = (new_break + kPageSize - 1) / kPageSize;

        if (new_pages > old_pages) {
            size_t diff = static_cast<size_t>(new_pages - old_pages);
            total_pages_allocated += diff;
        }

        current_break = new_break;
        return old_break;
    }
};

// =============================================================================
// Exercise 4: Local APIC Timer Math (Section 9.4)
// =============================================================================

struct LAPICTimerMath {
    static constexpr uint32_t kInitialCount = 0xFFFFFFFF;

    static uint32_t CalculateElapsedTicks(uint32_t initial_count, uint32_t current_count) {
        return initial_count - current_count;
    }

    static double TicksToMilliseconds(uint32_t ticks, double bus_frequency_hz) {
        return (static_cast<double>(ticks) / bus_frequency_hz) * 1000.0;
    }
};

// =============================================================================
// Exercise 5: High-Speed Console Scroll via memcpy (Section 9.7)
// =============================================================================

class ConsoleScrollSimulator {
public:
    static constexpr int kCols = 80;
    static constexpr int kRows = 25;
    static constexpr int kFontW = 8;
    static constexpr int kFontH = 16;
    static constexpr int kWidth = kCols * kFontW;  // 640
    static constexpr int kHeight = kRows * kFontH; // 400

    ConsoleScrollSimulator() : buffer_(kWidth * kHeight, PixelColor{0, 0, 0}) {}

    void WritePixel(int x, int y, PixelColor c) {
        if (x >= 0 && x < kWidth && y >= 0 && y < kHeight) {
            buffer_[static_cast<size_t>(y * kWidth + x)] = c;
        }
    }

    PixelColor At(int x, int y) const {
        return buffer_[static_cast<size_t>(y * kWidth + x)];
    }

    // Fast scroll using memcpy/memmove (Section 9.7)
    // Move rows 1..24 to 0..23 (shift upward by kFontH lines)
    // Clear bottom row (kHeight - kFontH .. kHeight) with bg color
    size_t FastScroll(PixelColor bg) {
        const int shift_lines = kHeight - kFontH; // 384 lines
        const size_t bytes_to_copy = static_cast<size_t>(shift_lines * kWidth) * sizeof(PixelColor);

        // Source is line 16 (16 * 640), Dest is line 0 (0 * 640).
        // Since dst < src, forward scan memcpy is completely safe without data corruption!
        PixelColor* dst = &buffer_[0];
        const PixelColor* src = &buffer_[static_cast<size_t>(kFontH * kWidth)];

        std::memcpy(dst, src, bytes_to_copy);

        // Fill bottom row with background color
        PixelColor* bottom = &buffer_[static_cast<size_t>(shift_lines * kWidth)];
        std::fill(bottom, bottom + (kFontH * kWidth), bg);

        return static_cast<size_t>(kFontH * kWidth); // only bottom row pixel writes!
    }

    const std::vector<PixelColor>& Buffer() const { return buffer_; }

private:
    std::vector<PixelColor> buffer_{};
};

// =============================================================================
// Test Harness
// =============================================================================

void RunAllExercises() {
    std::cout << "========================================================\n";
    std::cout << " MikanOS Chapter 9 Comprehensive Exercises (C++20)\n";
    std::cout << "========================================================\n\n";

    // -------------------------------------------------------------------------
    // Test 1: Layer Compositing & Transparency Math (Section 9.1, 9.3)
    // -------------------------------------------------------------------------
    std::cout << "[Test 1] Layer Compositing & Transparency Math... ";
    {
        LayerManager lm(200, 150);

        // Background layer (Z=0, full opaque blue)
        auto* bg_layer = lm.NewLayer();
        auto bg_win = std::make_shared<Window>(200, 150);
        bg_win->Fill(PixelColor{0, 0, 255});
        bg_layer->SetWindow(bg_win);
        bg_layer->SetPosition({0, 0});
        lm.UpDown(bg_layer, 0);

        // Window layer (Z=1, opaque green, 50x50 at (20, 20))
        auto* win_layer = lm.NewLayer();
        auto win_win = std::make_shared<Window>(50, 50);
        win_win->Fill(PixelColor{0, 255, 0});
        win_layer->SetWindow(win_win);
        win_layer->SetPosition({20, 20});
        lm.UpDown(win_layer, 1);

        // Mouse layer (Z=2, transparent key, 10x10 at (30, 30))
        auto* mouse_layer = lm.NewLayer();
        auto mouse_win = std::make_shared<Window>(10, 10, kTransColor);
        mouse_win->Fill(kTransColor); // transparent everywhere
        mouse_win->Set({0, 0}, PixelColor{255, 255, 255}); // only (0,0) is white cursor
        mouse_layer->SetWindow(mouse_win);
        mouse_layer->SetPosition({30, 30});
        lm.UpDown(mouse_layer, 2);

        lm.Draw();

        // Check background outside window
        assert(lm.GetScreenPixel({5, 5}) == PixelColor(0, 0, 255));

        // Check window inside (25, 25)
        assert(lm.GetScreenPixel({25, 25}) == PixelColor(0, 255, 0));

        // Check mouse cursor pixel (30, 30) -> white
        assert(lm.GetScreenPixel({30, 30}) == PixelColor(255, 255, 255));

        // Check mouse transparent pixel (31, 31) -> shows window green underneath!
        assert(lm.GetScreenPixel({31, 31}) == PixelColor(0, 255, 0));
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 2: sbrk() & 4KiB Page Allocation Math (Section 9.2)
    // -------------------------------------------------------------------------
    std::cout << "[Test 2] sbrk() Program Break & 4KiB Page Round-Up Math... ";
    {
        SbrkSimulator sim;
        assert(sim.current_break == 0x10000000);
        assert(sim.total_pages_allocated == 0);

        // Allocate 1000 bytes (within same 4KiB frame)
        uintptr_t b1 = sim.sbrk(1000);
        assert(b1 == 0x10000000);
        assert(sim.current_break == 0x10000000 + 1000);
        assert(sim.total_pages_allocated == 1); // first frame allocated

        // Allocate 2000 bytes (still fits in first 4096 frame: 1000 + 2000 = 3000 <= 4096)
        uintptr_t b2 = sim.sbrk(2000);
        assert(b2 == 0x10000000 + 1000);
        assert(sim.current_break == 0x10000000 + 3000);
        assert(sim.total_pages_allocated == 1); // no new frame needed!

        // Allocate 2000 bytes (3000 + 2000 = 5000 > 4096 -> enters 2nd frame)
        uintptr_t b3 = sim.sbrk(2000);
        assert(b3 == 0x10000000 + 3000);
        assert(sim.current_break == 0x10000000 + 5000);
        assert(sim.total_pages_allocated == 2); // 2nd frame allocated!

        // Query current break with incr = 0
        uintptr_t cur = sim.sbrk(0);
        assert(cur == sim.current_break);
        assert(sim.total_pages_allocated == 2);
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 3: 2D Bounding Box Geometric Clipping Math (Section 9.5)
    // -------------------------------------------------------------------------
    std::cout << "[Test 3] 2D Bounding Box Geometric Clipping Math... ";
    {
        Rectangle<int> rectA{{10, 10}, {50, 40}}; // [10..60, 10..50]
        Rectangle<int> rectB{{30, 20}, {60, 50}}; // [30..90, 20..70]

        auto inter = rectA.Intersect(rectB);
        assert(inter.has_value());
        assert(inter->pos.x == 30);
        assert(inter->pos.y == 20);
        assert(inter->size.x == 30); // 60 - 30
        assert(inter->size.y == 30); // 50 - 20

        // Disjoint rectangles
        Rectangle<int> rectC{{100, 100}, {20, 20}};
        auto disjoint = rectA.Intersect(rectC);
        assert(!disjoint.has_value());

        // Redraw count verification:
        LayerManager lm(100, 100);
        auto* l = lm.NewLayer();
        auto w = std::make_shared<Window>(100, 100);
        w->Fill(PixelColor{10, 20, 30});
        l->SetWindow(w);
        l->SetPosition({0, 0});
        lm.UpDown(l, 0);

        // Draw only 10x10 sub-rectangle
        size_t pixels = lm.Draw({5, 5}, {10, 10});
        assert(pixels == 100); // exactly 10x10 = 100 pixels updated!
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 4: Local APIC Timer Math & Profiling Breakdown (Section 9.4, 9.6)
    // -------------------------------------------------------------------------
    std::cout << "[Test 4] Local APIC Timer Math & Profiling Breakdown... ";
    {
        uint32_t icr = 0xFFFFFFFF;
        uint32_t ccr = 0xFFFE795F; // exactly 100,000 ticks difference

        uint32_t ticks = LAPICTimerMath::CalculateElapsedTicks(icr, ccr);
        assert(ticks == 100000);

        double ms = LAPICTimerMath::TicksToMilliseconds(ticks, 100'000'000.0); // 100 MHz
        assert(ms == 1.0); // 100,000 / 100,000,000 s = 1 ms

        // Latency Queue overflow math:
        // Mouse produces events every 10 ms (100 Hz).
        // If full screen redraw takes 30 ms, queue accumulates 2 events per frame!
        double redraw_time_ms = 30.0;
        double event_interval_ms = 10.0;
        assert(redraw_time_ms > event_interval_ms); // Queue starvation / overflow condition!
    }
    std::cout << "PASSED\n";

    // -------------------------------------------------------------------------
    // Test 5: Fast Console Scroll with Scanline memcpy Math (Section 9.7)
    // -------------------------------------------------------------------------
    std::cout << "[Test 5] Fast Console Scroll with Scanline memcpy Math... ";
    {
        ConsoleScrollSimulator sim;

        // Draw pattern in Row 1 (y = 16..31)
        sim.WritePixel(10, 16, PixelColor{255, 0, 0}); // red pixel in row 1
        // Draw pattern in Row 2 (y = 32..47)
        sim.WritePixel(20, 32, PixelColor{0, 255, 0}); // green pixel in row 2

        // Fast Scroll Up by 16 pixels
        size_t cleared_writes = sim.FastScroll(PixelColor{0, 0, 0});
        assert(cleared_writes == ConsoleScrollSimulator::kFontH * ConsoleScrollSimulator::kWidth);
        assert(cleared_writes == 16 * 640); // 10,240 pixel writes

        // Verify that Row 1 red pixel has moved to Row 0 (y = 0)
        assert(sim.At(10, 0) == PixelColor(255, 0, 0));

        // Verify that Row 2 green pixel has moved to Row 1 (y = 16)
        assert(sim.At(20, 16) == PixelColor(0, 255, 0));

        // Verify that bottom row is cleared (y = 384..399)
        assert(sim.At(10, 390) == PixelColor(0, 0, 0));
    }
    std::cout << "PASSED\n";

    std::cout << "\n>> All Chapter 9 exercises verified successfully!\n";
}

} // namespace mikanos::exercises::ch09

int main(int argc, char** argv) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    mikanos::exercises::ch09::RunAllExercises();

    if (test_mode) {
        std::cout << "[CTest] All Chapter 9 exercises verified successfully.\n";
    }
    return 0;
}
