/**
 * @file os_chapter10_partial_draw.cpp
 * @brief Chapter 10.4: Eliminating Flicker with Partial Draw & Layer Redraw Limitations (C++20)
 *
 * Demonstrates:
 * 1. 2D Rectangle Intersection Math (operator&) and 3-way Clipped Buffer Copy:
 *    `dst_outline & src_outline & src_area_shifted`
 * 2. Partial FrameBuffer Copy with Arbitrary Source Area (`FrameBuffer::Copy`)
 * 3. LayerManager Targeted Redraw: `Draw(unsigned int id)`
 *    - Skips lower layers (Background, Console) below the updated layer
 *    - Redraws only layers at or above the target layer inside the window bounding box
 * 4. LayerManager Position Update: `Move(id, new_pos)`
 *    - Automatically redraws old position with full stack: `Draw({old_pos, window_size})`
 *    - Automatically redraws new position with target layer and above: `Draw(id)`
 * 5. Quantitative Redraw Pixel & Bandwidth Comparison:
 *    - Full Redraw (1,051,112 px / ~4.2 MB) vs. Layer Draw (8,320 px / ~33.28 KB): ~126.3x reduction
 * 6. Analysis of Persistent Flicker Limitation:
 *    - Window-level redraw solves background flashing, but intra-window clearing
 *      (`FillRectangle` background wipe -> `WriteString`) still exposes intermediate states
 *      directly to VRAM without double buffering (previewing Chapter 10.5).
 */

#include <algorithm>
#include <array>
#include <cassert>
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

namespace mikanos::study {

// =============================================================================
// 1. Geometric Data Types & 2D Rectangle Intersection Math
// =============================================================================

template <typename T>
struct Vector2D {
    T x, y;

    constexpr bool operator==(const Vector2D& rhs) const noexcept {
        return x == rhs.x && y == rhs.y;
    }

    constexpr Vector2D operator+(const Vector2D& rhs) const noexcept {
        return {x + rhs.x, y + rhs.y};
    }

    constexpr Vector2D operator-(const Vector2D& rhs) const noexcept {
        return {x - rhs.x, y - rhs.y};
    }
};

template <typename T>
struct Rectangle {
    Vector2D<T> pos;
    Vector2D<T> size;

    constexpr bool operator==(const Rectangle& rhs) const noexcept {
        return pos == rhs.pos && size == rhs.size;
    }
};

/**
 * @brief Calculate the 2D geometric intersection of two axis-aligned rectangles.
 * Returns a rectangle of size {0, 0} if there is no overlapping area.
 */
template <typename T>
constexpr Rectangle<T> operator&(const Rectangle<T>& lhs, const Rectangle<T>& rhs) noexcept {
    const T left = std::max(lhs.pos.x, rhs.pos.x);
    const T right = std::min(lhs.pos.x + lhs.size.x, rhs.pos.x + rhs.size.x);
    const T top = std::max(lhs.pos.y, rhs.pos.y);
    const T bottom = std::min(lhs.pos.y + lhs.size.y, rhs.pos.y + rhs.size.y);

    if (left >= right || top >= bottom) {
        return {{0, 0}, {0, 0}};
    }
    return {{left, top}, {right - left, bottom - top}};
}

// =============================================================================
// 2. Colors and PixelWriter
// =============================================================================

struct PixelColor {
    uint8_t r, g, b;

    constexpr bool operator==(const PixelColor& rhs) const noexcept {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }

    constexpr bool operator!=(const PixelColor& rhs) const noexcept {
        return !(*this == rhs);
    }
};

constexpr PixelColor ToColor(uint32_t c) noexcept {
    return {
        static_cast<uint8_t>((c >> 16) & 0xff),
        static_cast<uint8_t>((c >> 8) & 0xff),
        static_cast<uint8_t>(c & 0xff)
    };
}

constexpr PixelColor kDesktopBGColor{0x00, 0x80, 0x80};   // Teal
constexpr PixelColor kWindowBGColor{0xc6, 0xc6, 0xc6};    // Light Gray
constexpr PixelColor kConsoleBGColor{0x00, 0x00, 0x00};   // Black
constexpr PixelColor kConsoleFGColor{0xff, 0xff, 0xff};   // White
constexpr PixelColor kHighlightColor{0xff, 0xff, 0xff};   // White
constexpr PixelColor kShadowColor{0x84, 0x84, 0x84};      // Dark Gray
constexpr PixelColor kTitleBarColor{0x00, 0x00, 0x84};    // Classic Navy

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(Vector2D<int> pos, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

void FillRectangle(PixelWriter& writer, Vector2D<int> pos, Vector2D<int> size, const PixelColor& c) {
    for (int dy = 0; dy < size.y; ++dy) {
        for (int dx = 0; dx < size.x; ++dx) {
            writer.Write({pos.x + dx, pos.y + dy}, c);
        }
    }
}

// =============================================================================
// 3. FrameBuffer with Rectangular Clipping Copy
// =============================================================================

class FrameBuffer : public PixelWriter {
public:
    FrameBuffer(int width, int height, PixelColor default_color = {0, 0, 0})
        : width_{width}, height_{height}, buffer_(width * height, default_color) {}

    void Write(Vector2D<int> pos, const PixelColor& c) override {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            buffer_[pos.y * width_ + pos.x] = c;
        }
    }

    int Width() const override { return width_; }
    int Height() const override { return height_; }
    Vector2D<int> Size() const noexcept { return {width_, height_}; }

    PixelColor At(Vector2D<int> pos) const {
        if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
            return buffer_[pos.y * width_ + pos.x];
        }
        return {0, 0, 0};
    }

    const PixelColor* RawData() const noexcept { return buffer_.data(); }
    PixelColor* RawData() noexcept { return buffer_.data(); }

    /**
     * @brief Copy rectangular sub-region `src_area` from `src` to `dst_pos` with 3-way clipping.
     * Corresponds to `FrameBuffer::Copy` in `osbook_day10d`.
     */
    size_t Copy(Vector2D<int> dst_pos, const FrameBuffer& src, const Rectangle<int>& src_area) {
        const Rectangle<int> src_area_shifted{dst_pos, src_area.size};
        const Rectangle<int> src_outline{dst_pos - src_area.pos, src.Size()};
        const Rectangle<int> dst_outline{{0, 0}, Size()};

        // 3-way geometric intersection
        const auto copy_area = dst_outline & src_outline & src_area_shifted;
        if (copy_area.size.x <= 0 || copy_area.size.y <= 0) {
            return 0; // No intersection
        }

        const auto src_start_pos = copy_area.pos - (dst_pos - src_area.pos);

        for (int y = 0; y < copy_area.size.y; ++y) {
            const int dst_y = copy_area.pos.y + y;
            const int src_y = src_start_pos.y + y;
            for (int x = 0; x < copy_area.size.x; ++x) {
                const int dst_x = copy_area.pos.x + x;
                const int src_x = src_start_pos.x + x;
                buffer_[dst_y * width_ + dst_x] = src.At({src_x, src_y});
            }
        }

        return static_cast<size_t>(copy_area.size.x * copy_area.size.y);
    }

private:
    int width_, height_;
    std::vector<PixelColor> buffer_;
};

// =============================================================================
// 4. Window with Clipped DrawTo Support
// =============================================================================

class Window {
public:
    class WindowWriter : public PixelWriter {
    public:
        WindowWriter(Window& window) : window_{window} {}
        void Write(Vector2D<int> pos, const PixelColor& c) override {
            window_.Write(pos, c);
        }
        int Width() const override { return window_.Width(); }
        int Height() const override { return window_.Height(); }
    private:
        Window& window_;
    };

    Window(int width, int height, PixelColor default_color = kWindowBGColor)
        : width_{width}, height_{height}, shadow_buffer_(width, height, default_color), writer_{*this} {}

    void Write(Vector2D<int> pos, const PixelColor& c) {
        shadow_buffer_.Write(pos, c);
    }

    PixelColor At(Vector2D<int> pos) const {
        return shadow_buffer_.At(pos);
    }

    int Width() const noexcept { return width_; }
    int Height() const noexcept { return height_; }
    Vector2D<int> Size() const noexcept { return {width_, height_}; }
    PixelWriter* Writer() noexcept { return &writer_; }

    void SetTransparentColor(std::optional<PixelColor> c) {
        transparent_color_ = c;
    }

    std::optional<PixelColor> TransparentColor() const noexcept {
        return transparent_color_;
    }

    /**
     * @brief Draw window content to destination FrameBuffer restricted to given `area`.
     * Corresponds to `Window::DrawTo(dst, pos, area)` in `osbook_day10d`.
     */
    size_t DrawTo(FrameBuffer& dst, Vector2D<int> pos, const Rectangle<int>& area) {
        size_t rendered_pixels = 0;
        if (!transparent_color_) {
            // Opaque Window: Fast-path via clipped FrameBuffer::Copy
            const Rectangle<int> window_area{pos, Size()};
            const Rectangle<int> intersection = area & window_area;
            if (intersection.size.x > 0 && intersection.size.y > 0) {
                rendered_pixels = dst.Copy(intersection.pos, shadow_buffer_,
                                           {intersection.pos - pos, intersection.size});
            }
            return rendered_pixels;
        }

        // Transparent Window (e.g., Mouse Cursor): Pixel-by-pixel check
        const auto tc = transparent_color_.value();
        const int start_y = std::max(0, std::max(area.pos.y - pos.y, 0 - pos.y));
        const int end_y = std::min(Height(), std::min(area.pos.y + area.size.y - pos.y, dst.Height() - pos.y));
        const int start_x = std::max(0, std::max(area.pos.x - pos.x, 0 - pos.x));
        const int end_x = std::min(Width(), std::min(area.pos.x + area.size.x - pos.x, dst.Width() - pos.x));

        for (int y = start_y; y < end_y; ++y) {
            for (int x = start_x; x < end_x; ++x) {
                const auto c = At({x, y});
                if (c != tc) {
                    dst.Write(pos + Vector2D<int>{x, y}, c);
                    ++rendered_pixels;
                }
            }
        }
        return rendered_pixels;
    }

private:
    int width_, height_;
    FrameBuffer shadow_buffer_;
    WindowWriter writer_;
    std::optional<PixelColor> transparent_color_{std::nullopt};
};

// =============================================================================
// 5. Layer & LayerManager with Selective Redraw Draw(id)
// =============================================================================

class Layer {
public:
    explicit Layer(unsigned int id) : id_{id} {}

    unsigned int ID() const noexcept { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }

    std::shared_ptr<Window> GetWindow() const noexcept { return window_; }

    Vector2D<int> GetPosition() const noexcept { return pos_; }

    Layer& Move(Vector2D<int> pos) noexcept {
        pos_ = pos;
        return *this;
    }

    size_t DrawTo(FrameBuffer& screen, const Rectangle<int>& area) const {
        if (window_) {
            return window_->DrawTo(screen, pos_, area);
        }
        return 0;
    }

private:
    unsigned int id_;
    Vector2D<int> pos_{0, 0};
    std::shared_ptr<Window> window_{nullptr};
};

class LayerManager {
public:
    struct RedrawStats {
        size_t total_calls{0};
        size_t layers_drawn{0};
        size_t layers_skipped{0};
        size_t pixels_drawn{0};
    };

    void SetWriter(FrameBuffer* screen) noexcept {
        screen_ = screen;
    }

    Layer& NewLayer() {
        ++latest_id_;
        auto layer = std::make_unique<Layer>(latest_id_);
        Layer* ptr = layer.get();
        layers_.push_back(std::move(layer));
        return *ptr;
    }

    Layer* FindLayer(unsigned int id) const {
        for (const auto& l : layers_) {
            if (l->ID() == id) return l.get();
        }
        return nullptr;
    }

    void UpDown(unsigned int id, int new_height) {
        if (new_height < 0) {
            Hide(id);
            return;
        }
        if (new_height > static_cast<int>(layer_stack_.size())) {
            new_height = static_cast<int>(layer_stack_.size());
        }

        auto layer = FindLayer(id);
        auto it = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (it != layer_stack_.end()) {
            layer_stack_.erase(it);
        }
        layer_stack_.insert(layer_stack_.begin() + new_height, layer);
    }

    void Hide(unsigned int id) {
        auto layer = FindLayer(id);
        auto it = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (it != layer_stack_.end()) {
            layer_stack_.erase(it);
        }
    }

    /**
     * @brief Redraw all layers within given bounding area (from bottom to top).
     * Corresponds to `LayerManager::Draw(area)` in `osbook_day10d`.
     */
    void Draw(const Rectangle<int>& area) const {
        ++stats_.total_calls;
        for (auto layer : layer_stack_) {
            stats_.pixels_drawn += layer->DrawTo(*screen_, area);
            ++stats_.layers_drawn;
        }
    }

    /**
     * @brief Redraw only the specified layer and layers ABOVE it within the layer's bounding box.
     * Lower layers (e.g., Background, Console) are skipped, eliminating full-screen redraw.
     * Corresponds to `LayerManager::Draw(unsigned int id)` in `osbook_day10d`.
     */
    void Draw(unsigned int id) const {
        ++stats_.total_calls;
        bool draw = false;
        Rectangle<int> window_area;

        for (auto layer : layer_stack_) {
            if (layer->ID() == id) {
                window_area.size = layer->GetWindow()->Size();
                window_area.pos = layer->GetPosition();
                draw = true;
            }
            if (draw) {
                stats_.pixels_drawn += layer->DrawTo(*screen_, window_area);
                ++stats_.layers_drawn;
            } else {
                ++stats_.layers_skipped;
            }
        }
    }

    /**
     * @brief Move layer and automatically trigger:
     * 1. Full redraw of old rectangle to restore lower background layers
     * 2. Targeted redraw of new rectangle for target layer and above
     * Corresponds to `LayerManager::Move` in `osbook_day10d`.
     */
    void Move(unsigned int id, Vector2D<int> new_pos) {
        auto layer = FindLayer(id);
        const auto window_size = layer->GetWindow()->Size();
        const auto old_pos = layer->GetPosition();
        layer->Move(new_pos);

        // 1. Restore background at old position
        Draw({old_pos, window_size});
        // 2. Draw target layer and above at new position
        Draw(id);
    }

    const std::vector<Layer*>& Stack() const noexcept { return layer_stack_; }
    RedrawStats& Stats() noexcept { return stats_; }
    const RedrawStats& Stats() const noexcept { return stats_; }
    void ResetStats() noexcept { stats_ = RedrawStats{}; }

private:
    FrameBuffer* screen_{nullptr};
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<Layer*> layer_stack_;
    unsigned int latest_id_{0};
    mutable RedrawStats stats_;
};

} // namespace mikanos::study

// =============================================================================
// Main Verification Runner
// =============================================================================

int main() {
    using namespace mikanos::study;
    std::cout << "================================================================\n";
    std::cout << "MikanOS Chapter 10.4: Eliminating Flicker with Partial Draw\n";
    std::cout << "================================================================\n";

    // -------------------------------------------------------------------------
    // Test 1: 2D Rectangle Intersection Math (operator&)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 1: 2D Rectangle Intersection Math ---\n";
    Rectangle<int> r1{{10, 10}, {50, 40}};  // [10, 60) x [10, 50)
    Rectangle<int> r2{{30, 20}, {60, 50}};  // [30, 90) x [20, 70)
    Rectangle<int> inter = r1 & r2;
    assert(inter.pos == (Vector2D<int>{30, 20}));
    assert(inter.size == (Vector2D<int>{30, 30})); // [30, 60) x [20, 50) -> width=30, height=30
    std::cout << "[PASS] Standard partial overlap intersection verified: ("
              << inter.pos.x << ", " << inter.pos.y << ") size ("
              << inter.size.x << ", " << inter.size.y << ")\n";

    // Completely disjoint rectangles
    Rectangle<int> r_disjoint{{100, 100}, {20, 20}};
    Rectangle<int> inter_empty = r1 & r_disjoint;
    assert(inter_empty.size == (Vector2D<int>{0, 0}));
    std::cout << "[PASS] Disjoint rectangles intersection empty size (0, 0) verified\n";

    // Sub-rectangle containment
    Rectangle<int> r_inner{{20, 15}, {10, 10}};
    Rectangle<int> inter_inner = r1 & r_inner;
    assert(inter_inner == r_inner);
    std::cout << "[PASS] Full containment intersection verified\n";

    // -------------------------------------------------------------------------
    // Test 2: FrameBuffer::Copy with 3-Way Clipped Source Region
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 2: FrameBuffer Partial Copy & 3-Way Clipping ---\n";
    FrameBuffer screen(100, 100, {0, 0, 0});
    FrameBuffer src(40, 40, {0xff, 0, 0}); // 40x40 red buffer
    // Mark a small 10x10 sub-region in green
    FillRectangle(src, {10, 10}, {10, 10}, {0, 0xff, 0});

    // Copy src_area {10, 10, 10, 10} to screen at {50, 50}
    size_t copied = screen.Copy({50, 50}, src, {{10, 10}, {10, 10}});
    assert(copied == 100); // 10x10 = 100 pixels
    assert(screen.At({50, 50}) == (PixelColor{0, 0xff, 0}));
    assert(screen.At({49, 50}) == (PixelColor{0, 0, 0})); // Outside unchanged
    assert(screen.At({60, 50}) == (PixelColor{0, 0, 0}));
    std::cout << "[PASS] FrameBuffer::Copy sub-region exact transfer verified\n";

    // Copy with boundary clipping (partially outside destination screen)
    FrameBuffer screen_small(50, 50, {0, 0, 0});
    size_t clipped_copied = screen_small.Copy({45, 45}, src, {{0, 0}, {20, 20}});
    // clipped overlap is [45, 50) x [45, 50) = 5x5 = 25 pixels
    assert(clipped_copied == 25);
    std::cout << "[PASS] Screen boundary clipping copy verified (25 pixels copied out of 400)\n";

    // -------------------------------------------------------------------------
    // Test 3: Layer Stack Setup and Selective Redraw (Draw(id))
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 3: LayerManager::Draw(id) Selective Redraw ---\n";
    const int kScreenWidth = 1024;
    const int kScreenHeight = 768;
    FrameBuffer display(kScreenWidth, kScreenHeight, {0, 0, 0});

    LayerManager manager;
    manager.SetWriter(&display);

    // 1. Layer 0: Background Layer (1024x768, teal)
    auto bg_window = std::make_shared<Window>(kScreenWidth, kScreenHeight, kDesktopBGColor);
    auto& bg_layer = manager.NewLayer().SetWindow(bg_window).Move({0, 0});
    const unsigned int bg_id = bg_layer.ID();

    // 2. Layer 1: Console Layer (640x400, black, located at {0, 0})
    auto console_window = std::make_shared<Window>(80 * 8, 25 * 16, kConsoleBGColor);
    auto& console_layer = manager.NewLayer().SetWindow(console_window).Move({0, 0});
    const unsigned int console_id = console_layer.ID();

    // 3. Layer 2: Main Counter Window (160x52, gray, located at {300, 100})
    auto main_window = std::make_shared<Window>(160, 52, kWindowBGColor);
    auto& main_layer = manager.NewLayer().SetWindow(main_window).Move({300, 100});
    const unsigned int main_id = main_layer.ID();

    // 4. Layer 3: Mouse Cursor (15x24, located at {350, 110}, overlapping main_window)
    auto mouse_window = std::make_shared<Window>(15, 24, PixelColor{0, 0, 0});
    mouse_window->SetTransparentColor(PixelColor{0, 0, 0}); // transparent black
    // Give mouse cursor a distinctive white dot
    mouse_window->Write({0, 0}, {0xff, 0xff, 0xff});
    auto& mouse_layer = manager.NewLayer().SetWindow(mouse_window).Move({350, 110});
    const unsigned int mouse_id = mouse_layer.ID();

    // Set Up Stack order: BG (height 0), Console (height 1), MainWindow (height 2), Mouse (height 3)
    manager.UpDown(bg_id, 0);
    manager.UpDown(console_id, 1);
    manager.UpDown(main_id, 2);
    manager.UpDown(mouse_id, 3);

    assert(manager.Stack().size() == 4);
    assert(manager.Stack()[0]->ID() == bg_id);
    assert(manager.Stack()[1]->ID() == console_id);
    assert(manager.Stack()[2]->ID() == main_id);
    assert(manager.Stack()[3]->ID() == mouse_id);
    std::cout << "[PASS] 4-layer stack initialized in correct z-order\n";

    // Initial Full Screen Draw
    manager.ResetStats();
    manager.Draw({{0, 0}, display.Size()});
    std::cout << "Initial Full Draw: " << manager.Stats().layers_drawn << " layers drawn, "
              << manager.Stats().pixels_drawn << " pixels rendered\n";
    assert(manager.Stats().layers_drawn == 4);
    assert(display.At({300, 100}) == kWindowBGColor);
    assert(display.At({350, 110}) == (PixelColor{0xff, 0xff, 0xff})); // Mouse dot visible

    // -------------------------------------------------------------------------
    // Test 4: Selective Redraw `Draw(main_id)` Verification
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 4: Selective Redraw Draw(main_id) Verification ---\n";
    manager.ResetStats();

    // Simulate high-speed counter text update
    FillRectangle(*main_window->Writer(), {24, 28}, {80, 16}, kWindowBGColor);
    // Draw target layer only
    manager.Draw(main_id);

    std::cout << "Targeted Draw(main_id) Stats:\n"
              << "  Layers drawn:   " << manager.Stats().layers_drawn << " (MainWindow + Mouse)\n"
              << "  Layers skipped: " << manager.Stats().layers_skipped << " (BG + Console)\n"
              << "  Pixels drawn:   " << manager.Stats().pixels_drawn << "\n";

    // Assert that lower layers were skipped completely
    assert(manager.Stats().layers_skipped == 2); // BG and Console skipped!
    assert(manager.Stats().layers_drawn == 2);   // Main Window and Mouse above it
    assert(manager.Stats().pixels_drawn >= 160 * 52); // Window area copied
    assert(manager.Stats().pixels_drawn < 160 * 52 + 15 * 24); // Only small mouse intersection added
    std::cout << "[PASS] Background and Console layers strictly SKIPPED by Draw(main_id)\n";

    // -------------------------------------------------------------------------
    // Test 5: Quantitative Redraw Comparison & Bandwidth Reduction
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 5: Quantitative Pixel Redraw & Bandwidth Analysis ---\n";
    const size_t full_redraw_pixels = (kScreenWidth * kScreenHeight) + // BG
                                      (640 * 400) +                    // Console
                                      (160 * 52) +                     // Main Window
                                      (15 * 24);                       // Mouse
    const size_t partial_redraw_pixels = manager.Stats().pixels_drawn;
    const double reduction_factor = static_cast<double>(full_redraw_pixels) / partial_redraw_pixels;

    std::cout << "Full redraw pixels:      " << full_redraw_pixels << " px (~"
              << std::fixed << std::setprecision(2) << (full_redraw_pixels * 4.0) / (1024 * 1024) << " MB)\n";
    std::cout << "Partial Draw(id) pixels: " << partial_redraw_pixels << " px (~"
              << std::fixed << std::setprecision(2) << (partial_redraw_pixels * 4.0) / 1024.0 << " KB)\n";
    std::cout << "Pixel reduction ratio:   " << reduction_factor << "x (~"
              << std::setprecision(1) << (1.0 - 1.0 / reduction_factor) * 100.0 << "% reduction)\n";

    assert(reduction_factor > 100.0); // Must achieve over 100x reduction!
    std::cout << "[PASS] Pixel reduction ratio exceeds 100x (>120x observed)\n";

    // -------------------------------------------------------------------------
    // Test 6: LayerManager::Move Behavior
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 6: LayerManager::Move Behavior Verification ---\n";
    manager.ResetStats();
    manager.Move(main_id, {320, 120});

    assert(main_layer.GetPosition() == (Vector2D<int>{320, 120}));
    assert(manager.Stats().total_calls == 2); // 1. Draw(old_area), 2. Draw(id)
    std::cout << "[PASS] LayerManager::Move correctly invoked both old area restore and new area draw\n";

    // -------------------------------------------------------------------------
    // Test 7: Analysis of Persistent Intra-Window Flicker (Chapter 10.5 Preview)
    // -------------------------------------------------------------------------
    std::cout << "\n--- Test 7: Flicker Elimination Scope & Limitations ---\n";
    std::cout << "  [SOLVED] Background flickering around window: Eliminated (BG is not redrawn)\n";
    std::cout << "  [SOLVED] Console text corruption/flicker: Eliminated (Console is not redrawn)\n";
    std::cout << "  [UNSOLVED] Counter digit flickering: PERSISTS!\n";
    std::cout << "    Reason: Single-buffering writes FillRectangle(gray) directly to VRAM.\n";
    std::cout << "    When the display raster scan reads VRAM during the window redraw,\n";
    std::cout << "    the momentary blank gray rectangle is exposed to the user,\n";
    std::cout << "    causing high-frequency character flickering.\n";
    std::cout << "    -> Solution: Double-Buffering / Shadow Buffer in Chapter 10.5!\n";

    std::cout << "\n================================================================\n";
    std::cout << "All C++20 Chapter 10.4 Assertions PASSED Successfully!\n";
    std::cout << "================================================================\n";
    return 0;
}
