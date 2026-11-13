/**
 * @file os_chapter10_draggable_layer.cpp
 * @brief Chapter 10.7: Draggable Layer Flags (`osbook_day10g`) Standalone C++20 Verification
 *
 * Demonstrates and verifies:
 * 1. `Layer` class extension (`kernel/layer.hpp`, `kernel/layer.cpp`):
 *    - Non-static data member initializers: `pos_{}`, `window_{}`, `draggable_{false}`
 *    - Safe fail-safe default (whitelist approach: layers are non-draggable by default)
 *    - Fluent interface `Layer& SetDraggable(bool draggable)` & `bool IsDraggable() const`
 * 2. `KernelMainNewStack` configuration (`kernel/main.cpp`):
 *    - Only `main_window_layer_id` calls `.SetDraggable(true)`
 *    - `bglayer_id`, `console_layer_id`, and `mouse_layer_id` remain `draggable_ == false`
 * 3. `MouseObserver` guard (`if (layer && layer->IsDraggable())`):
 *    - Short-circuit null-pointer safety when `FindLayerByPosition` returns `nullptr`
 *    - Rejection of drag start on background (`bglayer_id`) and console (`con_id`)
 *    - Single Responsibility Principle (SRP): why `FindLayerByPosition` stays a pure visual
 *      hit-test rather than skipping non-draggable layers (prevents Click-Through bugs when
 *      a non-draggable layer occludes a draggable window).
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace mikanos::study::ch10_7 {

// =============================================================================
// 1. Geometry & Pixel Color
// =============================================================================

template <typename T>
struct Vector2D {
    T x{}, y{};

    constexpr bool operator==(const Vector2D& rhs) const noexcept {
        return x == rhs.x && y == rhs.y;
    }

    constexpr Vector2D operator+(const Vector2D& rhs) const noexcept {
        return {x + rhs.x, y + rhs.y};
    }

    constexpr Vector2D operator-(const Vector2D& rhs) const noexcept {
        return {x - rhs.x, y - rhs.y};
    }

    Vector2D& operator+=(const Vector2D& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
};

template <typename T>
constexpr Vector2D<T> ElementMin(const Vector2D<T>& lhs, const Vector2D<T>& rhs) noexcept {
    return {
        lhs.x < rhs.x ? lhs.x : rhs.x,
        lhs.y < rhs.y ? lhs.y : rhs.y,
    };
}

template <typename T>
constexpr Vector2D<T> ElementMax(const Vector2D<T>& lhs, const Vector2D<T>& rhs) noexcept {
    return {
        lhs.x > rhs.x ? lhs.x : rhs.x,
        lhs.y > rhs.y ? lhs.y : rhs.y,
    };
}

template <typename T>
struct Rectangle {
    Vector2D<T> pos{};
    Vector2D<T> size{};
};

template <typename T, typename U>
constexpr Rectangle<T> operator&(const Rectangle<T>& lhs, const Rectangle<U>& rhs) noexcept {
    const auto lhs_end = lhs.pos + lhs.size;
    const auto rhs_end = rhs.pos + rhs.size;
    if (lhs_end.x < rhs.pos.x || lhs_end.y < rhs.pos.y ||
        rhs_end.x < lhs.pos.x || rhs_end.y < lhs.pos.y) {
        return {{0, 0}, {0, 0}};
    }
    const auto new_pos = ElementMax(lhs.pos, rhs.pos);
    const auto new_size = ElementMin(lhs_end, rhs_end) - new_pos;
    return {new_pos, new_size};
}

struct PixelColor {
    uint8_t r{}, g{}, b{};

    constexpr bool operator==(const PixelColor& rhs) const noexcept {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }
};

// =============================================================================
// 2. FrameBuffer & Window (Double-Buffered Back-Buffer Foundation)
// =============================================================================

enum class PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer{nullptr};
    uint32_t pixels_per_scan_line{0};
    uint32_t horizontal_resolution{0};
    uint32_t vertical_resolution{0};
    PixelFormat pixel_format{PixelFormat::kPixelRGBResv8BitPerColor};
};

class FrameBuffer {
public:
    bool Initialize(const FrameBufferConfig& config) {
        config_ = config;
        if (config_.frame_buffer == nullptr) {
            config_.pixels_per_scan_line = config_.horizontal_resolution;
            const size_t total_bytes =
                static_cast<size_t>(config_.pixels_per_scan_line) *
                config_.vertical_resolution * 4;
            buffer_.assign(total_bytes, 0);
            config_.frame_buffer = buffer_.data();
        } else {
            buffer_.clear();
        }
        return true;
    }

    const FrameBufferConfig& Config() const noexcept {
        return config_;
    }

    void WritePixel(int x, int y, const PixelColor& c) {
        if (x < 0 || y < 0 ||
            x >= static_cast<int>(config_.horizontal_resolution) ||
            y >= static_cast<int>(config_.vertical_resolution)) {
            return;
        }
        uint8_t* p = FrameAddrAt(x, y);
        if (config_.pixel_format == PixelFormat::kPixelRGBResv8BitPerColor) {
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
        } else {
            p[0] = c.b;
            p[1] = c.g;
            p[2] = c.r;
        }
    }

    PixelColor ReadPixel(int x, int y) const {
        const uint8_t* p = FrameAddrAt(x, y);
        if (config_.pixel_format == PixelFormat::kPixelRGBResv8BitPerColor) {
            return {p[0], p[1], p[2]};
        }
        return {p[2], p[1], p[0]};
    }

    void Copy(Vector2D<int> dst_pos, const FrameBuffer& src, const Rectangle<int>& src_area) {
        const auto dst_outline = Rectangle<int>{
            {0, 0},
            {static_cast<int>(config_.horizontal_resolution),
             static_cast<int>(config_.vertical_resolution)}
        };
        const auto src_outline = Rectangle<int>{
            {0, 0},
            {static_cast<int>(src.config_.horizontal_resolution),
             static_cast<int>(src.config_.vertical_resolution)}
        };
        const auto src_area_shifted = Rectangle<int>{
            dst_pos,
            src_area.size
        };

        const auto dst_area = dst_outline & src_area_shifted;
        if (dst_area.size.x <= 0 || dst_area.size.y <= 0) {
            return;
        }

        const auto src_offset = dst_area.pos - dst_pos;
        const auto clipped_src_pos = src_area.pos + src_offset;
        const auto valid_src_area = src_outline & Rectangle<int>{clipped_src_pos, dst_area.size};
        if (valid_src_area.size.x <= 0 || valid_src_area.size.y <= 0) {
            return;
        }

        const int copy_w = valid_src_area.size.x;
        const int copy_h = valid_src_area.size.y;
        const int final_dst_x = dst_area.pos.x + (valid_src_area.pos.x - clipped_src_pos.x);
        const int final_dst_y = dst_area.pos.y + (valid_src_area.pos.y - clipped_src_pos.y);

        for (int dy = 0; dy < copy_h; ++dy) {
            const uint8_t* s = src.FrameAddrAt(valid_src_area.pos.x, valid_src_area.pos.y + dy);
            uint8_t* d = FrameAddrAt(final_dst_x, final_dst_y + dy);
            std::memcpy(d, s, static_cast<size_t>(copy_w) * 4);
        }
    }

private:
    uint8_t* FrameAddrAt(int x, int y) {
        return config_.frame_buffer +
               4 * (static_cast<size_t>(config_.pixels_per_scan_line) * y + x);
    }

    const uint8_t* FrameAddrAt(int x, int y) const {
        return config_.frame_buffer +
               4 * (static_cast<size_t>(config_.pixels_per_scan_line) * y + x);
    }

    FrameBufferConfig config_{};
    std::vector<uint8_t> buffer_{};
};

class Window {
public:
    Window(int width, int height, PixelFormat shadow_format)
        : width_{width}, height_{height} {
        data_.resize(static_cast<size_t>(width) * height);
        FrameBufferConfig config{};
        config.frame_buffer = nullptr;
        config.horizontal_resolution = static_cast<uint32_t>(width);
        config.vertical_resolution = static_cast<uint32_t>(height);
        config.pixel_format = shadow_format;
        shadow_buffer_.Initialize(config);
    }

    void SetTransparentColor(std::optional<PixelColor> c) {
        transparent_color_ = c;
    }

    void FillRectangle(Vector2D<int> pos, Vector2D<int> size, const PixelColor& c) {
        for (int dy = 0; dy < size.y; ++dy) {
            for (int dx = 0; dx < size.x; ++dx) {
                const int px = pos.x + dx;
                const int py = pos.y + dy;
                if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                    data_[static_cast<size_t>(py) * width_ + px] = c;
                    shadow_buffer_.WritePixel(px, py, c);
                }
            }
        }
    }

    void DrawTo(FrameBuffer& dst, Vector2D<int> pos, const Rectangle<int>& area) const {
        if (!transparent_color_) {
            const Rectangle<int> window_area{pos, Size()};
            const Rectangle<int> intersection = area & window_area;
            if (intersection.size.x <= 0 || intersection.size.y <= 0) {
                return;
            }
            dst.Copy(intersection.pos, shadow_buffer_,
                     {intersection.pos - pos, intersection.size});
            return;
        }

        const PixelColor tc = *transparent_color_;
        const auto dst_outline = Rectangle<int>{
            {0, 0},
            {static_cast<int>(dst.Config().horizontal_resolution),
             static_cast<int>(dst.Config().vertical_resolution)}
        };
        const auto draw_area = area & Rectangle<int>{pos, Size()} & dst_outline;
        if (draw_area.size.x <= 0 || draw_area.size.y <= 0) {
            return;
        }
        const int start_x = std::max(0, draw_area.pos.x - pos.x);
        const int end_x = std::min(width_, draw_area.pos.x + draw_area.size.x - pos.x);
        const int start_y = std::max(0, draw_area.pos.y - pos.y);
        const int end_y = std::min(height_, draw_area.pos.y + draw_area.size.y - pos.y);

        for (int y = start_y; y < end_y; ++y) {
            for (int x = start_x; x < end_x; ++x) {
                const PixelColor c = At(x, y);
                if (!(c == tc)) {
                    dst.WritePixel(pos.x + x, pos.y + y, c);
                }
            }
        }
    }

    Vector2D<int> Size() const noexcept {
        return {width_, height_};
    }

    PixelColor At(int x, int y) const {
        return data_[static_cast<size_t>(y) * width_ + x];
    }

private:
    int width_;
    int height_;
    std::vector<PixelColor> data_;
    std::optional<PixelColor> transparent_color_{std::nullopt};
    FrameBuffer shadow_buffer_{};
};

// =============================================================================
// 3. Layer & LayerManager with `draggable_` Flag (`osbook_day10g`)
// =============================================================================

class Layer {
public:
    explicit Layer(unsigned int id = 0) : id_{id} {}

    unsigned int ID() const noexcept { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }

    std::shared_ptr<Window> GetWindow() const noexcept { return window_; }

    Vector2D<int> GetPosition() const noexcept { return pos_; }

    /** @brief Enable or disable mouse dragging for this layer (Fluent Interface). */
    Layer& SetDraggable(bool draggable) noexcept {
        draggable_ = draggable;
        return *this;
    }

    /** @brief Return true if this layer can be dragged by the mouse. */
    bool IsDraggable() const noexcept {
        return draggable_;
    }

    Layer& Move(Vector2D<int> pos) noexcept {
        pos_ = pos;
        return *this;
    }

    Layer& MoveRelative(Vector2D<int> pos_diff) noexcept {
        pos_ += pos_diff;
        return *this;
    }

    void DrawTo(FrameBuffer& screen, const Rectangle<int>& area) const {
        if (window_) {
            window_->DrawTo(screen, pos_, area);
        }
    }

private:
    unsigned int id_;
    Vector2D<int> pos_{};
    std::shared_ptr<Window> window_{};
    bool draggable_{false};
};

class LayerManager {
public:
    void SetWriter(FrameBuffer* screen) {
        screen_ = screen;
        FrameBufferConfig back_config = screen->Config();
        back_config.frame_buffer = nullptr;
        back_buffer_.Initialize(back_config);
    }

    Layer& NewLayer() {
        ++latest_id_;
        return *layers_.emplace_back(std::make_unique<Layer>(latest_id_));
    }

    void Draw(const Rectangle<int>& area) const {
        for (auto* layer : layer_stack_) {
            layer->DrawTo(back_buffer_, area);
        }
        screen_->Copy(area.pos, back_buffer_, area);
    }

    void Draw(unsigned int id) const {
        bool draw = false;
        Rectangle<int> window_area{};
        for (auto* layer : layer_stack_) {
            if (layer->ID() == id) {
                if (const auto& win = layer->GetWindow()) {
                    window_area.size = win->Size();
                }
                window_area.pos = layer->GetPosition();
                draw = true;
            }
            if (draw) {
                layer->DrawTo(back_buffer_, window_area);
            }
        }
        screen_->Copy(window_area.pos, back_buffer_, window_area);
    }

    void Move(unsigned int id, Vector2D<int> new_pos) {
        auto* layer = FindLayer(id);
        const auto window_size = layer->GetWindow()->Size();
        const auto old_pos = layer->GetPosition();
        layer->Move(new_pos);
        Draw({old_pos, window_size});
        Draw(id);
    }

    void MoveRelative(unsigned int id, Vector2D<int> pos_diff) {
        auto* layer = FindLayer(id);
        const auto window_size = layer->GetWindow()->Size();
        const auto old_pos = layer->GetPosition();
        layer->MoveRelative(pos_diff);
        Draw({old_pos, window_size});
        Draw(id);
    }

    void UpDown(unsigned int id, int new_height) {
        if (new_height < 0) {
            Hide(id);
            return;
        }
        if (new_height > static_cast<int>(layer_stack_.size())) {
            new_height = static_cast<int>(layer_stack_.size());
        }
        auto* layer = FindLayer(id);
        auto old_pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        auto new_pos = layer_stack_.begin() + new_height;
        if (old_pos == layer_stack_.end()) {
            layer_stack_.insert(new_pos, layer);
            return;
        }
        if (new_pos == layer_stack_.end()) {
            --new_pos;
        }
        layer_stack_.erase(old_pos);
        layer_stack_.insert(new_pos, layer);
    }

    void Hide(unsigned int id) {
        auto* layer = FindLayer(id);
        auto pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (pos != layer_stack_.end()) {
            layer_stack_.erase(pos);
        }
    }

    /**
     * @brief Exact MikanOS `FindLayerByPosition` implementation.
     *
     * Intentionally does NOT filter by `IsDraggable()` so that non-draggable layers
     * occlude underlying draggable windows (preventing Click-Through bugs).
     */
    Layer* FindLayerByPosition(Vector2D<int> pos, unsigned int exclude_id) const {
        auto pred = [pos, exclude_id](Layer* layer) {
            if (layer->ID() == exclude_id) {
                return false;
            }
            const auto& win = layer->GetWindow();
            if (!win) {
                return false;
            }
            const auto win_pos = layer->GetPosition();
            const auto win_end_pos = win_pos + win->Size();
            return win_pos.x <= pos.x && pos.x < win_end_pos.x &&
                   win_pos.y <= pos.y && pos.y < win_end_pos.y;
        };
        auto it = std::find_if(layer_stack_.rbegin(), layer_stack_.rend(), pred);
        if (it == layer_stack_.rend()) {
            return nullptr;
        }
        return *it;
    }

    /**
     * @brief Anti-pattern comparison: filtering `IsDraggable()` inside FindLayerByPosition.
     *
     * Demonstrates why putting draggable checks inside `FindLayerByPosition` causes
     * Click-Through bugs when an upper non-draggable panel occludes a lower draggable window.
     */
    Layer* FindLayerByPositionNaiveSkipBug(Vector2D<int> pos, unsigned int exclude_id) const {
        auto pred = [pos, exclude_id](Layer* layer) {
            if (layer->ID() == exclude_id || !layer->IsDraggable()) {
                return false;
            }
            const auto& win = layer->GetWindow();
            if (!win) {
                return false;
            }
            const auto win_pos = layer->GetPosition();
            const auto win_end_pos = win_pos + win->Size();
            return win_pos.x <= pos.x && pos.x < win_end_pos.x &&
                   win_pos.y <= pos.y && pos.y < win_end_pos.y;
        };
        auto it = std::find_if(layer_stack_.rbegin(), layer_stack_.rend(), pred);
        if (it == layer_stack_.rend()) {
            return nullptr;
        }
        return *it;
    }

    Layer* FindLayer(unsigned int id) const {
        auto pred = [id](const std::unique_ptr<Layer>& elem) {
            return elem->ID() == id;
        };
        auto it = std::find_if(layers_.begin(), layers_.end(), pred);
        if (it == layers_.end()) {
            return nullptr;
        }
        return it->get();
    }

    const FrameBuffer& BackBuffer() const noexcept { return back_buffer_; }

private:
    FrameBuffer* screen_{nullptr};
    mutable FrameBuffer back_buffer_{};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    unsigned int latest_id_{0};
};

// =============================================================================
// 4. HIDMouseDriver & MouseObserver with `layer && layer->IsDraggable()`
// =============================================================================

class HIDMouseDriverSim {
public:
    using ObserverType = void(uint8_t buttons, int8_t displacement_x, int8_t displacement_y);

    void SubscribeMouseMove(std::function<ObserverType> observer) {
        if (num_observers_ < static_cast<int>(observers_.size())) {
            observers_[static_cast<size_t>(num_observers_++)] = std::move(observer);
        }
    }

    void OnDataReceived(std::span<const uint8_t, 3> report) const {
        const uint8_t buttons = report[0];
        const int8_t displacement_x = static_cast<int8_t>(report[1]);
        const int8_t displacement_y = static_cast<int8_t>(report[2]);
        NotifyMouseMove(buttons, displacement_x, displacement_y);
    }

private:
    void NotifyMouseMove(uint8_t buttons, int8_t displacement_x, int8_t displacement_y) const {
        for (int i = 0; i < num_observers_; ++i) {
            observers_[static_cast<size_t>(i)](buttons, displacement_x, displacement_y);
        }
    }

    std::array<std::function<ObserverType>, 4> observers_{};
    int num_observers_{0};
};

struct DesktopController {
    LayerManager* layer_manager{nullptr};
    unsigned int mouse_layer_id{0};
    Vector2D<int> screen_size{};
    Vector2D<int> mouse_position{};
    unsigned int mouse_drag_layer_id{0};
    uint8_t previous_buttons{0};

    void MouseObserver(uint8_t buttons, int8_t displacement_x, int8_t displacement_y) {
        const auto oldpos = mouse_position;
        auto newpos = mouse_position + Vector2D<int>{displacement_x, displacement_y};
        newpos = ElementMin(newpos, screen_size + Vector2D<int>{-1, -1});
        mouse_position = ElementMax(newpos, {0, 0});

        const auto posdiff = mouse_position - oldpos;

        layer_manager->Move(mouse_layer_id, mouse_position);

        const bool previous_left_pressed = (previous_buttons & 0x01) != 0;
        const bool left_pressed = (buttons & 0x01) != 0;
        if (!previous_left_pressed && left_pressed) {
            auto* layer = layer_manager->FindLayerByPosition(mouse_position, mouse_layer_id);
            if (layer && layer->IsDraggable()) {
                mouse_drag_layer_id = layer->ID();
            }
        } else if (previous_left_pressed && left_pressed) {
            if (mouse_drag_layer_id > 0) {
                layer_manager->MoveRelative(mouse_drag_layer_id, posdiff);
            }
        } else if (previous_left_pressed && !left_pressed) {
            mouse_drag_layer_id = 0;
        }

        previous_buttons = buttons;
    }
};

}  // namespace mikanos::study::ch10_7

int main() {
    using namespace mikanos::study::ch10_7;

    constexpr uint32_t kWidth = 320;
    constexpr uint32_t kHeight = 200;
    std::vector<uint8_t> vram(static_cast<size_t>(kWidth) * kHeight * 4, 0);

    FrameBuffer screen{};
    FrameBufferConfig screen_cfg{
        vram.data(), kWidth, kWidth, kHeight, PixelFormat::kPixelRGBResv8BitPerColor
    };
    screen.Initialize(screen_cfg);

    LayerManager mgr{};
    mgr.SetWriter(&screen);

    constexpr PixelColor kBgColor{45, 110, 142};
    constexpr PixelColor kConsoleColor{16, 24, 32};
    constexpr PixelColor kWinColor{198, 198, 198};
    constexpr PixelColor kTitleBarColor{0, 17, 132};
    constexpr PixelColor kMouseColor{255, 255, 255};
    constexpr PixelColor kMouseTrans{0, 0, 1};

    auto bg_win = std::make_shared<Window>(kWidth, kHeight, PixelFormat::kPixelRGBResv8BitPerColor);
    bg_win->FillRectangle({0, 0}, {static_cast<int>(kWidth), static_cast<int>(kHeight)}, kBgColor);

    auto con_win = std::make_shared<Window>(156, 108, PixelFormat::kPixelRGBResv8BitPerColor);
    con_win->FillRectangle({0, 0}, {156, 108}, kConsoleColor);

    auto main_win = std::make_shared<Window>(110, 48, PixelFormat::kPixelRGBResv8BitPerColor);
    main_win->FillRectangle({0, 0}, {110, 48}, kWinColor);
    main_win->FillRectangle({3, 3}, {104, 16}, kTitleBarColor);

    auto mouse_win = std::make_shared<Window>(12, 16, PixelFormat::kPixelRGBResv8BitPerColor);
    mouse_win->SetTransparentColor(kMouseTrans);
    mouse_win->FillRectangle({0, 0}, {12, 16}, kMouseTrans);
    mouse_win->FillRectangle({0, 0}, {6, 10}, kMouseColor);

    // Exact osbook_day10g setup: ONLY main_window_layer_id calls .SetDraggable(true)
    const unsigned int bg_id = mgr.NewLayer()
        .SetWindow(bg_win)
        .Move({0, 0})
        .ID();
    const unsigned int con_id = mgr.NewLayer()
        .SetWindow(con_win)
        .Move({0, 0})
        .ID();
    const unsigned int main_id = mgr.NewLayer()
        .SetWindow(main_win)
        .SetDraggable(true)
        .Move({130, 56})
        .ID();
    const unsigned int mouse_id = mgr.NewLayer()
        .SetWindow(mouse_win)
        .Move({250, 140})
        .ID();

    mgr.UpDown(bg_id, 0);
    mgr.UpDown(con_id, 1);
    mgr.UpDown(main_id, 2);
    mgr.UpDown(mouse_id, 3);
    mgr.Draw({{0, 0}, {static_cast<int>(kWidth), static_cast<int>(kHeight)}});

    // -------------------------------------------------------------------------
    // Test 1: Fail-Safe Default (`draggable_ == false`) & Fluent Interface
    // -------------------------------------------------------------------------
    assert(mgr.FindLayer(bg_id)->IsDraggable() == false);
    assert(mgr.FindLayer(con_id)->IsDraggable() == false);
    assert(mgr.FindLayer(mouse_id)->IsDraggable() == false);
    assert(mgr.FindLayer(main_id)->IsDraggable() == true);

    // -------------------------------------------------------------------------
    // Test 2: DesktopBG Drag Rejection (Bug Fixed from osbook_day10f!)
    // -------------------------------------------------------------------------
    DesktopController desktop{
        &mgr, mouse_id, {static_cast<int>(kWidth), static_cast<int>(kHeight)}, {250, 140}, 0, 0
    };
    HIDMouseDriverSim mouse_driver{};
    mouse_driver.SubscribeMouseMove([&desktop](uint8_t buttons, int8_t dx, int8_t dy) {
        desktop.MouseObserver(buttons, dx, dy);
    });

    // Click empty desktop at (250, 140) and drag (-40, -28)
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x01, 0, 0});
    assert(mgr.FindLayerByPosition(desktop.mouse_position, mouse_id)->ID() == bg_id);
    assert(desktop.mouse_drag_layer_id == 0);  // Rejected because !bg_layer->IsDraggable()

    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{
        0x01, static_cast<uint8_t>(int8_t{-40}), static_cast<uint8_t>(int8_t{-28})
    });
    assert((desktop.mouse_position == Vector2D<int>{210, 112}));
    assert((mgr.FindLayer(bg_id)->GetPosition() == Vector2D<int>{0, 0}));  // BG stays fixed!
    assert(screen.ReadPixel(310, 190) == kBgColor);

    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x00, 0, 0});

    // -------------------------------------------------------------------------
    // Test 3: Console Drag Rejection
    // -------------------------------------------------------------------------
    // Move mouse over Console-only region (50, 50): delta (-160) -> two steps (-100, -60)
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{
        0x00, static_cast<uint8_t>(int8_t{-100}), static_cast<uint8_t>(int8_t{-62})
    });
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{
        0x00, static_cast<uint8_t>(int8_t{-60}), 0
    });
    assert((desktop.mouse_position == Vector2D<int>{50, 50}));

    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x01, 0, 0});
    assert(mgr.FindLayerByPosition(desktop.mouse_position, mouse_id)->ID() == con_id);
    assert(desktop.mouse_drag_layer_id == 0);  // Rejected because !con_layer->IsDraggable()

    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x01, 25, 15});
    assert((mgr.FindLayer(con_id)->GetPosition() == Vector2D<int>{0, 0}));  // Console stays fixed!
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x00, 0, 0});

    // -------------------------------------------------------------------------
    // Test 4: Draggable HelloWindow Dragging + Clamped Edge Synchronization
    // -------------------------------------------------------------------------
    // Move to HelloWindow title bar at (180, 66): from (75, 65) -> dx=+105, dy=+1
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x00, 105, 1});
    assert((desktop.mouse_position == Vector2D<int>{180, 66}));

    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x01, 0, 0});
    assert(desktop.mouse_drag_layer_id == main_id);

    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x01, 70, 50});
    assert((desktop.mouse_position == Vector2D<int>{250, 116}));
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{200, 106}));

    // Edge clamp test: +100 -> clamped to 319 -> posdiff = 69
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x01, 100, 0});
    assert((desktop.mouse_position == Vector2D<int>{319, 116}));
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{269, 106}));
    mouse_driver.OnDataReceived(std::array<uint8_t, 3>{0x00, 0, 0});
    assert(desktop.mouse_drag_layer_id == 0);

    // -------------------------------------------------------------------------
    // Test 5: Occlusion vs Click-Through Design Proof (SRP)
    // -------------------------------------------------------------------------
    // Move main_win back to (130, 56) and raise Console (non-draggable) ABOVE main_win
    mgr.Move(main_id, {130, 56});
    mgr.UpDown(con_id, 2);  // Stack: bg(0) < main(1) < con(2) < mouse(3)
    const Vector2D<int> overlap_pt{140, 70};

    // MikanOS design: FindLayerByPosition returns topmost visual layer (con_id),
    // and MouseObserver rejects dragging because !con_layer->IsDraggable()!
    const auto* mikan_hit = mgr.FindLayerByPosition(overlap_pt, mouse_id);
    assert(mikan_hit != nullptr && mikan_hit->ID() == con_id);
    assert(!mikan_hit->IsDraggable());

    // Anti-pattern: skipping non-draggable inside FindLayerByPosition clicks through Console to main_id!
    const auto* bug_hit = mgr.FindLayerByPositionNaiveSkipBug(overlap_pt, mouse_id);
    assert(bug_hit != nullptr && bug_hit->ID() == main_id);

    // Restore Z-order and test SetDraggable(false) runtime toggle
    mgr.UpDown(main_id, 2);
    mgr.FindLayer(main_id)->SetDraggable(false);
    assert(!mgr.FindLayer(main_id)->IsDraggable());
    mgr.FindLayer(main_id)->SetDraggable(true);
    assert(mgr.FindLayer(main_id)->IsDraggable());

    // Short-circuit null safety when clicking outside screen
    assert(mgr.FindLayerByPosition({-1, -1}, mouse_id) == nullptr);

    // Pixel-by-pixel equivalence between screen (VRAM) and back_buffer_ (RAM)
    for (int y = 0; y < static_cast<int>(kHeight); ++y) {
        for (int x = 0; x < static_cast<int>(kWidth); ++x) {
            assert(screen.ReadPixel(x, y) == mgr.BackBuffer().ReadPixel(x, y));
        }
    }

    std::cout << "=== Chapter 10.7 Draggable Layer Flags Verification (osbook_day10g) ===\n";
    std::cout << "Fail-safe default draggable_=false (BG/Console/Mouse): PASS\n";
    std::cout << "DesktopBG & Console drag rejection (stay fixed at (0,0)): PASS\n";
    std::cout << "HelloWindow drag & clamped posdiff edge sync: PASS\n";
    std::cout << "SRP Occlusion vs Click-Through bug verification: PASS\n";
    std::cout << "All C++20 Chapter 10.7 Assertions PASSED Successfully!\n";
    return 0;
}
