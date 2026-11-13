/**
 * @file os_chapter10_dragging_windows.cpp
 * @brief Chapter 10.6: Window Dragging with Mouse (`osbook_day10f`) Standalone C++20 Verification
 *
 * Demonstrates and verifies:
 * 1. USB HID Boot Protocol 3-byte mouse report decoding:
 *    - Buffer()[0]: `buttons` bitfield (Bit 0 = Left 0x01, Bit 1 = Right 0x02, Bit 2 = Middle 0x04)
 *    - Buffer()[1]: `int8_t displacement_x` (-128..+127)
 *    - Buffer()[2]: `int8_t displacement_y` (-128..+127)
 *    - `HIDMouseDriver::ObserverType = void(uint8_t buttons, int8_t dx, int8_t dy)`
 * 2. `LayerManager::FindLayerByPosition(Vector2D<int> pos, unsigned int exclude_id) const`:
 *    - Reverse Z-order scan (`layer_stack_.rbegin()` to `rend()`)
 *    - Exclusion of `mouse_layer_id` (`exclude_id`)
 *    - Null-window guard (`!win`) and hidden-layer immunity (not in `layer_stack_`)
 *    - Half-open bounding box hit test:
 *      `win_pos.x <= pos.x && pos.x < win_end_pos.x && win_pos.y <= pos.y && pos.y < win_end_pos.y`
 * 3. `MouseObserver` 4-state edge-detection FSM:
 *    - Rising edge `(!previous_left_pressed && left_pressed)` -> select `mouse_drag_layer_id`
 *    - Hold `(previous_left_pressed && left_pressed)` -> `MoveRelative(mouse_drag_layer_id, posdiff)`
 *    - Falling edge `(previous_left_pressed && !left_pressed)` -> reset `mouse_drag_layer_id = 0`
 *    - Clamped `posdiff = mouse_position - oldpos` preventing cursor/window desync at screen edges
 * 4. Faithful reproduction of the `osbook_day10f` background-layer drag bug:
 *    - Clicking empty desktop hits `bglayer_id` because `bgwindow` is attached and draggable
 *      is not yet gated (fixed in Section 10.7 `osbook_day10g`).
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

namespace mikanos::study::ch10_6 {

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
// 2. FrameBuffer & Window (Double-Buffered Back-Buffer Foundation from 10.5)
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
// 3. Layer & LayerManager with `FindLayerByPosition` (osbook_day10f)
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
     * @brief Exact MikanOS `osbook_day10f` `FindLayerByPosition` implementation.
     *
     * Searches `layer_stack_` from top to bottom (`rbegin()` -> `rend()`) to locate
     * the topmost visible layer whose window rectangle `[win_pos, win_pos + size)`
     * contains `pos`, excluding `exclude_id` (e.g. `mouse_layer_id`).
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
// 4. HIDMouseDriver (3-byte Boot Protocol) & MouseObserver Drag FSM
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
            if (layer) {
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

}  // namespace mikanos::study::ch10_6

int main() {
    using namespace mikanos::study::ch10_6;

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
    constexpr PixelColor kTitleBarColor{22, 48, 98};
    constexpr PixelColor kMouseColor{255, 255, 255};
    constexpr PixelColor kMouseTrans{0, 0, 1};

    auto bg_win = std::make_shared<Window>(kWidth, kHeight, PixelFormat::kPixelRGBResv8BitPerColor);
    bg_win->FillRectangle({0, 0}, {static_cast<int>(kWidth), static_cast<int>(kHeight)}, kBgColor);

    auto con_win = std::make_shared<Window>(140, 90, PixelFormat::kPixelRGBResv8BitPerColor);
    con_win->FillRectangle({0, 0}, {140, 90}, kConsoleColor);

    auto main_win = std::make_shared<Window>(100, 44, PixelFormat::kPixelRGBResv8BitPerColor);
    main_win->FillRectangle({0, 0}, {100, 44}, kWinColor);
    main_win->FillRectangle({3, 3}, {94, 16}, kTitleBarColor);

    auto mouse_win = std::make_shared<Window>(10, 14, PixelFormat::kPixelRGBResv8BitPerColor);
    mouse_win->SetTransparentColor(kMouseTrans);
    mouse_win->FillRectangle({0, 0}, {10, 14}, kMouseTrans);
    mouse_win->FillRectangle({0, 0}, {6, 10}, kMouseColor);

    const unsigned int bg_id = mgr.NewLayer().SetWindow(bg_win).Move({0, 0}).ID();
    const unsigned int con_id = mgr.NewLayer().SetWindow(con_win).Move({0, 0}).ID();
    const unsigned int main_id = mgr.NewLayer().SetWindow(main_win).Move({100, 50}).ID();
    // Overlapping layer with NO window attached -> must be skipped by FindLayerByPosition
    const unsigned int no_win_id = mgr.NewLayer().Move({110, 55}).ID();
    // Hidden window layer overlapping main_win -> must be skipped when hidden
    auto hidden_win = std::make_shared<Window>(60, 30, PixelFormat::kPixelRGBResv8BitPerColor);
    const unsigned int hidden_id = mgr.NewLayer().SetWindow(hidden_win).Move({105, 52}).ID();
    const unsigned int mouse_id = mgr.NewLayer().SetWindow(mouse_win).Move({120, 60}).ID();

    // Note: Layer IDs start at 1 (latest_id_ = 0 -> ++latest_id_)
    assert(bg_id == 1);

    mgr.UpDown(bg_id, 0);
    mgr.UpDown(con_id, 1);
    mgr.UpDown(main_id, 2);
    mgr.UpDown(no_win_id, 3);
    mgr.UpDown(hidden_id, -1);  // Hidden!
    mgr.UpDown(mouse_id, 4);

    mgr.Draw({{0, 0}, {static_cast<int>(kWidth), static_cast<int>(kHeight)}});

    // -------------------------------------------------------------------------
    // Test 1: FindLayerByPosition (exclude_id, Z-order, null window, hidden, half-open)
    // -------------------------------------------------------------------------
    // Inside main_win AND mouse_win overlap -> mouse excluded, no_win skipped, hidden skipped -> main_id
    assert(mgr.FindLayerByPosition({120, 60}, mouse_id)->ID() == main_id);
    // Inside con_win AND main_win overlap (105, 65) -> topmost is main_id
    assert(mgr.FindLayerByPosition({105, 65}, mouse_id)->ID() == main_id);
    // Inside con_win only (20, 20) -> con_id
    assert(mgr.FindLayerByPosition({20, 20}, mouse_id)->ID() == con_id);
    // Half-open boundary checks on main_win [100, 200) x [50, 94)
    assert(mgr.FindLayerByPosition({100, 50}, mouse_id)->ID() == main_id);
    assert(mgr.FindLayerByPosition({199, 93}, mouse_id)->ID() == main_id);
    // Exact right/bottom exclusive edge falls through to con_win or bg_win!
    assert(mgr.FindLayerByPosition({200, 60}, mouse_id)->ID() == bg_id);
    assert(mgr.FindLayerByPosition({120, 94}, mouse_id)->ID() == bg_id);
    // Outside entire screen -> nullptr
    assert(mgr.FindLayerByPosition({320, 100}, mouse_id) == nullptr);
    assert(mgr.FindLayerByPosition({-1, 10}, mouse_id) == nullptr);

    // -------------------------------------------------------------------------
    // Test 2: USB HID 3-byte Mouse Report + MouseObserver Drag FSM
    // -------------------------------------------------------------------------
    DesktopController desktop{
        &mgr, mouse_id, {static_cast<int>(kWidth), static_cast<int>(kHeight)}, {120, 60}, 0, 0
    };
    HIDMouseDriverSim mouse_driver{};
    mouse_driver.SubscribeMouseMove([&desktop](uint8_t buttons, int8_t dx, int8_t dy) {
        desktop.MouseObserver(buttons, dx, dy);
    });

    // Packet 1: Move without button press (dx=+10, dy=+5) -> mouse moves to (130, 65), main_win stays at (100, 50)
    const std::array<uint8_t, 3> pkt_move_idle{0x00, 10, 5};
    mouse_driver.OnDataReceived(pkt_move_idle);
    assert((desktop.mouse_position == Vector2D<int>{130, 65}));
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{100, 50}));
    assert(desktop.mouse_drag_layer_id == 0);

    // Packet 2: Left button pressed (0x01) with dx=0, dy=0 -> Rising Edge selects main_id
    const std::array<uint8_t, 3> pkt_press{0x01, 0, 0};
    mouse_driver.OnDataReceived(pkt_press);
    assert(desktop.mouse_drag_layer_id == main_id);
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{100, 50}));

    // Packet 3: Hold left button + negative two's complement displacement (dx=+40, dy=-20 = 0xEC)
    const std::array<uint8_t, 3> pkt_drag1{0x01, 40, static_cast<uint8_t>(int8_t{-20})};
    mouse_driver.OnDataReceived(pkt_drag1);
    assert((desktop.mouse_position == Vector2D<int>{170, 45}));
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{140, 30}));
    // Old exposed pixels at (105, 55) must now show console (16, 24, 32) since con_win covers [0,140)x[0,90)!
    assert(screen.ReadPixel(105, 55) == kConsoleColor);
    // New window title bar pixel at (145, 35) must show kTitleBarColor!
    assert(screen.ReadPixel(145, 35) == kTitleBarColor);

    // Packet 4: Screen-edge clamping desync prevention test!
    // Move far right (+200 -> raw 170+200=370, clamped to 319 -> posdiff = 149)
    // Wait: int8_t max is +127. Let's send +127 twice while holding left button!
    const std::array<uint8_t, 3> pkt_drag_edge1{0x01, 127, 0};
    mouse_driver.OnDataReceived(pkt_drag_edge1);
    // 170 + 127 = 297 (not clamped yet -> posdiff = 127, main_win = 140 + 127 = 267)
    assert((desktop.mouse_position == Vector2D<int>{297, 45}));
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{267, 30}));

    // Now send +50 -> raw x = 347, clamped to 319 -> actual posdiff.x = 319 - 297 = 22 (NOT 50!)
    const std::array<uint8_t, 3> pkt_drag_edge2{0x01, 50, 0};
    mouse_driver.OnDataReceived(pkt_drag_edge2);
    assert((desktop.mouse_position == Vector2D<int>{319, 45}));
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{289, 30}));

    // Packet 5: Left button released (0x00) -> Falling Edge resets mouse_drag_layer_id = 0
    const std::array<uint8_t, 3> pkt_release{0x00, static_cast<uint8_t>(int8_t{-30}), 10};
    mouse_driver.OnDataReceived(pkt_release);
    assert(desktop.mouse_drag_layer_id == 0);
    assert((desktop.mouse_position == Vector2D<int>{289, 55}));
    // Window stayed at (289, 30) on release packet (falling edge branch does not MoveRelative)
    assert((mgr.FindLayer(main_id)->GetPosition() == Vector2D<int>{289, 30}));

    // -------------------------------------------------------------------------
    // Test 3: Background Layer Drag Bug (`osbook_day10f`)
    // -------------------------------------------------------------------------
    // Move mouse to empty desktop at (250, 150), press left button and drag (+15, +10)
    const std::array<uint8_t, 3> pkt_to_bg{0x00, static_cast<uint8_t>(int8_t{-39}), 95};
    mouse_driver.OnDataReceived(pkt_to_bg);
    assert((desktop.mouse_position == Vector2D<int>{250, 150}));

    const std::array<uint8_t, 3> pkt_bg_press{0x01, 0, 0};
    mouse_driver.OnDataReceived(pkt_bg_press);
    // Bug in osbook_day10f: FindLayerByPosition returns bg_id!
    assert(desktop.mouse_drag_layer_id == bg_id);

    const std::array<uint8_t, 3> pkt_bg_drag{0x01, 15, 10};
    mouse_driver.OnDataReceived(pkt_bg_drag);
    assert((mgr.FindLayer(bg_id)->GetPosition() == Vector2D<int>{15, 10}));

    // Pixel-by-pixel equivalence between screen (VRAM) and back_buffer_ (RAM)
    for (int y = 0; y < static_cast<int>(kHeight); ++y) {
        for (int x = 0; x < static_cast<int>(kWidth); ++x) {
            assert(screen.ReadPixel(x, y) == mgr.BackBuffer().ReadPixel(x, y));
        }
    }

    std::cout << "=== Chapter 10.6 Window Dragging Verification (osbook_day10f) ===\n";
    std::cout << "FindLayerByPosition Z-order + half-open [pos, pos+size) checks: PASS\n";
    std::cout << "Clamped posdiff edge sync: main_win moved to (289, 30) with cursor at (319, 45)\n";
    std::cout << "osbook_day10f Background Drag Bug reproduced: bg_id=" << bg_id
              << " dragged to (" << mgr.FindLayer(bg_id)->GetPosition().x << ", "
              << mgr.FindLayer(bg_id)->GetPosition().y << ")\n";
    std::cout << "All C++20 Chapter 10.6 Assertions PASSED Successfully!\n";
    return 0;
}
