/**
 * @file os_chapter9_profiling_compositing.cpp
 * @brief Chapter 9.4: Compositing Performance Profiling using Local APIC Timer (C++20)
 *
 * Demonstrates:
 * 1. Local APIC Timer MMIO register architecture (0xFEE00000 base, DCR, LVT, ICR, CCR)
 * 2. Emulated Local APIC Timer and high-resolution cycle counter
 * 3. MikanOS timer API: InitializeLAPICTimer(), StartLAPICTimer(), LAPICTimerElapsed(), StopLAPICTimer()
 * 4. Full compositing benchmark for LayerManager::Draw() with Desktop, Console, and Mouse layers
 * 5. MouseObserver profiling and latency accumulation analysis
 * 6. Mathematical complexity analysis: O(W * H) full redraw vs O(w * h) dirty-rect redraw
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mikanos::study {

// =============================================================================
// 1. Local APIC Timer Emulation and MMIO Registers
// =============================================================================

struct LAPICTimerMMIO {
    uint32_t divide_config{0b1011};       // 0xFEE003E0 (divide by 1 default)
    uint32_t lvt_timer{(1u << 16) | 32u}; // 0xFEE00320 (masked, one-shot, vector 32)
    uint32_t initial_count{0};            // 0xFEE00380
    uint32_t current_count{0};            // 0xFEE00390
};

class SimulatedLAPICTimer {
public:
    static constexpr uint32_t kCountMax = 0xffffffffu;

    void Initialize() {
        mmio_.divide_config = 0b1011;           // 1:1 分周
        mmio_.lvt_timer = (0b001 << 16) | 32;   // masked, one-shot, vector 32
        mmio_.initial_count = 0;
        mmio_.current_count = 0;
        is_running_ = false;
    }

    void Start() {
        mmio_.initial_count = kCountMax;
        mmio_.current_count = kCountMax;
        is_running_ = true;
        start_time_ = std::chrono::steady_clock::now();
    }

    void TickCycles(uint32_t cycles) {
        if (!is_running_) return;
        uint32_t divider = DecodeDivider(mmio_.divide_config);
        uint32_t ticks = cycles / divider;
        if (ticks >= mmio_.current_count) {
            mmio_.current_count = 0;
            is_running_ = false;
        } else {
            mmio_.current_count -= ticks;
        }
    }

    [[nodiscard]] uint32_t Elapsed() const {
        if (!is_running_ && mmio_.initial_count == 0) {
            return 0;
        }
        return kCountMax - mmio_.current_count;
    }

    void Stop() {
        mmio_.initial_count = 0;
        is_running_ = false;
    }

    [[nodiscard]] static uint32_t DecodeDivider(uint32_t val) {
        uint32_t b3 = (val >> 3) & 1;
        uint32_t b1 = (val >> 1) & 1;
        uint32_t b0 = val & 1;
        uint32_t code = (b3 << 2) | (b1 << 1) | b0;
        switch (code) {
            case 0b000: return 2;
            case 0b001: return 4;
            case 0b010: return 8;
            case 0b011: return 16;
            case 0b100: return 32;
            case 0b101: return 64;
            case 0b110: return 128;
            case 0b111: return 1; // 0b1011 -> divide by 1
            default: return 1;
        }
    }

    [[nodiscard]] const LAPICTimerMMIO& MMIO() const { return mmio_; }
    [[nodiscard]] bool IsRunning() const { return is_running_; }

private:
    LAPICTimerMMIO mmio_{};
    bool is_running_{false};
    std::chrono::steady_clock::time_point start_time_{};
};

// Global simulated LAPIC timer matching MikanOS kernel API
SimulatedLAPICTimer g_lapic_timer;

void InitializeLAPICTimer() {
    g_lapic_timer.Initialize();
}

void StartLAPICTimer() {
    g_lapic_timer.Start();
}

uint32_t LAPICTimerElapsed() {
    return g_lapic_timer.Elapsed();
}

void StopLAPICTimer() {
    g_lapic_timer.Stop();
}

// =============================================================================
// 2. Geometry and Compositing Types (MikanOS Chapter 9 Core)
// =============================================================================

template <typename T>
struct Vector2D {
    T x, y;
    template <typename U>
    Vector2D<T>& operator+=(const Vector2D<U>& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
};

template <typename T, typename U>
auto operator+(const Vector2D<T>& lhs, const Vector2D<U>& rhs)
    -> Vector2D<decltype(lhs.x + rhs.x)> {
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}

struct PixelColor {
    uint8_t r, g, b;
    bool operator==(const PixelColor& rhs) const = default;
};

class PixelWriter {
public:
    virtual ~PixelWriter() = default;
    virtual void Write(int x, int y, const PixelColor& c) = 0;
    virtual int Width() const = 0;
    virtual int Height() const = 0;
};

class FrameBufferWriter : public PixelWriter {
public:
    FrameBufferWriter(int width, int height)
        : width_{width}, height_{height}, buffer_(height, std::vector<PixelColor>(width, {0, 0, 0})) {}

    void Write(int x, int y, const PixelColor& c) override {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            buffer_[y][x] = c;
            ++write_count_;
        }
    }

    [[nodiscard]] int Width() const override { return width_; }
    [[nodiscard]] int Height() const override { return height_; }
    [[nodiscard]] const PixelColor& At(int x, int y) const { return buffer_[y][x]; }
    [[nodiscard]] uint64_t WriteCount() const { return write_count_; }
    void ResetWriteCount() { write_count_ = 0; }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> buffer_;
    uint64_t write_count_{0};
};

class Window {
public:
    class WindowWriter : public PixelWriter {
    public:
        explicit WindowWriter(Window& window) : window_{window} {}
        void Write(int x, int y, const PixelColor& c) override {
            window_.Write(x, y, c);
        }
        [[nodiscard]] int Width() const override { return window_.Width(); }
        [[nodiscard]] int Height() const override { return window_.Height(); }
    private:
        Window& window_;
    };

    Window(int width, int height)
        : width_{width}, height_{height}, data_(height, std::vector<PixelColor>(width, {0, 0, 0})) {}

    void DrawTo(PixelWriter& writer, Vector2D<int> position) const {
        if (!transparent_color_) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    writer.Write(position.x + x, position.y + y, data_[y][x]);
                }
            }
        } else {
            const auto& tc = *transparent_color_;
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    const auto& c = data_[y][x];
                    if (c != tc) {
                        writer.Write(position.x + x, position.y + y, c);
                    }
                }
            }
        }
    }

    void SetTransparentColor(std::optional<PixelColor> c) {
        transparent_color_ = c;
    }

    WindowWriter* Writer() { return &writer_; }
    [[nodiscard]] int Width() const { return width_; }
    [[nodiscard]] int Height() const { return height_; }

    void Write(int x, int y, const PixelColor& c) {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            data_[y][x] = c;
        }
    }

private:
    int width_, height_;
    std::vector<std::vector<PixelColor>> data_;
    WindowWriter writer_{*this};
    std::optional<PixelColor> transparent_color_{std::nullopt};
};

class Layer {
public:
    explicit Layer(unsigned int id) : id_{id} {}
    [[nodiscard]] unsigned int ID() const { return id_; }

    Layer& SetWindow(const std::shared_ptr<Window>& window) {
        window_ = window;
        return *this;
    }

    [[nodiscard]] std::shared_ptr<Window> GetWindow() const { return window_; }

    Layer& Move(Vector2D<int> pos) {
        pos_ = pos;
        return *this;
    }

    Layer& MoveRelative(Vector2D<int> pos_diff) {
        pos_ += pos_diff;
        return *this;
    }

    [[nodiscard]] Vector2D<int> GetPosition() const { return pos_; }

    void DrawTo(PixelWriter& writer) const {
        if (window_) {
            window_->DrawTo(writer, pos_);
        }
    }

private:
    unsigned int id_{0};
    Vector2D<int> pos_{0, 0};
    std::shared_ptr<Window> window_{nullptr};
};

class LayerManager {
public:
    void SetWriter(PixelWriter* writer) { writer_ = writer; }

    Layer& NewLayer() {
        ++latest_id_;
        return *layers_.emplace_back(std::make_unique<Layer>(latest_id_));
    }

    void Draw() const {
        for (auto* layer : layer_stack_) {
            layer->DrawTo(*writer_);
        }
    }

    void Move(unsigned int id, Vector2D<int> new_position) {
        if (auto* layer = FindLayer(id)) {
            layer->Move(new_position);
        }
    }

    void MoveRelative(unsigned int id, Vector2D<int> pos_diff) {
        if (auto* layer = FindLayer(id)) {
            layer->MoveRelative(pos_diff);
        }
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
        if (!layer) return;

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
        if (!layer) return;
        auto pos = std::find(layer_stack_.begin(), layer_stack_.end(), layer);
        if (pos != layer_stack_.end()) {
            layer_stack_.erase(pos);
        }
    }

    [[nodiscard]] Layer* FindLayer(unsigned int id) {
        auto it = std::find_if(layers_.begin(), layers_.end(),
                               [id](const std::unique_ptr<Layer>& elem) { return elem->ID() == id; });
        return (it != layers_.end()) ? it->get() : nullptr;
    }

    [[nodiscard]] size_t LayerCount() const { return layers_.size(); }
    [[nodiscard]] size_t StackCount() const { return layer_stack_.size(); }

private:
    PixelWriter* writer_{nullptr};
    std::vector<std::unique_ptr<Layer>> layers_{};
    std::vector<Layer*> layer_stack_{};
    unsigned int latest_id_{0};
};

} // namespace mikanos::study

// =============================================================================
// 3. Verification Test Suite and Benchmarks
// =============================================================================

using namespace mikanos::study;

void TestLAPICTimerRegisters() {
    std::cout << "[Test 1] Testing Local APIC Timer MMIO register decoding and countdown..." << std::endl;
    InitializeLAPICTimer();

    const auto& mmio = g_lapic_timer.MMIO();
    assert(mmio.divide_config == 0b1011);
    assert(SimulatedLAPICTimer::DecodeDivider(mmio.divide_config) == 1);
    assert((mmio.lvt_timer & 0xFF) == 32);       // vector 32
    assert(((mmio.lvt_timer >> 16) & 1) == 1);   // masked
    assert(((mmio.lvt_timer >> 17) & 0b11) == 0);// one-shot

    // Start timer and verify initial count
    StartLAPICTimer();
    assert(g_lapic_timer.IsRunning());
    assert(LAPICTimerElapsed() == 0);

    // Simulate 12,345 cycles elapsed
    g_lapic_timer.TickCycles(12345);
    assert(LAPICTimerElapsed() == 12345);

    // Stop timer
    StopLAPICTimer();
    assert(!g_lapic_timer.IsRunning());
    std::cout << "  ✓ Local APIC Timer MMIO registers and countdown verified." << std::endl;
}

void TestCompositingLayerProfiling() {
    std::cout << "[Test 2] Testing Layer Compositing Full Redraw and Profiling..." << std::endl;

    constexpr int kWidth = 200;
    constexpr int kHeight = 150;
    FrameBufferWriter fb(kWidth, kHeight);

    LayerManager mgr;
    mgr.SetWriter(&fb);

    // 1. Desktop background layer (full screen)
    auto bg_window = std::make_shared<Window>(kWidth, kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            bg_window->Write(x, y, {45, 118, 237});
        }
    }
    auto& bg_layer = mgr.NewLayer();
    bg_layer.SetWindow(bg_window).Move({0, 0});
    mgr.UpDown(bg_layer.ID(), 0);

    // 2. Window layer (100x80)
    auto app_window = std::make_shared<Window>(100, 80);
    for (int y = 0; y < 80; ++y) {
        for (int x = 0; x < 100; ++x) {
            app_window->Write(x, y, {200, 200, 200});
        }
    }
    auto& app_layer = mgr.NewLayer();
    app_layer.SetWindow(app_window).Move({50, 30});
    mgr.UpDown(app_layer.ID(), 1);

    // 3. Mouse cursor layer (16x24) with transparent color
    auto mouse_window = std::make_shared<Window>(16, 24);
    mouse_window->SetTransparentColor(PixelColor{0, 0, 0});
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 16; ++x) {
            if (x == 0 || y == 0 || x == y) {
                mouse_window->Write(x, y, {255, 255, 255}); // cursor outline
            } else {
                mouse_window->Write(x, y, {0, 0, 0});       // transparent
            }
        }
    }
    auto& mouse_layer = mgr.NewLayer();
    mouse_layer.SetWindow(mouse_window).Move({80, 60});
    mgr.UpDown(mouse_layer.ID(), 2);

    // Measure Draw() using LAPIC Timer
    fb.ResetWriteCount();
    StartLAPICTimer();

    mgr.Draw();

    // In simulation, tick cycles proportional to pixels drawn
    uint32_t pixels_drawn = static_cast<uint32_t>(fb.WriteCount());
    g_lapic_timer.TickCycles(pixels_drawn * 10); // ~10 cycles per pixel

    auto elapsed = LAPICTimerElapsed();
    StopLAPICTimer();

    std::cout << "  LayerManager::Draw() written pixels: " << pixels_drawn
              << ", simulated LAPIC ticks: " << elapsed << std::endl;

    assert(pixels_drawn > 0);
    assert(elapsed == pixels_drawn * 10);
    std::cout << "  ✓ LayerManager::Draw() full compositing profiling verified." << std::endl;
}

void TestMouseObserverSimulation() {
    std::cout << "[Test 3] Simulating MouseObserver relative movements and redraw latency..." << std::endl;

    constexpr int kWidth = 100;
    constexpr int kHeight = 80;
    FrameBufferWriter fb(kWidth, kHeight);

    LayerManager mgr;
    mgr.SetWriter(&fb);

    auto bg = std::make_shared<Window>(kWidth, kHeight);
    auto& bg_layer = mgr.NewLayer();
    bg_layer.SetWindow(bg).Move({0, 0});
    mgr.UpDown(bg_layer.ID(), 0);

    auto mouse = std::make_shared<Window>(16, 24);
    mouse->SetTransparentColor(PixelColor{0, 0, 0});
    auto& mouse_layer = mgr.NewLayer();
    mouse_layer.SetWindow(mouse).Move({10, 10});
    mgr.UpDown(mouse_layer.ID(), 1);

    unsigned int mouse_id = mouse_layer.ID();

    // MouseObserver implementation mimicking MikanOS day09b
    auto mouse_observer = [&](int8_t dx, int8_t dy) -> uint32_t {
        mgr.MoveRelative(mouse_id, {dx, dy});
        StartLAPICTimer();
        mgr.Draw();
        g_lapic_timer.TickCycles(100 * 80 * 12); // ~96,000 cycles
        auto elapsed = LAPICTimerElapsed();
        StopLAPICTimer();
        return elapsed;
    };

    uint32_t t1 = mouse_observer(5, 5);
    assert(mouse_layer.GetPosition().x == 15 && mouse_layer.GetPosition().y == 15);
    assert(t1 > 0);

    uint32_t t2 = mouse_observer(-3, 2);
    assert(mouse_layer.GetPosition().x == 12 && mouse_layer.GetPosition().y == 17);
    assert(t2 > 0);

    std::cout << "  ✓ MouseObserver simulation with LAPIC timer verified." << std::endl;
}

void TestFullVsPartialRedrawComplexity() {
    std::cout << "[Test 4] Analyzing O(W * H) Full Redraw vs O(w * h) Dirty-Rect Redraw..." << std::endl;

    // Full screen 1024x768
    constexpr uint64_t kScreenWidth = 1024;
    constexpr uint64_t kScreenHeight = 768;
    constexpr uint64_t kTotalPixels = kScreenWidth * kScreenHeight; // 786,432 pixels

    // Mouse cursor 16x24
    constexpr uint64_t kMouseWidth = 16;
    constexpr uint64_t kMouseHeight = 24;
    constexpr uint64_t kMousePixels = kMouseWidth * kMouseHeight;  // 384 pixels

    // Moving mouse dirty bounding box = old position (384) + new position (384) = 768 pixels
    constexpr uint64_t kDirtyRectPixels = kMousePixels * 2;

    double ratio = static_cast<double>(kTotalPixels) / static_cast<double>(kDirtyRectPixels);
    std::cout << "  Full Screen Redraw Pixels : " << kTotalPixels << " px\n";
    std::cout << "  Dirty Rect Redraw Pixels  : " << kDirtyRectPixels << " px\n";
    std::cout << "  Theoretical Speedup Ratio : " << std::fixed << std::setprecision(1) << ratio << "x\n";

    assert(ratio > 1000.0); // More than 1000x speedup expected!
    std::cout << "  ✓ Complexity disparity verified (>1000x speedup potential)." << std::endl;
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  MikanOS Chapter 9.4: Compositing Performance Profiling & LAPIC   \n";
    std::cout << "====================================================================\n";

    TestLAPICTimerRegisters();
    TestCompositingLayerProfiling();
    TestMouseObserverSimulation();
    TestFullVsPartialRedrawComplexity();

    std::cout << "\n[PASS] All Chapter 9.4 C++20 profiling and LAPIC timer tests passed!\n";
    return 0;
}
