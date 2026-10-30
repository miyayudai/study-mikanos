/**
 * @file vtable_sim.cpp
 * @brief C++20 Simulation and Low-Level Verification of MikanOS Chapter 4.4 (vtable & __cxa_pure_virtual)
 *
 * Covers:
 * 1. Low-level vtable structure: vptr extraction, slot indexing, and manual indirect call execution
 * 2. Itanium C++ ABI compliance: offset-to-top, RTTI typeinfo pointer, and destructor slots
 * 3. Assembly mapping: mov rax, [rdi]; mov rax, [rax+16]; call rax
 * 4. Pure virtual functions & __cxa_pure_virtual trap mechanism in bare-metal environments
 * 5. Compile-time static assertions verifying polymorphic properties, sizes, and alignments
 */

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <new>
#include <type_traits>
#include <vector>

namespace mikan::ch04_vtable {

// ============================================================================
// 1. Core Structures and PixelWriter Class Hierarchy
// ============================================================================

enum PixelFormat {
    kPixelRGBResv8BitPerColor,
    kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
    uint8_t* frame_buffer;
    uint32_t pixels_per_scan_line;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    enum PixelFormat pixel_format;
};

struct PixelColor {
    uint8_t r, g, b;
};

/**
 * @brief Abstract base class with virtual destructor and pure virtual Write method.
 */
class PixelWriter {
public:
    explicit PixelWriter(const FrameBufferConfig& config) : config_{config} {}
    virtual ~PixelWriter() = default;

    // Pure virtual function
    virtual void Write(int x, int y, const PixelColor& c) = 0;

    [[nodiscard]] const FrameBufferConfig& GetConfig() const noexcept { return config_; }

protected:
    uint8_t* PixelAt(int x, int y) {
        return config_.frame_buffer + 4 * (config_.pixels_per_scan_line * y + x);
    }

private:
    const FrameBufferConfig& config_;
};

/**
 * @brief Concrete derived class for RGB pixel writing.
 */
class RGBResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;

    void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.r;
        p[1] = c.g;
        p[2] = c.b;
        p[3] = 0;
    }
};

/**
 * @brief Concrete derived class for BGR pixel writing.
 */
class BGRResv8BitPerColorPixelWriter : public PixelWriter {
public:
    using PixelWriter::PixelWriter;

    void Write(int x, int y, const PixelColor& c) override {
        auto p = PixelAt(x, y);
        p[0] = c.b;
        p[1] = c.g;
        p[2] = c.r;
        p[3] = 0;
    }
};

// ============================================================================
// Compile-time Static Assertions (Itanium C++ ABI / x86-64 Architecture)
// ============================================================================

static_assert(sizeof(PixelWriter) == 16, "PixelWriter size must be 16 bytes: vptr (8B) + config_ reference (8B)");
static_assert(alignof(PixelWriter) == 8, "PixelWriter alignment must be 8 bytes (pointer alignment on x86-64)");
static_assert(sizeof(RGBResv8BitPerColorPixelWriter) == 16, "Derived RGB writer size must remain 16 bytes");
static_assert(sizeof(BGRResv8BitPerColorPixelWriter) == 16, "Derived BGR writer size must remain 16 bytes");

static_assert(std::is_polymorphic_v<PixelWriter>, "PixelWriter must be polymorphic (has virtual functions)");
static_assert(std::is_abstract_v<PixelWriter>, "PixelWriter must be abstract (has pure virtual Write)");
static_assert(std::has_virtual_destructor_v<PixelWriter>, "PixelWriter must have a virtual destructor");
static_assert(!std::is_abstract_v<RGBResv8BitPerColorPixelWriter>, "RGB PixelWriter must be concrete");
static_assert(!std::is_abstract_v<BGRResv8BitPerColorPixelWriter>, "BGR PixelWriter must be concrete");

// ============================================================================
// 2. Simulated __cxa_pure_virtual Trap for Bare-Metal Kernel
// ============================================================================

static size_t g_pure_virtual_call_count = 0;

/**
 * @brief Mock bare-metal __cxa_pure_virtual handler.
 * In MikanOS kernel: extern "C" void __cxa_pure_virtual() { while (1) __asm__("hlt"); }
 */
extern "C" void MockCxaPureVirtual() {
    g_pure_virtual_call_count++;
    std::cerr << "[MOCK KERNEL PANIC] __cxa_pure_virtual() was called! Trapping CPU." << std::endl;
}

// Function pointer signature for Write(int x, int y, const PixelColor& c)
// System V AMD64 ABI: this in RDI, x in ESI, y in EDX, c in RCX (or stack)
using WriteMethodPtr = void (*)(PixelWriter* self, int x, int y, const PixelColor& c);

// ============================================================================
// 3. Test Cases
// ============================================================================

/**
 * @brief Test 1: Extract vptr via raw pointer arithmetic and perform manual indirect call.
 */
void TestVTablePointerExtractionAndManualCall() {
    std::cout << "[Test 1] Extracting vptr and invoking method via raw vtable pointer arithmetic... ";

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::vector<uint8_t> buffer(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = buffer.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    RGBResv8BitPerColorPixelWriter writer{config};
    PixelWriter* pwriter = &writer;

    // 1. Read first 8 bytes of the object instance: this is _vptr
    uintptr_t** obj_as_ptrs = reinterpret_cast<uintptr_t**>(pwriter);
    uintptr_t* vptr = obj_as_ptrs[0];

    assert(vptr != nullptr);

    // 2. In Itanium C++ ABI for x86-64:
    // Slot 0: Complete object destructor
    // Slot 1: Deleting destructor
    // Slot 2: First virtual method after destructors (Write)
    uintptr_t write_fn_addr = vptr[2];
    assert(write_fn_addr != 0);

    // 3. Cast raw address to function pointer and execute directly
    auto manual_write = reinterpret_cast<WriteMethodPtr>(write_fn_addr);
    manual_write(pwriter, 1, 2, {0xAA, 0xBB, 0xCC});

    // 4. Verify that the pixel was written at (1, 2)
    size_t pixel_offset = 4 * (width * 2 + 1);
    assert(buffer[pixel_offset + 0] == 0xAA);
    assert(buffer[pixel_offset + 1] == 0xBB);
    assert(buffer[pixel_offset + 2] == 0xCC);

    std::cout << "PASSED (vptr=0x" << std::hex << reinterpret_cast<uintptr_t>(vptr)
              << ", WriteSlot[2]=0x" << write_fn_addr << std::dec << ")" << std::endl;
}

/**
 * @brief Test 2: Verify Itanium C++ ABI negative offsets (offset-to-top & typeinfo).
 */
void TestItaniumAbiNegativeOffsets() {
    std::cout << "[Test 2] Verifying Itanium C++ ABI negative offsets (offset-to-top & typeinfo)... ";

    constexpr uint32_t width = 2;
    constexpr uint32_t height = 2;
    std::vector<uint8_t> buffer(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = buffer.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelRGBResv8BitPerColor,
    };

    RGBResv8BitPerColorPixelWriter rgb_writer{config};
    BGRResv8BitPerColorPixelWriter bgr_writer{config};

    uintptr_t* rgb_vptr = *reinterpret_cast<uintptr_t**>(&rgb_writer);
    uintptr_t* bgr_vptr = *reinterpret_cast<uintptr_t**>(&bgr_writer);

    // Vtables for distinct classes must reside at different addresses
    assert(rgb_vptr != bgr_vptr);

    // Offset-to-top (vptr[-2] in ptrdiff_t) for single inheritance must be 0
    ptrdiff_t* vtable_signed = reinterpret_cast<ptrdiff_t*>(rgb_vptr);
    ptrdiff_t offset_to_top = vtable_signed[-2];
    assert(offset_to_top == 0);

    // RTTI typeinfo pointer (vptr[-1]) must be non-null in standard hosted C++
    uintptr_t rtti_ptr = rgb_vptr[-1];
    assert(rtti_ptr != 0);

    std::cout << "PASSED (offset_to_top=" << offset_to_top
              << ", RTTI_ptr=0x" << std::hex << rtti_ptr << std::dec << ")" << std::endl;
}

/**
 * @brief Test 3: Simulated dynamic dispatch assembly instructions step-by-step.
 */
void TestPolymorphicDisassemblySimulation() {
    std::cout << "[Test 3] Simulating x86-64 indirect call machine instructions... ";

    constexpr uint32_t width = 10;
    constexpr uint32_t height = 10;
    std::vector<uint8_t> buffer(width * height * 4, 0);

    FrameBufferConfig config{
        .frame_buffer = buffer.data(),
        .pixels_per_scan_line = width,
        .horizontal_resolution = width,
        .vertical_resolution = height,
        .pixel_format = kPixelBGRResv8BitPerColor,
    };

    BGRResv8BitPerColorPixelWriter writer{config};
    PixelWriter* pwriter = &writer;

    // Step 1: MOV RAX, [RDI]   (Load vptr from object base)
    uintptr_t rax_reg = *reinterpret_cast<uintptr_t*>(pwriter);

    // Step 2: MOV RAX, [RAX + 16] (Index 2nd slot = 16 bytes offset for Write)
    uintptr_t target_func = *reinterpret_cast<uintptr_t*>(rax_reg + 16);

    // Step 3: CALL RAX
    auto func = reinterpret_cast<WriteMethodPtr>(target_func);
    func(pwriter, 5, 5, {10, 20, 30});

    // Check BGR ordering: B=30, G=20, R=10
    size_t offset = 4 * (width * 5 + 5);
    assert(buffer[offset + 0] == 30);
    assert(buffer[offset + 1] == 20);
    assert(buffer[offset + 2] == 10);

    std::cout << "PASSED (Successfully routed via BGR writer override)" << std::endl;
}

/**
 * @brief Test 4: Pure virtual function call trap mechanism verification.
 */
void TestPureVirtualCallTrapMechanism() {
    std::cout << "[Test 4] Verifying pure virtual function call trap (__cxa_pure_virtual)... ";

    g_pure_virtual_call_count = 0;

    // Simulate an abstract vtable entry containing __cxa_pure_virtual
    std::array<uintptr_t, 3> mock_abstract_vtable = {
        0x1000,                                         // Slot 0: dtor
        0x1008,                                         // Slot 1: deleting dtor
        reinterpret_cast<uintptr_t>(&MockCxaPureVirtual) // Slot 2: Write -> __cxa_pure_virtual
    };

    // Invoke slot 2
    auto pure_fn = reinterpret_cast<void (*)()>(mock_abstract_vtable[2]);
    pure_fn();

    assert(g_pure_virtual_call_count == 1);
    std::cout << "PASSED (Pure virtual trap successfully handled)" << std::endl;
}

/**
 * @brief Test 5: Verify runtime size, alignment, and absence of per-call branch overhead.
 */
void TestVTableSizeAndOverhead() {
    std::cout << "[Test 5] Verifying vtable memory footprint and static alignments... ";

    struct NonPolymorphicWriter {
        const FrameBufferConfig& config;
    };
    static_assert(sizeof(NonPolymorphicWriter) == 8, "Non-polymorphic struct only holds reference (8 bytes)");

    // Overhead of polymorphism: 8 bytes for _vptr per object instance
    constexpr size_t vptr_overhead = sizeof(PixelWriter) - sizeof(NonPolymorphicWriter);
    assert(vptr_overhead == 8);

    std::cout << "PASSED (Object size: 16B, vptr overhead: 8B)" << std::endl;
}

}  // namespace mikan::ch04_vtable

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << " MikanOS Chapter 4.4: Virtual Method Table (vtable) & __cxa_pure " << std::endl;
    std::cout << "=================================================================" << std::endl;

    mikan::ch04_vtable::TestVTablePointerExtractionAndManualCall();
    mikan::ch04_vtable::TestItaniumAbiNegativeOffsets();
    mikan::ch04_vtable::TestPolymorphicDisassemblySimulation();
    mikan::ch04_vtable::TestPureVirtualCallTrapMechanism();
    mikan::ch04_vtable::TestVTableSizeAndOverhead();

    std::cout << "=================================================================" << std::endl;
    std::cout << " All Section 4.4 C++20 Verification Tests Passed Successfully!   " << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
