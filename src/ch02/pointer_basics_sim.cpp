/**
 * @file pointer_basics_sim.cpp
 * @brief C++20 Simulation and Verification of Pointer Architecture and Memory Addressing
 *
 * Simulates and verifies MikanOS Chapter 2.8:
 * - 64-bit address space, memory cell byte arrays, and variable addresses (& operator)
 * - Pointer variables, pointer size invariant (sizeof(T*) == 8 bytes on x86-64)
 * - Dereference operator (* operator), load/store operations, and little-endian storage
 * - Pointer types, access widths, and pointer arithmetic stride:
 *     Address(p + n) = Address(p) + n * sizeof(*p)
 * - Memory map traversal casting rule (uintptr_t / uint8_t* byte stride vs T* stride)
 * - Bare-metal hardware abstraction: VRAM direct memory-mapped pixel access simulation
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::sim {

// ============================================================================
// 1. Virtual Byte-Addressable Memory Simulation
// ============================================================================

class VirtualMemoryBank {
public:
    explicit VirtualMemoryBank(size_t capacity = 4096, uintptr_t base_addr = 0x1000)
        : base_address_(base_addr), storage_(capacity, 0) {}

    [[nodiscard]] uintptr_t BaseAddress() const noexcept { return base_address_; }
    [[nodiscard]] size_t Capacity() const noexcept { return storage_.size(); }

    void WriteBytes(uintptr_t addr, const void* data, size_t size) {
        CheckBounds(addr, size);
        std::memcpy(&storage_[addr - base_address_], data, size);
    }

    void ReadBytes(uintptr_t addr, void* dest, size_t size) const {
        CheckBounds(addr, size);
        std::memcpy(dest, &storage_[addr - base_address_], size);
    }

    template <typename T>
    void Write(uintptr_t addr, const T& val) {
        WriteBytes(addr, &val, sizeof(T));
    }

    template <typename T>
    [[nodiscard]] T Read(uintptr_t addr) const {
        T val{};
        ReadBytes(addr, &val, sizeof(T));
        return val;
    }

    [[nodiscard]] uint8_t ReadByte(uintptr_t addr) const {
        CheckBounds(addr, 1);
        return storage_[addr - base_address_];
    }

private:
    uintptr_t base_address_;
    std::vector<uint8_t> storage_;

    void CheckBounds(uintptr_t addr, size_t size) const {
        assert(addr >= base_address_ && (addr + size) <= (base_address_ + storage_.size()) &&
               "VirtualMemoryBank: Address out of bounds!");
    }
};

// ============================================================================
// 2. Low-Level Pointer and Addressing Verification Engine
// ============================================================================

struct PointerVerificationReport {
    bool pointer_size_valid{false};
    bool dereference_read_valid{false};
    bool dereference_write_valid{false};
    bool pointer_arithmetic_scaling_valid{false};
    bool little_endian_representation_valid{false};
    bool uefi_descriptor_casting_valid{false};
    bool vram_mmio_simulation_valid{false};

    [[nodiscard]] bool AllPassed() const noexcept {
        return pointer_size_valid &&
               dereference_read_valid &&
               dereference_write_valid &&
               pointer_arithmetic_scaling_valid &&
               little_endian_representation_valid &&
               uefi_descriptor_casting_valid &&
               vram_mmio_simulation_valid;
    }
};

class PointerBasicsEngine {
public:
    static PointerVerificationReport RunVerification() {
        PointerVerificationReport report;

        // 1. Verify 64-bit Pointer Size Invariance
        report.pointer_size_valid = VerifyPointerSizes();

        // 2. Verify Address and Dereference Semantics
        report.dereference_read_valid = VerifyDereferenceRead();
        report.dereference_write_valid = VerifyDereferenceWrite();

        // 3. Verify Little-Endian Byte Order Representation
        report.little_endian_representation_valid = VerifyLittleEndian();

        // 4. Verify Pointer Arithmetic Stride Scaling
        report.pointer_arithmetic_scaling_valid = VerifyPointerArithmeticScaling();

        // 5. Verify UEFI Memory Map Traversal Casting
        report.uefi_descriptor_casting_valid = VerifyUefiDescriptorCasting();

        // 6. Verify Bare-Metal VRAM MMIO Emulation
        report.vram_mmio_simulation_valid = VerifyVramSimulation();

        return report;
    }

    static bool VerifyPointerSizes() {
        // In 64-bit architecture (x86-64), all pointer variables are exactly 8 bytes
        static_assert(sizeof(void*) == 8, "Target must be 64-bit architecture");
        static_assert(sizeof(char*) == 8, "char* must be 8 bytes");
        static_assert(sizeof(int*) == 8, "int* must be 8 bytes");
        static_assert(sizeof(double*) == 8, "double* must be 8 bytes");
        static_assert(sizeof(uint64_t*) == 8, "uint64_t* must be 8 bytes");
        static_assert(sizeof(uintptr_t) == 8, "uintptr_t must be 8 bytes");

        struct LargeStruct {
            uint8_t buffer[1024];
        };
        static_assert(sizeof(LargeStruct*) == 8, "LargeStruct* must still be 8 bytes");
        return true;
    }

    static bool VerifyDereferenceRead() {
        int target_val = 0x5A5AA5A5;
        int* ptr = &target_val;

        // Verify that *ptr reads the exact value of target_val
        if (*ptr != target_val) return false;
        if (reinterpret_cast<uintptr_t>(ptr) != reinterpret_cast<uintptr_t>(&target_val)) return false;
        return true;
    }

    static bool VerifyDereferenceWrite() {
        int target_val = 100;
        int* ptr = &target_val;

        // Write through dereferenced pointer
        *ptr = 999;
        if (target_val != 999) return false;

        *ptr += 1;
        if (target_val != 1000) return false;
        return true;
    }

    static bool VerifyLittleEndian() {
        uint32_t multi_byte = 0x12345678;
        const auto* byte_ptr = reinterpret_cast<const uint8_t*>(&multi_byte);

        // In little-endian, byte[0] is LSB (0x78), byte[3] is MSB (0x12)
        if (byte_ptr[0] != 0x78) return false;
        if (byte_ptr[1] != 0x56) return false;
        if (byte_ptr[2] != 0x34) return false;
        if (byte_ptr[3] != 0x12) return false;
        return true;
    }

    static bool VerifyPointerArithmeticScaling() {
        // Address(p + n) == Address(p) + n * sizeof(*p)
        alignas(8) uint8_t raw_buffer[64] = {0};

        char* c_ptr = reinterpret_cast<char*>(raw_buffer);
        short* s_ptr = reinterpret_cast<short*>(raw_buffer);
        int* i_ptr = reinterpret_cast<int*>(raw_buffer);
        uint64_t* q_ptr = reinterpret_cast<uint64_t*>(raw_buffer);

        uintptr_t base = reinterpret_cast<uintptr_t>(raw_buffer);

        // Step by 1
        if (reinterpret_cast<uintptr_t>(c_ptr + 1) != base + 1 * sizeof(char)) return false;
        if (reinterpret_cast<uintptr_t>(s_ptr + 1) != base + 1 * sizeof(short)) return false;
        if (reinterpret_cast<uintptr_t>(i_ptr + 1) != base + 1 * sizeof(int)) return false;
        if (reinterpret_cast<uintptr_t>(q_ptr + 1) != base + 1 * sizeof(uint64_t)) return false;

        // Step by 3
        if (reinterpret_cast<uintptr_t>(c_ptr + 3) != base + 3 * sizeof(char)) return false;
        if (reinterpret_cast<uintptr_t>(s_ptr + 3) != base + 3 * sizeof(short)) return false;
        if (reinterpret_cast<uintptr_t>(i_ptr + 3) != base + 3 * sizeof(int)) return false;
        if (reinterpret_cast<uintptr_t>(q_ptr + 3) != base + 3 * sizeof(uint64_t)) return false;

        return true;
    }

    static bool VerifyUefiDescriptorCasting() {
        // Simulate EFI_MEMORY_DESCRIPTOR traversal from MikanOS Day02b
        struct DummyDescriptor {
            uint32_t type;
            uint32_t pad;
            uint64_t physical_start;
            uint64_t virtual_start;
            uint64_t number_of_pages;
            uint64_t attribute;
        };
        static_assert(sizeof(DummyDescriptor) == 40, "Descriptor should be 40 bytes");

        constexpr size_t descriptor_size = 48; // In UEFI, descriptor_size can be larger than sizeof (e.g., 48)
        constexpr size_t count = 3;
        std::vector<uint8_t> buffer(descriptor_size * count, 0);

        // Populate entries
        for (size_t i = 0; i < count; ++i) {
            auto* desc = reinterpret_cast<DummyDescriptor*>(buffer.data() + i * descriptor_size);
            desc->type = static_cast<uint32_t>(i + 1);
            desc->physical_start = 0x100000 + i * 0x40000;
        }

        // Test 1: Wrong traversal (without cast to byte or uintptr_t)
        // If someone treats buffer as DummyDescriptor* and advances by descriptor_size:
        auto* wrong_ptr = reinterpret_cast<DummyDescriptor*>(buffer.data());
        uintptr_t wrong_step_addr = reinterpret_cast<uintptr_t>(wrong_ptr + descriptor_size);
        uintptr_t expected_wrong = reinterpret_cast<uintptr_t>(buffer.data()) + descriptor_size * sizeof(DummyDescriptor);
        if (wrong_step_addr != expected_wrong) return false;
        // Notice expected_wrong jumps by 48 * 40 = 1920 bytes!

        // Test 2: Correct MikanOS traversal via uintptr_t integer arithmetic
        for (size_t iter = 0; iter < descriptor_size * count; iter += descriptor_size) {
            auto* desc = reinterpret_cast<DummyDescriptor*>(reinterpret_cast<uintptr_t>(buffer.data()) + iter);
            size_t idx = iter / descriptor_size;
            if (desc->type != (idx + 1)) return false;
            if (desc->physical_start != (0x100000 + idx * 0x40000)) return false;
        }

        // Test 3: Correct traversal via uint8_t* byte pointer arithmetic
        const uint8_t* byte_ptr = buffer.data();
        for (size_t i = 0; i < count; ++i) {
            const auto* desc = reinterpret_cast<const DummyDescriptor*>(byte_ptr + i * descriptor_size);
            if (desc->type != (i + 1)) return false;
        }

        return true;
    }

    static bool VerifyVramSimulation() {
        // Simulate Chapter 3 VRAM GOP direct frame buffer writing
        constexpr size_t width = 4;
        constexpr size_t height = 4;
        std::vector<uint32_t> vram(width * height, 0x000000); // 32bpp (RGBR)

        uintptr_t vram_base = reinterpret_cast<uintptr_t>(vram.data());

        // Kernel writes a white pixel (0xFFFFFF) at (x=2, y=1)
        size_t target_x = 2;
        size_t target_y = 1;
        uint32_t color = 0x00FFFFFF;

        // Pointer cast from hardware integer address:
        auto* pixel_ptr = reinterpret_cast<uint32_t*>(vram_base);
        pixel_ptr[target_y * width + target_x] = color;

        if (vram[target_y * width + target_x] != color) return false;
        if (vram[0] != 0x000000) return false; // Other pixels untouched
        return true;
    }
};

} // namespace mikan::sim

// ============================================================================
// Main CLI Runner & Test Harness
// ============================================================================

int main(int argc, char** argv) {
    bool run_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            run_tests = true;
        }
    }

    std::cout << "============================================================\n";
    std::cout << "  MikanOS Ch 2.8: Pointer & Addressing Verification Engine\n";
    std::cout << "============================================================\n\n";

    auto report = mikan::sim::PointerBasicsEngine::RunVerification();

    std::cout << "[Verification Report]\n";
    std::cout << "  1. 64-bit Pointer Size Invariance (sizeof(T*) == 8): "
              << (report.pointer_size_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  2. Address (&) and Dereference (*) Read:            "
              << (report.dereference_read_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  3. Dereference (*) Store / Mutation:                "
              << (report.dereference_write_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  4. Little-Endian Byte Order Storage:                "
              << (report.little_endian_representation_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  5. Pointer Arithmetic Stride Scaling:               "
              << (report.pointer_arithmetic_scaling_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  6. UEFI Memory Map Traversal Casting (uintptr_t):   "
              << (report.uefi_descriptor_casting_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  7. VRAM Direct Frame Buffer Access Emulation:       "
              << (report.vram_mmio_simulation_valid ? "PASSED" : "FAILED") << "\n\n";

    if (report.AllPassed()) {
        std::cout << ">>> ALL 7 POINTER ARCHITECTURE VERIFICATIONS PASSED! <<<\n";
    } else {
        std::cerr << ">>> ERROR: ONE OR MORE VERIFICATIONS FAILED! <<<\n";
        return 1;
    }

    if (!run_tests) {
        // Detailed interactive explanation output
        std::cout << "\n[Pointer Math Demonstration]\n";
        alignas(8) uint8_t sample_mem[32] = {0};
        uintptr_t base = reinterpret_cast<uintptr_t>(sample_mem);
        std::cout << "  Base Address: 0x" << std::hex << base << std::dec << "\n";

        char* p_char = reinterpret_cast<char*>(base);
        short* p_short = reinterpret_cast<short*>(base);
        int* p_int = reinterpret_cast<int*>(base);
        uint64_t* p_qword = reinterpret_cast<uint64_t*>(base);

        std::cout << "  char*     p + 1 -> offset +" << (reinterpret_cast<uintptr_t>(p_char + 1) - base) << " bytes\n";
        std::cout << "  short*    p + 1 -> offset +" << (reinterpret_cast<uintptr_t>(p_short + 1) - base) << " bytes\n";
        std::cout << "  int*      p + 1 -> offset +" << (reinterpret_cast<uintptr_t>(p_int + 1) - base) << " bytes\n";
        std::cout << "  uint64_t* p + 1 -> offset +" << (reinterpret_cast<uintptr_t>(p_qword + 1) - base) << " bytes\n";
    }

    return 0;
}
