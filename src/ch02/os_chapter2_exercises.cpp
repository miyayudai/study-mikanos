/**
 * @file os_chapter2_exercises.cpp
 * @brief Chapter 2 Comprehensive Exercises Implementation & Verification (C++20)
 *
 * This program implements and verifies key systems concepts introduced in Chapter 2:
 * 1. UEFI EFI_MEMORY_DESCRIPTOR 40-byte Layout, Packing, and 64-bit Alignment
 * 2. Safe Variable-length Descriptor Scanning using (uintptr_t)p += DescriptorSize
 * 3. Memory Map Analysis & Usable RAM Calculation after ExitBootServices()
 * 4. 64-bit Pointer Arithmetic Scaling & Struct Natural Alignment Padding
 * 5. Double Pointer (T**) Address Reassignment Pattern (UEFI OpenRootDir simulation)
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::exercises::ch02 {

// =============================================================================
// Exercise 1: UEFI Memory Descriptor & Struct Alignment Verification
// =============================================================================

#pragma pack(push, 1)
struct PackedMemoryDescriptor {
    uint32_t type;
    uint64_t physical_start;
    uint64_t virtual_start;
    uint64_t number_of_pages;
    uint64_t attribute;
};
#pragma pack(pop)

struct AlignedMemoryDescriptor {
    uint32_t type;              // 4 bytes
    uint32_t padding;           // 4 bytes padding for 8-byte alignment
    uint64_t physical_start;    // 8 bytes
    uint64_t virtual_start;     // 8 bytes
    uint64_t number_of_pages;   // 8 bytes
    uint64_t attribute;         // 8 bytes
};

enum class MemoryType : uint32_t {
    kEfiReservedMemoryType = 0,
    kEfiLoaderCode,
    kEfiLoaderData,
    kEfiBootServicesCode,
    kEfiBootServicesData,
    kEfiRuntimeServicesCode,
    kEfiRuntimeServicesData,
    kEfiConventionalMemory,
    kEfiUnusableMemory,
    kEfiACPIReclaimMemory,
    kEfiACPIMemoryNVS,
    kEfiMemoryMappedIO,
    kEfiMemoryMappedIOPortSpace,
    kEfiPalCode,
    kEfiPersistentMemory,
    kEfiMaxMemoryType
};

// =============================================================================
// Exercise 2: Safe Memory Map Scanner Simulation (Variable DescriptorSize)
// =============================================================================

class MemoryMapScanner {
public:
    MemoryMapScanner(const uint8_t* buffer, size_t map_size, size_t descriptor_size)
        : buffer_(buffer), map_size_(map_size), descriptor_size_(descriptor_size) {}

    size_t count_descriptors() const {
        if (descriptor_size_ == 0) return 0;
        return map_size_ / descriptor_size_;
    }

    const AlignedMemoryDescriptor* get_descriptor(size_t index) const {
        if (index >= count_descriptors()) return nullptr;
        uintptr_t addr = reinterpret_cast<uintptr_t>(buffer_) + (index * descriptor_size_);
        return reinterpret_cast<const AlignedMemoryDescriptor*>(addr);
    }

    uint64_t calculate_usable_memory_after_exit_boot_services() const {
        uint64_t total_pages = 0;
        size_t count = count_descriptors();
        for (size_t i = 0; i < count; ++i) {
            const auto* desc = get_descriptor(i);
            if (!desc) continue;
            auto type = static_cast<MemoryType>(desc->type);
            // Conventional RAM + Reclaimable BootServices Code/Data
            if (type == MemoryType::kEfiConventionalMemory ||
                type == MemoryType::kEfiBootServicesCode ||
                type == MemoryType::kEfiBootServicesData) {
                total_pages += desc->number_of_pages;
            }
        }
        return total_pages * 4096ULL; // 4KiB pages to bytes
    }

private:
    const uint8_t* buffer_;
    size_t map_size_;
    size_t descriptor_size_;
};

// =============================================================================
// Exercise 3: Struct Natural Alignment & Padding Calculator
// =============================================================================

struct MixedStruct {
    uint8_t a;      // Offset 0 (1B) -> Pad 3B
    uint32_t b;     // Offset 4 (4B)
    uint16_t c;     // Offset 8 (2B) -> Pad 2B (for struct alignment = 4B)
};

inline void verify_struct_alignment() {
    assert(sizeof(MixedStruct) == 12);
    assert(alignof(MixedStruct) == 4);
    assert(offsetof(MixedStruct, a) == 0);
    assert(offsetof(MixedStruct, b) == 4);
    assert(offsetof(MixedStruct, c) == 8);
}

// =============================================================================
// Exercise 4: Double Pointer Pattern Simulation (Output Arguments)
// =============================================================================

struct MockProtocol {
    int id;
    const char* name;
};

inline void allocate_protocol(MockProtocol** out_protocol, int id, const char* name) {
    assert(out_protocol != nullptr);
    static MockProtocol static_instances[4];
    static size_t next_idx = 0;
    assert(next_idx < 4);

    static_instances[next_idx] = MockProtocol{id, name};
    *out_protocol = &static_instances[next_idx];
    ++next_idx;
}

} // namespace mikanos::exercises::ch02

int main(int argc, char* argv[]) {
    using namespace mikanos::exercises::ch02;
    bool test_mode = (argc > 1 && std::string_view(argv[1]) == "--test");

    std::cout << "=========================================================\n";
    std::cout << " MikanOS Chapter 2 Comprehensive Exercises (C++20 Verify)\n";
    std::cout << "=========================================================\n\n";

    // 1. Verify Memory Descriptor Struct Layouts
    std::cout << "[Ex 1] Verifying EFI_MEMORY_DESCRIPTOR Struct Layout...\n";
    assert(sizeof(PackedMemoryDescriptor) == 36);
    assert(sizeof(AlignedMemoryDescriptor) == 40);
    assert(sizeof(uintptr_t) == 8);
    assert(sizeof(void*) == 8);
    std::cout << "  - Packed descriptor size:  " << sizeof(PackedMemoryDescriptor) << " bytes\n";
    std::cout << "  - Aligned descriptor size: " << sizeof(AlignedMemoryDescriptor) << " bytes (matches UEFI spec)\n";
    std::cout << "  ✓ Verified descriptor layout & 40-byte natural alignment!\n\n";

    // 2. Simulate Variable-length Descriptor Buffer (e.g., 48-byte future UEFI descriptor)
    std::cout << "[Ex 2] Simulating Safe Memory Map Scanning (DescriptorSize = 48 bytes)...\n";
    const size_t descriptor_size = 48; // Extensible size (> 40)
    const size_t num_descriptors = 4;
    std::vector<uint8_t> buffer(num_descriptors * descriptor_size, 0);

    // Populate descriptors
    auto setup_desc = [&](size_t idx, MemoryType type, uint64_t phys, uint64_t pages) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(buffer.data()) + (idx * descriptor_size);
        auto* desc = reinterpret_cast<AlignedMemoryDescriptor*>(addr);
        desc->type = static_cast<uint32_t>(type);
        desc->physical_start = phys;
        desc->virtual_start = phys;
        desc->number_of_pages = pages;
        desc->attribute = 0xF;
    };

    setup_desc(0, MemoryType::kEfiConventionalMemory, 0x100000, 256);      // 1 MiB Conventional RAM
    setup_desc(1, MemoryType::kEfiBootServicesCode,    0x200000, 128);      // 512 KiB BS Code (Reclaimable)
    setup_desc(2, MemoryType::kEfiRuntimeServicesData, 0x300000, 64);       // 256 KiB RT Data (Keep)
    setup_desc(3, MemoryType::kEfiBootServicesData,    0x400000, 256);      // 1 MiB BS Data (Reclaimable)

    MemoryMapScanner scanner(buffer.data(), buffer.size(), descriptor_size);
    assert(scanner.count_descriptors() == num_descriptors);

    uint64_t usable_bytes = scanner.calculate_usable_memory_after_exit_boot_services();
    uint64_t expected_pages = 256 + 128 + 256; // 640 pages
    uint64_t expected_bytes = expected_pages * 4096ULL;
    assert(usable_bytes == expected_bytes);
    std::cout << "  - Total descriptors: " << scanner.count_descriptors() << "\n";
    std::cout << "  - Usable RAM after ExitBootServices: " << (usable_bytes / (1024 * 1024)) << " MiB ("
              << usable_bytes << " bytes)\n";
    std::cout << "  ✓ Verified variable-length descriptor traversal and RAM reclamation!\n\n";

    // 3. Struct Alignment & Padding Verification
    std::cout << "[Ex 3] Verifying Natural Alignment & Padding Mathematics...\n";
    verify_struct_alignment();
    std::cout << "  - MixedStruct sizeof: " << sizeof(MixedStruct) << " bytes (aligned to 4 bytes)\n";
    std::cout << "  ✓ Verified natural alignment offsets!\n\n";

    // 4. Double Pointer Pattern Verification
    std::cout << "[Ex 4] Verifying Double Pointer (T**) Address Reassignment Pattern...\n";
    MockProtocol* protocol_ptr = nullptr;
    assert(protocol_ptr == nullptr);

    allocate_protocol(&protocol_ptr, 42, "EFI_SIMPLE_FILE_SYSTEM_PROTOCOL");
    assert(protocol_ptr != nullptr);
    assert(protocol_ptr->id == 42);
    assert(std::string_view(protocol_ptr->name) == "EFI_SIMPLE_FILE_SYSTEM_PROTOCOL");
    std::cout << "  - Successfully assigned protocol address: " << static_cast<void*>(protocol_ptr) << "\n";
    std::cout << "  - Protocol Name: " << protocol_ptr->name << " (ID: " << protocol_ptr->id << ")\n";
    std::cout << "  ✓ Verified double pointer out-parameter semantics!\n\n";

    std::cout << "🎉 ALL CHAPTER 2 C++20 EXERCISES PASSED SUCCESSFULLY!\n";
    if (test_mode) {
        std::cout << "Running in --test mode: verification complete.\n";
    }
    return 0;
}
