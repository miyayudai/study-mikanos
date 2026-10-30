/**
 * @file get_memory_map_sim.cpp
 * @brief C++20 Simulation and Verification of gBS->GetMemoryMap() and MemoryMap Struct
 *
 * Simulates MikanOS Chapter 2.5 (osbook_day02b) memory map retrieval,
 * buffer sizing, EFI_BUFFER_TOO_SMALL recovery, MapKey lifecycle,
 * and byte-pointer descriptor stepping arithmetic.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::uefi {

// ============================================================================
// UEFI Basic Types and Status Codes (UEFI Specification 2.8+ Chapter 2 & 7)
// ============================================================================

using EFI_STATUS = uint64_t;
using EFI_PHYSICAL_ADDRESS = uint64_t;
using EFI_VIRTUAL_ADDRESS = uint64_t;
using EFI_HANDLE = void*;

constexpr EFI_STATUS EFI_SUCCESS             = 0ULL;
constexpr EFI_STATUS EFI_ERROR_MASK          = 0x8000'0000'0000'0000ULL;
constexpr EFI_STATUS EFI_INVALID_PARAMETER   = EFI_ERROR_MASK | 2ULL;
constexpr EFI_STATUS EFI_BUFFER_TOO_SMALL    = EFI_ERROR_MASK | 5ULL;
constexpr EFI_STATUS EFI_UNSUPPORTED         = EFI_ERROR_MASK | 3ULL;

constexpr bool EFI_ERROR(EFI_STATUS status) noexcept {
    return (status & EFI_ERROR_MASK) != 0;
}

constexpr std::string_view StatusToString(EFI_STATUS status) noexcept {
    switch (status) {
        case EFI_SUCCESS:           return "EFI_SUCCESS";
        case EFI_INVALID_PARAMETER: return "EFI_INVALID_PARAMETER";
        case EFI_BUFFER_TOO_SMALL:  return "EFI_BUFFER_TOO_SMALL";
        case EFI_UNSUPPORTED:       return "EFI_UNSUPPORTED";
        default:                    return "EFI_UNKNOWN_ERROR";
    }
}

// ============================================================================
// UEFI Memory Types (EFI_MEMORY_TYPE)
// ============================================================================

enum class MemoryType : uint32_t {
    EfiReservedMemoryType      = 0,
    EfiLoaderCode              = 1,
    EfiLoaderData              = 2,
    EfiBootServicesCode        = 3,
    EfiBootServicesData        = 4,
    EfiRuntimeServicesCode     = 5,
    EfiRuntimeServicesData     = 6,
    EfiConventionalMemory      = 7,
    EfiUnusableMemory          = 8,
    EfiACPIReclaimMemory       = 9,
    EfiACPIMemoryNVS           = 10,
    EfiMemoryMappedIO          = 11,
    EfiMemoryMappedIOPortSpace = 12,
    EfiPalCode                 = 13,
    EfiPersistentMemory        = 14,
    EfiMaxMemoryType           = 15
};

constexpr std::string_view MemoryTypeToString(MemoryType type) noexcept {
    switch (type) {
        case MemoryType::EfiReservedMemoryType:      return "EfiReservedMemoryType";
        case MemoryType::EfiLoaderCode:              return "EfiLoaderCode";
        case MemoryType::EfiLoaderData:              return "EfiLoaderData";
        case MemoryType::EfiBootServicesCode:        return "EfiBootServicesCode";
        case MemoryType::EfiBootServicesData:        return "EfiBootServicesData";
        case MemoryType::EfiRuntimeServicesCode:     return "EfiRuntimeServicesCode";
        case MemoryType::EfiRuntimeServicesData:     return "EfiRuntimeServicesData";
        case MemoryType::EfiConventionalMemory:      return "EfiConventionalMemory";
        case MemoryType::EfiUnusableMemory:          return "EfiUnusableMemory";
        case MemoryType::EfiACPIReclaimMemory:       return "EfiACPIReclaimMemory";
        case MemoryType::EfiACPIMemoryNVS:           return "EfiACPIMemoryNVS";
        case MemoryType::EfiMemoryMappedIO:          return "EfiMemoryMappedIO";
        case MemoryType::EfiMemoryMappedIOPortSpace: return "EfiMemoryMappedIOPortSpace";
        case MemoryType::EfiPalCode:                 return "EfiPalCode";
        case MemoryType::EfiPersistentMemory:        return "EfiPersistentMemory";
        default:                                     return "InvalidMemoryType";
    }
}

// ============================================================================
// EFI_MEMORY_DESCRIPTOR (UEFI Spec Section 7.2)
// ============================================================================

#pragma pack(push, 1)
struct MemoryDescriptor {
    MemoryType           Type;
    uint32_t             Padding;
    EFI_PHYSICAL_ADDRESS PhysicalStart;
    EFI_VIRTUAL_ADDRESS  VirtualStart;
    uint64_t             NumberOfPages;
    uint64_t             Attribute;
};
#pragma pack(pop)

static_assert(sizeof(MemoryDescriptor) == 40, "MemoryDescriptor struct must be exactly 40 bytes (0x28)");

// ============================================================================
// MikanOS struct MemoryMap (MikanLoaderPkg/Main.c)
// ============================================================================

struct MemoryMap {
    uint64_t buffer_size;
    void*    buffer;
    uint64_t map_size;
    uint64_t map_key;
    uint64_t descriptor_size;
    uint32_t descriptor_version;
};

// ============================================================================
// Simulated Mock Boot Services Table (EFI_BOOT_SERVICES)
// ============================================================================

class MockBootServices {
public:
    explicit MockBootServices(uint64_t initial_map_key = 42)
        : current_map_key_(initial_map_key), boot_services_active_(true) {
        InitializeDefaultDescriptorPool();
    }

    EFI_STATUS GetMemoryMap(
        uint64_t* memory_map_size,
        MemoryDescriptor* memory_map,
        uint64_t* map_key,
        uint64_t* descriptor_size,
        uint32_t* descriptor_version
    ) {
        if (!boot_services_active_) {
            return EFI_UNSUPPORTED;
        }
        if (memory_map_size == nullptr || map_key == nullptr ||
            descriptor_size == nullptr || descriptor_version == nullptr) {
            return EFI_INVALID_PARAMETER;
        }

        const uint64_t actual_desc_size = 48; // 40-byte struct + 8-byte padding (OVMF / EDK II standard)
        const uint64_t needed_size = descriptors_.size() * actual_desc_size;

        *descriptor_size = actual_desc_size;
        *descriptor_version = 1;

        if (memory_map == nullptr || *memory_map_size < needed_size) {
            *memory_map_size = needed_size;
            return EFI_BUFFER_TOO_SMALL;
        }

        // Copy each descriptor with stride actual_desc_size
        uint8_t* out_ptr = reinterpret_cast<uint8_t*>(memory_map);
        for (size_t i = 0; i < descriptors_.size(); ++i) {
            std::memset(out_ptr, 0, actual_desc_size);
            std::memcpy(out_ptr, &descriptors_[i], sizeof(MemoryDescriptor));
            out_ptr += actual_desc_size;
        }
        *memory_map_size = needed_size;
        *map_key = current_map_key_;

        return EFI_SUCCESS;
    }

    EFI_STATUS AllocatePages(MemoryType type, uint64_t pages, EFI_PHYSICAL_ADDRESS* address) {
        if (!boot_services_active_) {
            return EFI_UNSUPPORTED;
        }
        if (address == nullptr || pages == 0) {
            return EFI_INVALID_PARAMETER;
        }

        // Find conventional memory region
        for (size_t i = 0; i < descriptors_.size(); ++i) {
            if (descriptors_[i].Type == MemoryType::EfiConventionalMemory &&
                descriptors_[i].NumberOfPages >= pages) {
                EFI_PHYSICAL_ADDRESS start = descriptors_[i].PhysicalStart;
                *address = start;

                MemoryDescriptor allocated{
                    .Type = type,
                    .Padding = 0,
                    .PhysicalStart = start,
                    .VirtualStart = 0,
                    .NumberOfPages = pages,
                    .Attribute = descriptors_[i].Attribute,
                };

                if (descriptors_[i].NumberOfPages == pages) {
                    descriptors_[i] = allocated;
                } else {
                    MemoryDescriptor remaining{
                        .Type = MemoryType::EfiConventionalMemory,
                        .Padding = 0,
                        .PhysicalStart = start + pages * 4096,
                        .VirtualStart = 0,
                        .NumberOfPages = descriptors_[i].NumberOfPages - pages,
                        .Attribute = descriptors_[i].Attribute,
                    };
                    descriptors_[i] = allocated;
                    descriptors_.insert(descriptors_.begin() + static_cast<ptrdiff_t>(i) + 1, remaining);
                }

                // Any memory allocation invalidates previous MapKey
                ++current_map_key_;
                return EFI_SUCCESS;
            }
        }
        return EFI_BUFFER_TOO_SMALL;
    }

    EFI_STATUS ExitBootServices(EFI_HANDLE /*image_handle*/, uint64_t map_key) {
        if (!boot_services_active_) {
            return EFI_INVALID_PARAMETER;
        }
        if (map_key != current_map_key_) {
            // Mismatch! Memory map was altered between GetMemoryMap and ExitBootServices
            return EFI_INVALID_PARAMETER;
        }
        boot_services_active_ = false;
        return EFI_SUCCESS;
    }

    [[nodiscard]] uint64_t GetCurrentMapKey() const noexcept {
        return current_map_key_;
    }

    [[nodiscard]] size_t GetDescriptorCount() const noexcept {
        return descriptors_.size();
    }

    [[nodiscard]] bool IsBootServicesActive() const noexcept {
        return boot_services_active_;
    }

private:
    void InitializeDefaultDescriptorPool() {
        descriptors_.clear();
        descriptors_.push_back({MemoryType::EfiBootServicesCode, 0, 0x00000000, 0, 159, 0xF});
        descriptors_.push_back({MemoryType::EfiReservedMemoryType, 0, 0x0009F000, 0, 1, 0xF});
        descriptors_.push_back({MemoryType::EfiConventionalMemory, 0, 0x00100000, 0, 32512, 0xF});
        descriptors_.push_back({MemoryType::EfiLoaderCode, 0, 0x06000000, 0, 128, 0xF});
        descriptors_.push_back({MemoryType::EfiLoaderData, 0, 0x06080000, 0, 256, 0xF});
        descriptors_.push_back({MemoryType::EfiConventionalMemory, 0, 0x06180000, 0, 498816, 0xF});
        descriptors_.push_back({MemoryType::EfiACPIReclaimMemory, 0, 0x7FE00000, 0, 128, 0xF});
        descriptors_.push_back({MemoryType::EfiACPIMemoryNVS, 0, 0x7FE80000, 0, 64, 0xF});
        descriptors_.push_back({MemoryType::EfiRuntimeServicesCode, 0, 0x7FEC0000, 0, 128, 0xF | (1ULL << 31)});
        descriptors_.push_back({MemoryType::EfiRuntimeServicesData, 0, 0x7FF40000, 0, 192, 0xF | (1ULL << 31)});
        descriptors_.push_back({MemoryType::EfiMemoryMappedIO, 0, 0xC0000000, 0, 4096, 0x3 | (1ULL << 31)});
        descriptors_.push_back({MemoryType::EfiConventionalMemory, 0, 0x100000000, 0, 524288, 0xF});
    }

    uint64_t current_map_key_;
    bool boot_services_active_;
    std::vector<MemoryDescriptor> descriptors_;
};

// Global simulated gBS pointer
inline MockBootServices* gSimBS = nullptr;

// ============================================================================
// MikanOS GetMemoryMap Function Implementation (MikanLoaderPkg/Main.c)
// ============================================================================

EFI_STATUS GetMemoryMap(struct MemoryMap* map) {
    if (map == nullptr || map->buffer == nullptr) {
        return EFI_BUFFER_TOO_SMALL;
    }

    map->map_size = map->buffer_size;
    return gSimBS->GetMemoryMap(
        &map->map_size,
        static_cast<MemoryDescriptor*>(map->buffer),
        &map->map_key,
        &map->descriptor_size,
        &map->descriptor_version
    );
}

} // namespace mikan::uefi

// ============================================================================
// Verification and Unit Test Functions
// ============================================================================

static void TestBufferTooSmallHandling() {
    using namespace mikan::uefi;
    MockBootServices bs(100);
    gSimBS = &bs;

    // Provide an intentionally tiny buffer (only 32 bytes, need >= 12 * 48 = 576 bytes)
    uint8_t tiny_buffer[32];
    MemoryMap memmap{
        .buffer_size = sizeof(tiny_buffer),
        .buffer = tiny_buffer,
        .map_size = 0,
        .map_key = 0,
        .descriptor_size = 0,
        .descriptor_version = 0,
    };

    EFI_STATUS status = GetMemoryMap(&memmap);
    assert(status == EFI_BUFFER_TOO_SMALL);
    assert(memmap.map_size >= bs.GetDescriptorCount() * 48);
    assert(memmap.descriptor_size == 48);
    assert(memmap.descriptor_version == 1);
    std::cout << "[PASS] TestBufferTooSmallHandling: Correctly returned EFI_BUFFER_TOO_SMALL and required size "
              << memmap.map_size << " bytes.\n";
}

static void TestFixed16KiBBufferSuccess() {
    using namespace mikan::uefi;
    MockBootServices bs(200);
    gSimBS = &bs;

    // MikanOS Fixed Buffer: 16 KiB (4096 * 4)
    alignas(16) uint8_t memmap_buf[4096 * 4];
    MemoryMap memmap{
        .buffer_size = sizeof(memmap_buf),
        .buffer = memmap_buf,
        .map_size = 0,
        .map_key = 0,
        .descriptor_size = 0,
        .descriptor_version = 0,
    };

    EFI_STATUS status = GetMemoryMap(&memmap);
    assert(status == EFI_SUCCESS);
    assert(memmap.map_size > 0);
    assert(memmap.map_size <= memmap.buffer_size);
    assert(memmap.map_key == 200);
    assert(memmap.descriptor_size == 48);
    assert(memmap.descriptor_version == 1);

    // Verify pointer stepping iteration
    uint64_t total_pages = 0;
    uint64_t conventional_pages = 0;
    size_t count = 0;

    auto iter = reinterpret_cast<uintptr_t>(memmap.buffer);
    const auto end = iter + memmap.map_size;

    while (iter < end) {
        auto* desc = reinterpret_cast<const MemoryDescriptor*>(iter);
        total_pages += desc->NumberOfPages;
        if (desc->Type == MemoryType::EfiConventionalMemory) {
            conventional_pages += desc->NumberOfPages;
        }
        iter += memmap.descriptor_size;
        ++count;
    }

    assert(count == bs.GetDescriptorCount());
    assert(conventional_pages > 0);
    assert(total_pages >= conventional_pages);

    std::cout << "[PASS] TestFixed16KiBBufferSuccess: Retrieved " << count
              << " descriptors (" << memmap.map_size << " bytes), MapKey=" << memmap.map_key
              << ", Conventional RAM=" << (conventional_pages * 4096 / (1024 * 1024)) << " MiB.\n";
}

static void TestMapKeyLifecycleAndExitBootServices() {
    using namespace mikan::uefi;
    MockBootServices bs(500);
    gSimBS = &bs;

    alignas(16) uint8_t memmap_buf[4096 * 4];
    MemoryMap memmap{sizeof(memmap_buf), memmap_buf, 0, 0, 0, 0};

    // 1. Get memory map (MapKey = 500)
    EFI_STATUS status = GetMemoryMap(&memmap);
    assert(status == EFI_SUCCESS);
    assert(memmap.map_key == 500);

    // 2. Simulate intermediate memory allocation (e.g. driver or firmware event)
    EFI_PHYSICAL_ADDRESS alloc_addr = 0;
    status = bs.AllocatePages(MemoryType::EfiBootServicesData, 16, &alloc_addr);
    assert(status == EFI_SUCCESS);
    assert(bs.GetCurrentMapKey() == 501); // Incremented!

    // 3. Attempt ExitBootServices with stale MapKey (500)
    status = bs.ExitBootServices(nullptr, memmap.map_key);
    assert(status == EFI_INVALID_PARAMETER); // Must fail!

    // 4. Re-fetch memory map to get latest MapKey (501)
    status = GetMemoryMap(&memmap);
    assert(status == EFI_SUCCESS);
    assert(memmap.map_key == 501);

    // 5. ExitBootServices with matching MapKey (501)
    status = bs.ExitBootServices(nullptr, memmap.map_key);
    assert(status == EFI_SUCCESS); // Must succeed!
    assert(!bs.IsBootServicesActive());

    std::cout << "[PASS] TestMapKeyLifecycleAndExitBootServices: Stale MapKey correctly rejected; re-acquired key successfully terminated Boot Services.\n";
}

static void TestPointerSteppingVsTypeIncrement() {
    using namespace mikan::uefi;

    // Demonstrate why iter += descriptor_size is strictly required by UEFI spec
    constexpr uint64_t kFutureExtendedDescSize = 56; // 40-byte struct + 16 bytes of padding/extension

    assert(kFutureExtendedDescSize > sizeof(MemoryDescriptor));

    uint8_t dummy_buffer[kFutureExtendedDescSize * 4];
    std::memset(dummy_buffer, 0, sizeof(dummy_buffer));

    // Place identifiable markers at each descriptor
    for (size_t i = 0; i < 4; ++i) {
        auto* desc = reinterpret_cast<MemoryDescriptor*>(&dummy_buffer[i * kFutureExtendedDescSize]);
        desc->Type = MemoryType::EfiConventionalMemory;
        desc->PhysicalStart = 0x1000'0000ULL * (i + 1);
        desc->NumberOfPages = 100 * (i + 1);
    }

    // Correct traversal using byte stepping
    uint64_t correct_sum_pages = 0;
    auto iter = reinterpret_cast<uintptr_t>(dummy_buffer);
    for (size_t i = 0; i < 4; ++i) {
        auto* desc = reinterpret_cast<const MemoryDescriptor*>(iter);
        correct_sum_pages += desc->NumberOfPages;
        iter += kFutureExtendedDescSize;
    }
    assert(correct_sum_pages == (100 + 200 + 300 + 400));

    // Incorrect traversal using desc++ (steps only 48 bytes instead of 56)
    const auto* bad_desc = reinterpret_cast<const MemoryDescriptor*>(dummy_buffer);
    uint64_t incorrect_sum_pages = 0;
    for (size_t i = 0; i < 4; ++i) {
        incorrect_sum_pages += bad_desc->NumberOfPages;
        ++bad_desc; // Advances by 48 bytes -> misaligned from 2nd entry onwards!
    }
    // They must not match because of stride divergence
    assert(correct_sum_pages != incorrect_sum_pages);

    std::cout << "[PASS] TestPointerSteppingVsTypeIncrement: Byte stride stepping verified against padding divergence.\n";
}

int main(int argc, char* argv[]) {
    bool run_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            run_tests = true;
        }
    }
    (void)run_tests;

    std::cout << "=== MikanOS Section 2.5: GetMemoryMap & MemoryMap C++20 Simulation ===\n";

    TestBufferTooSmallHandling();
    TestFixed16KiBBufferSuccess();
    TestMapKeyLifecycleAndExitBootServices();
    TestPointerSteppingVsTypeIncrement();

    std::cout << "\n[ALL C++20 TESTS PASSED SUCCESSFULLY]\n";
    return 0;
}
