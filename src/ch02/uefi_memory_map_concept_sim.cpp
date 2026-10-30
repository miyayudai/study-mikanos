/**
 * @file uefi_memory_map_concept_sim.cpp
 * @brief C++20 Simulation and Verification of UEFI Memory Map & EFI_MEMORY_DESCRIPTOR
 *
 * Implements EFI_MEMORY_DESCRIPTOR layout verification, UEFI memory type
 * categorization, pointer stepping arithmetic, and memory lifecycle analysis.
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

/**
 * @brief UEFI Memory Types defined in UEFI Specification Section 7.2 (EFI_MEMORY_TYPE)
 */
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

constexpr std::string_view GetMemoryTypeName(MemoryType type) noexcept {
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
        case MemoryType::EfiMaxMemoryType:           return "EfiMaxMemoryType";
        default:                                     return "InvalidMemoryType";
    }
}

/**
 * @brief Memory Attributes (UEFI Specification Table 7.3)
 */
inline constexpr uint64_t EFI_MEMORY_UC            = 0x0000000000000001ULL;
inline constexpr uint64_t EFI_MEMORY_WC            = 0x0000000000000002ULL;
inline constexpr uint64_t EFI_MEMORY_WT            = 0x0000000000000004ULL;
inline constexpr uint64_t EFI_MEMORY_WB            = 0x0000000000000008ULL;
inline constexpr uint64_t EFI_MEMORY_UCE           = 0x0000000000000010ULL;
inline constexpr uint64_t EFI_MEMORY_WP            = 0x0000000000001000ULL;
inline constexpr uint64_t EFI_MEMORY_RP            = 0x0000000000002000ULL;
inline constexpr uint64_t EFI_MEMORY_XP            = 0x0000000000004000ULL;
inline constexpr uint64_t EFI_MEMORY_NV            = 0x0000000000008000ULL;
inline constexpr uint64_t EFI_MEMORY_MORE_RELIABLE = 0x0000000000010000ULL;
inline constexpr uint64_t EFI_MEMORY_RO            = 0x0000000000020000ULL;
inline constexpr uint64_t EFI_MEMORY_SP            = 0x0000000000040000ULL;
inline constexpr uint64_t EFI_MEMORY_CPU_CRYPTO    = 0x0000000000080000ULL;
inline constexpr uint64_t EFI_MEMORY_RUNTIME       = 0x8000000000000000ULL;

/**
 * @brief EFI_MEMORY_DESCRIPTOR exact struct layout on x86-64 EDK II
 */
struct MemoryDescriptor {
    uint32_t Type;           // 0x00: EFI_MEMORY_TYPE
    uint32_t Pad;            // 0x04: 32-bit padding for 64-bit alignment
    uint64_t PhysicalStart;  // 0x08: EFI_PHYSICAL_ADDRESS
    uint64_t VirtualStart;   // 0x10: EFI_VIRTUAL_ADDRESS
    uint64_t NumberOfPages;  // 0x18: Count of 4KiB pages
    uint64_t Attribute;      // 0x20: Memory cacheability / protection attributes

    static constexpr uint64_t kPageSize = 4096ULL;

    [[nodiscard]] constexpr uint64_t SizeBytes() const noexcept {
        return NumberOfPages * kPageSize;
    }

    [[nodiscard]] constexpr uint64_t PhysicalEnd() const noexcept {
        return PhysicalStart + SizeBytes();
    }

    [[nodiscard]] constexpr bool IsConventional() const noexcept {
        return static_cast<MemoryType>(Type) == MemoryType::EfiConventionalMemory;
    }

    [[nodiscard]] constexpr bool IsBootServicesReclaimable() const noexcept {
        const auto t = static_cast<MemoryType>(Type);
        return (t == MemoryType::EfiBootServicesCode ||
                t == MemoryType::EfiBootServicesData ||
                t == MemoryType::EfiLoaderCode ||
                t == MemoryType::EfiLoaderData);
    }

    [[nodiscard]] constexpr bool IsRuntimeServices() const noexcept {
        const auto t = static_cast<MemoryType>(Type);
        return (t == MemoryType::EfiRuntimeServicesCode ||
                t == MemoryType::EfiRuntimeServicesData);
    }

    [[nodiscard]] constexpr bool IsRuntimeRequired() const noexcept {
        return (Attribute & EFI_MEMORY_RUNTIME) != 0;
    }

    [[nodiscard]] constexpr bool IsPageAligned() const noexcept {
        return (PhysicalStart % kPageSize) == 0;
    }
};

static_assert(sizeof(MemoryDescriptor) == 40, "MemoryDescriptor must be exactly 40 bytes (0x28)");
static_assert(offsetof(MemoryDescriptor, Type) == 0x00, "Type offset must be 0x00");
static_assert(offsetof(MemoryDescriptor, Pad) == 0x04, "Pad offset must be 0x04");
static_assert(offsetof(MemoryDescriptor, PhysicalStart) == 0x08, "PhysicalStart offset must be 0x08");
static_assert(offsetof(MemoryDescriptor, VirtualStart) == 0x10, "VirtualStart offset must be 0x10");
static_assert(offsetof(MemoryDescriptor, NumberOfPages) == 0x18, "NumberOfPages offset must be 0x18");
static_assert(offsetof(MemoryDescriptor, Attribute) == 0x20, "Attribute offset must be 0x20");

/**
 * @brief Memory Map Iterator and Processor
 */
class MemoryMapSimulator {
public:
    explicit MemoryMapSimulator(uint64_t descriptor_size = sizeof(MemoryDescriptor),
                                uint32_t descriptor_version = 1)
        : descriptor_size_(descriptor_size), descriptor_version_(descriptor_version) {}

    void AddDescriptor(const MemoryDescriptor& desc) {
        descriptors_.push_back(desc);
    }

    [[nodiscard]] size_t EntryCount() const noexcept {
        return descriptors_.size();
    }

    [[nodiscard]] uint64_t DescriptorSize() const noexcept {
        return descriptor_size_;
    }

    [[nodiscard]] uint64_t TotalMapBytes() const noexcept {
        return descriptors_.size() * descriptor_size_;
    }

    [[nodiscard]] uint64_t TotalConventionalBytes() const noexcept {
        uint64_t total = 0;
        for (const auto& d : descriptors_) {
            if (d.IsConventional()) {
                total += d.SizeBytes();
            }
        }
        return total;
    }

    [[nodiscard]] uint64_t TotalBootServicesReclaimableBytes() const noexcept {
        uint64_t total = 0;
        for (const auto& d : descriptors_) {
            if (d.IsBootServicesReclaimable()) {
                total += d.SizeBytes();
            }
        }
        return total;
    }

    [[nodiscard]] uint64_t TotalRuntimeServicesBytes() const noexcept {
        uint64_t total = 0;
        for (const auto& d : descriptors_) {
            if (d.IsRuntimeServices()) {
                total += d.SizeBytes();
            }
        }
        return total;
    }

    [[nodiscard]] std::vector<uint8_t> Serialize() const {
        std::vector<uint8_t> buffer(descriptors_.size() * descriptor_size_, 0);
        for (size_t i = 0; i < descriptors_.size(); ++i) {
            uint8_t* dest = buffer.data() + (i * descriptor_size_);
            std::memcpy(dest, &descriptors_[i], sizeof(MemoryDescriptor));
        }
        return buffer;
    }

    static MemoryMapSimulator Deserialize(const uint8_t* buffer,
                                          size_t buffer_size,
                                          uint64_t descriptor_size,
                                          uint32_t descriptor_version = 1) {
        MemoryMapSimulator sim(descriptor_size, descriptor_version);
        for (size_t offset = 0; offset + descriptor_size <= buffer_size; offset += descriptor_size) {
            MemoryDescriptor desc{};
            std::memcpy(&desc, buffer + offset, sizeof(MemoryDescriptor));
            sim.AddDescriptor(desc);
        }
        return sim;
    }

    static MemoryMapSimulator CreateStandardQemuOvmfMap(uint64_t descriptor_size = sizeof(MemoryDescriptor)) {
        MemoryMapSimulator sim(descriptor_size, 1);
        // Low memory
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiBootServicesCode), 0, 0x00000000ULL, 0, 159,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiReservedMemoryType), 0, 0x0009F000ULL, 0, 1,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiConventionalMemory), 0, 0x00100000ULL, 0, 32512,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        // Boot loader
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiLoaderCode), 0, 0x06000000ULL, 0, 128,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiLoaderData), 0, 0x06080000ULL, 0, 256,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        // Main memory
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiConventionalMemory), 0, 0x06180000ULL, 0, 498816,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        // ACPI tables
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiACPIReclaimMemory), 0, 0x7FE00000ULL, 0, 128,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiACPIMemoryNVS), 0, 0x7FE80000ULL, 0, 64,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        // Runtime services
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiRuntimeServicesCode), 0, 0x7FEC0000ULL, 0, 128,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC | EFI_MEMORY_RUNTIME});
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiRuntimeServicesData), 0, 0x7FF40000ULL, 0, 192,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC | EFI_MEMORY_RUNTIME});
        // MMIO Devices
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiMemoryMappedIO), 0, 0xC0000000ULL, 0, 4096,
                           EFI_MEMORY_UC | EFI_MEMORY_WC | EFI_MEMORY_RUNTIME});
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiMemoryMappedIO), 0, 0xFEE00000ULL, 0, 1,
                           EFI_MEMORY_UC | EFI_MEMORY_RUNTIME});
        // High RAM (> 4GB)
        sim.AddDescriptor({static_cast<uint32_t>(MemoryType::EfiConventionalMemory), 0, 0x100000000ULL, 0, 524288,
                           EFI_MEMORY_WB | EFI_MEMORY_WT | EFI_MEMORY_WC | EFI_MEMORY_UC});
        return sim;
    }

    const std::vector<MemoryDescriptor>& Descriptors() const noexcept {
        return descriptors_;
    }

private:
    std::vector<MemoryDescriptor> descriptors_;
    uint64_t descriptor_size_{sizeof(MemoryDescriptor)};
    uint32_t descriptor_version_{1};
};

/**
 * @brief Self-test verification suite
 */
int RunSelfTests() {
    std::cout << "[TEST] Running UEFI Memory Map & Descriptor Verification Tests...\n";

    // 1. Struct layout & padding
    assert(sizeof(MemoryDescriptor) == 40);
    assert(offsetof(MemoryDescriptor, Type) == 0);
    assert(offsetof(MemoryDescriptor, Pad) == 4);
    assert(offsetof(MemoryDescriptor, PhysicalStart) == 8);
    assert(offsetof(MemoryDescriptor, VirtualStart) == 16);
    assert(offsetof(MemoryDescriptor, NumberOfPages) == 24);
    assert(offsetof(MemoryDescriptor, Attribute) == 32);
    std::cout << "  ✓ EFI_MEMORY_DESCRIPTOR struct alignment & offsets verified (40 bytes)\n";

    // 2. Standard map creation
    auto map48 = MemoryMapSimulator::CreateStandardQemuOvmfMap(48);
    assert(map48.EntryCount() == 13);
    assert(map48.TotalMapBytes() == 13 * 48);
    assert(map48.TotalConventionalBytes() > 0);
    assert(map48.TotalBootServicesReclaimableBytes() > 0);
    assert(map48.TotalRuntimeServicesBytes() > 0);
    std::cout << "  ✓ Standard UEFI Memory Map created with 13 descriptors\n";

    // 3. Serialization and Deserialization with standard 48-byte descriptors
    auto buf48 = map48.Serialize();
    assert(buf48.size() == 13 * 48);
    auto restored48 = MemoryMapSimulator::Deserialize(buf48.data(), buf48.size(), 48);
    assert(restored48.EntryCount() == 13);
    assert(restored48.TotalConventionalBytes() == map48.TotalConventionalBytes());
    std::cout << "  ✓ 48-byte descriptor serialization & deserialization passed\n";

    // 4. Extended descriptor size (e.g. 64-byte descriptor alignment in future firmware)
    auto map64 = MemoryMapSimulator::CreateStandardQemuOvmfMap(64);
    auto buf64 = map64.Serialize();
    assert(buf64.size() == 13 * 64);
    auto restored64 = MemoryMapSimulator::Deserialize(buf64.data(), buf64.size(), 64);
    assert(restored64.EntryCount() == 13);
    assert(restored64.TotalConventionalBytes() == map64.TotalConventionalBytes());
    std::cout << "  ✓ Extended 64-byte descriptor pointer stepping verified (iter += descriptor_size)\n";

    // 5. Conventional memory calculation check
    // Conventional pages: 32512 + 498816 + 524288 = 1055616 pages * 4096 = 4323803136 bytes (~4.026 GB)
    uint64_t expected_conv_pages = 32512 + 498816 + 524288;
    assert(map48.TotalConventionalBytes() == expected_conv_pages * 4096);
    std::cout << "  ✓ Conventional free RAM calculation verified: "
              << (map48.TotalConventionalBytes() / (1024 * 1024)) << " MiB\n";

    // 6. BootServices reclaimable memory check
    // Code/Data/Loader: 159 + 128 + 256 = 543 pages * 4096 = 2224128 bytes (~2.12 MiB)
    uint64_t expected_bs_pages = 159 + 128 + 256;
    assert(map48.TotalBootServicesReclaimableBytes() == expected_bs_pages * 4096);
    std::cout << "  ✓ BootServices reclaimable memory calculation verified: "
              << (map48.TotalBootServicesReclaimableBytes() / 1024) << " KiB\n";

    std::cout << "[SUCCESS] All UEFI Memory Map self-tests passed cleanly!\n";
    return 0;
}

} // namespace mikan::uefi

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--test") {
            return mikan::uefi::RunSelfTests();
        }
    }

    std::cout << "=================================================================\n";
    std::cout << "🍊 MikanOS EDK II EFI_MEMORY_DESCRIPTOR & Memory Map Inspector\n";
    std::cout << "=================================================================\n\n";

    auto map = mikan::uefi::MemoryMapSimulator::CreateStandardQemuOvmfMap(48);

    std::cout << std::left
              << std::setw(5)  << "Idx"
              << std::setw(28) << "Type (Name)"
              << std::setw(18) << "PhysicalStart"
              << std::setw(14) << "NumberOfPages"
              << std::setw(14) << "SizeBytes"
              << "Attribute\n";
    std::cout << std::string(90, '-') << "\n";

    size_t idx = 0;
    for (const auto& d : map.Descriptors()) {
        std::stringstream ss_phys;
        ss_phys << "0x" << std::hex << std::setw(12) << std::setfill('0') << d.PhysicalStart;

        std::stringstream ss_size;
        if (d.SizeBytes() >= 1024 * 1024 * 1024) {
            ss_size << (d.SizeBytes() / (1024 * 1024 * 1024)) << " GiB";
        } else if (d.SizeBytes() >= 1024 * 1024) {
            ss_size << (d.SizeBytes() / (1024 * 1024)) << " MiB";
        } else {
            ss_size << (d.SizeBytes() / 1024) << " KiB";
        }

        std::cout << std::left
                  << std::dec
                  << std::setw(5)  << idx++
                  << std::setw(28) << mikan::uefi::GetMemoryTypeName(static_cast<mikan::uefi::MemoryType>(d.Type))
                  << std::setw(18) << ss_phys.str()
                  << std::setw(14) << d.NumberOfPages
                  << std::setw(14) << ss_size.str()
                  << "0x" << std::hex << d.Attribute << std::dec << "\n";
    }

    std::cout << "\n📊 Memory Map Summary:\n";
    std::cout << "  - Total Descriptors: " << map.EntryCount() << "\n";
    std::cout << "  - Descriptor Size:   " << map.DescriptorSize() << " bytes\n";
    std::cout << "  - Total Map Size:    " << map.TotalMapBytes() << " bytes\n";
    std::cout << "  - Initial Usable RAM (Conventional): "
              << (map.TotalConventionalBytes() / (1024 * 1024)) << " MiB ("
              << (map.TotalConventionalBytes() / (1024 * 1024 * 1024.0)) << " GiB)\n";
    std::cout << "  - Reclaimable after ExitBootServices: "
              << (map.TotalBootServicesReclaimableBytes() / 1024) << " KiB\n";
    std::cout << "  - Runtime Services Preserved: "
              << (map.TotalRuntimeServicesBytes() / 1024) << " KiB\n";

    return 0;
}
