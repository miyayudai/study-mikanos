/**
 * @file main_memory_architecture_sim.cpp
 * @brief Standalone C++20 verification tool for Section 2.3: Main Memory Architecture & MMIO Simulation.
 *
 * Demonstrates and verifies:
 * 1. 64-bit Physical Address Space layout, bus width limits (48-bit / 52-bit), and address decoding.
 * 2. RAM vs MMIO (Memory-Mapped I/O) distinction, side effects, and cacheability attributes.
 * 3. Memory hole detection (PCI hole below 4GB, sparse DRAM mapping).
 * 4. Page alignment math (4KiB, 2MiB, 1GiB boundaries) and overlap detection algorithms.
 * 5. Simulation of modern x86-64 QEMU/OVMF physical memory map and legacy 1MB PC/XT map.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <algorithm>

namespace mikan::memory {

// =============================================================================
// 1. Types and Constants
// =============================================================================

using PhysicalAddress = uint64_t;
using ByteCount = uint64_t;
using PageCount = uint64_t;

constexpr ByteCount PAGE_SIZE_4K = 4096ULL;
constexpr ByteCount PAGE_SIZE_2M = 2ULL * 1024ULL * 1024ULL;
constexpr ByteCount PAGE_SIZE_1G = 1024ULL * 1024ULL * 1024ULL;

enum class RegionType : uint8_t {
    UsableRam,
    Reserved,
    AcpiReclaim,
    AcpiNvs,
    Mmio,
    MmioPortSpace,
    Framebuffer,
    BootServicesCode,
    BootServicesData,
    RuntimeServicesCode,
    RuntimeServicesData,
    FirmwareRom,
    MemoryHole
};

[[nodiscard]] constexpr std::string_view GetRegionTypeName(RegionType type) noexcept {
    switch (type) {
        case RegionType::UsableRam:           return "Usable RAM (DRAM)";
        case RegionType::Reserved:            return "Reserved Platform";
        case RegionType::AcpiReclaim:         return "ACPI Reclaim Memory";
        case RegionType::AcpiNvs:             return "ACPI NVS Storage";
        case RegionType::Mmio:                return "PCIe MMIO Space";
        case RegionType::MmioPortSpace:       return "MMIO Port Space";
        case RegionType::Framebuffer:         return "GOP Framebuffer (VRAM)";
        case RegionType::BootServicesCode:    return "UEFI Boot Services Code";
        case RegionType::BootServicesData:    return "UEFI Boot Services Data";
        case RegionType::RuntimeServicesCode: return "UEFI Runtime Services Code";
        case RegionType::RuntimeServicesData: return "UEFI Runtime Services Data";
        case RegionType::FirmwareRom:         return "Firmware / Option ROM";
        case RegionType::MemoryHole:          return "Unmapped Memory Hole";
    }
    return "Unknown";
}

// =============================================================================
// 2. Alignment & Math Utilities
// =============================================================================

[[nodiscard]] constexpr bool IsAligned(PhysicalAddress addr, ByteCount align) noexcept {
    assert(align > 0 && (align & (align - 1)) == 0 && "align must be a power of two");
    return (addr & (align - 1)) == 0;
}

[[nodiscard]] constexpr PhysicalAddress AlignUp(PhysicalAddress addr, ByteCount align) noexcept {
    assert(align > 0 && (align & (align - 1)) == 0 && "align must be a power of two");
    return (addr + align - 1) & ~(align - 1);
}

[[nodiscard]] constexpr PhysicalAddress AlignDown(PhysicalAddress addr, ByteCount align) noexcept {
    assert(align > 0 && (align & (align - 1)) == 0 && "align must be a power of two");
    return addr & ~(align - 1);
}

[[nodiscard]] constexpr PageCount To4KPageCount(ByteCount bytes) noexcept {
    return (bytes + PAGE_SIZE_4K - 1) / PAGE_SIZE_4K;
}

[[nodiscard]] std::string FormatByteSize(ByteCount bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (bytes >= (1ULL << 30)) {
        oss << (static_cast<double>(bytes) / (1ULL << 30)) << " GiB";
    } else if (bytes >= (1ULL << 20)) {
        oss << (static_cast<double>(bytes) / (1ULL << 20)) << " MiB";
    } else if (bytes >= (1ULL << 10)) {
        oss << (static_cast<double>(bytes) / (1ULL << 10)) << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

// =============================================================================
// 3. Memory Region Structure
// =============================================================================

struct MemoryRegion {
    PhysicalAddress base{0};
    ByteCount length{0};
    RegionType type{RegionType::Reserved};
    std::string description;
    bool is_cacheable{true};

    [[nodiscard]] constexpr PhysicalAddress EndAddress() const noexcept {
        return base + length;
    }

    [[nodiscard]] constexpr PageCount Pages() const noexcept {
        return To4KPageCount(length);
    }

    [[nodiscard]] constexpr bool IsPageAligned() const noexcept {
        return IsAligned(base, PAGE_SIZE_4K) && IsAligned(length, PAGE_SIZE_4K);
    }

    [[nodiscard]] constexpr bool Contains(PhysicalAddress addr) const noexcept {
        return addr >= base && addr < EndAddress();
    }

    [[nodiscard]] constexpr bool OverlapsWith(const MemoryRegion& other) const noexcept {
        return std::max(base, other.base) < std::min(EndAddress(), other.EndAddress());
    }

    [[nodiscard]] constexpr bool IsFreelyUsableAfterBoot() const noexcept {
        return type == RegionType::UsableRam ||
               type == RegionType::BootServicesCode ||
               type == RegionType::BootServicesData;
    }
};

// =============================================================================
// 4. Physical Address Space Simulator
// =============================================================================

class PhysicalAddressSpaceSimulator {
public:
    explicit PhysicalAddressSpaceSimulator(uint32_t address_width = 48)
        : address_width_bits_(address_width),
          max_address_((address_width == 64) ? ~0ULL : ((1ULL << address_width) - 1)) {
        assert((address_width == 32 || address_width == 36 || address_width == 40 ||
                address_width == 48 || address_width == 52) &&
               "Valid physical address widths for x86/x86-64 are 32, 36, 40, 48, 52 bits");
    }

    [[nodiscard]] uint32_t AddressWidth() const noexcept { return address_width_bits_; }
    [[nodiscard]] PhysicalAddress MaxAddress() const noexcept { return max_address_; }

    [[nodiscard]] const std::vector<MemoryRegion>& Regions() const noexcept {
        return regions_;
    }

    bool AddRegion(MemoryRegion region, std::string* error_out = nullptr) {
        if (region.length == 0) {
            if (error_out) *error_out = "Region length cannot be zero";
            return false;
        }
        if (region.EndAddress() > max_address_ + 1 && region.EndAddress() != 0) {
            if (error_out) {
                std::ostringstream oss;
                oss << "Region end 0x" << std::hex << region.EndAddress()
                    << " exceeds address width limit 0x" << max_address_;
                *error_out = oss.str();
            }
            return false;
        }
        for (const auto& existing : regions_) {
            if (region.OverlapsWith(existing)) {
                if (error_out) {
                    std::ostringstream oss;
                    oss << "Collision: '" << region.description << "' [0x" << std::hex
                        << region.base << "-0x" << region.EndAddress() << ") overlaps with '"
                        << existing.description << "' [0x" << existing.base << "-0x"
                        << existing.EndAddress() << ")";
                    *error_out = oss.str();
                }
                return false;
            }
        }
        regions_.push_back(std::move(region));
        std::sort(regions_.begin(), regions_.end(), [](const MemoryRegion& a, const MemoryRegion& b) {
            return a.base < b.base;
        });
        return true;
    }

    [[nodiscard]] const MemoryRegion* ResolveAddress(PhysicalAddress addr) const noexcept {
        if (addr > max_address_) return nullptr;
        for (const auto& r : regions_) {
            if (r.Contains(addr)) return &r;
        }
        return nullptr;
    }

    [[nodiscard]] std::vector<std::pair<PhysicalAddress, PhysicalAddress>> DetectMemoryHoles() const {
        std::vector<std::pair<PhysicalAddress, PhysicalAddress>> holes;
        if (regions_.empty()) {
            holes.emplace_back(0, max_address_);
            return holes;
        }

        if (regions_.front().base > 0) {
            holes.emplace_back(0, regions_.front().base);
        }

        for (size_t i = 0; i < regions_.size() - 1; ++i) {
            PhysicalAddress curr_end = regions_[i].EndAddress();
            PhysicalAddress next_start = regions_[i + 1].base;
            if (curr_end < next_start) {
                holes.emplace_back(curr_end, next_start);
            }
        }
        return holes;
    }

    [[nodiscard]] std::optional<PhysicalAddress> FindContiguousRam(
        ByteCount size, ByteCount alignment = PAGE_SIZE_4K) const noexcept {
        for (const auto& r : regions_) {
            if (r.type == RegionType::UsableRam) {
                PhysicalAddress aligned_start = AlignUp(r.base, alignment);
                if (aligned_start + size <= r.EndAddress()) {
                    return aligned_start;
                }
            }
        }
        return std::nullopt;
    }

    struct Summary {
        ByteCount total_usable_ram{0};
        ByteCount total_mmio{0};
        ByteCount total_reserved{0};
        ByteCount total_acpi{0};
        ByteCount total_firmware{0};
        ByteCount total_uefi_services{0};
        ByteCount total_mapped{0};
        ByteCount total_holes{0};
    };

    [[nodiscard]] Summary ComputeSummary() const {
        Summary sum;
        for (const auto& r : regions_) {
            sum.total_mapped += r.length;
            switch (r.type) {
                case RegionType::UsableRam:
                    sum.total_usable_ram += r.length;
                    break;
                case RegionType::Mmio:
                case RegionType::MmioPortSpace:
                case RegionType::Framebuffer:
                    sum.total_mmio += r.length;
                    break;
                case RegionType::Reserved:
                    sum.total_reserved += r.length;
                    break;
                case RegionType::AcpiReclaim:
                case RegionType::AcpiNvs:
                    sum.total_acpi += r.length;
                    break;
                case RegionType::FirmwareRom:
                    sum.total_firmware += r.length;
                    break;
                case RegionType::BootServicesCode:
                case RegionType::BootServicesData:
                case RegionType::RuntimeServicesCode:
                case RegionType::RuntimeServicesData:
                    sum.total_uefi_services += r.length;
                    break;
                default:
                    break;
            }
        }
        for (const auto& hole : DetectMemoryHoles()) {
            sum.total_holes += (hole.second - hole.first);
        }
        return sum;
    }

    void DumpMemoryMap(std::ostream& os) const {
        os << "========================================================================================\n";
        os << " x86-64 Physical Address Space Map (" << address_width_bits_ << "-bit Physical Bus Width)\n";
        os << "========================================================================================\n";
        os << std::left
           << std::setw(6)  << "Idx"
           << std::setw(20) << "Start Address"
           << std::setw(20) << "End Address"
           << std::setw(12) << "Size"
           << std::setw(26) << "Type"
           << "Description\n";
        os << "----------------------------------------------------------------------------------------\n";

        size_t idx = 0;
        for (const auto& r : regions_) {
            std::ostringstream s_start, s_end;
            s_start << "0x" << std::hex << std::setw(12) << std::setfill('0') << r.base;
            s_end   << "0x" << std::hex << std::setw(12) << std::setfill('0') << r.EndAddress();

            os << std::left
               << std::setw(6)  << idx++
               << std::setw(20) << s_start.str()
               << std::setw(20) << s_end.str()
               << std::setw(12) << FormatByteSize(r.length)
               << std::setw(26) << GetRegionTypeName(r.type)
               << r.description << "\n";
        }

        os << "----------------------------------------------------------------------------------------\n";
        auto sum = ComputeSummary();
        os << "Summary:\n";
        os << "  Total Mapped Memory:   " << FormatByteSize(sum.total_mapped) << "\n";
        os << "  - Usable DRAM:         " << FormatByteSize(sum.total_usable_ram) << "\n";
        os << "  - MMIO / Framebuffer:  " << FormatByteSize(sum.total_mmio) << "\n";
        os << "  - UEFI Services:       " << FormatByteSize(sum.total_uefi_services) << "\n";
        os << "  - ACPI Tables:         " << FormatByteSize(sum.total_acpi) << "\n";
        os << "  - Firmware ROMs:       " << FormatByteSize(sum.total_firmware) << "\n";
        os << "  - Reserved Hardware:   " << FormatByteSize(sum.total_reserved) << "\n";
        os << "  Detected Memory Holes: " << DetectMemoryHoles().size() << " gap(s)\n";
        os << "========================================================================================\n";
    }

    [[nodiscard]] static PhysicalAddressSpaceSimulator CreateQemuStandardPc() {
        PhysicalAddressSpaceSimulator sim(48);
        sim.AddRegion({0x0000'0000, 0x1000, RegionType::Reserved, "Real-Mode IVT & BDA", true});
        sim.AddRegion({0x0000'1000, 0x7'F000, RegionType::UsableRam, "Low Conventional RAM (<640KB)", true});
        sim.AddRegion({0x0008'0000, 0x2'0000, RegionType::Reserved, "Extended BIOS Data Area (EBDA)", true});
        sim.AddRegion({0x000A'0000, 0x2'0000, RegionType::Framebuffer, "Legacy VGA Framebuffer MMIO", false});
        sim.AddRegion({0x000C'0000, 0x2'0000, RegionType::FirmwareRom, "Video BIOS & Option ROMs", true});
        sim.AddRegion({0x000E'0000, 0x2'0000, RegionType::FirmwareRom, "Motherboard UEFI / BIOS ROM", true});
        sim.AddRegion({0x0010'0000, 0x7FE0'0000, RegionType::UsableRam, "Main Conventional RAM (1MB-2GB)", true});
        sim.AddRegion({0x7FF0'0000, 0x10'0000, RegionType::BootServicesData, "UEFI Boot Services Pool", true});
        sim.AddRegion({0x8000'0000, 0x10'0000, RegionType::AcpiReclaim, "ACPI Tables & Reclaim Memory", true});
        sim.AddRegion({0xC000'0000, 0x0100'0000, RegionType::Framebuffer, "GOP High-Res Framebuffer MMIO", false});
        sim.AddRegion({0xC100'0000, 0x3DB0'0000, RegionType::Mmio, "PCI Express Device MMIO Space (xHCI/NVMe/NIC)", false});
        sim.AddRegion({0xFEC0'0000, 0x1000, RegionType::Mmio, "I/O APIC MMIO Register Window", false});
        sim.AddRegion({0xFEE0'0000, 0x1000, RegionType::Mmio, "Local APIC MMIO Register Window", false});
        sim.AddRegion({0xFFC0'0000, 0x40'0000, RegionType::FirmwareRom, "OVMF High Firmware Flash ROM (Reset Vector)", true});
        sim.AddRegion({0x1'0000'0000ULL, 0x8000'0000ULL, RegionType::UsableRam, "High RAM above 4GB (>4GB Conventional RAM)", true});
        return sim;
    }

    [[nodiscard]] static PhysicalAddressSpaceSimulator CreateLegacy1MbPc() {
        PhysicalAddressSpaceSimulator sim(32);
        sim.AddRegion({0x00000, 0x00400, RegionType::Reserved, "Interrupt Vector Table (IVT)", true});
        sim.AddRegion({0x00400, 0x00100, RegionType::Reserved, "BIOS Data Area (BDA)", true});
        sim.AddRegion({0x00500, 0x07700, RegionType::UsableRam, "Low DOS RAM", true});
        sim.AddRegion({0x07C00, 0x00200, RegionType::UsableRam, "MBR Bootsector (0x7C00)", true});
        sim.AddRegion({0x07E00, 0x78200, RegionType::UsableRam, "Conventional DOS Free RAM", true});
        sim.AddRegion({0x80000, 0x20000, RegionType::Reserved, "EBDA (128KB)", true});
        sim.AddRegion({0xA0000, 0x10000, RegionType::Framebuffer, "VGA Graphics MMIO (64KB)", false});
        sim.AddRegion({0xB0000, 0x08000, RegionType::Framebuffer, "MDA Text Buffer (32KB)", false});
        sim.AddRegion({0xB8000, 0x08000, RegionType::Framebuffer, "CGA Text Buffer (32KB)", false});
        sim.AddRegion({0xC0000, 0x08000, RegionType::FirmwareRom, "Video BIOS ROM", true});
        sim.AddRegion({0xC8000, 0x18000, RegionType::FirmwareRom, "Option ROMs", true});
        sim.AddRegion({0xE0000, 0x20000, RegionType::FirmwareRom, "System BIOS ROM", true});
        return sim;
    }

private:
    uint32_t address_width_bits_{48};
    PhysicalAddress max_address_{0xFFFF'FFFF'FFFFULL};
    std::vector<MemoryRegion> regions_;
};

// =============================================================================
// 5. Self Tests
// =============================================================================

void RunSelfTests() {
    std::cout << "[TEST] Running self-tests for Main Memory Architecture & MMIO Simulator...\n";

    // 1. Math and alignment tests
    assert(IsAligned(0x1000, PAGE_SIZE_4K));
    assert(!IsAligned(0x1001, PAGE_SIZE_4K));
    assert(AlignUp(0x1001, PAGE_SIZE_4K) == 0x2000);
    assert(AlignDown(0x1FFF, PAGE_SIZE_4K) == 0x1000);
    assert(To4KPageCount(4096) == 1);
    assert(To4KPageCount(4097) == 2);
    assert(To4KPageCount(0) == 0);

    // 2. Overlap detection
    MemoryRegion r1{0x1000, 0x1000, RegionType::UsableRam, "R1", true}; // [0x1000, 0x2000)
    MemoryRegion r2{0x2000, 0x1000, RegionType::UsableRam, "R2", true}; // [0x2000, 0x3000) - Adjacent, no overlap
    MemoryRegion r3{0x1800, 0x1000, RegionType::UsableRam, "R3", true}; // [0x1800, 0x2800) - Overlaps r1 and r2
    assert(!r1.OverlapsWith(r2));
    assert(r1.OverlapsWith(r3));
    assert(r2.OverlapsWith(r3));

    // 3. Address resolution
    PhysicalAddressSpaceSimulator sim = PhysicalAddressSpaceSimulator::CreateQemuStandardPc();
    const auto* reg1 = sim.ResolveAddress(0x0010'0000); // 1MB start
    assert(reg1 != nullptr);
    assert(reg1->type == RegionType::UsableRam);

    const auto* reg_apic = sim.ResolveAddress(0xFEE0'0020); // Local APIC
    assert(reg_apic != nullptr);
    assert(reg_apic->type == RegionType::Mmio);
    assert(!reg_apic->is_cacheable);

    const auto* reg_fb = sim.ResolveAddress(0xC000'1000); // Framebuffer
    assert(reg_fb != nullptr);
    assert(reg_fb->type == RegionType::Framebuffer);

    // 4. Memory holes
    auto holes = sim.DetectMemoryHoles();
    assert(!holes.empty());
    // There should be a PCI hole between GOP Framebuffer and PCIe MMIO or around APIC
    bool found_gap = false;
    for (const auto& hole : holes) {
        if (hole.second > hole.first) found_gap = true;
    }
    assert(found_gap);

    // 5. Summary calculations
    auto sum = sim.ComputeSummary();
    assert(sum.total_usable_ram > 0);
    assert(sum.total_mmio > 0);
    assert(sum.total_mapped > 0);

    // 6. Contiguous RAM finder
    auto free_chunk = sim.FindContiguousRam(16 * 1024 * 1024); // 16MB
    assert(free_chunk.has_value());
    assert(IsAligned(*free_chunk, PAGE_SIZE_4K));

    // 7. Address collision rejection
    std::string err;
    bool added = sim.AddRegion({0x0010'0000, 0x1000, RegionType::UsableRam, "Duplicate", true}, &err);
    assert(!added);
    assert(!err.empty());

    std::cout << "[TEST] All self-tests passed successfully!\n";
}

} // namespace mikan::memory

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string_view(argv[1]) == "--test") {
        mikan::memory::RunSelfTests();
        return 0;
    }

    std::cout << "MikanOS Main Memory Architecture & MMIO Simulator (C++20 Standalone)\n";
    auto qemu_sim = mikan::memory::PhysicalAddressSpaceSimulator::CreateQemuStandardPc();
    qemu_sim.DumpMemoryMap(std::cout);

    std::cout << "\nLegacy 1MB PC Architecture Map:\n";
    auto legacy_sim = mikan::memory::PhysicalAddressSpaceSimulator::CreateLegacy1MbPc();
    legacy_sim.DumpMemoryMap(std::cout);

    return 0;
}
