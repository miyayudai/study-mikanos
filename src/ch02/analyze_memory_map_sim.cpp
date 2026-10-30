/**
 * @file analyze_memory_map_sim.cpp
 * @brief C++20 Simulation and Statistical Analysis of UEFI Memory Map Output
 *
 * Simulates MikanOS Chapter 2.7:
 * Parses memory map CSV dumps, analyzes usable EfiConventionalMemory,
 * calculates reclaimable EfiBootServicesCode / EfiBootServicesData after ExitBootServices(),
 * evaluates physical address fragmentation, and verifies memory layout.
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::uefi {

// ============================================================================
// UEFI Memory Types and Basic Definitions
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
    EfiMaxMemoryType           = 15,
};

inline std::string_view GetMemoryTypeName(MemoryType type) noexcept {
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

// ============================================================================
// Parsed Memory Map Record
// ============================================================================

struct MemoryRecord {
    uint32_t index{0};
    uint32_t type_raw{0};
    std::string type_name;
    uint64_t physical_start{0};
    uint64_t number_of_pages{0};
    uint64_t attribute{0};

    [[nodiscard]] constexpr uint64_t SizeBytes() const noexcept {
        return number_of_pages * 4096ULL;
    }

    [[nodiscard]] constexpr uint64_t PhysicalEnd() const noexcept {
        return physical_start + SizeBytes();
    }

    [[nodiscard]] constexpr double SizeMiB() const noexcept {
        return static_cast<double>(SizeBytes()) / (1024.0 * 1024.0);
    }

    [[nodiscard]] constexpr double SizeGiB() const noexcept {
        return static_cast<double>(SizeBytes()) / (1024.0 * 1024.0 * 1024.0);
    }

    [[nodiscard]] constexpr bool IsConventional() const noexcept {
        return type_raw == 7 || type_name == "EfiConventionalMemory";
    }

    [[nodiscard]] constexpr bool IsBootServices() const noexcept {
        return type_raw == 3 || type_raw == 4 ||
               type_name == "EfiBootServicesCode" || type_name == "EfiBootServicesData";
    }
};

// ============================================================================
// Memory Map Parser and Analyzer
// ============================================================================

class MemoryMapAnalyzer {
public:
    explicit MemoryMapAnalyzer(std::vector<MemoryRecord> records)
        : records_(std::move(records)) {
        std::sort(records_.begin(), records_.end(), [](const MemoryRecord& a, const MemoryRecord& b) {
            return a.physical_start < b.physical_start;
        });
        total_pages_ = 0;
        for (const auto& r : records_) {
            total_pages_ += r.number_of_pages;
        }
        total_bytes_ = total_pages_ * 4096ULL;
    }

    static MemoryMapAnalyzer FromCsv(const std::string& csv_text) {
        std::vector<MemoryRecord> records;
        std::istringstream stream(csv_text);
        std::string line;

        bool header_parsed = false;
        while (std::getline(stream, line)) {
            // Trim whitespace
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            if (line.empty()) continue;

            if (!header_parsed) {
                if (line.find("Index") == std::string::npos || line.find("PhysicalStart") == std::string::npos) {
                    throw std::runtime_error("Invalid CSV header format");
                }
                header_parsed = true;
                continue;
            }

            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> parts;
            while (std::getline(ss, token, ',')) {
                // Trim token
                size_t first = token.find_first_not_of(" \t");
                size_t last = token.find_last_not_of(" \t");
                if (first == std::string::npos) {
                    parts.emplace_back("");
                } else {
                    parts.emplace_back(token.substr(first, last - first + 1));
                }
            }

            if (parts.size() < 6) continue;

            MemoryRecord rec{};
            rec.index = static_cast<uint32_t>(std::stoul(parts[0], nullptr, 10));
            rec.type_raw = static_cast<uint32_t>(std::stoul(parts[1], nullptr, 16));
            rec.type_name = parts[2];
            rec.physical_start = std::stoull(parts[3], nullptr, 16);
            rec.number_of_pages = std::stoull(parts[4], nullptr, 16);
            rec.attribute = std::stoull(parts[5], nullptr, 16);

            records.push_back(rec);
        }

        return MemoryMapAnalyzer(std::move(records));
    }

    [[nodiscard]] const std::vector<MemoryRecord>& Records() const noexcept { return records_; }
    [[nodiscard]] size_t TotalRecords() const noexcept { return records_.size(); }
    [[nodiscard]] uint64_t TotalPages() const noexcept { return total_pages_; }
    [[nodiscard]] uint64_t TotalBytes() const noexcept { return total_bytes_; }
    [[nodiscard]] double TotalMiB() const noexcept { return static_cast<double>(total_bytes_) / (1024.0 * 1024.0); }
    [[nodiscard]] double TotalGiB() const noexcept { return static_cast<double>(total_bytes_) / (1024.0 * 1024.0 * 1024.0); }

    struct TypeStats {
        uint32_t type_code{0};
        std::string type_name;
        size_t count{0};
        uint64_t pages{0};
        uint64_t bytes{0};
        double mib{0.0};
        double percentage{0.0};
    };

    [[nodiscard]] std::map<std::string, TypeStats> ComputeTypeBreakdown() const {
        std::map<std::string, TypeStats> breakdown;
        for (const auto& r : records_) {
            auto& entry = breakdown[r.type_name];
            entry.type_code = r.type_raw;
            entry.type_name = r.type_name;
            entry.count++;
            entry.pages += r.number_of_pages;
            entry.bytes += r.SizeBytes();
        }
        for (auto& [_, entry] : breakdown) {
            entry.mib = static_cast<double>(entry.bytes) / (1024.0 * 1024.0);
            entry.percentage = total_bytes_ > 0 ? (static_cast<double>(entry.bytes) / total_bytes_ * 100.0) : 0.0;
        }
        return breakdown;
    }

    struct ConventionalSummary {
        size_t count{0};
        uint64_t pages{0};
        uint64_t bytes{0};
        double mib{0.0};
        double gib{0.0};
        double percentage{0.0};
        uint64_t max_block_pages{0};
        uint64_t max_block_start{0};
        double max_block_mib{0.0};
    };

    [[nodiscard]] ConventionalSummary SummarizeConventionalMemory() const {
        ConventionalSummary summary{};
        for (const auto& r : records_) {
            if (r.IsConventional()) {
                summary.count++;
                summary.pages += r.number_of_pages;
                summary.bytes += r.SizeBytes();
                if (r.number_of_pages > summary.max_block_pages) {
                    summary.max_block_pages = r.number_of_pages;
                    summary.max_block_start = r.physical_start;
                }
            }
        }
        summary.mib = static_cast<double>(summary.bytes) / (1024.0 * 1024.0);
        summary.gib = static_cast<double>(summary.bytes) / (1024.0 * 1024.0 * 1024.0);
        summary.percentage = total_bytes_ > 0 ? (static_cast<double>(summary.bytes) / total_bytes_ * 100.0) : 0.0;
        summary.max_block_mib = static_cast<double>(summary.max_block_pages * 4096ULL) / (1024.0 * 1024.0);
        return summary;
    }

    struct ReclaimableBootServicesSummary {
        size_t count{0};
        uint64_t pages{0};
        uint64_t bytes{0};
        double mib{0.0};
        double percentage{0.0};
    };

    [[nodiscard]] ReclaimableBootServicesSummary SummarizeReclaimableBootServices() const {
        ReclaimableBootServicesSummary summary{};
        for (const auto& r : records_) {
            if (r.IsBootServices()) {
                summary.count++;
                summary.pages += r.number_of_pages;
                summary.bytes += r.SizeBytes();
            }
        }
        summary.mib = static_cast<double>(summary.bytes) / (1024.0 * 1024.0);
        summary.percentage = total_bytes_ > 0 ? (static_cast<double>(summary.bytes) / total_bytes_ * 100.0) : 0.0;
        return summary;
    }

    struct AvailablePostBootSummary {
        uint64_t pages{0};
        uint64_t bytes{0};
        double mib{0.0};
        double gib{0.0};
        double percentage{0.0};
        double conventional_mib{0.0};
        double reclaimable_bs_mib{0.0};
    };

    [[nodiscard]] AvailablePostBootSummary SummarizeAvailableAfterBoot() const {
        auto conv = SummarizeConventionalMemory();
        auto bs = SummarizeReclaimableBootServices();

        AvailablePostBootSummary summary{};
        summary.pages = conv.pages + bs.pages;
        summary.bytes = summary.pages * 4096ULL;
        summary.mib = static_cast<double>(summary.bytes) / (1024.0 * 1024.0);
        summary.gib = static_cast<double>(summary.bytes) / (1024.0 * 1024.0 * 1024.0);
        summary.percentage = total_bytes_ > 0 ? (static_cast<double>(summary.bytes) / total_bytes_ * 100.0) : 0.0;
        summary.conventional_mib = conv.mib;
        summary.reclaimable_bs_mib = bs.mib;
        return summary;
    }

    struct FragmentationMetrics {
        size_t num_blocks{0};
        double avg_block_mib{0.0};
        double largest_block_ratio{0.0};
        double fragmentation_index{0.0}; // 0.0 = completely contiguous, 1.0 = highly fragmented
    };

    [[nodiscard]] FragmentationMetrics ComputeFragmentation() const {
        auto conv = SummarizeConventionalMemory();
        FragmentationMetrics metrics{};
        metrics.num_blocks = conv.count;
        if (metrics.num_blocks == 0) return metrics;

        metrics.avg_block_mib = conv.mib / static_cast<double>(conv.count);
        metrics.largest_block_ratio = conv.mib > 0.0 ? (conv.max_block_mib / conv.mib) : 1.0;
        metrics.fragmentation_index = 1.0 - metrics.largest_block_ratio;
        return metrics;
    }

    struct MemoryGap {
        size_t gap_index{0};
        uint32_t after_record_index{0};
        uint32_t before_record_index{0};
        uint64_t start_addr{0};
        uint64_t end_addr{0};
        uint64_t gap_bytes{0};
        double gap_mib{0.0};
    };

    [[nodiscard]] std::vector<MemoryGap> DetectGaps() const {
        std::vector<MemoryGap> gaps;
        for (size_t i = 0; i + 1 < records_.size(); ++i) {
            uint64_t curr_end = records_[i].PhysicalEnd();
            uint64_t next_start = records_[i + 1].physical_start;
            if (next_start > curr_end) {
                uint64_t diff = next_start - curr_end;
                MemoryGap gap{};
                gap.gap_index = gaps.size();
                gap.after_record_index = records_[i].index;
                gap.before_record_index = records_[i + 1].index;
                gap.start_addr = curr_end;
                gap.end_addr = next_start;
                gap.gap_bytes = diff;
                gap.gap_mib = static_cast<double>(diff) / (1024.0 * 1024.0);
                gaps.push_back(gap);
            }
        }
        return gaps;
    }

private:
    std::vector<MemoryRecord> records_;
    uint64_t total_pages_{0};
    uint64_t total_bytes_{0};
};

// ============================================================================
// Unit Verifications and Assertions
// ============================================================================

void TestMemoryMapCsvParsing() {
    std::cout << "[Test 1] Testing MemoryMap CSV parsing and field deserialization...\n";

    const std::string sample_csv =
        "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n"
        "0, 0, EfiReservedMemoryType, 00000000, 1, f\n"
        "1, 7, EfiConventionalMemory, 00001000, 9f, f\n"
        "2, 0, EfiReservedMemoryType, 000a0000, 60, f\n"
        "3, 7, EfiConventionalMemory, 00100000, 7ef00, f\n"
        "4, 3, EfiBootServicesCode, 7f000000, 1000, f\n"
        "5, 4, EfiBootServicesData, 80000000, 2000, f\n"
        "6, b, EfiMemoryMappedIO, fee00000, 10, 1\n";

    auto analyzer = MemoryMapAnalyzer::FromCsv(sample_csv);
    assert(analyzer.TotalRecords() == 7);

    const auto& recs = analyzer.Records();
    assert(recs[0].physical_start == 0x00000000ULL);
    assert(recs[0].number_of_pages == 1ULL);
    assert(recs[0].SizeBytes() == 4096ULL);
    assert(recs[0].PhysicalEnd() == 4096ULL);

    assert(recs[1].IsConventional());
    assert(recs[1].physical_start == 0x00001000ULL);
    assert(recs[1].number_of_pages == 0x9fULL);

    assert(recs[3].IsConventional());
    assert(recs[3].physical_start == 0x00100000ULL); // 1 MiB
    assert(recs[3].number_of_pages == 0x7ef00ULL);

    assert(recs[4].IsBootServices());
    assert(recs[5].IsBootServices());
    assert(recs[6].type_raw == 0xb);

    std::cout << "  -> Total Descriptors: " << analyzer.TotalRecords()
              << ", Total Memory: " << std::fixed << std::setprecision(2)
              << analyzer.TotalMiB() << " MiB (" << analyzer.TotalGiB() << " GiB)\n";
    std::cout << "  [PASS] CSV parsing and numeric conversion verified.\n";
}

void TestMemoryBreakdownAndUsableRam() {
    std::cout << "[Test 2] Testing memory type breakdown and usable RAM calculation...\n";

    const std::string sample_csv =
        "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n"
        "0, 0, EfiReservedMemoryType, 00000000, 1, f\n"
        "1, 7, EfiConventionalMemory, 00001000, 9f, f\n"
        "2, 0, EfiReservedMemoryType, 000a0000, 60, f\n"
        "3, 7, EfiConventionalMemory, 00100000, 70000, f\n" // ~1792 MiB
        "4, 3, EfiBootServicesCode, 70100000, 2000, f\n"   // 32 MiB
        "5, 4, EfiBootServicesData, 72100000, 4000, f\n"   // 64 MiB
        "6, 5, EfiRuntimeServicesCode, 76100000, 400, f\n" // 4 MiB
        "7, b, EfiMemoryMappedIO, 80000000, 10000, 1\n";

    auto analyzer = MemoryMapAnalyzer::FromCsv(sample_csv);
    auto breakdown = analyzer.ComputeTypeBreakdown();

    assert(breakdown.count("EfiConventionalMemory") == 1);
    assert(breakdown.count("EfiBootServicesCode") == 1);
    assert(breakdown.count("EfiBootServicesData") == 1);
    assert(breakdown.count("EfiRuntimeServicesCode") == 1);

    auto conv = analyzer.SummarizeConventionalMemory();
    assert(conv.count == 2);
    assert(conv.pages == (0x9fULL + 0x70000ULL));
    assert(conv.max_block_pages == 0x70000ULL);
    assert(conv.max_block_start == 0x00100000ULL);

    auto bs = analyzer.SummarizeReclaimableBootServices();
    assert(bs.count == 2);
    assert(bs.pages == (0x2000ULL + 0x4000ULL)); // 6000 pages = 24576 KiB = 24 MiB (Wait: 0x6000 pages = 24576 pages * 4K = 96 MiB)
    assert(bs.bytes == 0x6000ULL * 4096ULL);

    auto post_boot = analyzer.SummarizeAvailableAfterBoot();
    assert(post_boot.pages == conv.pages + bs.pages);
    assert(post_boot.bytes == conv.bytes + bs.bytes);

    std::cout << "  -> Immediately Usable Conventional RAM: " << conv.mib << " MiB\n";
    std::cout << "  -> Reclaimable Boot Services RAM: " << bs.mib << " MiB\n";
    std::cout << "  -> Total Available RAM after ExitBootServices: " << post_boot.mib << " MiB\n";
    std::cout << "  [PASS] Usable RAM and post-boot reclaimability verified.\n";
}

void TestFragmentationAndGapAnalysis() {
    std::cout << "[Test 3] Testing fragmentation metrics and MMIO gap detection...\n";

    const std::string sample_csv =
        "Index, Type, Type(name), PhysicalStart, NumberOfPages, Attribute\n"
        "0, 0, EfiReservedMemoryType, 00000000, 1, f\n"        // 0x00000000 - 0x00001000
        "1, 7, EfiConventionalMemory, 00001000, 9f, f\n"       // 0x00001000 - 0x000a0000
        "2, 0, EfiReservedMemoryType, 000a0000, 60, f\n"       // 0x000a0000 - 0x00100000
        "3, 7, EfiConventionalMemory, 00100000, 10000, f\n"    // 0x00100000 - 0x10100000 (256 MiB)
        // GAP HERE: 0x10100000 to 0x20000000 (255 MiB gap)
        "4, 7, EfiConventionalMemory, 20000000, 20000, f\n"    // 0x20000000 - 0x40000000 (512 MiB)
        "5, b, EfiMemoryMappedIO, fed00000, 1, 1\n";

    auto analyzer = MemoryMapAnalyzer::FromCsv(sample_csv);
    auto frag = analyzer.ComputeFragmentation();

    assert(frag.num_blocks == 3);
    assert(frag.largest_block_ratio > 0.6); // Largest block is 512MiB out of ~768MiB
    assert(frag.fragmentation_index < 0.4);

    auto gaps = analyzer.DetectGaps();
    assert(!gaps.empty());
    // Verify first detected gap between record 3 and 4
    assert(gaps[0].start_addr == 0x10100000ULL);
    assert(gaps[0].end_addr == 0x20000000ULL);
    assert(gaps[0].gap_bytes == (0x20000000ULL - 0x10100000ULL));

    std::cout << "  -> Fragmentation Index: " << frag.fragmentation_index
              << " (Blocks: " << frag.num_blocks << ")\n";
    std::cout << "  -> Detected Address Gaps: " << gaps.size()
              << " (First gap: " << std::hex << gaps[0].start_addr << " - " << gaps[0].end_addr
              << " (" << std::dec << gaps[0].gap_mib << " MiB))\n";
    std::cout << "  [PASS] Fragmentation and gap analysis verified.\n";
}

} // namespace mikan::uefi

int main() {
    std::cout << "======================================================================\n";
    std::cout << "MikanOS Chapter 2.7: UEFI Memory Map Output Analysis C++20 Verification\n";
    std::cout << "======================================================================\n";

    mikan::uefi::TestMemoryMapCsvParsing();
    mikan::uefi::TestMemoryBreakdownAndUsableRam();
    mikan::uefi::TestFragmentationAndGapAnalysis();

    std::cout << "\nAll Section 2.7 C++20 verifications passed successfully!\n";
    return 0;
}
