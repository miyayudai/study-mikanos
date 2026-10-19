// src/ch00/os_scale_analyzer.cpp
// Section 0.1: Quantitative comparison of OS codebase sizes and kernel architectures
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace mikanos::ch00 {

enum class KernelArchitecture {
    kMonolithic,
    kMicrokernel,
    kHybrid,
};

struct OSSpec {
    std::string_view name;
    int release_year;
    std::size_t lines_of_code;
    KernelArchitecture arch;
    bool is_64bit;
    bool is_uefi;
};

constexpr std::array<OSSpec, 4> kHistoricalOSList = {{
    {"Haribote OS (30-day OS)", 2006, 4'249, KernelArchitecture::kMonolithic, false, false},
    {"Linux 0.01", 1991, 10'239, KernelArchitecture::kMonolithic, false, false},
    {"MikanOS (Final Kernel)", 2021, 11'071, KernelArchitecture::kMonolithic, true, true},
    {"Linux 4.10", 2017, 21'000'000, KernelArchitecture::kMonolithic, true, true},
}};

constexpr double ComputeScaleRatio(const OSSpec& target, const OSSpec& baseline) noexcept {
    return static_cast<double>(target.lines_of_code) /
           static_cast<double>(baseline.lines_of_code);
}

constexpr double ComputeDailyLOC(const OSSpec& spec, std::size_t days = 30) noexcept {
    return static_cast<double>(spec.lines_of_code) / static_cast<double>(days);
}

}  // namespace mikanos::ch00

int main() {
    using namespace mikanos::ch00;

    static_assert(kHistoricalOSList[2].lines_of_code == 11'071,
                  "MikanOS final kernel LOC must match book specification (11,071 lines)");
    static_assert(kHistoricalOSList[2].is_64bit && kHistoricalOSList[2].is_uefi,
                  "MikanOS targets 64-bit Intel 64 mode and UEFI BIOS");

    const auto& haribote = kHistoricalOSList[0];
    const auto& linux_001 = kHistoricalOSList[1];
    const auto& mikanos = kHistoricalOSList[2];
    const auto& linux_410 = kHistoricalOSList[3];

    const double ratio_to_haribote = ComputeScaleRatio(mikanos, haribote);
    const double ratio_to_linux001 = ComputeScaleRatio(mikanos, linux_001);
    const double ratio_linux410_to_mikan = ComputeScaleRatio(linux_410, mikanos);
    const double daily_loc = ComputeDailyLOC(mikanos, 30);

    assert(ratio_to_haribote > 2.5 && ratio_to_haribote < 2.7);
    assert(ratio_to_linux001 > 1.0 && ratio_to_linux001 < 1.1);
    assert(ratio_linux410_to_mikan > 1800.0);
    assert(daily_loc > 360.0 && daily_loc < 370.0);

    std::cout << "=== Section 0.1: OS Codebase Scale Analysis (C++20) ===\n";
    for (const auto& os : kHistoricalOSList) {
        std::cout << std::left << std::setw(26) << os.name
                  << " | Year: " << os.release_year
                  << " | LOC: " << std::right << std::setw(10) << os.lines_of_code
                  << " | 64-bit: " << (os.is_64bit ? "Yes" : "No ")
                  << " | UEFI: " << (os.is_uefi ? "Yes" : "No ") << "\n";
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[Analysis]\n"
              << "- MikanOS / Haribote ratio   : " << ratio_to_haribote << "x\n"
              << "- MikanOS / Linux 0.01 ratio : " << ratio_to_linux001 << "x\n"
              << "- Linux 4.10 / MikanOS ratio : " << ratio_linux410_to_mikan << "x\n"
              << "- MikanOS average LOC / day  : " << daily_loc << " lines/day (over 30 days)\n";
    return 0;
}
