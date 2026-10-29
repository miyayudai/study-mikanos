/**
 * @file toolchain_inspector.cpp
 * @brief Standalone C++20 verification tool for Section 1.8: Tools for OS Development.
 *
 * Demonstrates and verifies target triple decomposition, freestanding environment header
 * compliance, compiler flag safety checks (e.g. -ffreestanding, -mno-red-zone, -fno-stack-protector),
 * and the 4-stage C/C++ compilation pipeline to PE32+ UEFI / ELF binaries.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <vector>
#include <set>
#include <array>
#include <cstdint>
#include <cassert>
#include <sstream>

namespace mikan {

/**
 * @brief Decomposition of a LLVM/Clang target triple.
 */
struct TargetTriple {
    std::string arch;
    std::string vendor;
    std::string os_name;
    std::string environment;

    static TargetTriple parse(std::string_view str) {
        TargetTriple tt;
        std::vector<std::string> parts;
        std::string current;
        for (char c : str) {
            if (c == '-') {
                if (!current.empty()) {
                    parts.push_back(current);
                    current.clear();
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            parts.push_back(current);
        }

        if (parts.size() == 1) {
            tt.arch = parts[0];
        } else if (parts.size() == 2) {
            tt.arch = parts[0];
            tt.vendor = "unknown";
            tt.os_name = parts[1];
        } else if (parts.size() == 3) {
            tt.arch = parts[0];
            tt.vendor = parts[1];
            tt.os_name = parts[2];
        } else if (parts.size() >= 4) {
            tt.arch = parts[0];
            tt.vendor = parts[1];
            tt.os_name = parts[2];
            std::string env;
            for (size_t i = 3; i < parts.size(); ++i) {
                if (i > 3) env += "-";
                env += parts[i];
            }
            tt.environment = env;
        }
        return tt;
    }

    [[nodiscard]] std::string to_string() const {
        std::string s = arch;
        if (!vendor.empty()) {
            s += "-" + vendor;
        }
        if (!os_name.empty()) {
            s += "-" + os_name;
        }
        if (!environment.empty()) {
            s += "-" + environment;
        }
        return s;
    }

    [[nodiscard]] bool is_cross_compile(const TargetTriple& host) const {
        return arch != host.arch || os_name != host.os_name || environment != host.environment;
    }

    [[nodiscard]] bool is_uefi_compatible() const {
        if (arch != "x86_64") return false;
        return (os_name.find("win32") != std::string::npos ||
                os_name.find("windows") != std::string::npos ||
                environment.find("coff") != std::string::npos);
    }

    [[nodiscard]] std::string object_format() const {
        if (is_uefi_compatible() || environment.find("coff") != std::string::npos) return "COFF";
        if (os_name.find("linux") != std::string::npos || os_name.find("elf") != std::string::npos || environment.find("elf") != std::string::npos || os_name == "none") return "ELF";
        if (os_name.find("darwin") != std::string::npos || os_name.find("macos") != std::string::npos) return "Mach-O";
        return "Unknown";
    }

    [[nodiscard]] std::string executable_format() const {
        if (is_uefi_compatible()) return "PE32+";
        if (os_name.find("linux") != std::string::npos || os_name.find("elf") != std::string::npos || environment.find("elf") != std::string::npos) return "ELF64";
        if (os_name.find("darwin") != std::string::npos || os_name.find("macos") != std::string::npos) return "Mach-O 64";
        return "Unknown";
    }
};

/**
 * @brief Checks ISO C/C++ freestanding environment constraints.
 */
class FreestandingChecker {
public:
    static inline const std::set<std::string> standard_freestanding_headers = {
        "float.h", "iso646.h", "limits.h", "stdalign.h",
        "stdarg.h", "stdbool.h", "stddef.h", "stdint.h", "stdnoreturn.h"
    };

    static inline const std::set<std::string> hosted_only_headers = {
        "stdio.h", "stdlib.h", "string.h", "math.h", "time.h",
        "assert.h", "signal.h", "setjmp.h", "errno.h", "ctype.h",
        "locale.h", "wchar.h", "wctype.h", "unistd.h", "fcntl.h",
        "sys/types.h", "pthread.h"
    };

    static bool is_header_allowed(const std::string& header) {
        return standard_freestanding_headers.contains(header);
    }

    struct IncludeAnalysis {
        std::vector<std::string> allowed;
        std::vector<std::string> forbidden;
        bool is_compliant;
    };

    static IncludeAnalysis analyze_headers(const std::vector<std::string>& headers) {
        IncludeAnalysis res;
        for (const auto& h : headers) {
            if (standard_freestanding_headers.contains(h)) {
                res.allowed.push_back(h);
            } else if (hosted_only_headers.contains(h)) {
                res.forbidden.push_back(h);
            }
        }
        res.is_compliant = res.forbidden.empty();
        return res;
    }
};

/**
 * @brief Audits compiler options for bare-metal OS development.
 */
class CompilerFlagsAuditor {
public:
    struct AuditResult {
        double score_percent;
        std::vector<std::string> passed_flags;
        std::vector<std::pair<std::string, std::string>> warnings;
        bool is_safe;
    };

    static AuditResult audit(const std::vector<std::string>& flags) {
        std::set<std::string> flag_set(flags.begin(), flags.end());
        AuditResult result{0.0, {}, {}, false};

        const std::vector<std::pair<std::string, std::string>> essential = {
            {"-ffreestanding", "ホスト標準ライブラリ (libc) 依存を排除し、ベアメタル環境としてコンパイル"},
            {"-mno-red-zone", "割り込み・ファームウェア呼び出し時のスタック128バイト Red Zone 破壊を防止"},
            {"-fno-stack-protector", "スタックカナリア未定義参照 (__stack_chk_fail) によるリンクエラー・クラッシュを防止"},
            {"-fshort-wchar", "UEFI仕様の16ビット Unicode (CHAR16) に wchar_t を整合"}
        };

        for (const auto& [flag, desc] : essential) {
            if (flag_set.contains(flag)) {
                result.passed_flags.push_back(flag);
            } else {
                result.warnings.emplace_back(flag, desc);
            }
        }

        result.score_percent = (static_cast<double>(result.passed_flags.size()) / essential.size()) * 100.0;
        result.is_safe = result.warnings.empty();
        return result;
    }
};

} // namespace mikan

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::cout << "======================================================================\n";
    std::cout << "  MikanOS Section 1.8: OSを作る道具 (Compiler Toolchain Inspector)\n";
    std::cout << "======================================================================\n\n";

    // 1. Target Triple Decomposition
    auto host = mikan::TargetTriple::parse("x86_64-unknown-linux-gnu");
    auto target_uefi = mikan::TargetTriple::parse("x86_64-pc-win32-coff");
    auto target_kernel = mikan::TargetTriple::parse("x86_64-elf");

    std::cout << "[1] Target Triple Analysis:\n";
    std::cout << "  - Host Triple:   " << host.to_string()
              << " (Object: " << host.object_format() << ", Executable: " << host.executable_format() << ")\n";
    std::cout << "  - UEFI Target:   " << target_uefi.to_string()
              << " (Object: " << target_uefi.object_format() << ", Executable: " << target_uefi.executable_format() << ")\n";
    std::cout << "  - Kernel Target: " << target_kernel.to_string()
              << " (Object: " << target_kernel.object_format() << ", Executable: " << target_kernel.executable_format() << ")\n";
    std::cout << "  - Cross-compilation (Host -> UEFI): "
              << (target_uefi.is_cross_compile(host) ? "YES (クロスコンパイル必須)" : "NO") << "\n";
    std::cout << "  - UEFI Compatibility: "
              << (target_uefi.is_uefi_compatible() ? "YES (PE/COFF 生成可能)" : "NO") << "\n\n";

    // 2. Freestanding Headers Check
    std::vector<std::string> sample_headers = {"stdint.h", "stddef.h", "stdio.h", "string.h", "stdarg.h"};
    auto inc_res = mikan::FreestandingChecker::analyze_headers(sample_headers);
    std::cout << "[2] Freestanding Environment Header Compliance:\n";
    std::cout << "  - Allowed (Freestanding): ";
    for (const auto& h : inc_res.allowed) std::cout << "<" << h << "> ";
    std::cout << "\n  - Forbidden (Hosted Only): ";
    for (const auto& h : inc_res.forbidden) std::cout << "<" << h << "> ";
    std::cout << "\n  - Result: "
              << (inc_res.is_compliant ? "PASS (フリースタンディング適合)" : "VIOLATION (ホストlibc依存あり！OSカーネルでは利用不可)")
              << "\n\n";

    // 3. Compiler Flags Safety Audit
    std::vector<std::string> test_flags = {"-ffreestanding", "-mno-red-zone", "-fno-stack-protector", "-fshort-wchar", "-O2", "-Wall"};
    auto audit_res = mikan::CompilerFlagsAuditor::audit(test_flags);
    std::cout << "[3] Compiler Flags Safety Audit (MikanOS UEFI Toolchain):\n";
    std::cout << "  - Safety Score: " << std::fixed << std::setprecision(1) << audit_res.score_percent << "%\n";
    std::cout << "  - Passed Essential Flags: ";
    for (const auto& f : audit_res.passed_flags) std::cout << f << " ";
    std::cout << "\n  - Status: " << (audit_res.is_safe ? "SAFE (OS自作の全必須フラグ完備)" : "WARNING") << "\n\n";

    // Self-verification assertions
    assert(target_uefi.is_cross_compile(host));
    assert(target_uefi.is_uefi_compatible());
    assert(target_uefi.executable_format() == "PE32+");
    assert(target_uefi.object_format() == "COFF");
    assert(!inc_res.is_compliant); // contains stdio.h and string.h
    assert(audit_res.is_safe);

    std::cout << "[+] All self-verification assertions passed successfully!\n";
    return 0;
}
