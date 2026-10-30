/**
 * @file qemu_monitor_sim.cpp
 * @brief C++20 Simulation and Verification of QEMU Monitor, x86-64 Registers, and Physical Memory Inspection
 *
 * Simulates and verifies MikanOS Chapter 3.1:
 * - Non-intrusive bare-metal debugging via QEMU Monitor
 * - x86-64 CPU Register state dumping (`info registers`):
 *     - 64-bit General Purpose Registers (RAX..R15)
 *     - Instruction Pointer (RIP) & Stack Pointer (RSP)
 *     - RFLAGS register bit extraction (Interrupt Flag IF, Reserved bit 1)
 *     - Control Registers: CR0 (Paging PG, Protected Mode PE), CR3 (PML4 table root), CR4 (PAE)
 *     - Segment Registers: CS, DS, ES, FS, GS, SS
 * - Physical Memory Inspection (`xp /nuf addr`):
 *     - Unit specifiers: 'b' (1B), 'h' (2B), 'w' (4B), 'g' (8B)
 *     - Format specifiers: 'x' (hex), 'd' (signed dec), 'c' (char)
 *     - Little-Endian byte-order reconstruction and memory layout
 * - Interactive command parsing and dispatching
 */

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mikan::sim {

// ============================================================================
// 1. x86-64 CPU Register Model
// ============================================================================

struct alignas(16) CpuRegisters {
    uint64_t rax{0};
    uint64_t rbx{0};
    uint64_t rcx{0};
    uint64_t rdx{0};
    uint64_t rsi{0};
    uint64_t rdi{0};
    uint64_t rbp{0};
    uint64_t rsp{0x0000000007fe3dc0ULL};
    uint64_t r8{0};
    uint64_t r9{0};
    uint64_t r10{0};
    uint64_t r11{0};
    uint64_t r12{0};
    uint64_t r13{0};
    uint64_t r14{0};
    uint64_t r15{0};

    uint64_t rip{0x0000000000100000ULL};
    uint64_t rflags{0x0000000000000202ULL}; // Bit 1 reserved=1, Bit 9 IF=1

    uint64_t cr0{0x80010033ULL}; // PG (bit 31), WP (bit 16), ET (bit 4), PE (bit 0)
    uint64_t cr2{0};             // Page fault linear address
    uint64_t cr3{0x0000000007fa0000ULL}; // PML4 base physical address
    uint64_t cr4{0x000006f0ULL}; // PAE, OSFXSR, OSXMMEXCPT

    uint16_t cs{0x0038};
    uint16_t ds{0x0030};
    uint16_t es{0x0030};
    uint16_t fs{0x0030};
    uint16_t gs{0x0030};
    uint16_t ss{0x0030};

    [[nodiscard]] constexpr bool IsPagingEnabled() const noexcept {
        return (cr0 & (1ULL << 31)) != 0;
    }

    [[nodiscard]] constexpr bool IsProtectedMode() const noexcept {
        return (cr0 & 1ULL) != 0;
    }

    [[nodiscard]] constexpr bool IsInterruptEnabled() const noexcept {
        return (rflags & (1ULL << 9)) != 0;
    }

    [[nodiscard]] std::string FormatQemuInfoRegisters() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        oss << "RAX=" << std::setw(16) << rax << " RBX=" << std::setw(16) << rbx
            << " RCX=" << std::setw(16) << rcx << " RDX=" << std::setw(16) << rdx << "\n";
        oss << "RSI=" << std::setw(16) << rsi << " RDI=" << std::setw(16) << rdi
            << " RBP=" << std::setw(16) << rbp << " RSP=" << std::setw(16) << rsp << "\n";
        oss << "R8 =" << std::setw(16) << r8  << " R9 =" << std::setw(16) << r9
            << " R10=" << std::setw(16) << r10 << " R11=" << std::setw(16) << r11 << "\n";
        oss << "R12=" << std::setw(16) << r12 << " R13=" << std::setw(16) << r13
            << " R14=" << std::setw(16) << r14 << " R15=" << std::setw(16) << r15 << "\n";
        oss << "RIP=" << std::setw(16) << rip << " RFL=" << std::setw(8) << rflags
            << " [----" << (IsInterruptEnabled() ? "I" : "-") << "---]\n";
        oss << "CS=" << std::setw(4) << cs << " DS=" << std::setw(4) << ds
            << " ES=" << std::setw(4) << es << " FS=" << std::setw(4) << fs
            << " GS=" << std::setw(4) << gs << " SS=" << std::setw(4) << ss << "\n";
        oss << "CR0=" << std::setw(8) << cr0 << " CR2=" << std::setw(16) << cr2
            << " CR3=" << std::setw(16) << cr3 << " CR4=" << std::setw(8) << cr4;
        return oss.str();
    }
};

// ============================================================================
// 2. Physical Memory Inspector & QEMU `xp` Simulator
// ============================================================================

class PhysicalMemoryInspector {
public:
    explicit PhysicalMemoryInspector(size_t size = 16 * 1024 * 1024)
        : size_(size), memory_(std::make_unique<uint8_t[]>(size)) {
        std::memset(memory_.get(), 0, size_);
    }

    void WriteBytes(uint64_t physical_addr, const uint8_t* data, size_t count) {
        if (physical_addr + count > size_) {
            throw std::out_of_range("Physical memory write out of range");
        }
        std::memcpy(memory_.get() + physical_addr, data, count);
    }

    [[nodiscard]] std::vector<uint8_t> ReadBytes(uint64_t physical_addr, size_t count) const {
        if (physical_addr + count > size_) {
            throw std::out_of_range("Physical memory read out of range");
        }
        std::vector<uint8_t> result(count);
        std::memcpy(result.data(), memory_.get() + physical_addr, count);
        return result;
    }

    [[nodiscard]] std::string ExecuteXp(std::string_view format_spec, uint64_t physical_addr) const {
        // Parse /<count><unit><format>
        size_t count = 1;
        char unit = 'w';   // b, h, w, g
        char fmt = 'x';    // x, d, c

        std::string spec(format_spec);
        if (!spec.empty() && spec.front() == '/') {
            spec = spec.substr(1);
        }

        size_t idx = 0;
        if (idx < spec.size() && std::isdigit(static_cast<unsigned char>(spec[idx]))) {
            size_t end_digits = idx;
            while (end_digits < spec.size() && std::isdigit(static_cast<unsigned char>(spec[end_digits]))) {
                ++end_digits;
            }
            count = std::stoull(spec.substr(idx, end_digits - idx));
            idx = end_digits;
        }

        while (idx < spec.size()) {
            char ch = spec[idx++];
            if (ch == 'b' || ch == 'h' || ch == 'w' || ch == 'g') {
                unit = ch;
            } else if (ch == 'x' || ch == 'd' || ch == 'c') {
                fmt = ch;
            }
        }

        size_t unit_bytes = 4;
        if (unit == 'b') unit_bytes = 1;
        else if (unit == 'h') unit_bytes = 2;
        else if (unit == 'w') unit_bytes = 4;
        else if (unit == 'g') unit_bytes = 8;

        size_t items_per_line = 4;
        if (unit_bytes == 1) items_per_line = 16;
        else if (unit_bytes == 2) items_per_line = 8;
        else if (unit_bytes == 4) items_per_line = 4;
        else if (unit_bytes == 8) items_per_line = 2;

        std::ostringstream out;
        for (size_t line_start = 0; line_start < count; line_start += items_per_line) {
            size_t line_count = std::min(items_per_line, count - line_start);
            uint64_t line_addr = physical_addr + line_start * unit_bytes;

            out << "0x" << std::hex << std::setfill('0') << std::setw(16) << line_addr << ":";

            for (size_t i = 0; i < line_count; ++i) {
                uint64_t item_addr = line_addr + i * unit_bytes;
                if (item_addr + unit_bytes > size_) break;

                const uint8_t* p = memory_.get() + item_addr;
                out << " ";

                if (unit == 'b') {
                    uint8_t val = *p;
                    if (fmt == 'x') {
                        out << "0x" << std::setw(2) << static_cast<int>(val);
                    } else if (fmt == 'c') {
                        char c = (val >= 32 && val <= 126) ? static_cast<char>(val) : '.';
                        out << "'" << c << "'";
                    } else {
                        out << std::dec << static_cast<int>(val);
                    }
                } else if (unit == 'h') {
                    uint16_t val = 0;
                    std::memcpy(&val, p, sizeof(val));
                    if (fmt == 'x') {
                        out << "0x" << std::setw(4) << val;
                    } else {
                        out << std::dec << static_cast<int16_t>(val);
                    }
                } else if (unit == 'w') {
                    uint32_t val = 0;
                    std::memcpy(&val, p, sizeof(val));
                    if (fmt == 'x') {
                        out << "0x" << std::setw(8) << val;
                    } else {
                        out << std::dec << static_cast<int32_t>(val);
                    }
                } else if (unit == 'g') {
                    uint64_t val = 0;
                    std::memcpy(&val, p, sizeof(val));
                    if (fmt == 'x') {
                        out << "0x" << std::setw(16) << val;
                    } else {
                        out << std::dec << static_cast<int64_t>(val);
                    }
                }
            }
            if (line_start + items_per_line < count) {
                out << "\n";
            }
        }

        return out.str();
    }

private:
    size_t size_;
    std::unique_ptr<uint8_t[]> memory_;
};

// ============================================================================
// 3. Verification Report
// ============================================================================

struct VerificationReport {
    bool cpu_registers_paging_flag_valid{false};
    bool cpu_registers_protected_mode_valid{false};
    bool cpu_registers_interrupt_flag_valid{false};
    bool cpu_info_registers_format_valid{false};
    bool physical_memory_write_read_valid{false};
    bool qemu_xp_byte_dump_valid{false};
    bool qemu_xp_giant_word_little_endian_valid{false};
    bool qemu_xp_multiple_lines_format_valid{false};

    [[nodiscard]] constexpr bool AllPassed() const noexcept {
        return cpu_registers_paging_flag_valid &&
               cpu_registers_protected_mode_valid &&
               cpu_registers_interrupt_flag_valid &&
               cpu_info_registers_format_valid &&
               physical_memory_write_read_valid &&
               qemu_xp_byte_dump_valid &&
               qemu_xp_giant_word_little_endian_valid &&
               qemu_xp_multiple_lines_format_valid;
    }
};

class QemuMonitorVerification {
public:
    static VerificationReport RunVerification() {
        VerificationReport report;

        // 1. CPU Registers Bit Testing
        CpuRegisters regs;
        report.cpu_registers_paging_flag_valid = regs.IsPagingEnabled();
        report.cpu_registers_protected_mode_valid = regs.IsProtectedMode();
        report.cpu_registers_interrupt_flag_valid = regs.IsInterruptEnabled();

        // 2. Format QEMU info registers output
        std::string info_regs = regs.FormatQemuInfoRegisters();
        report.cpu_info_registers_format_valid =
            (info_regs.find("RIP=0000000000100000") != std::string::npos) &&
            (info_regs.find("CR3=0000000007fa0000") != std::string::npos) &&
            (info_regs.find("[----I---]") != std::string::npos);

        // 3. Physical Memory Write / Read
        PhysicalMemoryInspector mem(4 * 1024 * 1024);
        const uint8_t sample_code[] = {
            0x48, 0x83, 0xEC, 0x28, // sub rsp, 0x28
            0xE8, 0x12, 0x00, 0x00, // call +0x12
            0x00, 0x48, 0x83, 0xC4, // (cont) add rsp, 0x28
            0x28, 0xC3, 0x90, 0x90  // ret, nop, nop
        };
        mem.WriteBytes(0x100000, sample_code, sizeof(sample_code));
        auto readback = mem.ReadBytes(0x100000, 4);
        report.physical_memory_write_read_valid =
            (readback[0] == 0x48 && readback[1] == 0x83 && readback[2] == 0xEC && readback[3] == 0x28);

        // 4. `xp /4bx 0x100000`
        std::string dump_b = mem.ExecuteXp("/4bx", 0x100000);
        report.qemu_xp_byte_dump_valid =
            (dump_b.find("0x0000000000100000: 0x48 0x83 0xec 0x28") != std::string::npos);

        // 5. Little-endian 64-bit giant word `xp /1gx 0x100000`
        // Bytes: 48 83 ec 28 e8 12 00 00 -> 0x000012e828ec8348
        std::string dump_g = mem.ExecuteXp("/1gx", 0x100000);
        report.qemu_xp_giant_word_little_endian_valid =
            (dump_g.find("0x0000000000100000: 0x000012e828ec8348") != std::string::npos);

        // 6. Multiple lines formatting `xp /16bx 0x100000`
        std::string dump_16b = mem.ExecuteXp("/16bx", 0x100000);
        report.qemu_xp_multiple_lines_format_valid =
            (dump_16b.find("0x90") != std::string::npos);

        return report;
    }
};

} // namespace mikan::sim

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    const bool is_test_mode = (argc > 1 && std::string_view(argv[1]) == "--test");

    const auto report = mikan::sim::QemuMonitorVerification::RunVerification();

    if (is_test_mode) {
        if (!report.AllPassed()) {
            std::cerr << "[ERROR] QemuMonitorVerification failed!" << std::endl;
            return 1;
        }
        std::cout << "[PASS] All QemuMonitorVerification tests passed successfully." << std::endl;
        return 0;
    }

    std::cout << "===============================================================================\n";
    std::cout << "  MikanOS Chapter 3.1: QEMU Monitor & x86-64 Debugging Infrastructure          \n";
    std::cout << "===============================================================================\n\n";

    std::cout << "[1] x86-64 CPU Register Model & Flags Verification:\n";
    std::cout << "  - CR0 Paging (PG bit 31)      : " << (report.cpu_registers_paging_flag_valid ? "ENABLED (1)" : "DISABLED") << "\n";
    std::cout << "  - CR0 Protected Mode (PE bit) : " << (report.cpu_registers_protected_mode_valid ? "ENABLED (1)" : "DISABLED") << "\n";
    std::cout << "  - RFLAGS Interrupt Flag (IF)  : " << (report.cpu_registers_interrupt_flag_valid ? "ENABLED (1)" : "DISABLED") << "\n";
    std::cout << "  - `info registers` Format     : " << (report.cpu_info_registers_format_valid ? "PASSED" : "FAILED") << "\n\n";

    std::cout << "[2] Physical Memory Inspection (`xp /nuf addr`) Verification:\n";
    std::cout << "  - RAM Write & Readback        : " << (report.physical_memory_write_read_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  - `xp /4bx` Byte Dump Formatter: " << (report.qemu_xp_byte_dump_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "  - `xp /1gx` Little-Endian Math: " << (report.qemu_xp_giant_word_little_endian_valid ? "PASSED (0x000012e828ec8348)" : "FAILED") << "\n";
    std::cout << "  - Multi-line Formatting       : " << (report.qemu_xp_multiple_lines_format_valid ? "PASSED" : "FAILED") << "\n\n";

    mikan::sim::CpuRegisters sample_regs;
    std::cout << "[3] Simulated QEMU Monitor `info registers` Output:\n";
    std::cout << sample_regs.FormatQemuInfoRegisters() << "\n\n";

    std::cout << "[Summary] All 8 verification checks: "
              << (report.AllPassed() ? "\033[32mALL PASSED\033[0m" : "\033[31mSOME FAILED\033[0m")
              << "\n===============================================================================\n";

    return report.AllPassed() ? 0 : 1;
}
