// src/ch00/os_build_steps.cpp
// Section 0.3: Bare-metal OS Build Pipeline, Freestanding Environment & UEFI Media Layout
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mikanos::ch00 {

// Standard PE32+ and ELF64 machine constants
enum class BinaryMachineType : uint16_t {
    kUnknown = 0,
    kX86_64_PE = 0x8664,  // IMAGE_FILE_MACHINE_AMD64 for UEFI
    kX86_64_ELF = 0x3E,   // EM_X86_64 for Linux / ELF64 kernel
};

enum class BinaryFormat {
    kPe32Plus,
    kElf64,
};

// Represents a required compiler flag in bare-metal OS compilation
struct CompilerFlagDef {
    std::string_view name;
    std::string_view description;
    bool is_mandatory_freestanding;
};

inline constexpr std::array<CompilerFlagDef, 5> kFreestandingFlags{{
    {"-ffreestanding", "Assume no host standard library or runtime wrapper exists", true},
    {"-nostdlib", "Prevent linking against host C/C++ runtime (crt0.o, libc)", true},
    {"-mno-red-zone", "Disable 128-byte red zone below %rsp to protect interrupt handlers", true},
    {"-fno-exceptions", "Disable C++ exception handling tables and unwinder", true},
    {"-fno-rtti", "Disable C++ Run-Time Type Information (dynamic_cast / typeid)", true},
}};

// Models the stages of the OS build pipeline
enum class PipelineStage {
    kPreprocessing,
    kCompilation,
    kAssembly,
    kLinking,
};

struct PipelineArtifact {
    PipelineStage stage;
    std::string filename;
    std::string format;
    size_t approximate_size;
};

// Simulates the bare-metal build toolchain (Clang/LLVM, LLD)
class ToolchainPipeline {
public:
    explicit ToolchainPipeline(std::vector<std::string> active_flags)
        : active_flags_(std::move(active_flags)) {}

    [[nodiscard]] bool HasFlag(std::string_view flag_name) const {
        return std::any_of(active_flags_.begin(), active_flags_.end(),
                           [flag_name](const std::string& f) { return f == flag_name; });
    }

    [[nodiscard]] bool ValidateFreestandingEnvironment(std::vector<std::string>& missing_flags) const {
        missing_flags.clear();
        for (const auto& flag : kFreestandingFlags) {
            if (flag.is_mandatory_freestanding && !HasFlag(flag.name)) {
                missing_flags.emplace_back(flag.name);
            }
        }
        return missing_flags.empty();
    }

    [[nodiscard]] std::vector<PipelineArtifact> Execute(BinaryFormat target_format) const {
        std::vector<PipelineArtifact> artifacts;

        artifacts.push_back({
            PipelineStage::kPreprocessing,
            "main.i",
            "Preprocessed C++ Translation Unit",
            16384,
        });

        artifacts.push_back({
            PipelineStage::kCompilation,
            "main.s",
            "x86-64 Assembly Source",
            12288,
        });

        artifacts.push_back({
            PipelineStage::kAssembly,
            "main.o",
            "Relocatable Object File",
            8192,
        });

        if (target_format == BinaryFormat::kPe32Plus) {
            artifacts.push_back({
                PipelineStage::kLinking,
                "BOOTX64.EFI",
                "PE32+ (UEFI x86-64 Executable)",
                12288,
            });
        } else {
            artifacts.push_back({
                PipelineStage::kLinking,
                "kernel.elf",
                "ELF64 Kernel Executable",
                16384,
            });
        }

        return artifacts;
    }

private:
    std::vector<std::string> active_flags_;
};

// Validates UEFI FAT32 filesystem layout and fallback bootloader path
class UefiDiskLayoutValidator {
public:
    static constexpr std::string_view kMandatoryBootloader = "/EFI/BOOT/BOOTX64.EFI";

    struct FileEntry {
        std::string path;
        size_t size_bytes;
        BinaryMachineType machine_type;
    };

    void AddFile(std::string_view path, size_t size, BinaryMachineType machine) {
        std::string normalized(path);
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        if (!normalized.empty() && normalized.front() != '/') {
            normalized.insert(normalized.begin(), '/');
        }
        // Store path in uppercase for FAT32 case-insensitive matching
        std::string key = normalized;
        std::transform(key.begin(), key.end(), key.begin(), ::toupper);
        entries_.push_back({std::move(normalized), size, machine});
    }

    [[nodiscard]] bool Validate(std::vector<std::string>& out_errors) const {
        out_errors.clear();
        bool found_bootloader = false;

        for (const auto& entry : entries_) {
            std::string upper_path = entry.path;
            std::transform(upper_path.begin(), upper_path.end(), upper_path.begin(), ::toupper);

            if (upper_path == kMandatoryBootloader) {
                found_bootloader = true;
                if (entry.size_bytes == 0) {
                    out_errors.emplace_back("Bootloader file is 0 bytes");
                }
                if (entry.machine_type != BinaryMachineType::kX86_64_PE) {
                    out_errors.emplace_back("Bootloader machine type must be x86-64 PE (0x8664)");
                }
            }
        }

        if (!found_bootloader) {
            out_errors.emplace_back("Missing mandatory /EFI/BOOT/BOOTX64.EFI path");
        }

        return out_errors.empty();
    }

    [[nodiscard]] const std::vector<FileEntry>& GetEntries() const {
        return entries_;
    }

private:
    std::vector<FileEntry> entries_;
};

}  // namespace mikanos::ch00

int main() {
    using namespace mikanos::ch00;

    std::cout << "=== Section 0.3: OS Build Pipeline & UEFI Media Validation (C++20) ===\n";

    // 1. Verify Compiler Flags & Freestanding Environment
    std::vector<std::string> correct_flags = {
        "-ffreestanding",
        "-nostdlib",
        "-mno-red-zone",
        "-fno-exceptions",
        "-fno-rtti",
        "-O2",
    };

    ToolchainPipeline valid_pipeline(correct_flags);
    std::vector<std::string> missing;
    bool is_valid = valid_pipeline.ValidateFreestandingEnvironment(missing);
    assert(is_valid);
    assert(missing.empty());
    std::cout << "[+] Freestanding Environment Check Passed: All 5 mandatory flags present.\n";

    // Test rejection when a critical flag is omitted (e.g. forgot -mno-red-zone)
    std::vector<std::string> broken_flags = {"-ffreestanding", "-nostdlib", "-O2"};
    ToolchainPipeline broken_pipeline(broken_flags);
    bool broken_valid = broken_pipeline.ValidateFreestandingEnvironment(missing);
    assert(!broken_valid);
    assert(missing.size() == 3);  // missing -mno-red-zone, -fno-exceptions, -fno-rtti
    std::cout << "[+] Protection Check Passed: Missing -mno-red-zone properly identified.\n";

    // 2. Execute Build Pipeline Simulation for UEFI Application
    auto uefi_artifacts = valid_pipeline.Execute(BinaryFormat::kPe32Plus);
    assert(uefi_artifacts.size() == 4);
    assert(uefi_artifacts.back().filename == "BOOTX64.EFI");
    assert(uefi_artifacts.back().stage == PipelineStage::kLinking);
    std::cout << "[+] Pipeline Execution Passed: Successfully generated "
              << uefi_artifacts.back().filename << " (" << uefi_artifacts.back().format << ").\n";

    // 3. Execute Build Pipeline Simulation for Kernel ELF64
    auto kernel_artifacts = valid_pipeline.Execute(BinaryFormat::kElf64);
    assert(kernel_artifacts.size() == 4);
    assert(kernel_artifacts.back().filename == "kernel.elf");
    std::cout << "[+] Kernel Linking Passed: Successfully generated "
              << kernel_artifacts.back().filename << " (" << kernel_artifacts.back().format << ").\n";

    // 4. Validate UEFI FAT32 Boot Disk Layout
    UefiDiskLayoutValidator disk;
    disk.AddFile("\\EFI\\BOOT\\BOOTX64.EFI", 24576, BinaryMachineType::kX86_64_PE);
    disk.AddFile("\\kernel.elf", 65536, BinaryMachineType::kX86_64_ELF);

    std::vector<std::string> disk_errors;
    bool disk_ok = disk.Validate(disk_errors);
    assert(disk_ok);
    assert(disk_errors.empty());
    std::cout << "[+] UEFI Media Validation Passed: Mandatory /EFI/BOOT/BOOTX64.EFI verified.\n";

    // Test rejection of invalid machine type (e.g., ELF instead of PE for bootloader)
    UefiDiskLayoutValidator bad_disk;
    bad_disk.AddFile("/EFI/BOOT/BOOTX64.EFI", 1024, BinaryMachineType::kX86_64_ELF);
    bool bad_disk_ok = bad_disk.Validate(disk_errors);
    assert(!bad_disk_ok);
    assert(disk_errors.size() == 1);
    std::cout << "[+] Media Specification Check Passed: Non-PE bootloader rejected.\n";

    std::cout << "\n[Summary]\n"
              << "- Bare-metal compilation requires -ffreestanding, -nostdlib, and -mno-red-zone.\n"
              << "- UEFI firmware requires FAT32 media and /EFI/BOOT/BOOTX64.EFI formatted in PE32+ (0x8664).\n"
              << "- QEMU and OVMF allow rapid test cycles without physical reboot.\n";

    return 0;
}
