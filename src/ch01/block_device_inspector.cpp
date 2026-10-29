/**
 * @file block_device_inspector.cpp
 * @brief Linux Block Device Inspector and Safe USB Detection for MikanOS (Chapter 1.2)
 *
 * Implements block device inspection, major/minor number decoding, sysfs tree traversal,
 * partition detection, and accidental data loss prevention protocols for UEFI USB boot media.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <regex>

namespace fs = std::filesystem;

namespace mikanos::ch01 {

/**
 * @brief Decoded Linux device number (dev_t)
 */
struct DeviceNumber {
    uint32_t major{0};
    uint32_t minor{0};

    [[nodiscard]] std::string to_string() const {
        return std::to_string(major) + ":" + std::to_string(minor);
    }
};

/**
 * @brief Parse Linux major:minor string (e.g., "8:16")
 */
inline DeviceNumber parse_device_number(const std::string& dev_str) {
    DeviceNumber num;
    auto colon = dev_str.find(':');
    if (colon != std::string::npos) {
        num.major = static_cast<uint32_t>(std::stoul(dev_str.substr(0, colon)));
        num.minor = static_cast<uint32_t>(std::stoul(dev_str.substr(colon + 1)));
    }
    return num;
}

/**
 * @brief Partition entry representation
 */
struct PartitionInfo {
    std::string name;
    DeviceNumber dev_num;
    uint64_t sector_count{0};
    uint64_t size_bytes{0};
    std::string mountpoint;
};

/**
 * @brief Block device representation
 */
struct BlockDevice {
    std::string name;             // e.g. "sdb", "nvme0n1"
    DeviceNumber dev_num;         // major:minor
    bool is_removable{false};     // from /sys/block/<name>/removable
    bool is_usb_bus{false};       // true if sysfs path contains "/usb"
    uint64_t sector_count{0};     // 512-byte sectors
    uint64_t size_bytes{0};       // total bytes
    std::string vendor{"Unknown"};
    std::string model{"Unknown"};
    std::string transport{"Unknown"}; // USB, NVMe, ATA, SCSI, Virtual
    std::vector<PartitionInfo> partitions;

    [[nodiscard]] double size_gib() const {
        return static_cast<double>(size_bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    [[nodiscard]] bool is_candidate_for_boot_usb() const {
        // Removable and USB-connected devices are safe candidates
        return is_removable && (is_usb_bus || transport == "USB");
    }
};

/**
 * @brief Helper to read a single trimmed line from a sysfs file
 */
inline std::optional<std::string> read_sysfs_line(const fs::path& file_path) {
    if (!fs::exists(file_path)) return std::nullopt;
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) return std::nullopt;
    std::string line;
    if (std::getline(ifs, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }
        return line;
    }
    return std::nullopt;
}

/**
 * @brief Inspector engine for Linux block subsystem
 */
class BlockDeviceInspector {
public:
    static std::vector<BlockDevice> scan_system(const fs::path& sysfs_block_root = "/sys/block") {
        std::vector<BlockDevice> devices;
        if (!fs::exists(sysfs_block_root) || !fs::is_directory(sysfs_block_root)) {
            return devices;
        }

        for (const auto& entry : fs::directory_iterator(sysfs_block_root)) {
            std::string name = entry.path().filename().string();
            // Filter out ram and loop devices for clarity unless requested
            if (name.rfind("ram", 0) == 0 || name.rfind("loop", 0) == 0) {
                continue;
            }

            BlockDevice dev;
            dev.name = name;

            // 1. Device number (major:minor)
            auto dev_str = read_sysfs_line(entry.path() / "dev");
            if (dev_str) {
                dev.dev_num = parse_device_number(*dev_str);
            }

            // 2. Removable flag
            auto rem_str = read_sysfs_line(entry.path() / "removable");
            if (rem_str && *rem_str == "1") {
                dev.is_removable = true;
            }

            // 3. Size in sectors (512-byte units in kernel sysfs)
            auto size_str = read_sysfs_line(entry.path() / "size");
            if (size_str) {
                try {
                    dev.sector_count = std::stoull(*size_str);
                    dev.size_bytes = dev.sector_count * 512ULL;
                } catch (...) {
                    dev.sector_count = 0;
                    dev.size_bytes = 0;
                }
            }

            // 4. Vendor / Model (under device/ directory)
            auto vendor_str = read_sysfs_line(entry.path() / "device" / "vendor");
            if (vendor_str) dev.vendor = *vendor_str;

            auto model_str = read_sysfs_line(entry.path() / "device" / "model");
            if (model_str) dev.model = *model_str;

            // 5. Inspect sysfs canonical target to detect bus type
            try {
                fs::path canonical_p = fs::canonical(entry.path());
                std::string path_str = canonical_p.string();
                if (path_str.find("/usb") != std::string::npos) {
                    dev.is_usb_bus = true;
                    dev.transport = "USB";
                } else if (path_str.find("/nvme") != std::string::npos) {
                    dev.transport = "NVMe";
                } else if (path_str.find("/ata") != std::string::npos) {
                    dev.transport = "SATA/ATA";
                } else if (path_str.find("/virtio") != std::string::npos) {
                    dev.transport = "VirtIO";
                } else {
                    dev.transport = "SCSI/Internal";
                }
            } catch (...) {
                dev.transport = "Unknown";
            }

            // 6. Find partitions (subdirectories with matching prefix e.g., sda1, nvme0n1p1)
            for (const auto& sub : fs::directory_iterator(entry.path())) {
                std::string sub_name = sub.path().filename().string();
                if (fs::exists(sub.path() / "partition")) {
                    PartitionInfo part;
                    part.name = sub_name;
                    auto part_dev = read_sysfs_line(sub.path() / "dev");
                    if (part_dev) part.dev_num = parse_device_number(*part_dev);

                    auto part_size = read_sysfs_line(sub.path() / "size");
                    if (part_size) {
                        try {
                            part.sector_count = std::stoull(*part_size);
                            part.size_bytes = part.sector_count * 512ULL;
                        } catch (...) {}
                    }
                    dev.partitions.push_back(part);
                }
            }

            // Sort partitions by name
            std::sort(dev.partitions.begin(), dev.partitions.end(),
                      [](const auto& a, const auto& b) { return a.name < b.name; });

            devices.push_back(dev);
        }

        std::sort(devices.begin(), devices.end(),
                  [](const auto& a, const auto& b) { return a.name < b.name; });
        return devices;
    }

    /**
     * @brief Generate simulated devices for demonstration and testing
     */
    static std::vector<BlockDevice> generate_simulated_devices() {
        std::vector<BlockDevice> devices;

        // 1. Internal NVMe SSD (Host OS - DANGEROUS!)
        BlockDevice host_ssd;
        host_ssd.name = "nvme0n1";
        host_ssd.dev_num = {259, 0};
        host_ssd.is_removable = false;
        host_ssd.is_usb_bus = false;
        host_ssd.transport = "NVMe";
        host_ssd.vendor = "Samsung";
        host_ssd.model = "SSD 980 PRO 1TB";
        host_ssd.sector_count = 1953525168ULL;
        host_ssd.size_bytes = host_ssd.sector_count * 512ULL;
        host_ssd.partitions = {
            {"nvme0n1p1", {259, 1}, 1048576ULL, 1048576ULL * 512ULL, "/boot/efi"},
            {"nvme0n1p2", {259, 2}, 33554432ULL, 33554432ULL * 512ULL, "[SWAP]"},
            {"nvme0n1p3", {259, 3}, 1918920704ULL, 1918920704ULL * 512ULL, "/"}
        };
        devices.push_back(host_ssd);

        // 2. USB Flash Drive (Target for MikanOS Boot Media - SAFE!)
        BlockDevice usb_flash;
        usb_flash.name = "sdb";
        usb_flash.dev_num = {8, 16};
        usb_flash.is_removable = true;
        usb_flash.is_usb_bus = true;
        usb_flash.transport = "USB";
        usb_flash.vendor = "SanDisk";
        usb_flash.model = "Ultra USB 3.0";
        usb_flash.sector_count = 30031872ULL; // ~15.38 GB (16GB drive)
        usb_flash.size_bytes = usb_flash.sector_count * 512ULL;
        usb_flash.partitions = {
            {"sdb1", {8, 17}, 30029824ULL, 30029824ULL * 512ULL, "/media/student/MIKAN_USB"}
        };
        devices.push_back(usb_flash);

        return devices;
    }

    /**
     * @brief Formatted print of block device status
     */
    static void print_device_table(const std::vector<BlockDevice>& devices) {
        std::cout << "========================================================================================\n";
        std::cout << std::left << std::setw(10) << "NAME"
                  << std::setw(8)  << "MAJ:MIN"
                  << std::setw(6)  << "RM"
                  << std::setw(8)  << "BUS"
                  << std::setw(12) << "SIZE (GiB)"
                  << std::setw(14) << "VENDOR"
                  << std::setw(20) << "MODEL"
                  << "SAFE TARGET?\n";
        std::cout << "----------------------------------------------------------------------------------------\n";

        for (const auto& dev : devices) {
            bool safe = dev.is_candidate_for_boot_usb();
            std::cout << std::left << std::setw(10) << dev.name
                      << std::setw(8)  << dev.dev_num.to_string()
                      << std::setw(6)  << (dev.is_removable ? "1" : "0")
                      << std::setw(8)  << dev.transport
                      << std::setw(12) << std::fixed << std::setprecision(2) << dev.size_gib()
                      << std::setw(14) << dev.vendor.substr(0, 12)
                      << std::setw(20) << dev.model.substr(0, 18)
                      << (safe ? "✅ SAFE (USB Boot Candidate)" : "❌ DANGER (Host Disk)")
                      << "\n";

            for (const auto& part : dev.partitions) {
                double part_gib = static_cast<double>(part.size_bytes) / (1024.0 * 1024.0 * 1024.0);
                std::cout << "  └─ " << std::left << std::setw(6) << part.name
                          << std::setw(8)  << part.dev_num.to_string()
                          << std::setw(6)  << " "
                          << std::setw(8)  << " "
                          << std::setw(12) << std::fixed << std::setprecision(2) << part_gib
                          << "Mount: " << (part.mountpoint.empty() ? "(none)" : part.mountpoint)
                          << "\n";
            }
        }
        std::cout << "========================================================================================\n";
    }

    /**
     * @brief Safety verification protocol
     */
    static bool evaluate_safety(const std::string& target_name, const std::vector<BlockDevice>& devices) {
        for (const auto& dev : devices) {
            if (dev.name == target_name) {
                std::cout << "\n[Safety Check for /dev/" << target_name << "]\n";
                std::cout << "  - Removable Media: " << (dev.is_removable ? "YES (Pass)" : "NO (FAIL - Internal Drive!)") << "\n";
                std::cout << "  - Bus Type: " << dev.transport << (dev.transport == "USB" ? " (Pass)" : " (FAIL - Non-USB!)") << "\n";
                std::cout << "  - Size: " << std::fixed << std::setprecision(2) << dev.size_gib() << " GiB\n";

                if (dev.is_candidate_for_boot_usb()) {
                    std::cout << "  => VERDICT: [APPROVED] /dev/" << target_name
                              << " is a removable USB device suitable for MikanOS boot media.\n";
                    return true;
                } else {
                    std::cout << "  => VERDICT: [BLOCKED] /dev/" << target_name
                              << " is an internal host drive! NEVER write or format this device!\n";
                    return false;
                }
            }
        }
        std::cout << "  => VERDICT: [NOT FOUND] Device /dev/" << target_name << " not detected.\n";
        return false;
    }
};

} // namespace mikanos::ch01

int main(int argc, char* argv[]) {
    using namespace mikanos::ch01;

    std::cout << "MikanOS Chapter 1.2: Linux Block Device Inspector & USB Safety Checker\n\n";

    bool scan_real = false;
    bool demo_mode = false;
    std::string safety_target;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scan") {
            scan_real = true;
        } else if (arg == "--demo") {
            demo_mode = true;
        } else if (arg == "--check" && i + 1 < argc) {
            safety_target = argv[++i];
        }
    }

    if (!scan_real && !demo_mode && safety_target.empty()) {
        demo_mode = true;
        scan_real = true;
    }

    if (scan_real) {
        std::cout << ">>> Scanning Real System Block Devices (/sys/block) <<<\n";
        auto real_devs = BlockDeviceInspector::scan_system();
        if (real_devs.empty()) {
            std::cout << "(No physical block devices found or insufficient permissions in /sys/block)\n";
        } else {
            BlockDeviceInspector::print_device_table(real_devs);
        }
        std::cout << "\n";
    }

    if (demo_mode) {
        std::cout << ">>> Simulation / Educational Model (Host SSD vs Target USB Flash) <<<\n";
        auto sim_devs = BlockDeviceInspector::generate_simulated_devices();
        BlockDeviceInspector::print_device_table(sim_devs);
        std::cout << "\n";

        // Demonstration of accidental data loss check
        BlockDeviceInspector::evaluate_safety("nvme0n1", sim_devs);
        BlockDeviceInspector::evaluate_safety("sdb", sim_devs);
    }

    if (!safety_target.empty()) {
        auto real_devs = BlockDeviceInspector::scan_system();
        BlockDeviceInspector::evaluate_safety(safety_target, real_devs);
    }

    return 0;
}
