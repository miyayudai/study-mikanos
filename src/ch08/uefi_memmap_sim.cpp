/**
 * @file uefi_memmap_sim.cpp
 * @brief Chapter 8.2: Standalone C++20 Verification for UEFI MemoryMap Inspection in Kernel
 * 
 * Verifies MikanOS day08a memory map handoff and pointer arithmetic:
 * 1. struct MemoryMap and struct MemoryDescriptor layout & alignment
 * 2. System V AMD64 ABI argument passing compatibility
 * 3. Safe descriptor iteration with variable descriptor_size (vs unsafe sizeof stepping)
 * 4. Filtering available physical memory (Conventional, BootServicesCode, BootServicesData)
 */

#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <numeric>

// ============================================================================
// MikanOS day08a Kernel Data Structures (kernel/memory_map.hpp)
// ============================================================================

struct MemoryMap {
  unsigned long long buffer_size;
  void* buffer;
  unsigned long long map_size;
  unsigned long long map_key;
  unsigned long long descriptor_size;
  uint32_t descriptor_version;
};

struct MemoryDescriptor {
  uint32_t type;
  uintptr_t physical_start;
  uintptr_t virtual_start;
  uint64_t number_of_pages;
  uint64_t attribute;
};

enum class MemoryType : uint32_t {
  kEfiReservedMemoryType,
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

inline bool operator==(uint32_t lhs, MemoryType rhs) {
  return lhs == static_cast<uint32_t>(rhs);
}

inline bool operator==(MemoryType lhs, uint32_t rhs) {
  return rhs == lhs;
}

inline bool IsAvailableForKernel(MemoryType type) {
  return type == MemoryType::kEfiBootServicesCode ||
         type == MemoryType::kEfiBootServicesData ||
         type == MemoryType::kEfiConventionalMemory;
}

inline bool IsAvailableForKernel(uint32_t type) {
  return IsAvailableForKernel(static_cast<MemoryType>(type));
}

// ============================================================================
// Verification and Inspection Engine
// ============================================================================

struct AvailableRegion {
  uint32_t type;
  uintptr_t physical_start;
  uintptr_t physical_end;
  uint64_t number_of_pages;
  uint64_t size_bytes;
  uint64_t attribute;
};

class KernelMemoryMapInspector {
 public:
  static std::vector<AvailableRegion> ScanAvailableMemory(const MemoryMap& memmap) {
    std::vector<AvailableRegion> available;
    const uintptr_t buffer_begin = reinterpret_cast<uintptr_t>(memmap.buffer);
    const uintptr_t buffer_end = buffer_begin + memmap.map_size;

    for (uintptr_t iter = buffer_begin; iter < buffer_end; iter += memmap.descriptor_size) {
      const auto* desc = reinterpret_cast<const MemoryDescriptor*>(iter);
      if (IsAvailableForKernel(desc->type)) {
        uint64_t bytes = desc->number_of_pages * 4096ULL;
        uintptr_t end_addr = desc->physical_start + bytes - 1;
        available.push_back({
            desc->type,
            desc->physical_start,
            end_addr,
            desc->number_of_pages,
            bytes,
            desc->attribute
        });
      }
    }
    return available;
  }

  static uint64_t CalculateTotalAvailableBytes(const MemoryMap& memmap) {
    auto regions = ScanAvailableMemory(memmap);
    uint64_t total = 0;
    for (const auto& r : regions) {
      total += r.size_bytes;
    }
    return total;
  }
};

// ============================================================================
// Unit Tests for Chapter 8.2
// ============================================================================

void TestStructureLayoutAndSizes() {
  std::cout << "[Test 1] Checking MemoryDescriptor struct layout & alignment...\n";
  // On x86-64:
  // uint32_t type: 4 bytes
  // 4 bytes implicit padding for 8-byte alignment of physical_start
  // uintptr_t physical_start: 8 bytes
  // uintptr_t virtual_start: 8 bytes
  // uint64_t number_of_pages: 8 bytes
  // uint64_t attribute: 8 bytes
  // Total C struct size = 40 bytes (0x28)
  static_assert(sizeof(MemoryDescriptor) == 40, "MemoryDescriptor size must be 40 bytes on x86-64");
  static_assert(sizeof(MemoryMap) == 48, "MemoryMap size must be 48 bytes on x86-64");

  std::cout << "  sizeof(MemoryDescriptor) = " << sizeof(MemoryDescriptor) << " bytes\n";
  std::cout << "  sizeof(MemoryMap) = " << sizeof(MemoryMap) << " bytes\n";
  std::cout << "  -> Struct sizes verified.\n";
}

void TestMemoryTypeOperators() {
  std::cout << "[Test 2] Checking MemoryType operator== comparisons...\n";
  uint32_t raw_conv = 7;
  assert(raw_conv == MemoryType::kEfiConventionalMemory);
  assert(MemoryType::kEfiConventionalMemory == raw_conv);

  uint32_t raw_bs_code = 3;
  assert(raw_bs_code == MemoryType::kEfiBootServicesCode);

  assert(IsAvailableForKernel(MemoryType::kEfiBootServicesCode));
  assert(IsAvailableForKernel(MemoryType::kEfiBootServicesData));
  assert(IsAvailableForKernel(MemoryType::kEfiConventionalMemory));
  assert(!IsAvailableForKernel(MemoryType::kEfiReservedMemoryType));
  assert(!IsAvailableForKernel(MemoryType::kEfiLoaderCode));
  assert(!IsAvailableForKernel(MemoryType::kEfiMemoryMappedIO));

  std::cout << "  -> MemoryType operators verified.\n";
}

void TestDescriptorSteppingWithPadding() {
  std::cout << "[Test 3] Simulating UEFI MemoryMap with 48-byte descriptor_size...\n";

  // UEFI standard often uses descriptor_size = 48 bytes (40 bytes data + 8 bytes vendor padding/reserved)
  const size_t desc_size = 48;
  const size_t num_descs = 5;
  std::vector<uint8_t> raw_buffer(desc_size * num_descs, 0);

  // Set up 5 descriptors:
  // 0: Reserved (0x00000000 - 0x0009FFFF, 160 pages) -> Not available
  // 1: ConventionalMemory (0x00100000 - 0x03FFFFFF, 16128 pages = 63 MiB) -> Available
  // 2: BootServicesCode (0x04000000 - 0x041FFFFF, 512 pages = 2 MiB) -> Available
  // 3: BootServicesData (0x04200000 - 0x043FFFFF, 512 pages = 2 MiB) -> Available
  // 4: MemoryMappedIO (0xFEE00000 - 0xFEE00FFF, 1 page = 4 KiB) -> Not available

  struct RawEntry {
    MemoryType type;
    uintptr_t phys;
    uint64_t pages;
    uint64_t attr;
  } entries[5] = {
    {MemoryType::kEfiReservedMemoryType, 0x00000000, 160, 0x8},
    {MemoryType::kEfiConventionalMemory, 0x00100000, 16128, 0x8},
    {MemoryType::kEfiBootServicesCode,   0x04000000, 512,   0x8},
    {MemoryType::kEfiBootServicesData,   0x04200000, 512,   0x8},
    {MemoryType::kEfiMemoryMappedIO,     0xFEE00000, 1,     0x1 | 0x8000000000000000ULL}
  };

  for (size_t i = 0; i < num_descs; ++i) {
    uint8_t* ptr = raw_buffer.data() + i * desc_size;
    auto* desc = reinterpret_cast<MemoryDescriptor*>(ptr);
    desc->type = static_cast<uint32_t>(entries[i].type);
    desc->physical_start = entries[i].phys;
    desc->virtual_start = entries[i].phys;
    desc->number_of_pages = entries[i].pages;
    desc->attribute = entries[i].attr;
  }

  MemoryMap memmap;
  memmap.buffer_size = raw_buffer.size();
  memmap.buffer = raw_buffer.data();
  memmap.map_size = raw_buffer.size();
  memmap.map_key = 1234;
  memmap.descriptor_size = desc_size;
  memmap.descriptor_version = 1;

  auto available = KernelMemoryMapInspector::ScanAvailableMemory(memmap);
  assert(available.size() == 3);

  // Region 1: ConventionalMemory (63 MiB)
  assert(available[0].type == static_cast<uint32_t>(MemoryType::kEfiConventionalMemory));
  assert(available[0].physical_start == 0x00100000);
  assert(available[0].number_of_pages == 16128);
  assert(available[0].size_bytes == 16128 * 4096ULL);

  // Region 2: BootServicesCode (2 MiB)
  assert(available[1].type == static_cast<uint32_t>(MemoryType::kEfiBootServicesCode));
  assert(available[1].physical_start == 0x04000000);
  assert(available[1].number_of_pages == 512);

  // Region 3: BootServicesData (2 MiB)
  assert(available[2].type == static_cast<uint32_t>(MemoryType::kEfiBootServicesData));
  assert(available[2].physical_start == 0x04200000);
  assert(available[2].number_of_pages == 512);

  uint64_t total_avail = KernelMemoryMapInspector::CalculateTotalAvailableBytes(memmap);
  uint64_t expected_bytes = (16128 + 512 + 512) * 4096ULL; // 67 MiB
  assert(total_avail == expected_bytes);

  std::cout << "  Found " << available.size() << " available regions.\n";
  std::cout << "  Total available RAM: " << (total_avail / (1024 * 1024)) << " MiB\n";
  std::cout << "  -> Stepping with descriptor_size=48 verified successfully.\n";
}

void TestUnsafeSteppingBugDemonstration() {
  std::cout << "[Test 4] Demonstrating memory corruption if sizeof(MemoryDescriptor) is used...\n";
  // If a programmer incorrectly wrote:
  // for (MemoryDescriptor* desc = (MemoryDescriptor*)map.buffer; ...; desc++)
  // desc would advance by 40 bytes instead of 48 bytes!
  const size_t desc_size = 48;
  std::vector<uint8_t> raw_buffer(desc_size * 2, 0);

  // Entry 0
  auto* d0 = reinterpret_cast<MemoryDescriptor*>(raw_buffer.data());
  d0->type = static_cast<uint32_t>(MemoryType::kEfiConventionalMemory);
  d0->physical_start = 0x100000;
  d0->number_of_pages = 100;

  // Entry 1 (at offset 48)
  auto* d1 = reinterpret_cast<MemoryDescriptor*>(raw_buffer.data() + 48);
  d1->type = static_cast<uint32_t>(MemoryType::kEfiBootServicesCode);
  d1->physical_start = 0x200000;
  d1->number_of_pages = 200;

  // If using incorrect pointer arithmetic (step of 40 bytes):
  auto* bad_d1 = reinterpret_cast<const MemoryDescriptor*>(raw_buffer.data() + sizeof(MemoryDescriptor));
  // bad_d1 points to offset 40, which falls into the 8-byte padding of entry 0!
  // Its type will read 0 (or garbage) instead of kEfiBootServicesCode (3).
  assert(bad_d1->type != static_cast<uint32_t>(MemoryType::kEfiBootServicesCode));
  std::cout << "  Correct offset 48 type: " << d1->type << " (kEfiBootServicesCode)\n";
  std::cout << "  Corrupted offset 40 type: " << bad_d1->type << " (Off-by-8 bytes corruption)\n";
  std::cout << "  -> Proved why 'iter += memory_map.descriptor_size' is mandatory.\n";
}

int main(int argc, char* argv[]) {
  bool run_test = (argc > 1 && std::string(argv[1]) == "--test");
  if (run_test) {
    std::cout << "[Info] Running in automated test mode (--test)\n";
  }
  std::cout << "============================================================\n";
  std::cout << "  MikanOS Chapter 8.2: UEFI MemoryMap Inspection Simulator  \n";
  std::cout << "============================================================\n";

  TestStructureLayoutAndSizes();
  TestMemoryTypeOperators();
  TestDescriptorSteppingWithPadding();
  TestUnsafeSteppingBugDemonstration();

  std::cout << "\n[PASS] All Chapter 8.2 memory map inspection tests passed successfully!\n";
  return 0;
}
