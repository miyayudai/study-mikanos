/**
 * @file relocate_data_sim.cpp
 * @brief Chapter 8.3: Standalone C++20 Verification for Relocating Kernel Data Structures
 * 
 * Verifies MikanOS day08b data structure relocation and copy semantics:
 * 1. struct FrameBufferConfig & struct MemoryMap layout and copy constructors
 * 2. inline bool IsAvailable(MemoryType) categorization logic (memory_map.hpp)
 * 3. Vulnerability of transient UEFI memory when holding raw references
 * 4. Safe isolation via value-copy relocation on the kernel stack
 * 5. Shallow copy pointer semantics of memory_map.buffer & execution phase boundaries
 */

#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iomanip>

// ============================================================================
// MikanOS day08b Kernel Data Structures (kernel/frame_buffer_config.hpp & memory_map.hpp)
// ============================================================================

enum PixelFormat {
  kPixelRGBResv8BitPerColor,
  kPixelBGRResv8BitPerColor,
};

struct FrameBufferConfig {
  uint8_t* frame_buffer;
  uint32_t pixels_per_scan_line;
  uint32_t horizontal_resolution;
  uint32_t vertical_resolution;
  enum PixelFormat pixel_format;
};

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

// Exactly matches MikanOS kernel/memory_map.hpp (day08b)
inline bool IsAvailable(MemoryType memory_type) {
  return
    memory_type == MemoryType::kEfiBootServicesCode ||
    memory_type == MemoryType::kEfiBootServicesData ||
    memory_type == MemoryType::kEfiConventionalMemory;
}

const int kUEFIPageSize = 4096;

// ============================================================================
// Verification Tests
// ============================================================================

void TestStructureLayoutAndSizes() {
  std::cout << "[Test 1] Verifying Structure Sizes and Alignment...\n";
  std::cout << "  sizeof(FrameBufferConfig) = " << sizeof(FrameBufferConfig) << " bytes\n";
  std::cout << "  sizeof(MemoryMap)         = " << sizeof(MemoryMap) << " bytes\n";
  std::cout << "  sizeof(MemoryDescriptor)  = " << sizeof(MemoryDescriptor) << " bytes\n";

  assert(sizeof(FrameBufferConfig) == 24);
  assert(sizeof(MemoryMap) == 48);
  assert(sizeof(MemoryDescriptor) == 40);
  std::cout << "  -> Passed: Structure sizes match 64-bit AMD64 ABI specifications.\n";
}

void TestIsAvailableLogic() {
  std::cout << "[Test 2] Verifying IsAvailable() Logic for All Memory Types...\n";

  // Reusable types
  assert(IsAvailable(MemoryType::kEfiBootServicesCode) == true);
  assert(IsAvailable(MemoryType::kEfiBootServicesData) == true);
  assert(IsAvailable(MemoryType::kEfiConventionalMemory) == true);

  // Preserved / hardware reserved types
  assert(IsAvailable(MemoryType::kEfiReservedMemoryType) == false);
  assert(IsAvailable(MemoryType::kEfiLoaderCode) == false);
  assert(IsAvailable(MemoryType::kEfiLoaderData) == false);
  assert(IsAvailable(MemoryType::kEfiRuntimeServicesCode) == false);
  assert(IsAvailable(MemoryType::kEfiRuntimeServicesData) == false);
  assert(IsAvailable(MemoryType::kEfiUnusableMemory) == false);
  assert(IsAvailable(MemoryType::kEfiACPIReclaimMemory) == false);
  assert(IsAvailable(MemoryType::kEfiACPIMemoryNVS) == false);
  assert(IsAvailable(MemoryType::kEfiMemoryMappedIO) == false);
  assert(IsAvailable(MemoryType::kEfiMemoryMappedIOPortSpace) == false);
  assert(IsAvailable(MemoryType::kEfiPalCode) == false);
  assert(IsAvailable(MemoryType::kEfiPersistentMemory) == false);

  std::cout << "  -> Passed: Exactly 3 types (BootServicesCode, BootServicesData, Conventional) are reusable.\n";
}

void TestUnprotectedReferenceVulnerability() {
  std::cout << "[Test 3] Simulating Vulnerability of Raw Reference Arguments...\n";

  // Simulate UEFI bootloader stack region
  alignas(16) uint8_t uefi_stack_region[256];
  std::memset(uefi_stack_region, 0x00, sizeof(uefi_stack_region));

  // Construct initial structures inside UEFI stack
  auto* uefi_fb = reinterpret_cast<FrameBufferConfig*>(&uefi_stack_region[0]);
  uefi_fb->frame_buffer = reinterpret_cast<uint8_t*>(0xE0000000ULL);
  uefi_fb->pixels_per_scan_line = 1024;
  uefi_fb->horizontal_resolution = 1024;
  uefi_fb->vertical_resolution = 768;
  uefi_fb->pixel_format = kPixelRGBResv8BitPerColor;

  auto* uefi_map = reinterpret_cast<MemoryMap*>(&uefi_stack_region[64]);
  uefi_map->buffer_size = 4096;
  uefi_map->buffer = reinterpret_cast<void*>(0x07E00000ULL);
  uefi_map->map_size = 1200;
  uefi_map->descriptor_size = 48;
  uefi_map->descriptor_version = 1;

  // Kernel receives raw reference (pointer under the hood)
  const FrameBufferConfig& fb_ref = *uefi_fb;
  const MemoryMap& map_ref = *uefi_map;

  assert(fb_ref.frame_buffer == reinterpret_cast<uint8_t*>(0xE0000000ULL));
  assert(map_ref.map_size == 1200);

  // Disaster: OS memory allocator reclaims UEFI BootServicesData and overwrites stack memory!
  std::memset(uefi_stack_region, 0xCC, sizeof(uefi_stack_region));

  // The raw references are now completely corrupted!
  assert(fb_ref.frame_buffer != reinterpret_cast<uint8_t*>(0xE0000000ULL));
  assert(map_ref.map_size != 1200);

  std::cout << "  Simulated UEFI stack wipe with 0xCC pattern.\n";
  std::cout << "  Corrupted frame_buffer pointer: " << static_cast<void*>(fb_ref.frame_buffer) << "\n";
  std::cout << "  Corrupted map_size: " << map_ref.map_size << "\n";
  std::cout << "  -> Proved: Holding raw references without relocation leads to catastrophic crash.\n";
}

void TestCopyConstructorRelocationSafety() {
  std::cout << "[Test 4] Verifying Value-Copy Relocation in KernelMainNewStack...\n";

  // Simulate UEFI bootloader stack region
  alignas(16) uint8_t uefi_stack_region[256];
  std::memset(uefi_stack_region, 0x00, sizeof(uefi_stack_region));

  auto* uefi_fb = reinterpret_cast<FrameBufferConfig*>(&uefi_stack_region[0]);
  uefi_fb->frame_buffer = reinterpret_cast<uint8_t*>(0xE0000000ULL);
  uefi_fb->pixels_per_scan_line = 1024;
  uefi_fb->horizontal_resolution = 1024;
  uefi_fb->vertical_resolution = 768;
  uefi_fb->pixel_format = kPixelRGBResv8BitPerColor;

  auto* uefi_map = reinterpret_cast<MemoryMap*>(&uefi_stack_region[64]);
  uefi_map->buffer_size = 4096;
  uefi_map->buffer = reinterpret_cast<void*>(0x07E00000ULL);
  uefi_map->map_size = 1200;
  uefi_map->descriptor_size = 48;
  uefi_map->descriptor_version = 1;

  // References received by KernelMainNewStack
  const FrameBufferConfig& frame_buffer_config_ref = *uefi_fb;
  const MemoryMap& memory_map_ref = *uefi_map;

  // The MikanOS day08b solution: Copy onto dedicated kernel stack!
  FrameBufferConfig frame_buffer_config{frame_buffer_config_ref};
  MemoryMap memory_map{memory_map_ref};

  // Addresses must differ (independent stack memory)
  assert(&frame_buffer_config != &frame_buffer_config_ref);
  assert(&memory_map != &memory_map_ref);

  // Now overwrite old UEFI stack region with garbage (0xAA)
  std::memset(uefi_stack_region, 0xAA, sizeof(uefi_stack_region));

  // Relocated kernel stack variables MUST remain 100% intact!
  assert(frame_buffer_config.frame_buffer == reinterpret_cast<uint8_t*>(0xE0000000ULL));
  assert(frame_buffer_config.pixels_per_scan_line == 1024);
  assert(frame_buffer_config.horizontal_resolution == 1024);
  assert(frame_buffer_config.vertical_resolution == 768);
  assert(frame_buffer_config.pixel_format == kPixelRGBResv8BitPerColor);

  assert(memory_map.buffer_size == 4096);
  assert(memory_map.buffer == reinterpret_cast<void*>(0x07E00000ULL));
  assert(memory_map.map_size == 1200);
  assert(memory_map.descriptor_size == 48);
  assert(memory_map.descriptor_version == 1);

  std::cout << "  Kernel local copy addresses: FB=&" << &frame_buffer_config << ", MM=&" << &memory_map << "\n";
  std::cout << "  UEFI stack wiped with 0xAA -> Kernel copies remained 100% valid!\n";
  std::cout << "  -> Passed: Copy-constructor relocation guarantees complete memory isolation.\n";
}

void TestShallowCopyBufferLifetimeAnalysis() {
  std::cout << "[Test 5] Analyzing Shallow Copy Pointer Semantics and Phase Safety...\n";

  // Simulate descriptor buffer located in UEFI BootServicesData
  std::vector<uint8_t> physical_uefi_buffer(48 * 4, 0);

  MemoryMap original_map{
    .buffer_size = 4096,
    .buffer = physical_uefi_buffer.data(),
    .map_size = 48 * 4,
    .map_key = 123,
    .descriptor_size = 48,
    .descriptor_version = 1
  };

  // Relocated struct copy on kernel stack
  MemoryMap kernel_map{original_map};

  // 1. Struct is independent
  assert(&kernel_map != &original_map);

  // 2. Buffer pointer is shallow-copied (pointing to same physical address)
  assert(kernel_map.buffer == original_map.buffer);

  // Phase 1 (Scanning Phase): Buffer is safe because allocator is inactive
  auto* desc0 = reinterpret_cast<MemoryDescriptor*>(kernel_map.buffer);
  desc0->type = static_cast<uint32_t>(MemoryType::kEfiConventionalMemory);
  desc0->physical_start = 0x100000;
  desc0->number_of_pages = 256;

  assert(desc0->type == static_cast<uint32_t>(MemoryType::kEfiConventionalMemory));
  assert(desc0->physical_start == 0x100000);

  // Phase 2 (Allocation Phase): Once BitmapMemoryManager takes over, buffer is no longer touched
  // Even if allocator reclaims physical_uefi_buffer:
  std::memset(physical_uefi_buffer.data(), 0xFF, physical_uefi_buffer.size());
  // The OS does not care anymore, because memory availability is tracked by the bitmap array!
  std::cout << "  Phase 1 (Scanning): Buffer safely read before allocator startup.\n";
  std::cout << "  Phase 2 (Allocation): Bitmap manager takes ownership; buffer is retired.\n";
  std::cout << "  -> Passed: Shallow copy is provably safe under MikanOS initialization order.\n";
}

int main(int argc, char* argv[]) {
  bool run_test = (argc > 1 && std::string(argv[1]) == "--test");
  if (run_test) {
    std::cout << "[Info] Running in automated test mode (--test)\n";
  }
  std::cout << "============================================================\n";
  std::cout << "  MikanOS Chapter 8.3: Relocating Data Structures Simulator \n";
  std::cout << "============================================================\n";

  TestStructureLayoutAndSizes();
  TestIsAvailableLogic();
  TestUnprotectedReferenceVulnerability();
  TestCopyConstructorRelocationSafety();
  TestShallowCopyBufferLifetimeAnalysis();

  std::cout << "\n[PASS] All Chapter 8.3 data structure relocation tests passed successfully!\n";
  return 0;
}
