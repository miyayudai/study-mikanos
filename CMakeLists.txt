cmake_minimum_required(VERSION 3.20)
project(study_mikanos LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Host-side verification & OS data-structure simulation targets are added per chapter
enable_testing()

add_executable(ch00_os_scale_analyzer src/ch00/os_scale_analyzer.cpp)
add_test(NAME test_ch00_os_scale_analyzer COMMAND ch00_os_scale_analyzer)

add_executable(ch00_os_posix_architecture src/ch00/os_posix_architecture.cpp)
add_test(NAME test_ch00_os_posix_architecture COMMAND ch00_os_posix_architecture)

add_executable(ch00_os_build_steps src/ch00/os_build_steps.cpp)
add_test(NAME test_ch00_os_build_steps COMMAND ch00_os_build_steps)

add_executable(ch00_os_debug_strategies src/ch00/os_debug_strategies.cpp)
add_test(NAME test_ch00_os_debug_strategies COMMAND ch00_os_debug_strategies)

add_executable(ch00_os_architecture_roadmap src/ch00/os_architecture_roadmap.cpp)
add_test(NAME test_ch00_os_architecture_roadmap COMMAND ch00_os_architecture_roadmap)

add_executable(ch00_os_chapter0_exercises src/ch00/os_chapter0_exercises.cpp)
add_test(NAME test_ch00_os_chapter0_exercises COMMAND ch00_os_chapter0_exercises)

add_executable(ch01_pe32plus_parser src/ch01/pe32plus_parser.cpp)
add_test(NAME test_ch01_pe32plus_parser COMMAND ch01_pe32plus_parser ../vendor/mikanos-build/day01/bin/hello.efi)

add_executable(ch01_block_device_inspector src/ch01/block_device_inspector.cpp)
add_test(NAME test_ch01_block_device_inspector COMMAND ch01_block_device_inspector --demo)

add_executable(ch01_wsl_environment_inspector src/ch01/wsl_environment_inspector.cpp)
add_test(NAME test_ch01_wsl_environment_inspector COMMAND ch01_wsl_environment_inspector --demo)

add_executable(ch01_qemu_launch_builder src/ch01/qemu_launch_builder.cpp)
add_test(NAME test_ch01_qemu_launch_builder COMMAND ch01_qemu_launch_builder)

add_executable(ch01_boot_sequence_tracer src/ch01/boot_sequence_tracer.cpp)
add_test(NAME test_ch01_boot_sequence_tracer COMMAND ch01_boot_sequence_tracer --test)

add_executable(ch01_binary_mutation_tester src/ch01/binary_mutation_tester.cpp)
add_test(NAME test_ch01_binary_mutation_tester COMMAND ch01_binary_mutation_tester)

add_executable(ch01_bios_uefi_comparator src/ch01/bios_uefi_comparator.cpp)
add_test(NAME test_ch01_bios_uefi_comparator COMMAND ch01_bios_uefi_comparator)

add_executable(ch01_toolchain_inspector src/ch01/toolchain_inspector.cpp)
add_test(NAME test_ch01_toolchain_inspector COMMAND ch01_toolchain_inspector)

add_executable(ch01_uefi_c_hello_analyzer src/ch01/uefi_c_hello_analyzer.cpp)
add_test(NAME test_ch01_uefi_c_hello_analyzer COMMAND ch01_uefi_c_hello_analyzer)

add_executable(ch01_os_chapter1_exercises src/ch01/os_chapter1_exercises.cpp)
add_test(NAME test_ch01_os_chapter1_exercises COMMAND ch01_os_chapter1_exercises)

# Chapter 2
add_executable(ch02_edk2_package_analyzer src/ch02/edk2_package_analyzer.cpp)
add_test(NAME test_ch02_edk2_package_analyzer COMMAND ch02_edk2_package_analyzer)

add_executable(ch02_uefi_hello_world_sim src/ch02/uefi_hello_world_sim.cpp)
add_test(NAME test_ch02_uefi_hello_world_sim COMMAND ch02_uefi_hello_world_sim --test)

add_executable(ch02_main_memory_architecture_sim src/ch02/main_memory_architecture_sim.cpp)
add_test(NAME test_ch02_main_memory_architecture_sim COMMAND ch02_main_memory_architecture_sim --test)

add_executable(ch02_uefi_memory_map_concept_sim src/ch02/uefi_memory_map_concept_sim.cpp)
add_test(NAME test_ch02_uefi_memory_map_concept_sim COMMAND ch02_uefi_memory_map_concept_sim --test)

add_executable(ch02_get_memory_map_sim src/ch02/get_memory_map_sim.cpp)
add_test(NAME test_ch02_get_memory_map_sim COMMAND ch02_get_memory_map_sim --test)

add_executable(ch02_save_memory_map_sim src/ch02/save_memory_map_sim.cpp)
add_test(NAME test_ch02_save_memory_map_sim COMMAND ch02_save_memory_map_sim)

add_executable(ch02_analyze_memory_map_sim src/ch02/analyze_memory_map_sim.cpp)
add_test(NAME test_ch02_analyze_memory_map_sim COMMAND ch02_analyze_memory_map_sim)

add_executable(ch02_pointer_basics_sim src/ch02/pointer_basics_sim.cpp)
add_test(NAME test_ch02_pointer_basics_sim COMMAND ch02_pointer_basics_sim --test)

add_executable(ch02_arrow_operator_sim src/ch02/arrow_operator_sim.cpp)
add_test(NAME test_ch02_arrow_operator_sim COMMAND ch02_arrow_operator_sim --test)

add_executable(ch02_os_chapter2_exercises src/ch02/os_chapter2_exercises.cpp)
add_test(NAME test_ch02_os_chapter2_exercises COMMAND ch02_os_chapter2_exercises --test)
















