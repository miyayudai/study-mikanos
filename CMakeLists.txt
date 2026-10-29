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



