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

