cmake_minimum_required(VERSION 3.20)

project(AresEdgeSystem 
    VERSION 2.0.0
    DESCRIPTION "Autonomous Reconnaissance and Electronic Supremacy Edge System"
    LANGUAGES CXX CUDA
)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Configuration
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Build options
option(ARES_ENABLE_CUDA "Enable CUDA acceleration" ON)
option(ARES_ENABLE_TESTING "Build tests" ON)
option(ARES_ENABLE_DOCUMENTATION "Build documentation" OFF)
option(BUILD_UNREAL_PLUGIN "Build Unreal Engine plugin" OFF)
option(ARES_ENABLE_SECURITY_HARDENING "Enable security hardening" ON)

# Find required packages
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

if(ARES_ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
    if(ARES_ENABLE_SECURITY_HARDENING)
        add_compile_options(-fstack-protector-strong -D_FORTIFY_SOURCE=2)
        add_link_options(-Wl,-z,relro,-z,now)
    endif()
endif()

# CUDA flags
if(ARES_ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")  # Support V100, A100, RTX 30xx, H100
endif()

# Global definitions
add_definitions(-DARES_VERSION_MAJOR=2)
add_definitions(-DARES_VERSION_MINOR=0)
add_definitions(-DARES_VERSION_PATCH=0)

if(ARES_ENABLE_CUDA)
    add_definitions(-DARES_ENABLE_CUDA)
endif()

if(ARES_ENABLE_SECURITY_HARDENING)
    add_definitions(-DARES_SECURITY_HARDENED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core/include)

# Add subdirectories
add_subdirectory(src)

# Main executable
add_executable(ares_edge_system
    src/main.cpp
)

target_link_libraries(ares_edge_system
    ares_unified
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

if(ARES_ENABLE_CUDA)
    target_link_libraries(ares_edge_system
        CUDA::cublas
        CUDA::curand
        CUDA::cufft
        CUDA::cudnn
    )
endif()

# Testing
if(ARES_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS ares_edge_system
    RUNTIME DESTINATION bin
)

install(TARGETS ares_unified
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AresEdgeSystemConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/AresEdgeSystemConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AresEdgeSystemConfig.cmake"
    INSTALL_DESTINATION lib/cmake/AresEdgeSystem
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/AresEdgeSystemConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/AresEdgeSystemConfigVersion.cmake"
    DESTINATION lib/cmake/AresEdgeSystem
)