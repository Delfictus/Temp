cmake_minimum_required(VERSION 3.18)

# ARES Edge System - Production Build Configuration
project(ARES VERSION 2.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(ARES_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_STATIC_ANALYSIS "Enable static analysis tools" ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for different configurations
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# Enable CUDA if requested and available
if(ARES_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA requested but not available")
        set(ARES_ENABLE_CUDA OFF)
    endif()
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenMP)

# Try to find Eigen3, but make it optional
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3")
    set(ARES_HAS_EIGEN3 ON)
else()
    message(STATUS "Eigen3 not found - some features will be disabled")
    set(ARES_HAS_EIGEN3 OFF)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/config
)

# Add source directory
add_subdirectory(src)

# Add tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Add examples if enabled
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add benchmarks if enabled
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Static analysis configuration
if(ENABLE_STATIC_ANALYSIS)
    find_program(CPPCHECK_EXECUTABLE cppcheck)
    if(CPPCHECK_EXECUTABLE)
        add_custom_target(cppcheck
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c++20
                --platform=unix64
                --suppress=missingIncludeSystem
                --quiet
                --xml
                --xml-version=2
                --output-file=${CMAKE_BINARY_DIR}/cppcheck_report.xml
                ${CMAKE_SOURCE_DIR}/src
            COMMENT "Running cppcheck static analysis"
        )
    endif()

    find_program(CLANG_TIDY_EXECUTABLE clang-tidy)
    if(CLANG_TIDY_EXECUTABLE)
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXECUTABLE})
    endif()
endif()

# Installation - disabled for now to avoid conflicts
# if(TARGET ares_unified)
#     install(TARGETS ares_unified
#         EXPORT ARESTargets
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
#         RUNTIME DESTINATION bin
#         INCLUDES DESTINATION include
#     )
# endif()

# Package configuration (disabled for now)
# if(TARGET ares_unified)
#     include(CMakePackageConfigHelpers)
#     write_basic_package_version_file(
#         "${CMAKE_CURRENT_BINARY_DIR}/ARESConfigVersion.cmake"
#         VERSION ${PROJECT_VERSION}
#         COMPATIBILITY AnyNewerVersion
#     )
# 
#     export(EXPORT ARESTargets
#         FILE "${CMAKE_CURRENT_BINARY_DIR}/ARESTargets.cmake"
#         NAMESPACE ARES::
#     )
# 
#     configure_file(cmake/ARESConfig.cmake
#         "${CMAKE_CURRENT_BINARY_DIR}/ARESConfig.cmake"
#         COPYONLY
#     )
# 
#     install(EXPORT ARESTargets
#         FILE ARESTargets.cmake
#         NAMESPACE ARES::
#         DESTINATION lib/cmake/ARES
#     )
# 
#     install(FILES
#         "${CMAKE_CURRENT_BINARY_DIR}/ARESConfig.cmake"
#         "${CMAKE_CURRENT_BINARY_DIR}/ARESConfigVersion.cmake"
#         DESTINATION lib/cmake/ARES
#     )
# endif()

# Print configuration summary
message(STATUS "")
message(STATUS "ARES Edge System Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA support: ${ARES_ENABLE_CUDA}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Static analysis: ${ENABLE_STATIC_ANALYSIS}")
message(STATUS "")