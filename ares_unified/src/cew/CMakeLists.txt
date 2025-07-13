# CEW Module CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

# CEW library sources
set(CEW_SOURCES
    cew_unified_interface.cpp
    cpu/cew_cpu_module.cpp
)

set(CEW_HEADERS
    include/cew_unified_interface.h
    include/cew_adaptive_jamming.h
    include/spectrum_waterfall.h
    cpu/cew_cpu_module.h
)

# Create CEW library (CPU version always available)
add_library(ares_cew SHARED ${CEW_SOURCES})

target_include_directories(ares_cew PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# CPU optimizations
target_compile_options(ares_cew PRIVATE
    -march=native
    -O3
    -ffast-math
    -fopenmp
)

target_link_libraries(ares_cew PRIVATE
    pthread
    m
)

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # CUDA sources
    set(CEW_CUDA_SOURCES
        cuda/cew_cuda_module.cpp
        cuda/adaptive_jamming_kernel.cu
        cuda/spectrum_waterfall_kernel.cu
        cuda/threat_classifier_kernel.cu
    )
    
    set(CEW_CUDA_HEADERS
        cuda/cew_cuda_module.h
    )
    
    # Add CUDA sources to library
    target_sources(ares_cew PRIVATE ${CEW_CUDA_SOURCES})
    
    # CUDA properties
    set_target_properties(ares_cew PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
    )
    
    # CUDA compile options
    target_compile_options(ares_cew PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --extended-lambda
            -Xcompiler=-fPIC
            -O3
        >
    )
    
    # Link CUDA libraries
    target_link_libraries(ares_cew PRIVATE
        cudart
        cufft
        curand
    )
    
    # Define CUDA available
    target_compile_definitions(ares_cew PUBLIC CEW_CUDA_AVAILABLE)
    
    message(STATUS "CEW: CUDA support enabled")
else()
    message(STATUS "CEW: CUDA not available, using CPU-only implementation")
endif()

# Installation
install(TARGETS ares_cew
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CEW_HEADERS}
    DESTINATION include/ares/cew
)

# Tests (disabled for now)
# if(BUILD_TESTS)
#     add_subdirectory(tests)
# endif()