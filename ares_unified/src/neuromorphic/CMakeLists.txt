cmake_minimum_required(VERSION 3.18)

# Neuromorphic module library
add_library(ares_neuromorphic STATIC)

# Core headers
target_sources(ares_neuromorphic
    PUBLIC
        include/neuromorphic_unified_interface.h
        include/neuromorphic_core.h
        include/loihi2_hardware_abstraction.h
        include/loihi2_spike_encoder.h
        include/loihi2_spike_encoding.h
        include/spike_encoder.h
        include/sdr_neuromorphic_bridge.h
        include/tpu_neuromorphic_accelerator.h
        include/unified_neuromorphic_sensors.h
)

# Core sources
target_sources(ares_neuromorphic
    PRIVATE
        src/loihi2_hardware_abstraction.cpp
        cpp/neuromorphic_engine.cpp
        cpp/custom_neuron_models.cpp
        cpp/synaptic_models.cpp
        cpp/integrated_system.cpp
        cpp/lidar_audio_integration.cpp
        cpp/benchmark_neuromorphic.cpp
        cpp/neuromorphic_python_wrapper.cpp
        cpp/tpu_integration_demo.cpp
)

# CUDA sources if available
if(CMAKE_CUDA_COMPILER)
    target_sources(ares_neuromorphic
        PRIVATE
            kernels/neuromorphic_simulation_kernels.cu
            kernels/spike_encoding_kernel.cu
            cpp/neuromorphic_cuda_bridge.cu
    )
    set_target_properties(ares_neuromorphic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# MLIR support (optional)
if(MLIR_FOUND)
    target_sources(ares_neuromorphic
        PRIVATE
            mlir/neuromorphic_dialect.cpp
            mlir/neuromorphic_lowering.cpp
            mlir/mlir_neuromorphic_driver.cpp
            mlir/brian2_integration.cpp
    )
    target_link_libraries(ares_neuromorphic PRIVATE ${MLIR_LIBRARIES})
endif()

# Include directories
target_include_directories(ares_neuromorphic
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:include/ares/neuromorphic>
)

# Dependencies
target_link_libraries(ares_neuromorphic
    PUBLIC
        Threads::Threads
    PRIVATE
        ${CUDA_LIBRARIES}
)

# Optional Eigen3 support
if(TARGET Eigen3::Eigen)
    target_link_libraries(ares_neuromorphic PUBLIC Eigen3::Eigen)
    target_compile_definitions(ares_neuromorphic PUBLIC ARES_HAS_EIGEN3)
endif()

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(ares_neuromorphic_py cpp/neuromorphic_python_wrapper.cpp)
    target_link_libraries(ares_neuromorphic_py PRIVATE ares_neuromorphic)
endif()

# C++20 standard
target_compile_features(ares_neuromorphic PUBLIC cxx_std_20)

# Installation
install(DIRECTORY include/
    DESTINATION include/ares/neuromorphic
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY cpp/
    DESTINATION include/ares/neuromorphic/cpp
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install Lava integration files
install(DIRECTORY lava/
    DESTINATION share/ares/neuromorphic/lava
    FILES_MATCHING PATTERN "*.py" PATTERN "*.yaml"
)

# Install test datasets
install(DIRECTORY datasets/
    DESTINATION share/ares/neuromorphic/datasets
)