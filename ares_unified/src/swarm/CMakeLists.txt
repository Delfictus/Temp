cmake_minimum_required(VERSION 3.18)

# Swarm module library
add_library(ares_swarm STATIC)

# Headers
target_sources(ares_swarm
    PUBLIC
        include/swarm_unified_interface.h
        include/byzantine_consensus_engine.h
        include/distributed_task_auction.h
)

# Sources
target_sources(ares_swarm
    PRIVATE
        src/byzantine_consensus_engine.cpp
        src/distributed_task_auction.cpp
)

# CUDA sources if available
if(CMAKE_CUDA_COMPILER)
    target_sources(ares_swarm
        PRIVATE
            kernels/auction_optimization_kernels.cu
    )
    set_target_properties(ares_swarm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Include directories
target_include_directories(ares_swarm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/ares/swarm>
)

# Dependencies
target_link_libraries(ares_swarm
    PUBLIC
        ares_core
        Eigen3::Eigen
        Threads::Threads
    PRIVATE
        ${CUDA_LIBRARIES}
)

# C++20 standard
target_compile_features(ares_swarm PUBLIC cxx_std_20)

# Installation
install(DIRECTORY include/
    DESTINATION include/ares/swarm
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)