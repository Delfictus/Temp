cmake_minimum_required(VERSION 3.18)

# Add available subdirectories
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt)
    add_subdirectory(core)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cew/CMakeLists.txt)
    add_subdirectory(cew)
endif()

# Temporarily disable swarm module due to CUDA dependencies
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/swarm/CMakeLists.txt)
#     add_subdirectory(swarm)
# endif()

# Temporarily disable neuromorphic module due to CUDA dependencies
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuromorphic/CMakeLists.txt)
#     add_subdirectory(neuromorphic)
# endif()

# Add other modules as they become available
foreach(module digital_twin optical_stealth identity federated_learning countermeasures orchestrator cyber_em backscatter)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${module}/CMakeLists.txt)
        add_subdirectory(${module})
    endif()
endforeach()

# Unreal Engine plugin is built separately
if(BUILD_UNREAL_PLUGIN)
    add_subdirectory(unreal)
endif()

# Create unified ARES library
set(AVAILABLE_LIBRARIES)

if(TARGET ares_cew)
    list(APPEND AVAILABLE_LIBRARIES ares_cew)
endif()

if(TARGET ares_swarm)
    list(APPEND AVAILABLE_LIBRARIES ares_swarm)
endif()

if(TARGET ares_neuromorphic)
    list(APPEND AVAILABLE_LIBRARIES ares_neuromorphic)
endif()

# Only create unified library if we have components
if(AVAILABLE_LIBRARIES)
    add_library(ares_unified INTERFACE)
    target_link_libraries(ares_unified INTERFACE ${AVAILABLE_LIBRARIES})
endif()

# Installation
if(TARGET ares_unified)
    install(TARGETS ares_unified
        EXPORT ARESTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
endif()