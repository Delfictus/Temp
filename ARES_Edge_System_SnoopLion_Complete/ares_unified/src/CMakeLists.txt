cmake_minimum_required(VERSION 3.18)

# Add all subdirectories
add_subdirectory(core)
add_subdirectory(perception)
add_subdirectory(navigation)
add_subdirectory(communication)
add_subdirectory(swarm)
add_subdirectory(digital_twin)
add_subdirectory(optical_stealth)
add_subdirectory(identity)
add_subdirectory(federated_learning)
add_subdirectory(countermeasures)
add_subdirectory(orchestrator)
add_subdirectory(cyber_em)
add_subdirectory(backscatter)
add_subdirectory(neuromorphic)

# Unreal Engine plugin is built separately
if(BUILD_UNREAL_PLUGIN)
    add_subdirectory(unreal)
endif()

# Create unified ARES library
add_library(ares_unified INTERFACE)
target_link_libraries(ares_unified INTERFACE
    ares_core
    ares_perception
    ares_navigation
    ares_communication
    ares_swarm
    ares_digital_twin
    ares_optical_stealth
    ares_identity
    ares_federated_learning
    ares_countermeasures
    ares_orchestrator
    ares_cyber_em
    ares_backscatter
    ares_neuromorphic
)

# Export targets
install(TARGETS ares_unified
    EXPORT ARESTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)