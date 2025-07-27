# Algorithms library
add_library(ares_algorithms STATIC
    ares_transfer_entropy.cpp
    helios_he.cpp
    athena_adp.cpp
    ares_obfuscation_protocol.cpp
)

target_include_directories(ares_algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(ares_algorithms
    PUBLIC ares_hardware
    PUBLIC ares_security
    PUBLIC OpenSSL::SSL
    PUBLIC OpenSSL::Crypto
)

if(ARES_ENABLE_CUDA)
    target_compile_definitions(ares_algorithms PUBLIC ARES_ENABLE_CUDA)
endif()