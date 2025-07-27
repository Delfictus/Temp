# Security library
add_library(ares_security STATIC
    post_quantum_crypto.cpp
)

target_include_directories(ares_security
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ares_security
    PUBLIC OpenSSL::SSL
    PUBLIC OpenSSL::Crypto
)