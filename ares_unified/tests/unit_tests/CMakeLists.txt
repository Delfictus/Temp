# Unit Tests for CEW Module
cmake_minimum_required(VERSION 3.18)

# Create a simple test framework since we can't download GTest
add_executable(test_cew_unit
    test_cew_basic.cpp
    test_cew_performance.cpp
    test_cew_security.cpp
)

target_include_directories(test_cew_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cew/include
    ${CMAKE_SOURCE_DIR}/src/cew/cpu
)

target_link_libraries(test_cew_unit 
    ares_cew
    pthread
)

# Add test
add_test(NAME cew_unit_tests COMMAND test_cew_unit)