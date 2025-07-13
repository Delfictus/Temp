# ARES Edge System - MLIR Neuromorphic Build Configuration
# Copyright (c) 2024 DELFICTUS I/O LLC

cmake_minimum_required(VERSION 3.20)
project(ares_neuromorphic_mlir LANGUAGES CXX)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# MLIR/LLVM configuration
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen for dialect
set(LLVM_TARGET_DEFINITIONS neuromorphic_dialect.td)
mlir_tablegen(neuromorphic_dialect.h.inc -gen-dialect-decls)
mlir_tablegen(neuromorphic_dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(neuromorphic_ops.h.inc -gen-op-decls)
mlir_tablegen(neuromorphic_ops.cpp.inc -gen-op-defs)
mlir_tablegen(neuromorphic_types.h.inc -gen-typedef-decls)
mlir_tablegen(neuromorphic_types.cpp.inc -gen-typedef-defs)
mlir_tablegen(neuromorphic_attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(neuromorphic_attrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRNeuromorphicOpsIncGen)

# Neuromorphic dialect library
add_library(MLIRNeuromorphic
    neuromorphic_dialect.cpp
    neuromorphic_lowering.cpp
)

target_link_libraries(MLIRNeuromorphic
    PUBLIC
        MLIRIR
        MLIRArithDialect
        MLIRFuncDialect  
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRTensorDialect
        MLIRVectorDialect
        MLIRTransforms
        MLIRPass
)

add_dependencies(MLIRNeuromorphic MLIRNeuromorphicOpsIncGen)

# MLIR driver executable
add_executable(ares-mlir-neuromorphic
    mlir_neuromorphic_driver.cpp
)

target_link_libraries(ares-mlir-neuromorphic
    PRIVATE
        MLIRNeuromorphic
        MLIRExecutionEngine
        MLIRParser
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIROptLib
        ${LLVM_LIBS}
)

# Link with existing ARES libraries
target_link_libraries(ares-mlir-neuromorphic
    PRIVATE
        ares_neuromorphic_core
        ares_tpu_accelerator
)

# Optional CUDA support
find_package(CUDA)
if(CUDA_FOUND)
    target_compile_definitions(ares-mlir-neuromorphic PRIVATE USE_CUDA)
    target_link_libraries(ares-mlir-neuromorphic PRIVATE ${CUDA_LIBRARIES})
endif()

# Installation
install(TARGETS MLIRNeuromorphic ares-mlir-neuromorphic
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/neuromorphic_dialect.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/neuromorphic_ops.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/neuromorphic_types.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/neuromorphic_attrs.h.inc
    neuromorphic_dialect.h
    DESTINATION include/ares/mlir)

# Example files
install(FILES
    threat_detection_example.mlir
    DESTINATION share/ares/examples/mlir)

# Testing
enable_testing()
add_test(NAME neuromorphic_mlir_test
         COMMAND ares-mlir-neuromorphic ${CMAKE_CURRENT_SOURCE_DIR}/threat_detection_example.mlir --benchmark)
