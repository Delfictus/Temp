#!/bin/bash
# ARES Edge System - MLIR Build Script
# Copyright (c) 2024 DELFICTUS I/O LLC

set -e

echo "Building ARES MLIR Neuromorphic Components..."

# Check for MLIR installation
if [ -z "$MLIR_DIR" ]; then
    echo "Error: MLIR_DIR environment variable not set"
    echo "Please set MLIR_DIR to your MLIR installation directory"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR="$MLIR_DIR" \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang

# Build
echo "Building..."
make -j$(nproc)

# Run tests
echo "Running tests..."
ctest --verbose

echo "Build complete!"
echo "Executable: build/ares-mlir-neuromorphic"
echo "Library: build/libMLIRNeuromorphic.a"

# Example usage
echo ""
echo "Example usage:"
echo "  ./ares-mlir-neuromorphic ../threat_detection_example.mlir --target=cpu"
echo "  ./ares-mlir-neuromorphic ../threat_detection_example.mlir --benchmark"
echo "  ./ares-mlir-neuromorphic ../threat_detection_example.mlir --dump-lowered"
