#!/bin/bash
# Build Brian2-MLIR integration module

set -e

echo "Building Brian2-MLIR integration..."

# Install dependencies if needed
if ! python3 -c "import brian2" 2>/dev/null; then
    echo "Installing Brian2..."
    pip3 install brian2 --user
fi

if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "Installing pybind11..."
    pip3 install pybind11 --user
fi

# Build the C++ extension
python3 setup_brian2.py build_ext --inplace

# Run tests
echo "Running Brian2 benchmarks..."
python3 brian2_benchmark.py

echo "Build complete!"
