#!/usr/bin/env python3
"""
Setup script for Brian2-MLIR integration
Builds the C++ extension module for Python
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import glob

__version__ = "1.0.0"

ext_modules = [
    Pybind11Extension(
        "brian2_mlir_integration",
        ["brian2_integration.cpp"],
        include_dirs=[
            "..",  # For neuromorphic_core.h
            ".",   # For neuromorphic_dialect.h
        ],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-fopenmp",
            "-mavx2",
            "-mfma",
            "-std=c++17"
        ],
        extra_link_args=[
            "-fopenmp"
        ],
        language="c++"
    ),
]

setup(
    name="brian2_mlir_integration",
    version=__version__,
    author="DELFICTUS I/O LLC",
    description="Brian2-MLIR integration for ARES neuromorphic benchmarking",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
