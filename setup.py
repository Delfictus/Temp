#!/usr/bin/env python3
"""
ARES Edge System Setup Script
Production-grade installation for DARPA/DoD deployment
"""

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

__version__ = "1.0.0"

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# C++ Extensions for performance-critical components
ext_modules = [
    Pybind11Extension(
        "ares_neuromorphic_core",
        [
            "ares_unified/src/neuromorphic/mlir/brian2_integration.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir(),
            "ares_unified/src/neuromorphic/include",
            "ares_unified/src/core/include",
        ],
        extra_compile_args=[
            "-O3",
            "-march=native", 
            "-fopenmp",
            "-mavx2",
            "-mfma",
            "-std=c++17",
            "-DPRODUCTION_BUILD"
        ],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
]

setup(
    name="ares-edge-system",
    version=__version__,
    author="DELFICTUS I/O LLC",
    author_email="contact@delfictus.io",
    description="Tactical-grade autonomous threat mitigation engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Delfictus/AE",
    project_urls={
        "Bug Tracker": "https://github.com/Delfictus/AE/issues",
        "Documentation": "https://github.com/Delfictus/AE/tree/main/docs",
        "Source Code": "https://github.com/Delfictus/AE",
    },
    packages=find_packages(where="ares_unified"),
    package_dir={"": "ares_unified"},
    package_data={
        "ares": [
            "config/*.yaml",
            "config/*.json",
            "docs/*.md",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "cuda": [
            "cupy-cuda11x>=12.0.0",
            "numba>=0.57.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    entry_points={
        "console_scripts": [
            "ares-system=ares.cli:main",
            "ares-config=ares.config:validate_config",
        ],
    },
    keywords=[
        "artificial-intelligence",
        "neuromorphic",
        "edge-computing", 
        "threat-detection",
        "autonomous-systems",
        "quantum-resilient",
        "defense-technology",
    ],
    zip_safe=False,  # Required for C++ extensions
)