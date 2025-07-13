# Simple Makefile for CEW module testing
# Use: make cpu-only or make with-cuda

CXX = g++
NVCC = nvcc

# Common flags
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -fPIC
INCLUDES = -I. -Iinclude

# Libraries
LIBS = -lpthread -lm

# CPU sources
CPU_SOURCES = \
    cew_unified_interface.cpp \
    cpu/cew_cpu_module.cpp

# CUDA sources
CUDA_SOURCES = \
    cuda/cew_cuda_module.cpp \
    cuda/adaptive_jamming_kernel.cu \
    cuda/spectrum_waterfall_kernel.cu \
    cuda/threat_classifier_kernel.cu

# Test program
TEST_SOURCES = tests/test_cew_unified.cpp

# Check for CUDA
CUDA_PATH ?= /usr/local/cuda
CUDA_EXISTS := $(shell test -d $(CUDA_PATH) && echo yes)

.PHONY: all clean cpu-only with-cuda test

# Default target
all: $(if $(CUDA_EXISTS),with-cuda,cpu-only)

# CPU-only build
cpu-only: libares_cew_cpu.so test_cew_cpu

libares_cew_cpu.so: $(CPU_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -o $@ $^ $(LIBS)

test_cew_cpu: $(TEST_SOURCES) libares_cew_cpu.so
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(TEST_SOURCES) -L. -lares_cew_cpu $(LIBS) -Wl,-rpath,.

# CUDA build
with-cuda: libares_cew_cuda.so test_cew_cuda

CUDA_FLAGS = -std=c++17 -O3 --use_fast_math -Xcompiler=-fPIC
CUDA_INCLUDES = $(INCLUDES) -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcufft -lcurand

# Compile CUDA kernels
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c -o $@ $<

# Compile C++ files for CUDA build
%.cuda.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -DCEW_CUDA_AVAILABLE -c -o $@ $<

CUDA_OBJS = $(CUDA_SOURCES:.cu=.o) $(CPU_SOURCES:.cpp=.cuda.o) cuda/cew_cuda_module.cuda.o

libares_cew_cuda.so: $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LIBS) $(CUDA_LIBS)

test_cew_cuda: $(TEST_SOURCES) libares_cew_cuda.so
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -DCEW_CUDA_AVAILABLE -o $@ $(TEST_SOURCES) \
		-L. -lares_cew_cuda $(LIBS) $(CUDA_LIBS) -Wl,-rpath,.

# Run tests
test: test_cew_$(if $(CUDA_EXISTS),cuda,cpu)
	./test_cew_$(if $(CUDA_EXISTS),cuda,cpu)

# Clean
clean:
	rm -f *.o *.so test_cew_cpu test_cew_cuda
	rm -f cuda/*.o cpu/*.o

# Info
info:
	@echo "CUDA available: $(if $(CUDA_EXISTS),yes,no)"
	@echo "CUDA path: $(CUDA_PATH)"
	@echo "Build target: $(if $(CUDA_EXISTS),with-cuda,cpu-only)"