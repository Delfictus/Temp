# CEW (Cognitive Electronic Warfare) Module Architecture

## Overview

The unified CEW module provides a high-performance, real-time cognitive electronic warfare system with automatic CPU/GPU backend selection. The module achieves <100ms threat detection to jamming response using Q-learning with eligibility traces.

## Architecture

### 1. Unified Interface (`cew_unified_interface.h`)

- **ICEWModule**: Abstract interface for CEW implementations
- **CEWModuleFactory**: Factory pattern for creating appropriate backend
- **CEWManager**: Thread-safe wrapper for concurrent access

### 2. Backend Implementations

#### CUDA Backend (`cuda/`)
- High-performance GPU implementation
- Utilizes CUDA kernels for parallel processing
- Features:
  - Concurrent streams for compute/transfer overlap
  - Optimized memory access patterns
  - cuFFT for spectrum analysis
  - Warp-level primitives for efficiency

#### CPU Backend (`cpu/`)
- Optimized CPU fallback implementation
- Features:
  - SIMD vectorization (AVX2)
  - Thread pool for parallel processing
  - Cache-optimized data structures
  - NUMA-aware memory allocation

### 3. Key Components

#### Adaptive Jamming Module
- Q-learning with eligibility traces
- 16 jamming strategies
- 256 quantized threat states
- Real-time adaptation

#### Spectrum Waterfall
- Sliding window FFT analysis
- Configurable window functions
- Signal detection and classification
- Continuous spectrum monitoring

#### Threat Classifier
- CNN-based threat identification
- Protocol-aware classification
- Priority-based response

## Performance Characteristics

### Latency
- Target: <100ms threat-to-jamming response
- GPU: Typically 10-50ms
- CPU: Typically 50-100ms

### Throughput
- GPU: Process up to 128 simultaneous threats
- CPU: Process up to 64 simultaneous threats
- Spectrum analysis: 2 GSPS sample rate

### Memory Usage
- Base: ~100MB (Q-table, waveform bank)
- Per-threat: ~1KB
- Spectrum buffer: Configurable (default 64MB)

## Usage Example

```cpp
// Create CEW manager with automatic backend selection
CEWManager cew_manager(CEWBackend::AUTO);

// Initialize
cew_manager.initialize(0);  // Device ID 0

// Process spectrum
cew_manager.process_spectrum_threadsafe(
    spectrum_data,
    detected_threats,
    num_threats,
    jamming_params,
    timestamp_ns
);

// Update learning
cew_manager.update_qlearning_threadsafe(reward);

// Get metrics
CEWMetrics metrics = cew_manager.get_metrics();
```

## Build Configuration

### With CUDA
```cmake
cmake -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" ..
```

### CPU-only
```cmake
cmake -DCMAKE_CXX_FLAGS="-march=native -O3" ..
```

## Thread Safety

The CEWManager class provides thread-safe access to all operations:
- Concurrent spectrum processing
- Safe Q-learning updates
- Atomic metric updates
- Lock-free read operations where possible

## Future Enhancements

1. **Multi-GPU Support**: Distribute threats across multiple GPUs
2. **Deep Q-Networks**: Replace tabular Q-learning with DQN
3. **Distributed Learning**: Share Q-tables across multiple systems
4. **Hardware Acceleration**: Support for FPGA/ASIC backends