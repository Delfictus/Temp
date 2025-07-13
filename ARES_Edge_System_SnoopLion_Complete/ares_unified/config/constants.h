/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Unified Constants
 * 
 * This file consolidates all system-wide constants used across different modules
 * to avoid duplication and ensure consistency.
 */

#ifndef ARES_UNIFIED_CONSTANTS_H
#define ARES_UNIFIED_CONSTANTS_H

#include <cstdint>
#include <cmath>

namespace ares {
namespace constants {

// =============================================================================
// System-wide Constants
// =============================================================================

// Mathematical Constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr double PI = M_PI;
constexpr double TWO_PI = 2.0 * M_PI;
constexpr double PI_HALF = M_PI / 2.0;
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;

// Physical Constants
constexpr double SPEED_OF_LIGHT = 299792458.0;  // m/s
constexpr double BOLTZMANN_CONSTANT = 1.380649e-23;  // J/K
constexpr double PLANCK_CONSTANT = 6.62607015e-34;  // J⋅s

// =============================================================================
// GPU/CUDA Configuration
// =============================================================================

// CUDA Block and Grid Sizes
constexpr uint32_t DEFAULT_BLOCK_SIZE = 256;
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t WARP_SIZE = 32;

// Memory Alignment
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t GPU_MEMORY_ALIGNMENT = 256;

// =============================================================================
// System Limits
// =============================================================================

// Maximum Entities
constexpr uint32_t MAX_SWARM_SIZE = 256;
constexpr uint32_t MAX_SWARM_TARGETS = 256;
constexpr uint32_t MAX_EDGE_NODES = 1024;
constexpr uint32_t MAX_IDENTITIES = 256;
constexpr uint32_t MAX_HARDWARE_COMPONENTS = 64;
constexpr uint32_t MAX_BACKSCATTER_NODES = 1024;

// CEW/RF Configuration
constexpr uint32_t MAX_THREATS = 128;
constexpr uint32_t RF_SPOOFING_CHANNELS = 64;
constexpr uint32_t SIDE_CHANNEL_BANDS = 32;
constexpr uint32_t SPECTRUM_BINS = 1024;
constexpr uint32_t MAX_HISTORY_DEPTH = 256;

// Neural/Neuromorphic
constexpr uint32_t MAX_NEURONS = 1000000;      // 1M neurons
constexpr uint32_t MAX_SYNAPSES = 100000000;   // 100M synapses
constexpr uint32_t MAX_NEUROMORPHIC_CORES = 128;

// Cryptography
constexpr uint32_t SIGNATURE_DIMENSIONS = 128;
constexpr uint32_t MAX_KEY_SIZE = 4096;
constexpr uint32_t MAX_ATTESTATION_CHAIN = 8;
constexpr uint32_t POLY_MODULUS_DEGREE = 16384;

// Protocol Configuration
constexpr uint32_t MAX_MESSAGE_SIZE = 4096;
constexpr uint32_t PROTOCOL_TEMPLATE_SIZE = 1024;
constexpr uint32_t MAX_PROTOCOL_TYPES = 256;

// =============================================================================
// Timing and Performance
// =============================================================================

// Latency Requirements
constexpr uint32_t MAX_LATENCY_US = 100000;      // 100ms hard deadline
constexpr uint32_t TARGET_LATENCY_US = 10000;    // 10ms target
constexpr float MAX_TRANSITION_TIME_MS = 100.0f;

// Update Rates
constexpr float DEFAULT_UPDATE_RATE_HZ = 1000.0f;
constexpr float MIN_UPDATE_RATE_HZ = 10.0f;
constexpr float MAX_UPDATE_RATE_HZ = 10000.0f;

// =============================================================================
// RF/EM Spectrum Configuration
// =============================================================================

// Frequency Ranges
constexpr float FREQ_MIN_GHZ = 0.001f;    // 1 MHz
constexpr float FREQ_MAX_GHZ = 40.0f;     // 40 GHz
constexpr float FREQ_CENTER_GHZ = 2.4f;   // Default center frequency

// Common RF Frequencies
constexpr float SPOOF_FREQ_IR_THERMAL_HZ = 3e13f;    // 30 THz - thermal IR
constexpr float SPOOF_FREQ_UV_SIGNATURE_HZ = 1e15f;  // 1 PHz - UV signature
constexpr float SPOOF_FREQ_RADAR_XBAND_HZ = 10e9f;   // 10 GHz - X-band radar
constexpr float SPOOF_FREQ_LIDAR_HZ = 2e14f;         // 200 THz - LIDAR

// Power Levels
constexpr float MAX_TX_POWER_DBM = 30.0f;    // 1W
constexpr float MIN_RX_SENSITIVITY_DBM = -120.0f;
constexpr float NOISE_FLOOR_DBM = -174.0f;   // Thermal noise at room temp

// =============================================================================
// Chaos and Countermeasures
// =============================================================================

constexpr uint32_t CHAOS_PATTERN_VARIANTS = 32;
constexpr float MIN_CONFUSION_DISTANCE_M = 10.0f;
constexpr float MAX_CONFUSION_DISTANCE_M = 1000.0f;
constexpr float FRIENDLY_FIRE_THRESHOLD = 0.7f;

// =============================================================================
// Digital Twin and Simulation
// =============================================================================

constexpr uint32_t MAX_KEYFRAMES = 10000;
constexpr uint32_t MAX_LANDMARKS = 100000;
constexpr uint32_t MAX_LOOP_CLOSURES = 1000;
constexpr float SIMULATION_TIME_STEP = 0.001f;  // 1ms

// =============================================================================
// Security and Cryptography
// =============================================================================

constexpr uint32_t MAX_PARTIES = 128;
constexpr uint32_t MAX_CIRCUIT_DEPTH = 100;
constexpr uint32_t MAX_MULTIPLICATIVE_DEPTH = 20;
constexpr uint32_t MAX_CONCURRENT_TRANSITIONS = 8;

// =============================================================================
// Energy and Power
// =============================================================================

constexpr float MAX_HARVESTED_POWER_W = 0.1f;      // 100mW max
constexpr float MAX_CONVERSION_EFFICIENCY = 0.85f;  // 85% theoretical max
constexpr float MIN_OPERATING_VOLTAGE_V = 1.8f;
constexpr float MAX_OPERATING_VOLTAGE_V = 5.0f;

// =============================================================================
// Network and Communication
// =============================================================================

constexpr uint32_t MAX_PACKET_SIZE = 65536;
constexpr uint32_t DEFAULT_MTU = 1500;
constexpr uint32_t MAX_RETRIES = 3;
constexpr uint32_t CONNECTION_TIMEOUT_MS = 5000;

// =============================================================================
// Error Thresholds
// =============================================================================

constexpr float EPSILON = 1e-6f;
constexpr float MAX_POSITION_ERROR_M = 0.1f;
constexpr float MAX_TIMING_ERROR_US = 1.0f;
constexpr float MAX_FREQUENCY_ERROR_HZ = 1000.0f;

} // namespace constants
} // namespace ares

#endif // ARES_UNIFIED_CONSTANTS_H