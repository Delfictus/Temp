/**
 * @file cew_adaptive_jamming.h
 * @brief Cognitive Electronic Warfare Adaptive Jamming Module - Shared definitions
 * @author ARES Development Team
 * @date 2024
 * 
 * @details This header contains shared structures and constants used by both
 * CPU and CUDA implementations of the CEW module. It defines the core data
 * structures for threat detection, jamming strategies, and Q-learning based
 * adaptive response generation.
 * 
 * @section Performance
 * - Real-time constraint: < 100μs response time
 * - Spectrum coverage: 0.1 - 40 GHz
 * - Simultaneous threats: Up to 128
 * 
 * @section Security
 * All structures are designed to be memory-safe with explicit padding
 * to prevent information leakage through uninitialized memory.
 */

#ifndef ARES_CEW_ADAPTIVE_JAMMING_H
#define ARES_CEW_ADAPTIVE_JAMMING_H

#include <stdint.h>
#include <array>
#include <concepts>
#include <type_traits>

namespace ares::cew {

/**
 * @defgroup CEWConstants CEW Configuration Constants
 * @{
 */

/** @brief Number of frequency bins in spectrum analysis */
constexpr uint32_t SPECTRUM_BINS = 4096;

/** @brief Historical depth of spectrum waterfall */
constexpr uint32_t WATERFALL_HISTORY = 256;

/** @brief Minimum frequency coverage in GHz */
constexpr float FREQ_MIN_GHZ = 0.1f;

/** @brief Maximum frequency coverage in GHz */
constexpr float FREQ_MAX_GHZ = 40.0f;

/** @brief Maximum simultaneous threat tracks */
constexpr uint32_t MAX_THREATS = 128;

/** @brief Hard real-time deadline in microseconds */
constexpr uint32_t MAX_LATENCY_US = 100000;

/** @} */ // end of CEWConstants

/**
 * @defgroup QLearning Q-Learning Configuration
 * @{
 */

/** @brief Q-learning rate parameter (0.0-1.0) */
constexpr float ALPHA = 0.1f;

/** @brief Discount factor for future rewards */
constexpr float GAMMA = 0.95f;

/** @brief Exploration rate for action selection */
constexpr float EPSILON = 0.05f;

/** @brief Number of possible jamming actions */
constexpr uint32_t NUM_ACTIONS = 16;

/** @brief Number of quantized threat states */
constexpr uint32_t NUM_STATES = 256;

/** @} */ // end of QLearning

/**
 * @enum JammingStrategy
 * @brief Available jamming techniques with different effectiveness profiles
 * 
 * @details Each strategy has different power, bandwidth, and effectiveness
 * characteristics. Selection is performed by the Q-learning algorithm based
 * on observed effectiveness against specific threat types.
 */
enum class JammingStrategy : uint8_t {
    /** @brief Narrow-band barrage jamming (high power density) */
    BARRAGE_NARROW = 0,
    
    /** @brief Wide-band barrage jamming (lower power density) */
    BARRAGE_WIDE = 1,
    
    /** @brief Targeted single-frequency jamming */
    SPOT_JAMMING = 2,
    
    /** @brief Slow frequency sweep (< 100 MHz/s) */
    SWEEP_SLOW = 3,
    
    /** @brief Fast frequency sweep (> 1 GHz/s) */
    SWEEP_FAST = 4,
    
    /** @brief Pulsed jamming with configurable duty cycle */
    PULSE_JAMMING = 5,
    
    /** @brief Noise modulated with false data */
    NOISE_MODULATED = 6,
    
    /** @brief Repeater jamming with delay */
    DECEPTIVE_REPEAT = 7,
    
    /** @brief Protocol-specific jamming */
    PROTOCOL_AWARE = 8,
    
    /** @brief AI-driven adaptive jamming */
    COGNITIVE_ADAPTIVE = 9,
    
    /** @brief Synchronized frequency hopping */
    FREQUENCY_HOPPING = 10,
    
    /** @brief Time-division multiplexed jamming */
    TIME_SLICED = 11,
    
    /** @brief Dynamic power allocation */
    POWER_CYCLING = 12,
    
    /** @brief Multiple-input multiple-output spatial */
    MIMO_SPATIAL = 13,
    
    /** @brief Phase-coherent multi-emitter */
    PHASE_ALIGNED = 14,
    
    /** @brief Spatial null steering */
    NULL_STEERING = 15
};

/**
 * @struct ThreatSignature
 * @brief Detected threat characterization
 * 
 * @details Represents a detected RF threat with all parameters needed
 * for classification and response generation. Structure is aligned
 * for efficient GPU memory access.
 */
struct alignas(32) ThreatSignature {
    /** @brief Center frequency in GHz (0.1 - 40.0) */
    float center_freq_ghz;
    
    /** @brief Signal bandwidth in MHz */
    float bandwidth_mhz;
    
    /** @brief Received signal power in dBm */
    float power_dbm;
    
    /** @brief Detected modulation type (FSK, PSK, QAM, etc.) */
    uint8_t modulation_type;
    
    /** @brief Identified protocol (WiFi, LTE, custom, etc.) */
    uint8_t protocol_id;
    
    /** @brief Threat priority level (0-255, higher = more critical) */
    uint8_t priority;
    
    /** @brief Padding for alignment and security */
    uint8_t padding[2];
};

/**
 * @struct JammingParams
 * @brief Jamming signal generation parameters
 * 
 * @details Defines all parameters needed to generate a jamming signal.
 * Used by both software and hardware signal generators.
 */
struct alignas(32) JammingParams {
    /** @brief Jamming center frequency in GHz */
    float center_freq_ghz;
    
    /** @brief Jamming bandwidth in MHz */
    float bandwidth_mhz;
    
    /** @brief Output power in watts */
    float power_watts;
    
    /** @brief Selected jamming strategy */
    uint8_t strategy;
    
    /** @brief Waveform generator ID */
    uint8_t waveform_id;
    
    /** @brief Jamming duration in milliseconds */
    uint16_t duration_ms;
    
    /** @brief Initial phase offset in radians */
    float phase_offset;
    
    /** @brief Frequency sweep rate (MHz/second) */
    float sweep_rate_mhz_per_sec;
};

/**
 * @struct QTableState
 * @brief Q-Learning algorithm state
 * 
 * @details Maintains the complete state of the Q-learning algorithm
 * including Q-values, eligibility traces, and statistics. This structure
 * is designed for efficient GPU updates with aligned memory access.
 */
struct alignas(64) QTableState {
    /** @brief Q-value table [state][action] */
    float q_values[NUM_STATES][NUM_ACTIONS];
    
    /** @brief Eligibility traces for TD(λ) learning */
    float eligibility_traces[NUM_STATES][NUM_ACTIONS];
    
    /** @brief Visit count for each state */
    uint32_t visit_count[NUM_STATES];
    
    /** @brief Current state index */
    uint32_t current_state;
    
    /** @brief Last action taken */
    uint32_t last_action;
    
    /** @brief Cumulative reward */
    float total_reward;
};

/**
 * @struct CEWMetrics
 * @brief Performance and effectiveness metrics
 * 
 * @details Tracks key performance indicators for the CEW system
 * including detection rates, response times, and resource utilization.
 */
struct CEWMetrics {
    /** @brief Total threats detected */
    uint64_t threats_detected;
    
    /** @brief Total jamming activations */
    uint64_t jamming_activated;
    
    /** @brief Average response latency in microseconds */
    float average_response_time_us;
    
    /** @brief Jamming effectiveness ratio (0.0-1.0) */
    float jamming_effectiveness;
    
    /** @brief Count of missed real-time deadlines */
    uint32_t deadline_misses;
    
    /** @brief Count of CPU/GPU backend switches */
    uint32_t backend_switches;
    
    /** @brief Total processing time in microseconds */
    uint64_t total_processing_time_us;
    
    /** @brief CPU processing time in microseconds */
    uint64_t cpu_processing_time_us;
    
    /** @brief GPU processing time in microseconds */
    uint64_t gpu_processing_time_us;
};

/**
 * @defgroup Quantization Signal Quantization Functions
 * @{
 */

/**
 * @brief Quantize frequency to state space
 * @param freq_ghz Frequency in GHz
 * @return Quantized frequency index
 * 
 * @details Maps continuous frequency values to discrete state indices
 * for Q-learning. Uses uniform quantization with 10 GHz bins.
 */
inline constexpr uint32_t quantize_frequency(float freq_ghz) noexcept {
    return static_cast<uint32_t>((freq_ghz - FREQ_MIN_GHZ) / 10.0f);
}

/**
 * @brief Quantize power level to state space
 * @param power_dbm Power in dBm
 * @return Quantized power index
 * 
 * @details Maps power levels to discrete indices. Assumes input
 * range of -100 to +50 dBm with 25 dB quantization steps.
 */
inline constexpr uint32_t quantize_power(float power_dbm) noexcept {
    return static_cast<uint32_t>((power_dbm + 100.0f) / 25.0f);
}

/**
 * @brief Quantize bandwidth to state space
 * @param bandwidth_mhz Bandwidth in MHz
 * @return Quantized bandwidth index
 * 
 * @details Maps bandwidth to discrete indices with 50 MHz steps.
 * Useful for categorizing narrow vs. wide band signals.
 */
inline constexpr uint32_t quantize_bandwidth(float bandwidth_mhz) noexcept {
    return static_cast<uint32_t>(bandwidth_mhz / 50.0f);
}

} // namespace ares::cew

// Compile-time assertions for struct safety
static_assert(std::is_standard_layout_v<ares::cew::ThreatSignature>);
static_assert(std::is_trivially_copyable_v<ares::cew::ThreatSignature>);
static_assert(std::is_standard_layout_v<ares::cew::JammingParams>);
static_assert(std::is_trivially_copyable_v<ares::cew::JammingParams>);

#endif // ARES_CEW_ADAPTIVE_JAMMING_H