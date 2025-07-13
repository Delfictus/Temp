/**
 * @file realtime_state_sync.h
 * @brief Real-time state synchronization for digital twin with <1ms latency
 * 
 * Implements bidirectional state mirroring between physical and digital systems
 * using lock-free data structures and GPU acceleration
 */

#ifndef ARES_DIGITAL_TWIN_REALTIME_STATE_SYNC_H
#define ARES_DIGITAL_TWIN_REALTIME_STATE_SYNC_H

#include <cuda_runtime.h>
#include <atomic>
#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <immintrin.h>  // For SIMD

namespace ares::digital_twin {

// Synchronization constants
constexpr uint32_t MAX_STATE_DIMENSIONS = 16384;
constexpr uint32_t MAX_ENTITIES = 1024;
constexpr uint32_t SYNC_BUFFER_SIZE = 64;  // Ring buffer size
constexpr float SYNC_RATE_HZ = 1000.0f;    // 1kHz synchronization
constexpr float MAX_LATENCY_US = 1000.0f;  // 1ms target
constexpr uint32_t INTERPOLATION_STEPS = 10;

// State types
enum class StateType : uint8_t {
    POSITION = 0,
    VELOCITY = 1,
    ACCELERATION = 2,
    ORIENTATION = 3,
    ANGULAR_VELOCITY = 4,
    SENSOR_DATA = 5,
    ACTUATOR_STATE = 6,
    SYSTEM_HEALTH = 7,
    RESOURCE_LEVELS = 8,
    MISSION_STATE = 9,
    CUSTOM = 10
};

// Synchronization modes
enum class SyncMode : uint8_t {
    CONTINUOUS = 0,         // Always sync
    ON_CHANGE = 1,         // Sync only on state change
    PERIODIC = 2,          // Fixed interval sync
    PREDICTIVE = 3,        // Sync with prediction
    ADAPTIVE = 4           // Adjust sync rate based on divergence
};

// Quality of Service levels
enum class QoSLevel : uint8_t {
    BEST_EFFORT = 0,       // No guarantees
    RELIABLE = 1,          // Guaranteed delivery
    REAL_TIME = 2,         // Latency bounds
    MISSION_CRITICAL = 3   // Highest priority
};

// State vector representation
template<typename T, size_t N>
struct alignas(64) StateVector {
    std::array<T, N> data;
    uint64_t timestamp_ns;
    uint32_t entity_id;
    StateType type;
    uint8_t confidence;    // 0-100%
    
    // SIMD-optimized operations
    StateVector<T, N> operator+(const StateVector<T, N>& other) const;
    StateVector<T, N> operator*(T scalar) const;
    T dot(const StateVector<T, N>& other) const;
    T norm() const;
};

// Compressed state for bandwidth efficiency
struct CompressedState {
    uint32_t entity_id;
    StateType type;
    uint8_t compression_method;
    uint32_t compressed_size;
    uint8_t data[256];  // Variable size in practice
};

// State delta for efficient updates
struct StateDelta {
    uint32_t entity_id;
    StateType type;
    uint64_t base_timestamp_ns;
    uint64_t delta_timestamp_ns;
    std::vector<float> delta_values;
    std::vector<uint32_t> changed_indices;
};

// Synchronization statistics
struct SyncStatistics {
    uint64_t total_syncs;
    uint64_t successful_syncs;
    uint64_t failed_syncs;
    float average_latency_us;
    float max_latency_us;
    float min_latency_us;
    float divergence_metric;
    float bandwidth_usage_mbps;
    uint64_t compression_ratio;
};

// Entity state container
template<size_t StateDim>
class EntityState {
public:
    EntityState(uint32_t entity_id);
    
    // State access
    StateVector<float, StateDim> get_current_state() const;
    StateVector<float, StateDim> get_predicted_state(uint64_t future_ns) const;
    std::vector<StateVector<float, StateDim>> get_history(uint32_t count) const;
    
    // State update
    void update_state(const StateVector<float, StateDim>& new_state);
    void apply_delta(const StateDelta& delta);
    
    // Interpolation and extrapolation
    StateVector<float, StateDim> interpolate(uint64_t timestamp_ns) const;
    StateVector<float, StateDim> extrapolate(uint64_t timestamp_ns) const;
    
private:
    uint32_t entity_id_;
    
    // Circular buffer for state history
    alignas(64) StateVector<float, StateDim> state_buffer_[SYNC_BUFFER_SIZE];
    std::atomic<uint32_t> write_index_;
    std::atomic<uint32_t> read_index_;
    
    // Kalman filter state
    std::array<float, StateDim> kalman_state_;
    std::array<float, StateDim * StateDim> kalman_covariance_;
    
    // Interpolation cache
    mutable std::array<float, StateDim> interp_cache_;
    mutable uint64_t interp_timestamp_;
};

// Main state synchronization engine
class RealtimeStateSync {
public:
    RealtimeStateSync();
    ~RealtimeStateSync();
    
    // Initialize synchronization
    cudaError_t initialize(
        uint32_t num_entities,
        uint32_t state_dimensions,
        SyncMode mode = SyncMode::CONTINUOUS,
        QoSLevel qos = QoSLevel::REAL_TIME
    );
    
    // Entity management
    cudaError_t register_entity(
        uint32_t entity_id,
        StateType type,
        uint32_t dimensions
    );
    cudaError_t unregister_entity(uint32_t entity_id);
    
    // State synchronization
    cudaError_t sync_to_digital(
        uint32_t entity_id,
        const float* physical_state,
        uint32_t dimensions,
        uint64_t timestamp_ns
    );
    
    cudaError_t sync_from_digital(
        uint32_t entity_id,
        float* physical_state,
        uint32_t dimensions,
        uint64_t timestamp_ns
    );
    
    // Batch synchronization for efficiency
    cudaError_t batch_sync_to_digital(
        const uint32_t* entity_ids,
        const float* physical_states,
        uint32_t num_entities,
        uint64_t timestamp_ns
    );
    
    // Predictive synchronization
    cudaError_t enable_prediction(
        uint32_t entity_id,
        uint32_t prediction_horizon_ms
    );
    
    // Delta compression
    cudaError_t enable_delta_compression(
        uint32_t entity_id,
        float threshold
    );
    
    // Quality monitoring
    float get_divergence(uint32_t entity_id) const;
    float get_sync_confidence(uint32_t entity_id) const;
    SyncStatistics get_statistics() const;
    
    // Advanced features
    cudaError_t set_sync_rate(uint32_t entity_id, float rate_hz);
    cudaError_t set_interpolation_method(uint32_t entity_id, uint32_t method);
    cudaError_t enable_adaptive_sync(uint32_t entity_id, float threshold);
    
    // GPU memory management
    cudaError_t pin_entity_memory(uint32_t entity_id);
    cudaError_t prefetch_entity_data(uint32_t entity_id, int device);
    
private:
    // Configuration
    uint32_t num_entities_;
    uint32_t state_dimensions_;
    SyncMode sync_mode_;
    QoSLevel qos_level_;
    
    // Entity states
    std::unordered_map<uint32_t, std::unique_ptr<EntityState<MAX_STATE_DIMENSIONS>>> entities_;
    
    // GPU memory
    struct GPUMemory {
        float* d_physical_states;
        float* d_digital_states;
        float* d_state_deltas;
        uint64_t* d_timestamps;
        uint32_t* d_entity_ids;
        float* d_interpolation_coeffs;
        
        // Prediction buffers
        float* d_predicted_states;
        float* d_kalman_gains;
        
        // Compression buffers
        uint8_t* d_compressed_states;
        uint32_t* d_compression_indices;
        
        cudaStream_t sync_stream;
        cudaStream_t predict_stream;
        cudaEvent_t sync_event;
    } gpu_mem_;
    
    // Lock-free synchronization
    struct alignas(64) LockFreeSlot {
        std::atomic<uint64_t> version;
        float state[MAX_STATE_DIMENSIONS];
        uint64_t timestamp_ns;
        uint32_t entity_id;
    };
    
    LockFreeSlot* sync_slots_;
    std::atomic<uint32_t> slot_index_;
    
    // Worker threads
    std::thread sync_thread_;
    std::thread predict_thread_;
    std::thread compress_thread_;
    std::atomic<bool> running_;
    
    // Performance monitoring
    std::atomic<uint64_t> total_syncs_;
    std::atomic<uint64_t> successful_syncs_;
    std::atomic<float> total_latency_us_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Internal methods
    void sync_worker();
    void predict_worker();
    void compress_worker();
    
    cudaError_t sync_entity_gpu(
        uint32_t entity_id,
        uint64_t timestamp_ns
    );
    
    float calculate_divergence(
        const float* physical_state,
        const float* digital_state,
        uint32_t dimensions
    );
    
    cudaError_t apply_kalman_filter(
        uint32_t entity_id,
        const float* measurement,
        float* filtered_state
    );
};

// GPU Kernels for state synchronization
namespace sync_kernels {

__global__ void state_interpolation_kernel(
    const float* state_history,
    const uint64_t* timestamps,
    float* interpolated_state,
    uint64_t target_timestamp,
    uint32_t num_states,
    uint32_t state_dim
);

__global__ void state_extrapolation_kernel(
    const float* current_state,
    const float* velocity_state,
    const float* acceleration_state,
    float* predicted_state,
    float delta_time_s,
    uint32_t num_entities,
    uint32_t state_dim
);

__global__ void delta_compression_kernel(
    const float* current_state,
    const float* previous_state,
    float* delta_values,
    uint32_t* changed_indices,
    uint32_t* num_changes,
    float threshold,
    uint32_t state_dim
);

__global__ void kalman_update_kernel(
    float* state_estimate,
    float* covariance_matrix,
    const float* measurement,
    const float* measurement_noise,
    float* kalman_gain,
    uint32_t state_dim
);

__global__ void divergence_calculation_kernel(
    const float* physical_states,
    const float* digital_states,
    float* divergence_metrics,
    uint32_t num_entities,
    uint32_t state_dim
);

__global__ void batch_sync_kernel(
    const float* source_states,
    float* target_states,
    const uint32_t* entity_indices,
    const uint64_t* timestamps,
    uint32_t num_entities,
    uint32_t state_dim
);

__global__ void adaptive_sync_rate_kernel(
    const float* divergence_history,
    float* sync_rates,
    const float* thresholds,
    uint32_t num_entities,
    uint32_t history_length
);

} // namespace sync_kernels

// Utility functions
inline uint64_t get_timestamp_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// SIMD-optimized state operations
inline __m256 load_state_avx(const float* state) {
    return _mm256_load_ps(state);
}

inline void store_state_avx(float* state, __m256 value) {
    _mm256_store_ps(state, value);
}

inline __m256 interpolate_states_avx(
    __m256 state1, 
    __m256 state2, 
    float alpha
) {
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 one_minus_alpha = _mm256_set1_ps(1.0f - alpha);
    
    return _mm256_add_ps(
        _mm256_mul_ps(state1, one_minus_alpha),
        _mm256_mul_ps(state2, alpha_vec)
    );
}

} // namespace ares::digital_twin

#endif // ARES_DIGITAL_TWIN_REALTIME_STATE_SYNC_H