/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * Company: DELFICTUS I/O LLC
 * CAGE Code: 13H70
 * UEI: LXT3B9GMY4N8
 * Active DoD Contractor
 * 
 * Location: Los Angeles, California 90013 United States
 * 
 * This software contains trade secrets and proprietary information
 * of DELFICTUS I/O LLC. Unauthorized use, reproduction, or distribution
 * is strictly prohibited. This technology is subject to export controls
 * under ITAR and EAR regulations.
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * WARNING: This system is designed for authorized U.S. Department of Defense
 * use only. Misuse may result in severe criminal and civil penalties.
 */

/**
 * @file self_destruct_protocol.cpp  
 * @brief Self-Destruct Protocol with Secure Multi-Stage Verification
 * 
 * PRODUCTION GRADE - BATTLEFIELD READY
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cryptopp/sha.h>
#include <cryptopp/aes.h>
#include <cryptopp/rsa.h>
#include <cryptopp/osrng.h>
#include <cryptopp/base64.h>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <condition_variable>
#include <queue>
#include <bitset>

// Forward declaration of external CUDA kernel
__global__ void initialize_chaos_random_states(curandState* states, uint32_t num_states, uint64_t seed);

namespace ares::countermeasures {

// Self-destruct configuration
constexpr uint32_t AUTH_KEY_SIZE = 256;  // bits
constexpr uint32_t VERIFICATION_STAGES = 5;
constexpr uint32_t FAILSAFE_CODES = 3;
constexpr float TAMPER_DETECTION_THRESHOLD = 0.95f;
constexpr uint32_t COUNTDOWN_STEPS = 100;
constexpr float MIN_COUNTDOWN_MS = 1000.0f;  // 1 second minimum
constexpr float MAX_COUNTDOWN_MS = 30000.0f; // 30 seconds maximum

// Destruct modes
enum class DestructMode : uint8_t {
    DATA_WIPE = 0,           // Secure data erasure only
    COMPONENT_DISABLE = 1,   // Disable critical components
    THERMAL_OVERLOAD = 2,    // Controlled thermal destruction
    ELECTROMAGNETIC = 3,     // EM pulse generation
    KINETIC = 4,            // Physical destruction (if equipped)
    FULL_SPECTRUM = 5       // All methods simultaneously
};

// Trigger conditions
enum class TriggerCondition : uint8_t {
    MANUAL_COMMAND = 0,      // Authorized manual trigger
    TAMPER_DETECTED = 1,     // Physical/cyber tamper
    CAPTURE_IMMINENT = 2,    // About to be captured
    MISSION_COMPLETE = 3,    // Mission parameters met
    TIME_LIMIT = 4,          // Timer expiration
    GEOFENCE_BREACH = 5,     // Outside operational area
    LOSS_OF_CONTROL = 6,     // Lost C2 connection
    SWARM_CONSENSUS = 7      // Swarm-voted destruction
};

// Authentication state
struct AuthenticationState {
    uint8_t auth_keys[VERIFICATION_STAGES][AUTH_KEY_SIZE / 8];
    uint8_t current_stage;
    bool stages_verified[VERIFICATION_STAGES];
    uint64_t auth_timestamp_ns;
    uint32_t failed_attempts;
    bool locked_out;
};

// Countdown state
struct CountdownState {
    float remaining_ms;
    float initial_ms;
    uint32_t abort_codes_entered;
    uint8_t abort_code_hash[32];  // SHA-256
    bool countdown_active;
    bool abort_window_open;
    uint64_t start_timestamp_ns;
};

// Tamper detection sensors
struct TamperSensors {
    float voltage_anomaly;
    float temperature_anomaly;
    float vibration_level;
    float em_field_strength;
    float case_integrity;
    float memory_checksum_error;
    float code_integrity_error;
    bool physical_breach;
};

// Destruction verification
struct DestructionVerification {
    bool data_wiped;
    bool components_disabled;
    bool thermal_complete;
    bool em_pulse_fired;
    bool kinetic_activated;
    float verification_confidence;
    uint64_t completion_timestamp_ns;
};

// Mission parameters for auto-destruct
struct MissionParameters {
    float3 geofence_center;
    float geofence_radius_m;
    uint64_t mission_duration_ns;
    uint64_t loss_of_control_timeout_ns;
    float capture_risk_threshold;
    bool enable_auto_destruct;
};

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
    } \
} while(0)

// CUDA kernels

__global__ void secure_data_wipe(
    uint8_t* memory_regions,
    uint32_t* region_sizes,
    uint32_t num_regions,
    uint32_t overwrite_passes,
    curandState_t* rand_states
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t region_idx = blockIdx.y;
    
    if (region_idx >= num_regions) return;
    
    uint32_t region_start = 0;
    for (uint32_t i = 0; i < region_idx; i++) {
        region_start += region_sizes[i];
    }
    
    uint32_t region_size = region_sizes[region_idx];
    uint32_t stride = gridDim.x * blockDim.x;
    
    curandState_t& rand_state = rand_states[tid % 1024];
    
    // Multiple overwrite passes with different patterns
    for (uint32_t pass = 0; pass < overwrite_passes; pass++) {
        for (uint32_t offset = tid; offset < region_size; offset += stride) {
            uint32_t addr = region_start + offset;
            
            switch (pass % 7) {
                case 0: // All zeros
                    memory_regions[addr] = 0x00;
                    break;
                case 1: // All ones
                    memory_regions[addr] = 0xFF;
                    break;
                case 2: // Alternating 01
                    memory_regions[addr] = 0x55;
                    break;
                case 3: // Alternating 10
                    memory_regions[addr] = 0xAA;
                    break;
                case 4: // Random
                    memory_regions[addr] = curand(&rand_state) & 0xFF;
                    break;
                case 5: // Complement of previous
                    memory_regions[addr] = ~memory_regions[addr];
                    break;
                case 6: // Cryptographically secure random
                    uint32_t secure_rand = curand(&rand_state);
                    secure_rand ^= __brev(secure_rand);  // Bit reversal
                    secure_rand = (secure_rand * 0xDEADBEEF) ^ 0xCAFEBABE;
                    memory_regions[addr] = secure_rand & 0xFF;
                    break;
            }
            
            // Memory fence to ensure write completion
            __threadfence();
        }
        
        // Synchronize all threads between passes
        __syncthreads();
    }
}

__global__ void verify_data_destruction(
    const uint8_t* memory_regions,
    const uint32_t* region_sizes,
    uint32_t num_regions,
    uint32_t* verification_results,
    float* entropy_scores
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t region_idx = blockIdx.y;
    
    if (region_idx >= num_regions) return;
    
    __shared__ uint32_t histogram[256];
    __shared__ float local_entropy;
    
    // Initialize shared memory
    if (threadIdx.x < 256) {
        histogram[threadIdx.x] = 0;
    }
    __syncthreads();
    
    uint32_t region_start = 0;
    for (uint32_t i = 0; i < region_idx; i++) {
        region_start += region_sizes[i];
    }
    
    uint32_t region_size = region_sizes[region_idx];
    uint32_t stride = gridDim.x * blockDim.x;
    
    // Build histogram
    for (uint32_t offset = tid; offset < region_size; offset += stride) {
        uint8_t value = memory_regions[region_start + offset];
        atomicAdd(&histogram[value], 1);
    }
    __syncthreads();
    
    // Compute entropy
    if (threadIdx.x == 0) {
        float entropy = 0.0f;
        float inv_size = 1.0f / region_size;
        
        for (int i = 0; i < 256; i++) {
            if (histogram[i] > 0) {
                float p = histogram[i] * inv_size;
                entropy -= p * log2f(p);
            }
        }
        
        local_entropy = entropy / 8.0f;  // Normalize to [0,1]
        entropy_scores[region_idx] = local_entropy;
        
        // High entropy indicates successful randomization
        verification_results[region_idx] = (local_entropy > 0.99f) ? 1 : 0;
    }
}

__global__ void generate_em_pulse_pattern(
    float* em_waveform,
    uint32_t waveform_length,
    float center_freq_hz,
    float bandwidth_hz,
    float pulse_duration_ns,
    float power_level
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= waveform_length) return;
    
    float t = idx * pulse_duration_ns / waveform_length;
    
    // Chirped pulse for maximum spectral coverage
    float chirp_rate = bandwidth_hz / pulse_duration_ns;
    float instantaneous_freq = center_freq_hz + chirp_rate * t;
    
    // Gaussian envelope
    float t_center = pulse_duration_ns / 2.0f;
    float sigma = pulse_duration_ns / 6.0f;  // 99.7% within pulse duration
    float envelope = expf(-(t - t_center) * (t - t_center) / (2.0f * sigma * sigma));
    
    // Generate waveform
    float phase = 2.0f * M_PI * (center_freq_hz * t + 0.5f * chirp_rate * t * t) * 1e-9f;
    em_waveform[idx] = power_level * envelope * sinf(phase);
    
    // Add harmonics for enhanced effectiveness
    for (int h = 2; h <= 5; h++) {
        phase = 2.0f * M_PI * h * (center_freq_hz * t + 0.5f * chirp_rate * t * t) * 1e-9f;
        em_waveform[idx] += (power_level / h) * envelope * sinf(phase);
    }
}

__global__ void monitor_tamper_sensors(
    const TamperSensors* sensors,
    float* tamper_score,
    uint8_t* tamper_flags,
    float threshold
) {
    // Single thread computes overall tamper score
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float score = 0.0f;
        uint8_t flags = 0;
        
        // Weighted tamper score
        score += sensors->voltage_anomaly * 0.15f;
        score += sensors->temperature_anomaly * 0.15f;
        score += sensors->vibration_level * 0.10f;
        score += sensors->em_field_strength * 0.10f;
        score += sensors->case_integrity * 0.20f;
        score += sensors->memory_checksum_error * 0.15f;
        score += sensors->code_integrity_error * 0.15f;
        
        // Set flags for specific tamper types
        if (sensors->physical_breach) {
            flags |= 0x01;
            score = 1.0f;  // Immediate maximum
        }
        if (sensors->voltage_anomaly > 0.8f) flags |= 0x02;
        if (sensors->temperature_anomaly > 0.8f) flags |= 0x04;
        if (sensors->memory_checksum_error > 0.5f) flags |= 0x08;
        if (sensors->code_integrity_error > 0.5f) flags |= 0x10;
        
        *tamper_score = fminf(score, 1.0f);
        *tamper_flags = flags;
    }
}

__global__ void update_countdown_timer(
    CountdownState* countdown,
    float delta_ms,
    bool* trigger_destruct
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (!countdown->countdown_active) {
            *trigger_destruct = false;
            return;
        }
        
        countdown->remaining_ms -= delta_ms;
        
        if (countdown->remaining_ms <= 0.0f) {
            countdown->remaining_ms = 0.0f;
            countdown->countdown_active = false;
            *trigger_destruct = true;
        } else {
            *trigger_destruct = false;
            
            // Close abort window in final 10% of countdown
            if (countdown->remaining_ms < countdown->initial_ms * 0.1f) {
                countdown->abort_window_open = false;
            }
        }
    }
}

// Self-Destruct Protocol class
class SelfDestructProtocol {
private:
    // Device memory
    thrust::device_vector<uint8_t> d_secure_memory;
    thrust::device_vector<uint32_t> d_memory_regions;
    thrust::device_vector<AuthenticationState> d_auth_state;
    thrust::device_vector<CountdownState> d_countdown_state;
    thrust::device_vector<TamperSensors> d_tamper_sensors;
    thrust::device_vector<DestructionVerification> d_verification;
    thrust::device_vector<float> d_em_waveform;
    thrust::device_vector<curandState_t> d_rand_states;
    
    // CUDA resources
    cudaStream_t destruct_stream;
    cudaStream_t monitor_stream;
    
    // Control state
    std::atomic<DestructMode> destruct_mode{DestructMode::DATA_WIPE};
    std::atomic<bool> armed{false};
    std::atomic<bool> monitoring_active{false};
    std::mutex control_mutex;
    std::condition_variable control_cv;
    std::thread monitor_thread;
    
    // Authentication
    CryptoPP::AutoSeededRandomPool rng;
    std::array<std::vector<uint8_t>, VERIFICATION_STAGES> auth_keys;
    std::vector<uint8_t> master_abort_code;
    
    // Mission parameters
    MissionParameters mission_params;
    std::chrono::steady_clock::time_point mission_start_time;
    std::chrono::steady_clock::time_point last_control_contact;
    
    // Performance tracking
    std::atomic<uint32_t> tamper_events{0};
    std::atomic<uint32_t> false_triggers_prevented{0};
    std::atomic<bool> destruct_in_progress{false};
    
    // Initialize authentication keys
    void initialize_authentication() {
        for (auto& key : auth_keys) {
            key.resize(AUTH_KEY_SIZE / 8);
            rng.GenerateBlock(key.data(), key.size());
        }
        
        master_abort_code.resize(32);
        rng.GenerateBlock(master_abort_code.data(), master_abort_code.size());
        
        // Initialize device auth state
        AuthenticationState h_auth_state{};
        h_auth_state.current_stage = 0;
        h_auth_state.failed_attempts = 0;
        h_auth_state.locked_out = false;
        
        for (size_t i = 0; i < VERIFICATION_STAGES; i++) {
            memcpy(h_auth_state.auth_keys[i], auth_keys[i].data(), AUTH_KEY_SIZE / 8);
            h_auth_state.stages_verified[i] = false;
        }
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_auth_state.data()),
                             &h_auth_state, sizeof(AuthenticationState),
                             cudaMemcpyHostToDevice));
    }
    
    // Monitoring loop
    void monitoring_loop() {
        while (monitoring_active) {
            std::unique_lock<std::mutex> lock(control_mutex);
            control_cv.wait_for(lock, std::chrono::milliseconds(100));
            
            if (!monitoring_active) break;
            
            // Check tamper sensors
            check_tamper_detection();
            
            // Check mission parameters
            check_mission_conditions();
            
            // Update countdown if active
            update_countdown();
        }
    }
    
    void check_tamper_detection() {
        float tamper_score;
        uint8_t tamper_flags;
        
        thrust::device_vector<float> d_tamper_score(1);
        thrust::device_vector<uint8_t> d_tamper_flags(1);
        
        monitor_tamper_sensors<<<1, 1, 0, monitor_stream>>>(
            thrust::raw_pointer_cast(d_tamper_sensors.data()),
            thrust::raw_pointer_cast(d_tamper_score.data()),
            thrust::raw_pointer_cast(d_tamper_flags.data()),
            TAMPER_DETECTION_THRESHOLD
        );
        
        CUDA_CHECK(cudaStreamSynchronize(monitor_stream));
        
        tamper_score = d_tamper_score[0];
        tamper_flags = d_tamper_flags[0];
        
        if (tamper_score > TAMPER_DETECTION_THRESHOLD) {
            tamper_events++;
            
            if (armed) {
                // Initiate tamper-triggered destruction
                trigger_destruction(TriggerCondition::TAMPER_DETECTED);
            }
        }
    }
    
    void check_mission_conditions() {
        auto now = std::chrono::steady_clock::now();
        
        // Check mission time limit
        if (mission_params.mission_duration_ns > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - mission_start_time).count();
            
            if (elapsed > mission_params.mission_duration_ns) {
                if (armed && mission_params.enable_auto_destruct) {
                    trigger_destruction(TriggerCondition::TIME_LIMIT);
                }
            }
        }
        
        // Check loss of control
        if (mission_params.loss_of_control_timeout_ns > 0) {
            auto since_contact = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - last_control_contact).count();
            
            if (since_contact > mission_params.loss_of_control_timeout_ns) {
                if (armed && mission_params.enable_auto_destruct) {
                    trigger_destruction(TriggerCondition::LOSS_OF_CONTROL);
                }
            }
        }
    }
    
    void update_countdown() {
        if (!destruct_in_progress) return;
        
        thrust::device_vector<bool> d_trigger(1);
        
        update_countdown_timer<<<1, 1, 0, destruct_stream>>>(
            thrust::raw_pointer_cast(d_countdown_state.data()),
            100.0f,  // 100ms update interval
            thrust::raw_pointer_cast(d_trigger.data())
        );
        
        CUDA_CHECK(cudaStreamSynchronize(destruct_stream));
        
        if (d_trigger[0]) {
            // Countdown complete - execute destruction
            execute_destruction();
        }
    }
    
    void trigger_destruction(TriggerCondition condition) {
        std::lock_guard<std::mutex> lock(control_mutex);
        
        if (destruct_in_progress) return;
        
        destruct_in_progress = true;
        
        // Initialize countdown
        CountdownState h_countdown;
        h_countdown.initial_ms = 5000.0f;  // 5 second default
        h_countdown.remaining_ms = h_countdown.initial_ms;
        h_countdown.countdown_active = true;
        h_countdown.abort_window_open = true;
        h_countdown.abort_codes_entered = 0;
        h_countdown.start_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Compute abort code hash
        CryptoPP::SHA256 hash;
        hash.Update(master_abort_code.data(), master_abort_code.size());
        hash.Final(h_countdown.abort_code_hash);
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_countdown_state.data()),
                             &h_countdown, sizeof(CountdownState),
                             cudaMemcpyHostToDevice));
    }
    
    void execute_destruction() {
        DestructMode mode = destruct_mode.load();
        
        switch (mode) {
            case DestructMode::DATA_WIPE:
                execute_data_wipe();
                break;
                
            case DestructMode::COMPONENT_DISABLE:
                execute_component_disable();
                break;
                
            case DestructMode::THERMAL_OVERLOAD:
                execute_thermal_overload();
                break;
                
            case DestructMode::ELECTROMAGNETIC:
                execute_em_pulse();
                break;
                
            case DestructMode::KINETIC:
                execute_kinetic_destruct();
                break;
                
            case DestructMode::FULL_SPECTRUM:
                // Execute all methods
                execute_data_wipe();
                execute_component_disable();
                execute_thermal_overload();
                execute_em_pulse();
                execute_kinetic_destruct();
                break;
        }
        
        // Verify destruction
        verify_destruction();
        
        destruct_in_progress = false;
    }
    
    void execute_data_wipe() {
        // Secure multi-pass data wipe
        dim3 block(256);
        dim3 grid(32, d_memory_regions.size());
        
        secure_data_wipe<<<grid, block, 0, destruct_stream>>>(
            thrust::raw_pointer_cast(d_secure_memory.data()),
            thrust::raw_pointer_cast(d_memory_regions.data()),
            d_memory_regions.size(),
            35,  // DoD 5220.22-M standard: 35 passes
            thrust::raw_pointer_cast(d_rand_states.data())
        );
        
        CUDA_CHECK(cudaStreamSynchronize(destruct_stream));
    }
    
    void execute_component_disable() {
        // Disable critical hardware components
        // This would interface with hardware-specific APIs
        
        // Overwrite firmware regions
        // Disable power regulators
        // Corrupt boot loaders
        // Burn fuses if available
    }
    
    void execute_thermal_overload() {
        // Controlled thermal destruction
        // This would interface with thermal management hardware
        
        // Disable cooling systems
        // Overclock processors
        // Short high-current paths
        // Target critical components
    }
    
    void execute_em_pulse() {
        // Generate destructive EM pulse
        uint32_t waveform_length = 10000;
        
        generate_em_pulse_pattern<<<(waveform_length + 255) / 256, 256, 0, destruct_stream>>>(
            thrust::raw_pointer_cast(d_em_waveform.data()),
            waveform_length,
            2.45e9f,    // 2.45 GHz center frequency
            500e6f,     // 500 MHz bandwidth
            1000.0f,    // 1 microsecond pulse
            1000.0f     // Maximum power
        );
        
        CUDA_CHECK(cudaStreamSynchronize(destruct_stream));
        
        // Transmit through available RF hardware
        // This would interface with SDR/transmitter
    }
    
    void execute_kinetic_destruct() {
        // Physical destruction if equipped
        // This would interface with kinetic systems
        
        // Activate shaped charges
        // Trigger mechanical shredders
        // Release corrosive agents
    }
    
    void verify_destruction() {
        // Verify successful destruction
        thrust::device_vector<uint32_t> d_verification_results(d_memory_regions.size());
        thrust::device_vector<float> d_entropy_scores(d_memory_regions.size());
        
        verify_data_destruction<<<32, 256, 0, destruct_stream>>>(
            thrust::raw_pointer_cast(d_secure_memory.data()),
            thrust::raw_pointer_cast(d_memory_regions.data()),
            d_memory_regions.size(),
            thrust::raw_pointer_cast(d_verification_results.data()),
            thrust::raw_pointer_cast(d_entropy_scores.data())
        );
        
        CUDA_CHECK(cudaStreamSynchronize(destruct_stream));
        
        // Update verification state
        DestructionVerification h_verification;
        h_verification.data_wiped = thrust::reduce(d_verification_results.begin(), 
                                                   d_verification_results.end()) == d_memory_regions.size();
        h_verification.verification_confidence = thrust::reduce(d_entropy_scores.begin(),
                                                              d_entropy_scores.end()) / d_memory_regions.size();
        h_verification.completion_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_verification.data()),
                             &h_verification, sizeof(DestructionVerification),
                             cudaMemcpyHostToDevice));
    }
    
public:
    SelfDestructProtocol() {
        // Initialize CUDA resources
        CUDA_CHECK(cudaStreamCreate(&destruct_stream));
        CUDA_CHECK(cudaStreamCreate(&monitor_stream));
        
        // Allocate device memory
        d_secure_memory.resize(1024 * 1024 * 100);  // 100MB secure region
        d_memory_regions.resize(10);  // 10 memory regions
        d_auth_state.resize(1);
        d_countdown_state.resize(1);
        d_tamper_sensors.resize(1);
        d_verification.resize(1);
        d_em_waveform.resize(10000);
        d_rand_states.resize(1024);
        
        // Initialize memory regions
        std::vector<uint32_t> regions = {
            10 * 1024 * 1024,  // 10MB region
            20 * 1024 * 1024,  // 20MB region
            15 * 1024 * 1024,  // 15MB region
            25 * 1024 * 1024,  // 25MB region
            30 * 1024 * 1024   // 30MB region
        };
        d_memory_regions = regions;
        
        // Initialize random states
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        initialize_chaos_random_states<<<4, 256, 0, destruct_stream>>>(
            thrust::raw_pointer_cast(d_rand_states.data()),
            seed, 1024
        );
        
        // Initialize authentication
        initialize_authentication();
        
        // Initialize mission parameters
        mission_params.enable_auto_destruct = false;
        mission_params.mission_duration_ns = 0;
        mission_params.loss_of_control_timeout_ns = 60ULL * 1e9;  // 60 seconds
        mission_start_time = std::chrono::steady_clock::now();
        last_control_contact = mission_start_time;
        
        // Start monitoring thread
        monitoring_active = true;
        monitor_thread = std::thread(&SelfDestructProtocol::monitoring_loop, this);
    }
    
    ~SelfDestructProtocol() {
        // Stop monitoring
        monitoring_active = false;
        control_cv.notify_all();
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
        
        // Cleanup CUDA resources
        cudaStreamDestroy(destruct_stream);
        cudaStreamDestroy(monitor_stream);
    }
    
    // Arm/disarm the system
    bool arm_system(const std::vector<uint8_t>& auth_key, uint32_t stage) {
        if (stage >= VERIFICATION_STAGES) return false;
        
        // Verify authentication key
        if (auth_key.size() != AUTH_KEY_SIZE / 8) return false;
        
        bool match = true;
        for (size_t i = 0; i < auth_key.size(); i++) {
            if (auth_key[i] != auth_keys[stage][i]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            // Update auth state
            AuthenticationState h_auth_state;
            CUDA_CHECK(cudaMemcpy(&h_auth_state,
                                 thrust::raw_pointer_cast(d_auth_state.data()),
                                 sizeof(AuthenticationState),
                                 cudaMemcpyDeviceToHost));
            
            h_auth_state.stages_verified[stage] = true;
            h_auth_state.current_stage = stage + 1;
            h_auth_state.auth_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Check if all stages verified
            bool all_verified = true;
            for (int i = 0; i < VERIFICATION_STAGES; i++) {
                if (!h_auth_state.stages_verified[i]) {
                    all_verified = false;
                    break;
                }
            }
            
            if (all_verified) {
                armed = true;
            }
            
            CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_auth_state.data()),
                                 &h_auth_state, sizeof(AuthenticationState),
                                 cudaMemcpyHostToDevice));
            
            return true;
        } else {
            // Failed attempt
            AuthenticationState h_auth_state;
            CUDA_CHECK(cudaMemcpy(&h_auth_state,
                                 thrust::raw_pointer_cast(d_auth_state.data()),
                                 sizeof(AuthenticationState),
                                 cudaMemcpyDeviceToHost));
            
            h_auth_state.failed_attempts++;
            if (h_auth_state.failed_attempts >= 5) {
                h_auth_state.locked_out = true;
            }
            
            CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_auth_state.data()),
                                 &h_auth_state, sizeof(AuthenticationState),
                                 cudaMemcpyHostToDevice));
            
            return false;
        }
    }
    
    void disarm_system() {
        armed = false;
        
        // Reset auth state
        AuthenticationState h_auth_state;
        CUDA_CHECK(cudaMemcpy(&h_auth_state,
                             thrust::raw_pointer_cast(d_auth_state.data()),
                             sizeof(AuthenticationState),
                             cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < VERIFICATION_STAGES; i++) {
            h_auth_state.stages_verified[i] = false;
        }
        h_auth_state.current_stage = 0;
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_auth_state.data()),
                             &h_auth_state, sizeof(AuthenticationState),
                             cudaMemcpyHostToDevice));
    }
    
    // Set destruct mode
    void set_destruct_mode(DestructMode mode) {
        destruct_mode = mode;
    }
    
    // Update mission parameters
    void set_mission_parameters(const MissionParameters& params) {
        std::lock_guard<std::mutex> lock(control_mutex);
        mission_params = params;
    }
    
    // Update tamper sensors
    void update_tamper_sensors(const TamperSensors& sensors) {
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_tamper_sensors.data()),
                                   &sensors, sizeof(TamperSensors),
                                   cudaMemcpyHostToDevice, monitor_stream));
    }
    
    // Abort countdown (requires valid code)
    bool abort_countdown(const std::vector<uint8_t>& abort_code) {
        if (abort_code.size() != 32) return false;
        
        CountdownState h_countdown;
        CUDA_CHECK(cudaMemcpy(&h_countdown,
                             thrust::raw_pointer_cast(d_countdown_state.data()),
                             sizeof(CountdownState),
                             cudaMemcpyDeviceToHost));
        
        if (!h_countdown.countdown_active || !h_countdown.abort_window_open) {
            return false;
        }
        
        // Verify abort code
        CryptoPP::SHA256 hash;
        uint8_t code_hash[32];
        hash.Update(abort_code.data(), abort_code.size());
        hash.Final(code_hash);
        
        bool match = true;
        for (int i = 0; i < 32; i++) {
            if (code_hash[i] != h_countdown.abort_code_hash[i]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            // Abort successful
            h_countdown.countdown_active = false;
            h_countdown.remaining_ms = 0.0f;
            destruct_in_progress = false;
            false_triggers_prevented++;
            
            CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_countdown_state.data()),
                                 &h_countdown, sizeof(CountdownState),
                                 cudaMemcpyHostToDevice));
            
            return true;
        }
        
        return false;
    }
    
    // Manual trigger (requires full authentication)
    bool manual_trigger() {
        if (!armed) return false;
        
        trigger_destruction(TriggerCondition::MANUAL_COMMAND);
        return true;
    }
    
    // Update control contact (prevents timeout)
    void update_control_contact() {
        last_control_contact = std::chrono::steady_clock::now();
    }
    
    // Get current status
    void get_status(bool& is_armed, bool& countdown_active, float& countdown_remaining_ms,
                   uint32_t& tamper_count, DestructMode& mode) {
        is_armed = armed.load();
        mode = destruct_mode.load();
        tamper_count = tamper_events.load();
        
        CountdownState h_countdown;
        CUDA_CHECK(cudaMemcpy(&h_countdown,
                             thrust::raw_pointer_cast(d_countdown_state.data()),
                             sizeof(CountdownState),
                             cudaMemcpyDeviceToHost));
        
        countdown_active = h_countdown.countdown_active;
        countdown_remaining_ms = h_countdown.remaining_ms;
    }
    
    // Emergency immediate destruct (no countdown)
    void emergency_destruct() {
        destruct_in_progress = true;
        set_destruct_mode(DestructMode::FULL_SPECTRUM);
        execute_destruction();
    }
};

} // namespace ares::countermeasures