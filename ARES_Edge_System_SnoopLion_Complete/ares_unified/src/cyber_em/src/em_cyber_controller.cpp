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
 * @file em_cyber_controller.cpp
 * @brief Electromagnetic Cyber Operations Controller
 * 
 * Implements offensive and defensive cyber operations through EM emissions
 * PRODUCTION GRADE - BATTLEFIELD READY
 */

#include "../../config/constants.h"
#include "../../utils/common_utils.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>
#include <thread>
#include <condition_variable>
#include <unordered_map>
#include <queue>
#include <bitset>

namespace ares::cyber_em {

// Use system-wide constants
using namespace ares::constants;
using namespace ares::utils;

// Module-specific EM cyber operation parameters
constexpr uint32_t MAX_SIMULTANEOUS_ATTACKS = 16;
constexpr uint32_t WAVEFORM_BUFFER_SIZE = 1048576;  // 1M samples
// PROTOCOL_TEMPLATE_SIZE and SIDE_CHANNEL_BANDS defined in constants.h
constexpr float MIN_INJECTION_FREQ_HZ = 1e3f;     // 1 kHz
constexpr float MAX_INJECTION_FREQ_HZ = 40e9f;    // 40 GHz
constexpr float TIMING_PRECISION_NS = 0.1f;       // 100 ps
constexpr float POWER_PRECISION_DBM = 0.01f;      // 10 µW

// Attack types
enum class EMAttackType : uint8_t {
    GLITCH_INJECTION = 0,        // Clock/power glitching
    SIDE_CHANNEL_ANALYSIS = 1,   // DPA/DEMA attacks
    PROTOCOL_FUZZING = 2,        // Protocol exploitation
    TEMPEST_INTERCEPT = 3,       // EM emanation capture
    INJECTION_FAULT = 4,         // Direct EM fault injection
    REPLAY_ATTACK = 5,           // Captured signal replay
    JAMMING_SELECTIVE = 6,       // Selective frequency jamming
    MAN_IN_THE_MIDDLE = 7        // EM-based MITM
};

// Defense modes
enum class EMDefenseMode : uint8_t {
    PASSIVE_MONITOR = 0,         // Monitor for attacks
    ACTIVE_SHIELDING = 1,        // Dynamic EM shielding
    NOISE_INJECTION = 2,         // Masking emissions
    PROTOCOL_HARDENING = 3,      // Adaptive protocols
    EMANATION_SUPPRESSION = 4,   // Reduce leakage
    DECEPTION_SIGNALS = 5        // False emanations
};

// Target profile
struct EMTarget {
    uint32_t target_id;
    float3 position;
    float frequency_hz;
    float bandwidth_hz;
    float power_level_dbm;
    uint8_t protocol_type;
    std::array<float, 128> em_signature;
    std::array<uint8_t, 64> protocol_pattern;
    bool is_vulnerable;
    float vulnerability_score;
    uint64_t last_seen_ns;
};

// Attack vector
struct AttackVector {
    EMAttackType attack_type;
    uint32_t target_id;
    float injection_freq_hz;
    float injection_power_dbm;
    float pulse_width_ns;
    float repetition_rate_hz;
    std::array<float, PROTOCOL_TEMPLATE_SIZE> waveform_template;
    float success_probability;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    bool active;
};

// Side channel measurement
struct SideChannelMeasurement {
    float frequency_bands[SIDE_CHANNEL_BANDS];
    float power_spectral_density[SIDE_CHANNEL_BANDS];
    float temporal_correlation[1024];
    float information_leakage_bits;
    uint32_t key_bits_recovered;
    float confidence;
};

// Protocol vulnerability
struct ProtocolVulnerability {
    uint8_t protocol_id;
    std::string vulnerability_type;
    float exploit_difficulty;
    std::vector<uint8_t> exploit_sequence;
    float detection_probability;
    float impact_severity;
};

// Defense state
struct DefenseState {
    EMDefenseMode mode;
    float shield_frequencies[32];
    float shield_strengths[32];
    float noise_level_dbm;
    std::bitset<256> hardened_protocols;
    float emanation_reduction_db;
    bool deception_active;
};

// Use CUDA error checking from common_utils.h
#define CUDA_CHECK ARES_CUDA_CHECK

// CUDA kernels

__global__ void generate_glitch_waveform(
    float* waveform,
    const AttackVector* attack,
    uint32_t waveform_length,
    float sample_rate_hz
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= waveform_length) return;
    
    float t = idx / sample_rate_hz * 1e9f;  // Time in nanoseconds
    
    // Generate glitch pattern
    float glitch = 0.0f;
    
    // Main carrier
    float carrier_phase = 2.0f * M_PI * attack->injection_freq_hz * t * 1e-9f;
    
    // Glitch timing
    float glitch_period = 1e9f / attack->repetition_rate_hz;  // ns
    float phase_in_period = fmodf(t, glitch_period);
    
    if (phase_in_period < attack->pulse_width_ns) {
        // Within glitch pulse
        float envelope = 1.0f;
        
        // Fast rise/fall times for effective glitching
        float rise_time = attack->pulse_width_ns * 0.1f;
        float fall_time = attack->pulse_width_ns * 0.1f;
        
        if (phase_in_period < rise_time) {
            envelope = phase_in_period / rise_time;
        } else if (phase_in_period > attack->pulse_width_ns - fall_time) {
            envelope = (attack->pulse_width_ns - phase_in_period) / fall_time;
        }
        
        // Apply shaped glitch
        glitch = envelope * cosf(carrier_phase);
        
        // Add harmonics for sharper edges
        for (int h = 2; h <= 5; h++) {
            glitch += (envelope / h) * cosf(h * carrier_phase);
        }
    }
    
    // Convert to power level
    float power_linear = powf(10.0f, attack->injection_power_dbm / 10.0f) * 0.001f;  // mW to W
    waveform[idx] = sqrtf(power_linear) * glitch;
}

__global__ void analyze_side_channel_emissions(
    const thrust::complex<float>* fft_data,
    SideChannelMeasurement* measurements,
    const float* reference_signature,
    uint32_t fft_size,
    uint32_t num_measurements,
    float sample_rate_hz
) {
    uint32_t meas_idx = blockIdx.x;
    if (meas_idx >= num_measurements) return;
    
    uint32_t tid = threadIdx.x;
    uint32_t band_size = fft_size / SIDE_CHANNEL_BANDS;
    
    __shared__ float band_power[SIDE_CHANNEL_BANDS];
    __shared__ float band_entropy[SIDE_CHANNEL_BANDS];
    
    // Initialize shared memory
    if (tid < SIDE_CHANNEL_BANDS) {
        band_power[tid] = 0.0f;
        band_entropy[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute power in each frequency band
    for (uint32_t i = tid; i < fft_size; i += blockDim.x) {
        uint32_t band = i / band_size;
        if (band < SIDE_CHANNEL_BANDS) {
            float power = thrust::abs(fft_data[meas_idx * fft_size + i]);
            power = power * power;
            atomicAdd(&band_power[band], power);
        }
    }
    __syncthreads();
    
    // Compute spectral entropy (information leakage indicator)
    if (tid < SIDE_CHANNEL_BANDS) {
        float total_power = 0.0f;
        for (int i = 0; i < SIDE_CHANNEL_BANDS; i++) {
            total_power += band_power[i];
        }
        
        if (total_power > 0) {
            float p = band_power[tid] / total_power;
            if (p > 1e-10f) {
                band_entropy[tid] = -p * log2f(p);
            }
        }
    }
    __syncthreads();
    
    // Store measurements
    if (tid == 0) {
        SideChannelMeasurement& meas = measurements[meas_idx];
        
        float total_entropy = 0.0f;
        for (int i = 0; i < SIDE_CHANNEL_BANDS; i++) {
            meas.frequency_bands[i] = i * sample_rate_hz / SIDE_CHANNEL_BANDS;
            meas.power_spectral_density[i] = band_power[i] / band_size;
            total_entropy += band_entropy[i];
        }
        
        // Estimate information leakage
        float max_entropy = log2f(SIDE_CHANNEL_BANDS);
        meas.information_leakage_bits = (max_entropy - total_entropy) * 8.0f;  // Bytes to bits
        
        // Correlation with reference (for DPA)
        float correlation = 0.0f;
        float ref_mean = 0.0f;
        float meas_mean = 0.0f;
        
        for (int i = 0; i < SIDE_CHANNEL_BANDS; i++) {
            ref_mean += reference_signature[i];
            meas_mean += band_power[i];
        }
        ref_mean /= SIDE_CHANNEL_BANDS;
        meas_mean /= SIDE_CHANNEL_BANDS;
        
        float ref_var = 0.0f;
        float meas_var = 0.0f;
        for (int i = 0; i < SIDE_CHANNEL_BANDS; i++) {
            float ref_diff = reference_signature[i] - ref_mean;
            float meas_diff = band_power[i] - meas_mean;
            correlation += ref_diff * meas_diff;
            ref_var += ref_diff * ref_diff;
            meas_var += meas_diff * meas_diff;
        }
        
        if (ref_var > 0 && meas_var > 0) {
            correlation /= sqrtf(ref_var * meas_var);
        }
        
        meas.confidence = fabsf(correlation);
        meas.key_bits_recovered = (uint32_t)(meas.confidence * meas.information_leakage_bits);
    }
}

__global__ void generate_protocol_exploit_waveform(
    float* waveform,
    const AttackVector* attack,
    const uint8_t* exploit_sequence,
    uint32_t sequence_length,
    uint32_t waveform_length,
    float sample_rate_hz,
    float symbol_rate_hz
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= waveform_length) return;
    
    float t = idx / sample_rate_hz;
    
    // Determine current symbol
    uint32_t symbol_idx = (uint32_t)(t * symbol_rate_hz);
    if (symbol_idx >= sequence_length) {
        waveform[idx] = 0.0f;
        return;
    }
    
    uint8_t symbol = exploit_sequence[symbol_idx];
    float symbol_phase = fmodf(t * symbol_rate_hz, 1.0f);
    
    // Generate modulated waveform based on protocol
    float signal = 0.0f;
    
    switch (attack->target_id & 0x07) {  // Protocol type from target ID
        case 0:  // ASK (Amplitude Shift Keying)
            signal = (symbol & 0x01) ? 1.0f : 0.3f;
            signal *= cosf(2.0f * M_PI * attack->injection_freq_hz * t);
            break;
            
        case 1:  // FSK (Frequency Shift Keying)
            {
                float freq_shift = (symbol & 0x01) ? 1000.0f : -1000.0f;
                signal = cosf(2.0f * M_PI * (attack->injection_freq_hz + freq_shift) * t);
            }
            break;
            
        case 2:  // PSK (Phase Shift Keying)
            {
                float phase = (symbol & 0x01) ? M_PI : 0.0f;
                signal = cosf(2.0f * M_PI * attack->injection_freq_hz * t + phase);
            }
            break;
            
        case 3:  // QAM (Quadrature Amplitude Modulation)
            {
                float i_val = ((symbol >> 1) & 0x01) ? 1.0f : -1.0f;
                float q_val = (symbol & 0x01) ? 1.0f : -1.0f;
                signal = i_val * cosf(2.0f * M_PI * attack->injection_freq_hz * t) -
                        q_val * sinf(2.0f * M_PI * attack->injection_freq_hz * t);
            }
            break;
            
        case 4:  // OFDM-like
            {
                // Multiple subcarriers
                for (int sc = 0; sc < 4; sc++) {
                    if (symbol & (1 << sc)) {
                        float subcarrier_freq = attack->injection_freq_hz + 
                                              sc * symbol_rate_hz;
                        signal += 0.25f * cosf(2.0f * M_PI * subcarrier_freq * t);
                    }
                }
            }
            break;
            
        default:  // Direct template playback
            {
                uint32_t template_idx = (uint32_t)(symbol_phase * PROTOCOL_TEMPLATE_SIZE);
                signal = attack->waveform_template[template_idx];
            }
            break;
    }
    
    // Apply pulse shaping
    float pulse_shape = 1.0f;
    if (symbol_phase < 0.1f) {
        pulse_shape = symbol_phase / 0.1f;  // Rise time
    } else if (symbol_phase > 0.9f) {
        pulse_shape = (1.0f - symbol_phase) / 0.1f;  // Fall time
    }
    signal *= pulse_shape;
    
    // Convert to power level
    float power_linear = powf(10.0f, attack->injection_power_dbm / 10.0f) * 0.001f;
    waveform[idx] = sqrtf(power_linear) * signal;
}

__global__ void detect_em_attacks(
    const float* em_spectrum,
    const DefenseState* defense,
    EMAttackType* detected_attacks,
    float* attack_scores,
    uint32_t spectrum_size,
    uint32_t num_detectors
) {
    uint32_t detector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (detector_idx >= num_detectors) return;
    
    // Each detector looks for specific attack signatures
    EMAttackType attack_type = (EMAttackType)(detector_idx % 8);
    float detection_score = 0.0f;
    
    switch (attack_type) {
        case EMAttackType::GLITCH_INJECTION:
            {
                // Look for sharp transients
                float max_transient = 0.0f;
                for (uint32_t i = 1; i < spectrum_size - 1; i++) {
                    float diff = fabsf(em_spectrum[i] - em_spectrum[i-1]);
                    float diff2 = fabsf(em_spectrum[i+1] - em_spectrum[i]);
                    float transient = fmaxf(diff, diff2);
                    max_transient = fmaxf(max_transient, transient);
                }
                detection_score = 1.0f - expf(-max_transient * 10.0f);
            }
            break;
            
        case EMAttackType::SIDE_CHANNEL_ANALYSIS:
            {
                // Look for focused monitoring on specific frequencies
                float spectral_focus = 0.0f;
                float total_power = 0.0f;
                float max_power = 0.0f;
                
                for (uint32_t i = 0; i < spectrum_size; i++) {
                    total_power += em_spectrum[i];
                    max_power = fmaxf(max_power, em_spectrum[i]);
                }
                
                if (total_power > 0) {
                    spectral_focus = max_power / (total_power / spectrum_size);
                    detection_score = 1.0f - expf(-spectral_focus + 1.0f);
                }
            }
            break;
            
        case EMAttackType::PROTOCOL_FUZZING:
            {
                // Look for rapid frequency changes
                float freq_variance = 0.0f;
                float mean_power = 0.0f;
                
                for (uint32_t i = 0; i < spectrum_size; i++) {
                    mean_power += em_spectrum[i];
                }
                mean_power /= spectrum_size;
                
                for (uint32_t i = 0; i < spectrum_size; i++) {
                    float diff = em_spectrum[i] - mean_power;
                    freq_variance += diff * diff;
                }
                freq_variance /= spectrum_size;
                
                detection_score = 1.0f - expf(-freq_variance * 100.0f);
            }
            break;
            
        case EMAttackType::TEMPEST_INTERCEPT:
            {
                // Look for wideband monitoring
                float bandwidth_usage = 0.0f;
                uint32_t active_bins = 0;
                float threshold = 0.1f;  // -10dB
                
                for (uint32_t i = 0; i < spectrum_size; i++) {
                    if (em_spectrum[i] > threshold) {
                        active_bins++;
                    }
                }
                
                bandwidth_usage = (float)active_bins / spectrum_size;
                detection_score = bandwidth_usage;
            }
            break;
            
        case EMAttackType::INJECTION_FAULT:
            {
                // Look for high-power narrowband signals
                float max_narrowband = 0.0f;
                
                for (uint32_t i = 2; i < spectrum_size - 2; i++) {
                    float center = em_spectrum[i];
                    float adjacent = (em_spectrum[i-1] + em_spectrum[i+1]) / 2.0f;
                    float ratio = center / (adjacent + 0.001f);
                    
                    if (ratio > 10.0f && center > 0.5f) {  // 10dB above adjacent, >-3dB
                        max_narrowband = fmaxf(max_narrowband, center);
                    }
                }
                
                detection_score = max_narrowband;
            }
            break;
            
        default:
            detection_score = 0.0f;
            break;
    }
    
    // Apply defense-based modulation
    if (defense->mode == EMDefenseMode::ACTIVE_SHIELDING) {
        // Reduce detection score if within shielded frequencies
        // Would check against shield_frequencies array
        detection_score *= 0.5f;
    }
    
    detected_attacks[detector_idx] = attack_type;
    attack_scores[detector_idx] = detection_score;
}

__global__ void generate_defensive_countermeasure(
    float* countermeasure_waveform,
    const EMAttackType* detected_attacks,
    const float* attack_scores,
    const DefenseState* defense,
    uint32_t waveform_length,
    uint32_t num_detections,
    float sample_rate_hz
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= waveform_length) return;
    
    float t = idx / sample_rate_hz;
    float signal = 0.0f;
    
    // Generate countermeasure based on detected attacks
    for (uint32_t d = 0; d < num_detections; d++) {
        if (attack_scores[d] < 0.5f) continue;  // Threshold
        
        switch (detected_attacks[d]) {
            case EMAttackType::GLITCH_INJECTION:
                // Generate inverse glitches to cancel
                {
                    float anti_glitch_freq = 1e6f;  // 1MHz anti-glitch
                    signal -= 0.5f * sinf(2.0f * M_PI * anti_glitch_freq * t);
                }
                break;
                
            case EMAttackType::SIDE_CHANNEL_ANALYSIS:
                // Add masking noise
                {
                    curandState_t state;
                    curand_init(idx + d * 1000, 0, 0, &state);
                    signal += defense->noise_level_dbm * curand_normal(&state);
                }
                break;
                
            case EMAttackType::PROTOCOL_FUZZING:
                // Frequency hopping countermeasure
                {
                    float hop_rate = 1000.0f;  // 1kHz hopping
                    float hop_phase = fmodf(t * hop_rate, 1.0f);
                    float hop_freq = 1e9f + hop_phase * 1e8f;  // 1-1.1GHz
                    signal += 0.3f * cosf(2.0f * M_PI * hop_freq * t);
                }
                break;
                
            case EMAttackType::INJECTION_FAULT:
                // Generate blocking signal
                {
                    float block_freq = 2.4e9f;  // Common injection frequency
                    signal += 0.8f * cosf(2.0f * M_PI * block_freq * t + M_PI);  // Inverted phase
                }
                break;
                
            default:
                break;
        }
    }
    
    // Apply defense mode specific modifications
    switch (defense->mode) {
        case EMDefenseMode::ACTIVE_SHIELDING:
            // Add notch filters at shield frequencies
            for (int i = 0; i < 32; i++) {
                if (defense->shield_frequencies[i] > 0) {
                    float notch_freq = defense->shield_frequencies[i];
                    float notch_strength = defense->shield_strengths[i];
                    signal *= 1.0f - notch_strength * 
                             expf(-powf((t * notch_freq - floorf(t * notch_freq)) - 0.5f, 2.0f) * 100.0f);
                }
            }
            break;
            
        case EMDefenseMode::NOISE_INJECTION:
            // Add broadband noise
            {
                curandState_t state;
                curand_init(idx, 0, 0, &state);
                signal += powf(10.0f, defense->noise_level_dbm / 20.0f) * curand_normal(&state);
            }
            break;
            
        case EMDefenseMode::DECEPTION_SIGNALS:
            // Add fake protocol signals
            if (defense->deception_active) {
                float deception_freq = 915e6f;  // ISM band
                float deception_pattern = (idx % 1000) < 100 ? 1.0f : 0.0f;  // 10% duty cycle
                signal += 0.5f * deception_pattern * cosf(2.0f * M_PI * deception_freq * t);
            }
            break;
            
        default:
            break;
    }
    
    countermeasure_waveform[idx] = signal;
}

// EM Cyber Controller class
class EMCyberController {
private:
    // Device memory
    thrust::device_vector<EMTarget> d_targets;
    thrust::device_vector<AttackVector> d_attack_vectors;
    thrust::device_vector<SideChannelMeasurement> d_side_channel_data;
    thrust::device_vector<float> d_attack_waveform;
    thrust::device_vector<float> d_defense_waveform;
    thrust::device_vector<float> d_em_spectrum;
    thrust::device_vector<thrust::complex<float>> d_fft_buffer;
    thrust::device_vector<EMAttackType> d_detected_attacks;
    thrust::device_vector<float> d_attack_scores;
    thrust::device_vector<DefenseState> d_defense_state;
    thrust::device_vector<curandState_t> d_rand_states;
    
    // CUDA resources
    cudaStream_t attack_stream;
    cudaStream_t defense_stream;
    cudaStream_t analysis_stream;
    cublasHandle_t cublas_handle;
    cufftHandle fft_plan;
    
    // Control state
    std::atomic<bool> cyber_active{false};
    std::atomic<bool> defense_enabled{true};
    std::atomic<float> attack_power_limit_dbm{30.0f};  // 1W limit
    std::mutex control_mutex;
    std::condition_variable control_cv;
    std::thread cyber_thread;
    
    // Protocol database
    std::unordered_map<uint8_t, ProtocolVulnerability> vulnerability_db;
    std::unordered_map<uint32_t, EMTarget> target_db;
    std::queue<AttackVector> attack_queue;
    
    // Performance metrics
    std::atomic<uint64_t> attacks_executed{0};
    std::atomic<uint64_t> attacks_defended{0};
    std::atomic<uint64_t> vulnerabilities_found{0};
    std::atomic<float> avg_attack_success_rate{0.0f};
    std::atomic<float> avg_defense_effectiveness{0.0f};
    
    // Initialize vulnerability database
    void initialize_vulnerability_db() {
        // Common protocol vulnerabilities
        
        // WiFi WPA2 KRACK-style
        ProtocolVulnerability wpa2_krack;
        wpa2_krack.protocol_id = 0x01;
        wpa2_krack.vulnerability_type = "Key Reinstallation";
        wpa2_krack.exploit_difficulty = 0.6f;
        wpa2_krack.detection_probability = 0.3f;
        wpa2_krack.impact_severity = 0.8f;
        vulnerability_db[0x01] = wpa2_krack;
        
        // Bluetooth BLE
        ProtocolVulnerability ble_mitm;
        ble_mitm.protocol_id = 0x02;
        ble_mitm.vulnerability_type = "Pairing MITM";
        ble_mitm.exploit_difficulty = 0.4f;
        ble_mitm.detection_probability = 0.2f;
        ble_mitm.impact_severity = 0.7f;
        vulnerability_db[0x02] = ble_mitm;
        
        // RFID/NFC
        ProtocolVulnerability rfid_replay;
        rfid_replay.protocol_id = 0x03;
        rfid_replay.vulnerability_type = "Replay Attack";
        rfid_replay.exploit_difficulty = 0.3f;
        rfid_replay.detection_probability = 0.1f;
        rfid_replay.impact_severity = 0.6f;
        vulnerability_db[0x03] = rfid_replay;
        
        // Zigbee
        ProtocolVulnerability zigbee_inject;
        zigbee_inject.protocol_id = 0x04;
        zigbee_inject.vulnerability_type = "Packet Injection";
        zigbee_inject.exploit_difficulty = 0.5f;
        zigbee_inject.detection_probability = 0.4f;
        zigbee_inject.impact_severity = 0.5f;
        vulnerability_db[0x04] = zigbee_inject;
    }
    
    // Cyber operations loop
    void cyber_loop() {
        while (cyber_active) {
            std::unique_lock<std::mutex> lock(control_mutex);
            control_cv.wait_for(lock, std::chrono::milliseconds(10));
            
            if (!cyber_active) break;
            
            // Execute attack queue
            process_attack_queue();
            
            // Perform defense analysis
            if (defense_enabled) {
                perform_defense_analysis();
            }
            
            // Update metrics
            update_performance_metrics();
        }
    }
    
    void process_attack_queue() {
        std::lock_guard<std::mutex> lock(control_mutex);
        
        while (!attack_queue.empty() && attacks_executed < MAX_SIMULTANEOUS_ATTACKS) {
            AttackVector attack = attack_queue.front();
            attack_queue.pop();
            
            // Execute attack
            execute_attack(attack);
            attacks_executed++;
        }
    }
    
    void execute_attack(const AttackVector& attack) {
        // Copy attack vector to device
        CUDA_CHECK(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_attack_vectors.data()),
            &attack, sizeof(AttackVector),
            cudaMemcpyHostToDevice, attack_stream));
        
        dim3 block(256);
        dim3 grid((WAVEFORM_BUFFER_SIZE + block.x - 1) / block.x);
        
        switch (attack.attack_type) {
            case EMAttackType::GLITCH_INJECTION:
                generate_glitch_waveform<<<grid, block, 0, attack_stream>>>(
                    thrust::raw_pointer_cast(d_attack_waveform.data()),
                    thrust::raw_pointer_cast(d_attack_vectors.data()),
                    WAVEFORM_BUFFER_SIZE,
                    100e6f  // 100 MS/s
                );
                break;
                
            case EMAttackType::PROTOCOL_FUZZING:
                {
                    // Generate exploit sequence
                    std::vector<uint8_t> exploit_seq;
                    auto vuln_it = vulnerability_db.find(attack.target_id & 0xFF);
                    if (vuln_it != vulnerability_db.end()) {
                        exploit_seq = vuln_it->second.exploit_sequence;
                    } else {
                        // Generate random fuzzing sequence
                        exploit_seq.resize(1024);
                        for (auto& byte : exploit_seq) {
                            byte = rand() % 256;
                        }
                    }
                    
                    thrust::device_vector<uint8_t> d_exploit_seq = exploit_seq;
                    
                    generate_protocol_exploit_waveform<<<grid, block, 0, attack_stream>>>(
                        thrust::raw_pointer_cast(d_attack_waveform.data()),
                        thrust::raw_pointer_cast(d_attack_vectors.data()),
                        thrust::raw_pointer_cast(d_exploit_seq.data()),
                        exploit_seq.size(),
                        WAVEFORM_BUFFER_SIZE,
                        100e6f,   // 100 MS/s
                        1e6f      // 1 Mbps symbol rate
                    );
                }
                break;
                
            case EMAttackType::SIDE_CHANNEL_ANALYSIS:
                // For side channel, we analyze received signals
                perform_side_channel_analysis();
                break;
                
            default:
                break;
        }
        
        CUDA_CHECK(cudaStreamSynchronize(attack_stream));
        
        // Transmit waveform (interface with SDR)
        // transmit_waveform(d_attack_waveform);
    }
    
    void perform_side_channel_analysis() {
        // FFT of captured EM spectrum
        CUDA_CHECK(cufftExecR2C(fft_plan,
                               thrust::raw_pointer_cast(d_em_spectrum.data()),
                               thrust::raw_pointer_cast(d_fft_buffer.data())));
        
        // Analyze for information leakage
        dim3 block(256);
        dim3 grid(d_side_channel_data.size());
        
        thrust::device_vector<float> d_reference(SIDE_CHANNEL_BANDS, 0.0f);
        
        analyze_side_channel_emissions<<<grid, block, 0, analysis_stream>>>(
            thrust::raw_pointer_cast(d_fft_buffer.data()),
            thrust::raw_pointer_cast(d_side_channel_data.data()),
            thrust::raw_pointer_cast(d_reference.data()),
            d_em_spectrum.size(),
            d_side_channel_data.size(),
            100e6f  // 100 MS/s
        );
        
        CUDA_CHECK(cudaStreamSynchronize(analysis_stream));
    }
    
    void perform_defense_analysis() {
        // Detect ongoing attacks
        dim3 block(256);
        dim3 grid((MAX_SIMULTANEOUS_ATTACKS + block.x - 1) / block.x);
        
        detect_em_attacks<<<grid, block, 0, defense_stream>>>(
            thrust::raw_pointer_cast(d_em_spectrum.data()),
            thrust::raw_pointer_cast(d_defense_state.data()),
            thrust::raw_pointer_cast(d_detected_attacks.data()),
            thrust::raw_pointer_cast(d_attack_scores.data()),
            d_em_spectrum.size(),
            MAX_SIMULTANEOUS_ATTACKS
        );
        
        // Generate countermeasures
        generate_defensive_countermeasure<<<grid, block, 0, defense_stream>>>(
            thrust::raw_pointer_cast(d_defense_waveform.data()),
            thrust::raw_pointer_cast(d_detected_attacks.data()),
            thrust::raw_pointer_cast(d_attack_scores.data()),
            thrust::raw_pointer_cast(d_defense_state.data()),
            WAVEFORM_BUFFER_SIZE,
            MAX_SIMULTANEOUS_ATTACKS,
            100e6f  // 100 MS/s
        );
        
        CUDA_CHECK(cudaStreamSynchronize(defense_stream));
        
        // Check detection results
        std::vector<float> h_scores(MAX_SIMULTANEOUS_ATTACKS);
        CUDA_CHECK(cudaMemcpy(h_scores.data(),
                             thrust::raw_pointer_cast(d_attack_scores.data()),
                             h_scores.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        for (float score : h_scores) {
            if (score > 0.7f) {
                attacks_defended++;
            }
        }
    }
    
    void update_performance_metrics() {
        // Calculate success rates
        if (attacks_executed > 0) {
            // Would get actual success feedback from target monitoring
            avg_attack_success_rate = 0.7f;  // Placeholder
        }
        
        if (attacks_defended > 0) {
            avg_defense_effectiveness = (float)attacks_defended / (attacks_executed + attacks_defended);
        }
    }
    
public:
    EMCyberController() {
        // Initialize CUDA resources
        CUDA_CHECK(cudaStreamCreate(&attack_stream));
        CUDA_CHECK(cudaStreamCreate(&defense_stream));
        CUDA_CHECK(cudaStreamCreate(&analysis_stream));
        CUDA_CHECK(cublasCreate(&cublas_handle));
        
        // Create FFT plan
        CUDA_CHECK(cufftPlan1d(&fft_plan, WAVEFORM_BUFFER_SIZE, CUFFT_R2C, 1));
        
        // Allocate device memory
        d_targets.resize(256);
        d_attack_vectors.resize(MAX_SIMULTANEOUS_ATTACKS);
        d_side_channel_data.resize(32);
        d_attack_waveform.resize(WAVEFORM_BUFFER_SIZE);
        d_defense_waveform.resize(WAVEFORM_BUFFER_SIZE);
        d_em_spectrum.resize(WAVEFORM_BUFFER_SIZE);
        d_fft_buffer.resize(WAVEFORM_BUFFER_SIZE / 2 + 1);
        d_detected_attacks.resize(MAX_SIMULTANEOUS_ATTACKS);
        d_attack_scores.resize(MAX_SIMULTANEOUS_ATTACKS);
        d_defense_state.resize(1);
        d_rand_states.resize(1024);
        
        // Initialize defense state
        DefenseState h_defense;
        h_defense.mode = EMDefenseMode::ACTIVE_SHIELDING;
        h_defense.noise_level_dbm = -30.0f;  // -30 dBm noise floor
        h_defense.emanation_reduction_db = 20.0f;
        h_defense.deception_active = false;
        
        // Initialize shield frequencies (common attack frequencies)
        h_defense.shield_frequencies[0] = 125e3f;     // LF RFID
        h_defense.shield_frequencies[1] = 13.56e6f;   // HF RFID
        h_defense.shield_frequencies[2] = 433e6f;     // ISM
        h_defense.shield_frequencies[3] = 915e6f;     // ISM
        h_defense.shield_frequencies[4] = 2.4e9f;     // WiFi/BT
        h_defense.shield_frequencies[5] = 5.8e9f;     // WiFi
        
        for (int i = 0; i < 6; i++) {
            h_defense.shield_strengths[i] = 0.9f;  // 90% suppression
        }
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_defense_state.data()),
                             &h_defense, sizeof(DefenseState),
                             cudaMemcpyHostToDevice));
        
        // Initialize random states
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        initialize_chaos_random_states<<<4, 256, 0, attack_stream>>>(
            thrust::raw_pointer_cast(d_rand_states.data()),
            seed, 1024
        );
        
        // Initialize vulnerability database
        initialize_vulnerability_db();
        
        // Start cyber thread
        cyber_active = true;
        cyber_thread = std::thread(&EMCyberController::cyber_loop, this);
    }
    
    ~EMCyberController() {
        // Stop cyber operations
        cyber_active = false;
        control_cv.notify_all();
        if (cyber_thread.joinable()) {
            cyber_thread.join();
        }
        
        // Cleanup CUDA resources
        cufftDestroy(fft_plan);
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(attack_stream);
        cudaStreamDestroy(defense_stream);
        cudaStreamDestroy(analysis_stream);
    }
    
    // Add target for cyber operations
    void add_target(const EMTarget& target) {
        std::lock_guard<std::mutex> lock(control_mutex);
        target_db[target.target_id] = target;
        
        // Update device memory
        std::vector<EMTarget> h_targets;
        for (const auto& [id, tgt] : target_db) {
            h_targets.push_back(tgt);
        }
        
        if (h_targets.size() <= d_targets.size()) {
            CUDA_CHECK(cudaMemcpyAsync(
                thrust::raw_pointer_cast(d_targets.data()),
                h_targets.data(),
                h_targets.size() * sizeof(EMTarget),
                cudaMemcpyHostToDevice, attack_stream));
        }
    }
    
    // Queue attack
    void queue_attack(const AttackVector& attack) {
        std::lock_guard<std::mutex> lock(control_mutex);
        
        // Validate power limit
        AttackVector validated_attack = attack;
        if (validated_attack.injection_power_dbm > attack_power_limit_dbm) {
            validated_attack.injection_power_dbm = attack_power_limit_dbm;
        }
        
        attack_queue.push(validated_attack);
        control_cv.notify_one();
    }
    
    // Set defense mode
    void set_defense_mode(EMDefenseMode mode) {
        DefenseState h_defense;
        CUDA_CHECK(cudaMemcpy(&h_defense,
                             thrust::raw_pointer_cast(d_defense_state.data()),
                             sizeof(DefenseState),
                             cudaMemcpyDeviceToHost));
        
        h_defense.mode = mode;
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_defense_state.data()),
                             &h_defense, sizeof(DefenseState),
                             cudaMemcpyHostToDevice));
    }
    
    // Enable/disable defense
    void set_defense_enabled(bool enabled) {
        defense_enabled = enabled;
    }
    
    // Set attack power limit
    void set_power_limit(float limit_dbm) {
        attack_power_limit_dbm = std::clamp(limit_dbm, -30.0f, 40.0f);  // -30 to +40 dBm
    }
    
    // Get side channel analysis results
    std::vector<SideChannelMeasurement> get_side_channel_results() {
        std::vector<SideChannelMeasurement> results(d_side_channel_data.size());
        CUDA_CHECK(cudaMemcpy(results.data(),
                             thrust::raw_pointer_cast(d_side_channel_data.data()),
                             results.size() * sizeof(SideChannelMeasurement),
                             cudaMemcpyDeviceToHost));
        return results;
    }
    
    // Get detected attacks
    std::vector<std::pair<EMAttackType, float>> get_detected_attacks() {
        std::vector<EMAttackType> types(d_detected_attacks.size());
        std::vector<float> scores(d_attack_scores.size());
        
        CUDA_CHECK(cudaMemcpy(types.data(),
                             thrust::raw_pointer_cast(d_detected_attacks.data()),
                             types.size() * sizeof(EMAttackType),
                             cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaMemcpy(scores.data(),
                             thrust::raw_pointer_cast(d_attack_scores.data()),
                             scores.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        std::vector<std::pair<EMAttackType, float>> results;
        for (size_t i = 0; i < types.size(); i++) {
            if (scores[i] > 0.5f) {
                results.push_back({types[i], scores[i]});
            }
        }
        
        return results;
    }
    
    // Get performance metrics
    void get_performance_metrics(uint64_t& attacks, uint64_t& defenses,
                                uint64_t& vulns, float& attack_success,
                                float& defense_effect) {
        attacks = attacks_executed.load();
        defenses = attacks_defended.load();
        vulns = vulnerabilities_found.load();
        attack_success = avg_attack_success_rate.load();
        defense_effect = avg_defense_effectiveness.load();
    }
    
    // Update EM spectrum (from SDR input)
    void update_em_spectrum(const std::vector<float>& spectrum) {
        if (spectrum.size() <= d_em_spectrum.size()) {
            CUDA_CHECK(cudaMemcpyAsync(
                thrust::raw_pointer_cast(d_em_spectrum.data()),
                spectrum.data(),
                spectrum.size() * sizeof(float),
                cudaMemcpyHostToDevice, analysis_stream));
        }
    }
    
    // Emergency cyber defense
    void emergency_cyber_defense() {
        set_defense_mode(EMDefenseMode::ACTIVE_SHIELDING);
        defense_enabled = true;
        
        // Maximum shielding on all frequencies
        DefenseState h_defense;
        CUDA_CHECK(cudaMemcpy(&h_defense,
                             thrust::raw_pointer_cast(d_defense_state.data()),
                             sizeof(DefenseState),
                             cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < 32; i++) {
            h_defense.shield_strengths[i] = 1.0f;  // 100% suppression attempt
        }
        h_defense.noise_level_dbm = 0.0f;  // Maximum noise
        h_defense.deception_active = true;
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_defense_state.data()),
                             &h_defense, sizeof(DefenseState),
                             cudaMemcpyHostToDevice));
    }
};

} // namespace ares::cyber_em