/**
 * @file loihi2_spike_encoder.h
 * @brief Intel Loihi 2 spike encoding for multi-modal sensor fusion
 * 
 * Implements population coding, temporal contrast, and phase-of-firing
 * encoding for ultra-low latency (0.1-1ms) neuromorphic processing
 */

#ifndef ARES_NEUROMORPHIC_LOIHI2_SPIKE_ENCODER_H
#define ARES_NEUROMORPHIC_LOIHI2_SPIKE_ENCODER_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <memory>

namespace ares::neuromorphic {

// Loihi 2 Hardware Constants
constexpr uint32_t LOIHI2_CORES = 128;
constexpr uint32_t NEURONS_PER_CORE = 8192;
constexpr uint32_t SYNAPSES_PER_CORE = 131072;
constexpr uint32_t AXONS_PER_CORE = 4096;
constexpr float TIME_STEP_US = 1.0f;  // 1 microsecond resolution
constexpr uint32_t MAX_SPIKE_RATE_HZ = 1000;

// Encoding Types
enum class SpikeEncodingType : uint8_t {
    POPULATION = 0,          // Gaussian receptive fields
    TEMPORAL_CONTRAST = 1,   // Change detection
    PHASE_OF_FIRING = 2,     // Phase relationships
    BURST_CODING = 3,        // Urgency signaling
    RATE_CODING = 4,         // Traditional rate-based
    LATENCY_CODING = 5,      // First-spike latency
    RANK_ORDER = 6,          // Spike order encoding
    SYNCHRONY = 7            // Synchronized spiking
};

// Sensor Modalities
enum class SensorModality : uint8_t {
    VISUAL_RGB = 0,
    VISUAL_IR = 1,
    RADAR = 2,
    LIDAR = 3,
    ACOUSTIC = 4,
    RF_SPECTRUM = 5,
    IMU = 6,
    GPS = 7,
    MAGNETIC = 8,
    CHEMICAL = 9
};

// Spike Train Representation
struct SpikeTrain {
    uint32_t neuron_id;
    uint32_t* spike_times;      // In microseconds
    uint32_t num_spikes;
    float firing_rate;
    uint8_t modality;
    uint8_t encoding_type;
    uint16_t receptive_field_id;
};

// Population Coding Parameters
struct PopulationCodingParams {
    uint32_t num_neurons;       // Number of neurons in population
    float min_value;            // Minimum input value
    float max_value;            // Maximum input value
    float sigma;                // Gaussian width
    float max_rate;             // Maximum firing rate
    bool circular;              // For angular variables
};

// Temporal Contrast Parameters
struct TemporalContrastParams {
    float tau_fast_ms;          // Fast adaptation time constant
    float tau_slow_ms;          // Slow adaptation time constant
    float threshold;            // Spike threshold
    float refractory_ms;        // Refractory period
    bool on_off_cells;          // Separate ON/OFF pathways
};

// Multi-modal Sensor Data
struct MultiModalSensorData {
    SensorModality modality;
    float* data;
    uint32_t num_channels;
    uint32_t num_samples;
    uint64_t timestamp_us;
    float sampling_rate_hz;
};

class Loihi2SpikeEncoder {
public:
    Loihi2SpikeEncoder();
    ~Loihi2SpikeEncoder();
    
    // Initialize encoder with Loihi 2 configuration
    cudaError_t initialize(
        int gpu_device_id,
        bool use_hardware_loihi = false,
        const char* loihi_config = nullptr
    );
    
    // Configure encoding for specific modality
    cudaError_t configure_modality(
        SensorModality modality,
        SpikeEncodingType encoding,
        void* params  // PopulationCodingParams or TemporalContrastParams
    );
    
    // Encode multi-modal sensor data to spikes
    cudaError_t encode_sensor_data(
        const MultiModalSensorData* sensor_data,
        uint32_t num_modalities,
        SpikeTrain** spike_trains,
        uint32_t* num_spike_trains
    );
    
    // Specialized encoders for different modalities
    cudaError_t encode_visual(
        const float* image_data,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        SpikeTrain* spike_trains
    );
    
    cudaError_t encode_radar(
        const float* range_doppler_map,
        uint32_t range_bins,
        uint32_t doppler_bins,
        SpikeTrain* spike_trains
    );
    
    cudaError_t encode_rf_spectrum(
        const float* spectrum_db,
        uint32_t num_bins,
        float center_freq_ghz,
        SpikeTrain* spike_trains
    );
    
    // Get encoding latency statistics
    float get_average_encoding_latency_us() const { return avg_encoding_latency_us_; }
    
private:
    // Device memory allocations
    float* d_sensor_buffer_;
    uint32_t* d_spike_times_;
    float* d_membrane_potentials_;
    float* d_adaptation_states_;
    float* d_receptive_fields_;
    
    // Encoding configurations per modality
    struct ModalityConfig {
        SpikeEncodingType encoding_type;
        void* encoding_params;
        uint32_t num_neurons;
        uint32_t neuron_offset;
    };
    
    ModalityConfig modality_configs_[10];  // One per SensorModality
    
    // CUDA resources
    cudaStream_t encode_stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Performance tracking
    float avg_encoding_latency_us_;
    uint32_t encoding_count_;
    
    // Loihi 2 hardware interface (when available)
    void* loihi_context_;
    bool use_hardware_;
    
    // Internal encoding methods
    cudaError_t apply_population_coding(
        const float* input,
        uint32_t input_size,
        const PopulationCodingParams* params,
        SpikeTrain* output
    );
    
    cudaError_t apply_temporal_contrast(
        const float* input,
        uint32_t input_size,
        const TemporalContrastParams* params,
        SpikeTrain* output
    );
};

// CUDA Kernels for spike encoding
__global__ void population_coding_kernel(
    const float* __restrict__ input,
    const float* __restrict__ receptive_fields,
    uint32_t* __restrict__ spike_times,
    uint32_t* __restrict__ spike_counts,
    const PopulationCodingParams params,
    uint32_t time_step,
    uint32_t input_size
);

__global__ void temporal_contrast_kernel(
    const float* __restrict__ input,
    float* __restrict__ membrane_potentials,
    float* __restrict__ adaptation_fast,
    float* __restrict__ adaptation_slow,
    uint32_t* __restrict__ spike_times,
    uint32_t* __restrict__ spike_counts,
    const TemporalContrastParams params,
    uint32_t time_step,
    uint32_t num_neurons
);

__global__ void phase_encoding_kernel(
    const float* __restrict__ input,
    float* __restrict__ phase_oscillators,
    uint32_t* __restrict__ spike_times,
    float base_frequency,
    uint32_t time_window_us,
    uint32_t num_channels
);

__global__ void burst_coding_kernel(
    const float* __restrict__ input,
    float* __restrict__ burst_accumulators,
    uint32_t* __restrict__ spike_times,
    uint32_t* __restrict__ burst_lengths,
    float urgency_threshold,
    uint32_t num_inputs
);

// Utility kernels
__global__ void generate_gaussian_receptive_fields(
    float* __restrict__ receptive_fields,
    uint32_t num_neurons,
    uint32_t input_dim,
    float min_val,
    float max_val,
    float sigma
);

__global__ void spike_train_statistics_kernel(
    const uint32_t* __restrict__ spike_times,
    const uint32_t* __restrict__ spike_counts,
    float* __restrict__ firing_rates,
    float* __restrict__ isi_cv,  // Inter-spike interval coefficient of variation
    uint32_t num_neurons,
    uint32_t time_window_us
);

} // namespace ares::neuromorphic

#endif // ARES_NEUROMORPHIC_LOIHI2_SPIKE_ENCODER_H