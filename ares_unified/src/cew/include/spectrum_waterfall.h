/**
 * @file spectrum_waterfall.h
 * @brief Real-time spectrum waterfall analysis for CEW - Unified interface
 * 
 * Provides continuous spectrum monitoring with sliding window FFT
 * and threat detection capabilities for both CPU and GPU backends
 */

#ifndef ARES_CEW_SPECTRUM_WATERFALL_H
#define ARES_CEW_SPECTRUM_WATERFALL_H

#include <stdint.h>
#include <memory>
#include <vector>

namespace ares::cew {

// Waterfall configuration
constexpr uint32_t FFT_SIZE = 4096;
constexpr uint32_t OVERLAP_FACTOR = 4;  // 75% overlap
constexpr uint32_t WINDOW_STRIDE = FFT_SIZE / OVERLAP_FACTOR;
constexpr uint32_t MAX_HISTORY_DEPTH = 256;
constexpr float SAMPLE_RATE_MSPS = 2000.0f;  // 2 GSPS

// Detection parameters
constexpr float NOISE_FLOOR_DBM = -110.0f;
constexpr float DETECTION_THRESHOLD_DB = 10.0f;  // 10dB above noise floor
constexpr uint32_t MIN_SIGNAL_BINS = 4;  // Minimum width for detection

// Window functions
enum class WindowType : uint8_t {
    RECTANGULAR = 0,
    HANNING = 1,
    HAMMING = 2,
    BLACKMAN = 3,
    KAISER = 4,
    FLAT_TOP = 5
};

// Spectrum statistics
struct SpectrumStats {
    float noise_floor_dbm;
    float peak_power_dbm;
    float avg_power_dbm;
    uint32_t active_bins;
    uint32_t detected_signals;
};

// Signal detection result
struct DetectedSignal {
    uint32_t start_bin;
    uint32_t end_bin;
    float center_freq_mhz;
    float bandwidth_mhz;
    float peak_power_dbm;
    float avg_power_dbm;
    uint32_t duration_samples;
    uint64_t timestamp_ns;
};

// Abstract interface for spectrum waterfall processing
class ISpectrumWaterfall {
public:
    virtual ~ISpectrumWaterfall() = default;
    
    // Initialize with configuration
    virtual bool initialize(
        uint32_t fft_size = FFT_SIZE,
        WindowType window = WindowType::BLACKMAN,
        float sample_rate_msps = SAMPLE_RATE_MSPS
    ) = 0;
    
    // Process new IQ samples
    virtual bool process_samples(
        const float* iq_samples,
        uint32_t num_samples,
        uint64_t timestamp_ns
    ) = 0;
    
    // Get current waterfall data
    virtual bool get_waterfall(
        float* waterfall_out,
        uint32_t& width,
        uint32_t& height
    ) const = 0;
    
    // Detect signals in current spectrum
    virtual std::vector<DetectedSignal> detect_signals(
        float threshold_db = DETECTION_THRESHOLD_DB
    ) = 0;
    
    // Get spectrum statistics
    virtual SpectrumStats get_statistics() const = 0;
    
    // Check backend type
    virtual bool is_gpu_accelerated() const = 0;
};

// Factory for creating spectrum waterfall processor
class SpectrumWaterfallFactory {
public:
    static std::unique_ptr<ISpectrumWaterfall> create(bool use_gpu = true);
};

// Thread-safe wrapper for concurrent access
class SpectrumWaterfallManager {
public:
    explicit SpectrumWaterfallManager(bool use_gpu = true);
    ~SpectrumWaterfallManager();
    
    bool initialize(
        uint32_t fft_size = FFT_SIZE,
        WindowType window = WindowType::BLACKMAN,
        float sample_rate_msps = SAMPLE_RATE_MSPS
    );
    
    bool process_samples_threadsafe(
        const float* iq_samples,
        uint32_t num_samples,
        uint64_t timestamp_ns
    );
    
    bool get_waterfall_threadsafe(
        float* waterfall_out,
        uint32_t& width,
        uint32_t& height
    ) const;
    
    std::vector<DetectedSignal> detect_signals_threadsafe(
        float threshold_db = DETECTION_THRESHOLD_DB
    );
    
    SpectrumStats get_statistics_threadsafe() const;
    
    bool is_gpu_accelerated() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ares::cew

#endif // ARES_CEW_SPECTRUM_WATERFALL_H