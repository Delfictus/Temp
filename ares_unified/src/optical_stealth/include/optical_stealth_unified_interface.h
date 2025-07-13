#pragma once

#include <memory>
#include <vector>
#include <array>
#include <complex>

namespace ares::optical_stealth {

// Unified optical stealth control interface
class OpticalStealthController {
public:
    OpticalStealthController();
    ~OpticalStealthController();
    
    // Initialization
    bool initialize(const MetamaterialConfig& config);
    void shutdown();
    
    // Metamaterial control
    bool setMetamaterialPattern(const MetamaterialPattern& pattern);
    bool adaptToEnvironment(const EnvironmentalParameters& env);
    
    // Multi-spectral operations
    struct SpectralResponse {
        std::vector<float> wavelengths;  // nm
        std::vector<float> reflectance;
        std::vector<float> transmittance;
        std::vector<float> absorption;
    };
    
    SpectralResponse getCurrentSpectralResponse() const;
    bool optimizeForWavelength(float wavelengthNm, float targetReflectance);
    
    // Active camouflage
    bool enableActiveCamouflage(const CamouflageMode& mode);
    void updateCamouflagePattern(const std::vector<uint8_t>& backgroundImage);
    
    // Radar-infrared-optical stealth synthesis (RIOSS)
    struct RIOSSProfile {
        float radarCrossSection;     // m²
        float infraredSignature;      // W/sr
        float opticalVisibility;      // 0-1
        float optimizationScore;      // Combined metric
    };
    
    RIOSSProfile getCurrentProfile() const;
    bool optimizeRIOSS(const RIOSSConstraints& constraints);
    
    // Performance control
    void setUseCuda(bool useCuda);
    void setPowerBudget(float watts);
    
    // Real-time monitoring
    struct SystemStatus {
        bool isOperational;
        float powerConsumption;
        float thermalLoad;
        std::vector<std::string> warnings;
    };
    
    SystemStatus getStatus() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Metamaterial configuration
struct MetamaterialConfig {
    size_t gridResolutionX = 1024;
    size_t gridResolutionY = 1024;
    float elementSpacing = 0.1f;  // mm
    float frequencyRange[2] = {1e9, 100e9};  // Hz
    bool supportsDynamicTuning = true;
};

// Metamaterial pattern specification
struct MetamaterialPattern {
    std::vector<std::complex<float>> permittivity;
    std::vector<std::complex<float>> permeability;
    std::vector<float> conductivity;
};

// Environmental parameters for adaptation
struct EnvironmentalParameters {
    float ambientTemperature;  // Kelvin
    float humidity;            // 0-1
    float solarIrradiance;     // W/m²
    std::array<float, 3> observerDirection;
    std::vector<float> threatFrequencies;  // Hz
};

// Camouflage modes
enum class CamouflageMode {
    ADAPTIVE_PATTERN,
    BACKGROUND_MATCHING,
    DISRUPTIVE_COLORATION,
    COUNTERSHADING,
    MOTION_DAZZLE
};

// RIOSS optimization constraints
struct RIOSSConstraints {
    float maxRadarCrossSection;    // m²
    float maxInfraredSignature;    // W/sr
    float maxOpticalVisibility;    // 0-1
    float maxPowerConsumption;     // W
    std::vector<float> priorityWeights;  // [radar, IR, optical]
};

} // namespace ares::optical_stealth