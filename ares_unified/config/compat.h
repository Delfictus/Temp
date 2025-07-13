/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * ARES Edge System™ - Compatibility Header
 * 
 * This header provides backward compatibility mappings for the consolidation.
 * It helps transition from old scattered definitions to the new unified structure.
 */

#ifndef ARES_COMPAT_H
#define ARES_COMPAT_H

#include "constants.h"
#include "../src/utils/common_utils.h"

// =============================================================================
// Compatibility Mappings
// =============================================================================

// Old CUDA macros to new ones
#ifndef CUDA_CHECK
#define CUDA_CHECK ARES_CUDA_CHECK
#endif

#ifndef CUFFT_CHECK
#define CUFFT_CHECK ARES_CUFFT_CHECK
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK ARES_CUBLAS_CHECK
#endif

#ifndef CUDNN_CHECK
#define CUDNN_CHECK ARES_CUDNN_CHECK
#endif

// Common constants that modules might expect locally
namespace ares {

// Some modules expect these in the local namespace
using constants::MAX_SWARM_SIZE;
using constants::MAX_THREATS;
using constants::PROTOCOL_TEMPLATE_SIZE;
using constants::SIDE_CHANNEL_BANDS;
using constants::SIGNATURE_DIMENSIONS;
using constants::DEFAULT_BLOCK_SIZE;
using constants::WARP_SIZE;

// Common utility functions
using utils::Timer;
using utils::CudaTimer;
using utils::CudaBuffer;
using utils::initializeCuda;
using utils::getOptimalLaunchConfig;
using utils::dBmToWatts;
using utils::wattsTodBm;

// For CUDA code
#ifdef __CUDACC__
using utils::makeComplex;
using utils::atomicAddFloat;
using utils::atomicMaxFloat;
using utils::initCurandState;
#endif

} // namespace ares

// =============================================================================
// Deprecation Warnings
// =============================================================================

// These will help identify code that needs updating
#ifdef ARES_ENABLE_DEPRECATION_WARNINGS

#define DEPRECATED_CONSTANT(old_name, new_name) \
    [[deprecated("Use ares::constants::" #new_name " instead")]] \
    constexpr auto old_name = ares::constants::new_name;

#define DEPRECATED_FUNCTION(old_name, new_name) \
    [[deprecated("Use ares::utils::" #new_name " instead")]] \
    inline auto old_name = ares::utils::new_name;

#endif // ARES_ENABLE_DEPRECATION_WARNINGS

// =============================================================================
// Module-Specific Compatibility
// =============================================================================

// CEW Module
#ifdef ARES_MODULE_CEW
namespace ares::cew {
    using constants::MAX_THREATS;
    using constants::SPECTRUM_BINS;
    using constants::MAX_HISTORY_DEPTH;
    using constants::FREQ_MIN_GHZ;
    using constants::FREQ_MAX_GHZ;
}
#endif

// Swarm Module
#ifdef ARES_MODULE_SWARM
namespace ares::swarm {
    using constants::MAX_SWARM_SIZE;
    using constants::MAX_MESSAGE_SIZE;
}
#endif

// Countermeasures Module
#ifdef ARES_MODULE_COUNTERMEASURES
namespace ares::countermeasures {
    using constants::MAX_SWARM_TARGETS;
    using constants::RF_SPOOFING_CHANNELS;
    using constants::CHAOS_PATTERN_VARIANTS;
    using constants::FRIENDLY_FIRE_THRESHOLD;
}
#endif

// Identity Module
#ifdef ARES_MODULE_IDENTITY
namespace ares::identity {
    using constants::MAX_IDENTITIES;
    using constants::MAX_HARDWARE_COMPONENTS;
    using constants::MAX_ATTESTATION_CHAIN;
}
#endif

// Neuromorphic Module
#ifdef ARES_MODULE_NEUROMORPHIC
namespace ares::neuromorphic {
    using constants::MAX_NEURONS;
    using constants::MAX_SYNAPSES;
    using constants::MAX_NEUROMORPHIC_CORES;
}
#endif

#endif // ARES_COMPAT_H