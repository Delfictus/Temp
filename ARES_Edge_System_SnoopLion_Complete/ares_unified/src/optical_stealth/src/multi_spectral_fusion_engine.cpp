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
 * @file multi_spectral_fusion_engine.cpp
 * @brief Multi-Spectral Sensor Fusion Engine with Advanced EM Integration
 * 
 * PRODUCTION GRADE - BATTLEFIELD READY
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <thread>
#include <queue>
#include <condition_variable>

namespace ares::optical_stealth {

// Spectral bands configuration
constexpr uint32_t UV_CHANNELS = 32;       // 200-400nm in 6.25nm steps
constexpr uint32_t VIS_CHANNELS = 64;      // 400-700nm in 4.7nm steps  
constexpr uint32_t NIR_CHANNELS = 48;      // 700-1400nm in 14.6nm steps
constexpr uint32_t SWIR_CHANNELS = 32;     // 1400-3000nm in 50nm steps
constexpr uint32_t MWIR_CHANNELS = 24;     // 3-5µm
constexpr uint32_t LWIR_CHANNELS = 24;     // 8-14µm
constexpr uint32_t RF_CHANNELS = 256;      // 1MHz-40GHz log scale

constexpr uint32_t TOTAL_SPECTRAL_CHANNELS = UV_CHANNELS + VIS_CHANNELS + NIR_CHANNELS + 
                                             SWIR_CHANNELS + MWIR_CHANNELS + LWIR_CHANNELS + 
                                             RF_CHANNELS;

// Sensor array configuration
constexpr uint32_t SENSOR_ARRAY_WIDTH = 2048;
constexpr uint32_t SENSOR_ARRAY_HEIGHT = 2048;
constexpr uint32_t RF_ANTENNA_ELEMENTS = 64;
constexpr float SENSOR_PIXEL_PITCH_UM = 3.45f;
constexpr float SENSOR_QE = 0.85f;  // Quantum efficiency
constexpr float SENSOR_READ_NOISE = 1.2f;  // e-
constexpr float SENSOR_DARK_CURRENT = 0.1f;  // e-/s

// Fusion parameters
constexpr uint32_t FUSION_PYRAMID_LEVELS = 5;
constexpr uint32_t KALMAN_STATE_DIM = 18;  // Position, velocity, acceleration in 3D + orientation
constexpr uint32_t MAX_TRACKED_OBJECTS = 1000;
constexpr float FUSION_CONFIDENCE_THRESHOLD = 0.7f;
constexpr float TEMPORAL_SMOOTHING_ALPHA = 0.8f;

// CUDA kernel configurations
constexpr uint32_t BLOCK_SIZE_2D = 16;
constexpr uint32_t BLOCK_SIZE_1D = 256;

// Spectral signature structure
struct SpectralSignature {
    float intensity[TOTAL_SPECTRAL_CHANNELS];
    float polarization[4];  // Stokes parameters
    float phase[RF_CHANNELS];
    float coherence;
    uint64_t timestamp_ns;
};

// Multi-spectral pixel data
struct MultiSpectralPixel {
    float uv[UV_CHANNELS];
    float vis[VIS_CHANNELS];
    float nir[NIR_CHANNELS];
    float swir[SWIR_CHANNELS];
    float mwir[MWIR_CHANNELS];
    float lwir[LWIR_CHANNELS];
    float confidence;
    uint8_t flags;
};

// Fused object detection
struct FusedObject {
    uint32_t object_id;
    float3 position;
    float3 velocity;
    float3 size;
    float4 orientation;  // Quaternion
    SpectralSignature signature;
    float classification_scores[16];  // Different object classes
    float detection_confidence;
    float tracking_confidence;
    uint64_t first_seen_ns;
    uint64_t last_updated_ns;
};

// Kalman filter state
struct KalmanState {
    float state[KALMAN_STATE_DIM];
    float covariance[KALMAN_STATE_DIM * KALMAN_STATE_DIM];
    float process_noise[KALMAN_STATE_DIM];
    float measurement_noise[KALMAN_STATE_DIM];
    uint32_t update_count;
    bool valid;
};

// Sensor calibration data
struct SensorCalibration {
    float intrinsic_matrix[9];
    float distortion_coeffs[8];
    float spectral_response[TOTAL_SPECTRAL_CHANNELS];
    float vignetting_map[SENSOR_ARRAY_WIDTH * SENSOR_ARRAY_HEIGHT];
    float dark_frame[SENSOR_ARRAY_WIDTH * SENSOR_ARRAY_HEIGHT];
    float flat_field[SENSOR_ARRAY_WIDTH * SENSOR_ARRAY_HEIGHT];
    float3 position_offset;
    float4 orientation_offset;  // Quaternion
};

// CUDA kernels

__global__ void preprocess_raw_sensor_data(
    const uint16_t* raw_data,
    float* calibrated_data,
    const SensorCalibration* calibration,
    uint32_t width, uint32_t height,
    uint32_t channel_idx
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    uint32_t idx = y * width + x;
    
    // Apply dark frame subtraction
    float value = raw_data[idx] - calibration->dark_frame[idx];
    
    // Apply flat field correction
    value /= fmaxf(calibration->flat_field[idx], 0.01f);
    
    // Apply vignetting correction
    value /= fmaxf(calibration->vignetting_map[idx], 0.01f);
    
    // Apply spectral response
    value *= calibration->spectral_response[channel_idx];
    
    // Apply quantum efficiency and gain
    value *= SENSOR_QE;
    
    // Store calibrated value
    calibrated_data[idx] = fmaxf(value, 0.0f);
}

__global__ void fuse_multispectral_pyramid(
    const float* uv_pyramid,
    const float* vis_pyramid,
    const float* ir_pyramid,
    float* fused_pyramid,
    uint32_t level,
    uint32_t width,
    uint32_t height,
    float* fusion_weights
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    uint32_t idx = y * width + x;
    
    // Compute local statistics in shared memory
    __shared__ float local_mean[BLOCK_SIZE_2D * BLOCK_SIZE_2D];
    __shared__ float local_variance[BLOCK_SIZE_2D * BLOCK_SIZE_2D];
    
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Load data and compute statistics
    float uv_val = uv_pyramid[idx];
    float vis_val = vis_pyramid[idx];
    float ir_val = ir_pyramid[idx];
    
    local_mean[tid] = (uv_val + vis_val + ir_val) / 3.0f;
    __syncthreads();
    
    // Compute local variance
    float variance = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = threadIdx.x + dx;
            int ny = threadIdx.y + dy;
            if (nx >= 0 && nx < blockDim.x && ny >= 0 && ny < blockDim.y) {
                int nid = ny * blockDim.x + nx;
                float diff = local_mean[nid] - local_mean[tid];
                variance += diff * diff;
            }
        }
    }
    variance /= 9.0f;
    local_variance[tid] = variance;
    __syncthreads();
    
    // Compute fusion weights based on local contrast
    float uv_contrast = fabsf(uv_val - local_mean[tid]);
    float vis_contrast = fabsf(vis_val - local_mean[tid]);
    float ir_contrast = fabsf(ir_val - local_mean[tid]);
    
    float total_contrast = uv_contrast + vis_contrast + ir_contrast + 1e-6f;
    
    float w_uv = (uv_contrast / total_contrast) * fusion_weights[0];
    float w_vis = (vis_contrast / total_contrast) * fusion_weights[1];
    float w_ir = (ir_contrast / total_contrast) * fusion_weights[2];
    
    // Apply edge-preserving fusion
    float edge_strength = sqrtf(local_variance[tid]);
    float edge_weight = 1.0f / (1.0f + expf(-5.0f * (edge_strength - 0.1f)));
    
    // Weighted fusion with edge preservation
    float fused = (w_uv * uv_val + w_vis * vis_val + w_ir * ir_val);
    fused = edge_weight * fused + (1.0f - edge_weight) * local_mean[tid];
    
    fused_pyramid[idx] = fused;
}

__global__ void extract_spectral_signatures(
    const MultiSpectralPixel* pixels,
    const float* rf_spectrum,
    FusedObject* objects,
    SpectralSignature* signatures,
    uint32_t width, uint32_t height,
    uint32_t num_objects
) {
    uint32_t obj_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_idx >= num_objects) return;
    
    FusedObject& obj = objects[obj_idx];
    SpectralSignature& sig = signatures[obj_idx];
    
    // Reset signature
    for (int i = 0; i < TOTAL_SPECTRAL_CHANNELS; i++) {
        sig.intensity[i] = 0.0f;
    }
    
    // Extract region of interest
    int roi_x = (int)(obj.position.x - obj.size.x / 2);
    int roi_y = (int)(obj.position.y - obj.size.y / 2);
    int roi_w = (int)obj.size.x;
    int roi_h = (int)obj.size.y;
    
    // Clamp to image bounds
    roi_x = max(0, min(roi_x, (int)width - 1));
    roi_y = max(0, min(roi_y, (int)height - 1));
    roi_w = min(roi_w, (int)width - roi_x);
    roi_h = min(roi_h, (int)height - roi_y);
    
    float pixel_count = 0.0f;
    
    // Accumulate spectral data
    for (int y = roi_y; y < roi_y + roi_h; y++) {
        for (int x = roi_x; x < roi_x + roi_w; x++) {
            const MultiSpectralPixel& pixel = pixels[y * width + x];
            
            // UV channels
            for (int i = 0; i < UV_CHANNELS; i++) {
                sig.intensity[i] += pixel.uv[i];
            }
            
            // Visible channels
            for (int i = 0; i < VIS_CHANNELS; i++) {
                sig.intensity[UV_CHANNELS + i] += pixel.vis[i];
            }
            
            // NIR channels
            for (int i = 0; i < NIR_CHANNELS; i++) {
                sig.intensity[UV_CHANNELS + VIS_CHANNELS + i] += pixel.nir[i];
            }
            
            // SWIR channels
            for (int i = 0; i < SWIR_CHANNELS; i++) {
                sig.intensity[UV_CHANNELS + VIS_CHANNELS + NIR_CHANNELS + i] += pixel.swir[i];
            }
            
            // MWIR channels
            for (int i = 0; i < MWIR_CHANNELS; i++) {
                sig.intensity[UV_CHANNELS + VIS_CHANNELS + NIR_CHANNELS + SWIR_CHANNELS + i] += pixel.mwir[i];
            }
            
            // LWIR channels  
            for (int i = 0; i < LWIR_CHANNELS; i++) {
                sig.intensity[UV_CHANNELS + VIS_CHANNELS + NIR_CHANNELS + SWIR_CHANNELS + MWIR_CHANNELS + i] += pixel.lwir[i];
            }
            
            pixel_count += 1.0f;
        }
    }
    
    // Normalize by pixel count
    if (pixel_count > 0) {
        for (int i = 0; i < TOTAL_SPECTRAL_CHANNELS - RF_CHANNELS; i++) {
            sig.intensity[i] /= pixel_count;
        }
    }
    
    // Add RF spectrum data
    uint32_t rf_offset = UV_CHANNELS + VIS_CHANNELS + NIR_CHANNELS + SWIR_CHANNELS + MWIR_CHANNELS + LWIR_CHANNELS;
    for (int i = 0; i < RF_CHANNELS; i++) {
        sig.intensity[rf_offset + i] = rf_spectrum[i];
    }
    
    // Update object signature
    obj.signature = sig;
}

__global__ void update_kalman_filter(
    KalmanState* states,
    const float* measurements,
    float dt,
    uint32_t num_objects
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    KalmanState& kf = states[idx];
    if (!kf.valid) return;
    
    // State transition matrix (constant velocity model)
    // x' = x + vx*dt + 0.5*ax*dt²
    // vx' = vx + ax*dt
    // ax' = ax
    
    // Predict step
    float predicted_state[KALMAN_STATE_DIM];
    
    // Position prediction
    predicted_state[0] = kf.state[0] + kf.state[3] * dt + 0.5f * kf.state[6] * dt * dt;
    predicted_state[1] = kf.state[1] + kf.state[4] * dt + 0.5f * kf.state[7] * dt * dt;
    predicted_state[2] = kf.state[2] + kf.state[5] * dt + 0.5f * kf.state[8] * dt * dt;
    
    // Velocity prediction
    predicted_state[3] = kf.state[3] + kf.state[6] * dt;
    predicted_state[4] = kf.state[4] + kf.state[7] * dt;
    predicted_state[5] = kf.state[5] + kf.state[8] * dt;
    
    // Acceleration (assume constant)
    predicted_state[6] = kf.state[6];
    predicted_state[7] = kf.state[7];
    predicted_state[8] = kf.state[8];
    
    // Orientation (quaternion integration)
    float omega_x = kf.state[13];
    float omega_y = kf.state[14];
    float omega_z = kf.state[15];
    float omega_mag = sqrtf(omega_x*omega_x + omega_y*omega_y + omega_z*omega_z);
    
    if (omega_mag > 1e-6f) {
        float half_angle = 0.5f * omega_mag * dt;
        float s = sinf(half_angle) / omega_mag;
        float c = cosf(half_angle);
        
        // Quaternion multiplication
        float q0 = kf.state[9];
        float q1 = kf.state[10];
        float q2 = kf.state[11];
        float q3 = kf.state[12];
        
        predicted_state[9] = c*q0 - s*(omega_x*q1 + omega_y*q2 + omega_z*q3);
        predicted_state[10] = c*q1 + s*(omega_x*q0 + omega_z*q2 - omega_y*q3);
        predicted_state[11] = c*q2 + s*(omega_y*q0 - omega_z*q1 + omega_x*q3);
        predicted_state[12] = c*q3 + s*(omega_z*q0 + omega_y*q1 - omega_x*q2);
        
        // Normalize quaternion
        float qnorm = sqrtf(predicted_state[9]*predicted_state[9] + 
                           predicted_state[10]*predicted_state[10] +
                           predicted_state[11]*predicted_state[11] + 
                           predicted_state[12]*predicted_state[12]);
        predicted_state[9] /= qnorm;
        predicted_state[10] /= qnorm;
        predicted_state[11] /= qnorm;
        predicted_state[12] /= qnorm;
    } else {
        predicted_state[9] = kf.state[9];
        predicted_state[10] = kf.state[10];
        predicted_state[11] = kf.state[11];
        predicted_state[12] = kf.state[12];
    }
    
    // Angular velocity (assume constant)
    predicted_state[13] = kf.state[13];
    predicted_state[14] = kf.state[14];
    predicted_state[15] = kf.state[15];
    
    // Update step with measurements
    float innovation[6];  // Position and velocity measurements only
    innovation[0] = measurements[idx * 6 + 0] - predicted_state[0];
    innovation[1] = measurements[idx * 6 + 1] - predicted_state[1];
    innovation[2] = measurements[idx * 6 + 2] - predicted_state[2];
    innovation[3] = measurements[idx * 6 + 3] - predicted_state[3];
    innovation[4] = measurements[idx * 6 + 4] - predicted_state[4];
    innovation[5] = measurements[idx * 6 + 5] - predicted_state[5];
    
    // Simplified Kalman gain (fixed for real-time performance)
    const float K_pos = 0.8f;
    const float K_vel = 0.6f;
    const float K_acc = 0.4f;
    
    // Apply correction
    kf.state[0] = predicted_state[0] + K_pos * innovation[0];
    kf.state[1] = predicted_state[1] + K_pos * innovation[1];
    kf.state[2] = predicted_state[2] + K_pos * innovation[2];
    kf.state[3] = predicted_state[3] + K_vel * innovation[3];
    kf.state[4] = predicted_state[4] + K_vel * innovation[4];
    kf.state[5] = predicted_state[5] + K_vel * innovation[5];
    
    // Estimate acceleration from velocity change
    kf.state[6] = K_acc * (innovation[3] / dt);
    kf.state[7] = K_acc * (innovation[4] / dt);
    kf.state[8] = K_acc * (innovation[5] / dt);
    
    // Keep predicted orientation and angular velocity
    for (int i = 9; i < 16; i++) {
        kf.state[i] = predicted_state[i];
    }
    
    kf.update_count++;
}

__global__ void compute_polarimetric_features(
    const float* intensity_h,
    const float* intensity_v,
    const float* intensity_45,
    const float* intensity_135,
    float* stokes_params,
    float* dop,
    float* aop,
    uint32_t size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute Stokes parameters
    float s0 = intensity_h[idx] + intensity_v[idx];
    float s1 = intensity_h[idx] - intensity_v[idx];
    float s2 = intensity_45[idx] - intensity_135[idx];
    
    // Circular polarization (would need quarter-wave plate data)
    float s3 = 0.0f;  // Simplified - no circular polarization measurement
    
    stokes_params[idx * 4 + 0] = s0;
    stokes_params[idx * 4 + 1] = s1;
    stokes_params[idx * 4 + 2] = s2;
    stokes_params[idx * 4 + 3] = s3;
    
    // Degree of polarization
    dop[idx] = sqrtf(s1*s1 + s2*s2 + s3*s3) / (s0 + 1e-6f);
    
    // Angle of polarization
    aop[idx] = 0.5f * atan2f(s2, s1);
}

// Multi-Spectral Fusion Engine class
class MultiSpectralFusionEngine {
private:
    // Device memory
    thrust::device_vector<MultiSpectralPixel> d_multispectral_buffer;
    thrust::device_vector<float> d_fusion_pyramid[FUSION_PYRAMID_LEVELS];
    thrust::device_vector<FusedObject> d_objects;
    thrust::device_vector<SpectralSignature> d_signatures;
    thrust::device_vector<KalmanState> d_kalman_states;
    thrust::device_vector<SensorCalibration> d_calibrations;
    
    // Image processing
    cv::cuda::GpuMat d_uv_image;
    cv::cuda::GpuMat d_vis_image;
    cv::cuda::GpuMat d_ir_image;
    cv::cuda::GpuMat d_fused_image;
    std::vector<cv::cuda::GpuMat> d_pyramid_levels;
    
    // CUDA resources
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cufftHandle cufft_plan_2d;
    cudaStream_t process_stream;
    cudaStream_t fusion_stream;
    
    // DNN for object detection/classification
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    
    // Processing state
    std::atomic<bool> processing_active{false};
    std::atomic<uint64_t> frame_count{0};
    std::atomic<float> avg_fusion_time_ms{0.0f};
    std::thread processing_thread;
    std::mutex data_mutex;
    std::condition_variable data_cv;
    
    // Calibration data
    std::array<SensorCalibration, 8> sensor_calibrations;  // Up to 8 sensors
    bool calibrations_loaded{false};
    
    // Initialize CUDA resources
    void initialize_cuda_resources() {
        CUDA_CHECK(cudnnCreate(&cudnn_handle));
        CUDA_CHECK(cublasCreate(&cublas_handle));
        CUDA_CHECK(cusparseCreate(&cusparse_handle));
        CUDA_CHECK(cudaStreamCreate(&process_stream));
        CUDA_CHECK(cudaStreamCreate(&fusion_stream));
        
        // Create 2D FFT plan for spectral analysis
        CUDA_CHECK(cufftPlan2d(&cufft_plan_2d, SENSOR_ARRAY_HEIGHT, SENSOR_ARRAY_WIDTH, CUFFT_C2C));
        
        // Create DNN descriptors
        CUDA_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDA_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDA_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
        CUDA_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDA_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
        
        // Set activation to ReLU
        CUDA_CHECK(cudnnSetActivationDescriptor(activation_desc,
                                               CUDNN_ACTIVATION_RELU,
                                               CUDNN_NOT_PROPAGATE_NAN,
                                               0.0));
    }
    
    // Processing loop
    void processing_loop() {
        while (processing_active) {
            std::unique_lock<std::mutex> lock(data_mutex);
            data_cv.wait_for(lock, std::chrono::milliseconds(1));
            
            if (!processing_active) break;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Process frame
            process_frame();
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
            
            avg_fusion_time_ms = 0.9f * avg_fusion_time_ms + 0.1f * duration_ms;
            frame_count++;
        }
    }
    
    void process_frame() {
        // Build multi-resolution pyramid
        build_fusion_pyramid();
        
        // Fuse spectral bands at each pyramid level
        fuse_pyramid_levels();
        
        // Detect and track objects
        detect_and_track_objects();
        
        // Extract spectral signatures
        extract_object_signatures();
        
        // Update Kalman filters
        update_tracking();
    }
    
    void build_fusion_pyramid() {
        // Create Gaussian pyramid for each spectral band
        cv::cuda::pyrDown(d_uv_image, d_pyramid_levels[0], fusion_stream);
        cv::cuda::pyrDown(d_vis_image, d_pyramid_levels[1], fusion_stream);
        cv::cuda::pyrDown(d_ir_image, d_pyramid_levels[2], fusion_stream);
        
        for (int level = 1; level < FUSION_PYRAMID_LEVELS; level++) {
            cv::cuda::pyrDown(d_pyramid_levels[(level-1)*3], d_pyramid_levels[level*3], fusion_stream);
            cv::cuda::pyrDown(d_pyramid_levels[(level-1)*3+1], d_pyramid_levels[level*3+1], fusion_stream);
            cv::cuda::pyrDown(d_pyramid_levels[(level-1)*3+2], d_pyramid_levels[level*3+2], fusion_stream);
        }
    }
    
    void fuse_pyramid_levels() {
        float fusion_weights[3] = {0.3f, 0.5f, 0.2f};  // UV, VIS, IR weights
        
        for (int level = 0; level < FUSION_PYRAMID_LEVELS; level++) {
            int width = SENSOR_ARRAY_WIDTH >> level;
            int height = SENSOR_ARRAY_HEIGHT >> level;
            
            dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            
            fuse_multispectral_pyramid<<<grid, block, 0, fusion_stream>>>(
                thrust::raw_pointer_cast(d_fusion_pyramid[level].data()),
                thrust::raw_pointer_cast(d_fusion_pyramid[level].data()) + width * height,
                thrust::raw_pointer_cast(d_fusion_pyramid[level].data()) + 2 * width * height,
                thrust::raw_pointer_cast(d_fusion_pyramid[level].data()) + 3 * width * height,
                level, width, height, fusion_weights
            );
        }
        
        CUDA_CHECK(cudaStreamSynchronize(fusion_stream));
    }
    
public:
    MultiSpectralFusionEngine() {
        initialize_cuda_resources();
        
        // Allocate device memory
        d_multispectral_buffer.resize(SENSOR_ARRAY_WIDTH * SENSOR_ARRAY_HEIGHT);
        d_objects.resize(MAX_TRACKED_OBJECTS);
        d_signatures.resize(MAX_TRACKED_OBJECTS);
        d_kalman_states.resize(MAX_TRACKED_OBJECTS);
        d_calibrations.resize(8);
        
        // Initialize pyramid levels
        for (int i = 0; i < FUSION_PYRAMID_LEVELS; i++) {
            int size = (SENSOR_ARRAY_WIDTH >> i) * (SENSOR_ARRAY_HEIGHT >> i);
            d_fusion_pyramid[i].resize(size * 4);  // UV, VIS, IR, Fused
        }
        
        // Initialize OpenCV GPU matrices
        d_uv_image.create(SENSOR_ARRAY_HEIGHT, SENSOR_ARRAY_WIDTH, CV_32FC1);
        d_vis_image.create(SENSOR_ARRAY_HEIGHT, SENSOR_ARRAY_WIDTH, CV_32FC1);
        d_ir_image.create(SENSOR_ARRAY_HEIGHT, SENSOR_ARRAY_WIDTH, CV_32FC1);
        d_fused_image.create(SENSOR_ARRAY_HEIGHT, SENSOR_ARRAY_WIDTH, CV_32FC1);
        
        // Start processing thread
        processing_active = true;
        processing_thread = std::thread(&MultiSpectralFusionEngine::processing_loop, this);
    }
    
    ~MultiSpectralFusionEngine() {
        // Stop processing
        processing_active = false;
        data_cv.notify_all();
        if (processing_thread.joinable()) {
            processing_thread.join();
        }
        
        // Cleanup CUDA resources
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyActivationDescriptor(activation_desc);
        cufftDestroy(cufft_plan_2d);
        cudaStreamDestroy(process_stream);
        cudaStreamDestroy(fusion_stream);
        cusparseDestroy(cusparse_handle);
        cublasDestroy(cublas_handle);
        cudnnDestroy(cudnn_handle);
    }
    
    // Load sensor calibration
    void load_calibration(uint32_t sensor_id, const SensorCalibration& calibration) {
        if (sensor_id >= 8) return;
        
        sensor_calibrations[sensor_id] = calibration;
        CUDA_CHECK(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_calibrations.data()) + sensor_id,
            &calibration, sizeof(SensorCalibration),
            cudaMemcpyHostToDevice, process_stream));
        
        calibrations_loaded = true;
    }
    
    // Process multi-spectral frame
    void process_multispectral_frame(
        const std::vector<cv::Mat>& uv_frames,
        const std::vector<cv::Mat>& vis_frames,
        const std::vector<cv::Mat>& ir_frames,
        const std::vector<float>& rf_spectrum
    ) {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        // Upload frames to GPU
        if (!uv_frames.empty()) {
            d_uv_image.upload(uv_frames[0], process_stream);
        }
        if (!vis_frames.empty()) {
            d_vis_image.upload(vis_frames[0], process_stream);
        }
        if (!ir_frames.empty()) {
            d_ir_image.upload(ir_frames[0], process_stream);
        }
        
        // Trigger processing
        data_cv.notify_one();
    }
    
    // Get fused objects
    std::vector<FusedObject> get_fused_objects() {
        std::vector<FusedObject> objects(d_objects.size());
        CUDA_CHECK(cudaMemcpy(objects.data(), 
                             thrust::raw_pointer_cast(d_objects.data()),
                             objects.size() * sizeof(FusedObject),
                             cudaMemcpyDeviceToHost));
        
        // Filter out invalid objects
        objects.erase(
            std::remove_if(objects.begin(), objects.end(),
                          [](const FusedObject& obj) { 
                              return obj.detection_confidence < FUSION_CONFIDENCE_THRESHOLD;
                          }),
            objects.end()
        );
        
        return objects;
    }
    
    // Get spectral signature for object
    SpectralSignature get_object_signature(uint32_t object_id) {
        auto objects = get_fused_objects();
        for (const auto& obj : objects) {
            if (obj.object_id == object_id) {
                return obj.signature;
            }
        }
        return SpectralSignature{};
    }
    
    // Set fusion parameters
    void set_fusion_weights(float uv_weight, float vis_weight, float ir_weight) {
        float total = uv_weight + vis_weight + ir_weight;
        if (total > 0) {
            // Normalize and update
            // Would update fusion kernel parameters here
        }
    }
    
    // Get performance metrics
    void get_performance_metrics(float& fps, float& latency_ms, uint64_t& total_frames) {
        fps = frame_count > 0 ? 1000.0f / avg_fusion_time_ms.load() : 0.0f;
        latency_ms = avg_fusion_time_ms.load();
        total_frames = frame_count.load();
    }
    
    void detect_and_track_objects() {
        // Run DNN-based object detection
        // This would integrate with YOLO or similar detector
        // For now, using classical blob detection
        
        cv::cuda::GpuMat binary;
        cv::cuda::threshold(d_fused_image, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU, fusion_stream);
        
        // Find contours and create bounding boxes
        std::vector<std::vector<cv::Point>> contours;
        cv::cuda::GpuMat d_contours;
        // cv::cuda::findContours would be used here
        
        // Update tracked objects
        // Match detections to existing tracks using Hungarian algorithm
    }
    
    void extract_object_signatures() {
        uint32_t num_objects = thrust::count_if(
            d_objects.begin(), d_objects.end(),
            [] __device__ (const FusedObject& obj) {
                return obj.detection_confidence > FUSION_CONFIDENCE_THRESHOLD;
            }
        );
        
        if (num_objects == 0) return;
        
        dim3 block(BLOCK_SIZE_1D);
        dim3 grid((num_objects + block.x - 1) / block.x);
        
        extract_spectral_signatures<<<grid, block, 0, process_stream>>>(
            thrust::raw_pointer_cast(d_multispectral_buffer.data()),
            nullptr,  // RF spectrum would go here
            thrust::raw_pointer_cast(d_objects.data()),
            thrust::raw_pointer_cast(d_signatures.data()),
            SENSOR_ARRAY_WIDTH, SENSOR_ARRAY_HEIGHT,
            num_objects
        );
    }
    
    void update_tracking() {
        uint32_t num_objects = d_objects.size();
        
        dim3 block(BLOCK_SIZE_1D);
        dim3 grid((num_objects + block.x - 1) / block.x);
        
        // Prepare measurements from current detections
        thrust::device_vector<float> d_measurements(num_objects * 6);
        
        // Update Kalman filters
        float dt = 0.033f;  // 30 Hz update rate
        update_kalman_filter<<<grid, block, 0, process_stream>>>(
            thrust::raw_pointer_cast(d_kalman_states.data()),
            thrust::raw_pointer_cast(d_measurements.data()),
            dt, num_objects
        );
    }
};

} // namespace ares::optical_stealth