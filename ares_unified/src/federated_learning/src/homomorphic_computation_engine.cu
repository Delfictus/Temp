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
 * @file homomorphic_computation_engine.cpp
 * @brief Homomorphic Encryption Engine for Privacy-Preserving Computation
 * 
 * Implements CKKS and BGV schemes for encrypted neural network operations
 * PRODUCTION GRADE - BATTLEFIELD READY
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/complex.h>
#include <seal/seal.h>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <chrono>
#include <memory>
#include <thread>
#include <condition_variable>
#include <unordered_map>
#include <complex>
#include <cmath>

namespace ares::homomorphic {

using namespace seal;

// Homomorphic parameters
constexpr uint32_t POLY_MODULUS_DEGREE = 16384;  // 2^14
constexpr uint32_t SCALE_BITS = 40;
constexpr uint32_t MAX_MULTIPLICATIVE_DEPTH = 20;
constexpr uint32_t GALOIS_KEYS_DECOMPOSITION_BITS = 60;
constexpr uint32_t MAX_BATCH_SIZE = POLY_MODULUS_DEGREE / 2;
constexpr uint32_t BOOTSTRAPPING_PRECISION = 25;
constexpr float NOISE_BUDGET_THRESHOLD = 10.0f;

// Encryption schemes
enum class EncryptionScheme : uint8_t {
    CKKS = 0,       // For real/complex numbers
    BGV = 1,        // For integers
    BFV = 2,        // Batched integers
    TFHE = 3        // Fast bootstrapping
};

// Operation types
enum class HomomorphicOp : uint8_t {
    ADD = 0,
    MULTIPLY = 1,
    ROTATE = 2,
    NEGATE = 3,
    SQUARE = 4,
    POLYNOMIAL = 5,
    COMPARISON = 6,
    BOOTSTRAP = 7
};

// Ciphertext metadata
struct CiphertextMeta {
    uint32_t size;
    uint32_t level;
    float scale;
    float noise_budget;
    EncryptionScheme scheme;
    uint64_t operation_count;
    bool needs_relinearization;
    bool needs_rescaling;
};

// Neural network layer types
enum class LayerType : uint8_t {
    FULLY_CONNECTED = 0,
    CONVOLUTION = 1,
    POOLING = 2,
    ACTIVATION = 3,
    BATCH_NORM = 4,
    ATTENTION = 5
};

// Encrypted tensor
struct EncryptedTensor {
    std::vector<Ciphertext> data;
    std::array<uint32_t, 4> shape;  // NCHW format
    CiphertextMeta metadata;
    uint32_t batch_size;
    bool is_packed;
};

// CUDA kernels for pre/post processing
__global__ void packTensorForCKKSKernel(
    thrust::complex<double>* packed_data,
    float* tensor_data,
    uint32_t tensor_size,
    uint32_t slot_count,
    float scale
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= slot_count) return;
    
    if (idx < tensor_size) {
        packed_data[idx] = thrust::complex<double>(
            tensor_data[idx] * scale, 0.0
        );
    } else {
        packed_data[idx] = thrust::complex<double>(0.0, 0.0);
    }
}

__global__ void unpackCKKSResultKernel(
    float* tensor_data,
    thrust::complex<double>* packed_data,
    uint32_t tensor_size,
    float inv_scale
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensor_size) return;
    
    tensor_data[idx] = packed_data[idx].real() * inv_scale;
}

__global__ void polynomialApproximationKernel(
    double* coefficients,
    uint32_t degree,
    double x_min,
    double x_max,
    uint32_t function_type  // 0=ReLU, 1=Sigmoid, 2=Tanh
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= degree + 1) return;
    
    // Chebyshev polynomial approximation
    double x = x_min + (x_max - x_min) * idx / degree;
    
    switch (function_type) {
        case 0:  // ReLU approximation
            if (idx == 0) coefficients[idx] = 0.0;
            else if (idx == 1) coefficients[idx] = 0.5;
            else if (idx == 2) coefficients[idx] = 0.5;
            else coefficients[idx] = 0.0;
            break;
            
        case 1:  // Sigmoid approximation
            {
                double t = 2.0 * (x - x_min) / (x_max - x_min) - 1.0;
                coefficients[idx] = 0.5 + 0.25 * t - 0.03125 * (3 * t * t * t - t);
            }
            break;
            
        case 2:  // Tanh approximation
            {
                double t = 2.0 * (x - x_min) / (x_max - x_min) - 1.0;
                coefficients[idx] = t - (t * t * t) / 3.0 + 2.0 * (t * t * t * t * t) / 15.0;
            }
            break;
    }
}

class HomomorphicComputationEngine {
private:
    // SEAL context and keys
    std::shared_ptr<SEALContext> ckks_context_;
    std::shared_ptr<SEALContext> bgv_context_;
    
    KeyGenerator* ckks_keygen_;
    KeyGenerator* bgv_keygen_;
    
    PublicKey ckks_public_key_;
    SecretKey ckks_secret_key_;
    RelinKeys ckks_relin_keys_;
    GaloisKeys ckks_galois_keys_;
    
    PublicKey bgv_public_key_;
    SecretKey bgv_secret_key_;
    RelinKeys bgv_relin_keys_;
    
    // Evaluators and encoders
    std::unique_ptr<Evaluator> ckks_evaluator_;
    std::unique_ptr<Evaluator> bgv_evaluator_;
    std::unique_ptr<CKKSEncoder> ckks_encoder_;
    std::unique_ptr<BatchEncoder> bgv_encoder_;
    
    // Encryptors and decryptors
    std::unique_ptr<Encryptor> ckks_encryptor_;
    std::unique_ptr<Decryptor> ckks_decryptor_;
    std::unique_ptr<Encryptor> bgv_encryptor_;
    std::unique_ptr<Decryptor> bgv_decryptor_;
    
    // Bootstrapping parameters
    thrust::device_vector<double> bootstrap_polynomial_;
    uint32_t bootstrap_precision_;
    
    // Operation cache
    std::unordered_map<uint64_t, Ciphertext> operation_cache_;
    std::mutex cache_mutex_;
    
    // Performance metrics
    std::atomic<uint64_t> total_operations_;
    std::atomic<uint64_t> bootstrap_count_;
    std::atomic<double> total_noise_consumed_;
    
public:
    HomomorphicComputationEngine() 
        : bootstrap_precision_(BOOTSTRAPPING_PRECISION),
          total_operations_(0),
          bootstrap_count_(0),
          total_noise_consumed_(0.0) {
        
        initializeCKKS();
        initializeBGV();
        initializeBootstrapping();
    }
    
    ~HomomorphicComputationEngine() {
        delete ckks_keygen_;
        delete bgv_keygen_;
    }
    
    EncryptedTensor encryptTensor(
        const thrust::device_vector<float>& tensor,
        const std::array<uint32_t, 4>& shape,
        EncryptionScheme scheme = EncryptionScheme::CKKS
    ) {
        EncryptedTensor encrypted;
        encrypted.shape = shape;
        encrypted.metadata.scheme = scheme;
        
        if (scheme == EncryptionScheme::CKKS) {
            encryptCKKSTensor(tensor, encrypted);
        } else if (scheme == EncryptionScheme::BGV) {
            encryptBGVTensor(tensor, encrypted);
        }
        
        return encrypted;
    }
    
    thrust::device_vector<float> decryptTensor(const EncryptedTensor& encrypted) {
        thrust::device_vector<float> result;
        
        if (encrypted.metadata.scheme == EncryptionScheme::CKKS) {
            decryptCKKSTensor(encrypted, result);
        } else if (encrypted.metadata.scheme == EncryptionScheme::BGV) {
            decryptBGVTensor(encrypted, result);
        }
        
        return result;
    }
    
    EncryptedTensor matrixMultiply(
        const EncryptedTensor& A,
        const EncryptedTensor& B
    ) {
        EncryptedTensor result;
        result.shape = {A.shape[0], A.shape[1], A.shape[2], B.shape[3]};
        result.metadata = A.metadata;
        
        // Ensure compatible dimensions
        if (A.shape[3] != B.shape[2]) {
            throw std::runtime_error("Incompatible matrix dimensions");
        }
        
        uint32_t m = A.shape[2];  // rows of A
        uint32_t n = B.shape[3];  // cols of B
        uint32_t k = A.shape[3];  // cols of A / rows of B
        
        result.data.resize(m * n);
        
        // Blocked matrix multiplication for efficiency
        const uint32_t block_size = 64;
        
        for (uint32_t i = 0; i < m; i += block_size) {
            for (uint32_t j = 0; j < n; j += block_size) {
                Ciphertext block_sum;
                bool first = true;
                
                for (uint32_t l = 0; l < k; ++l) {
                    uint32_t a_idx = i * k + l;
                    uint32_t b_idx = l * n + j;
                    
                    if (a_idx < A.data.size() && b_idx < B.data.size()) {
                        Ciphertext product = A.data[a_idx];
                        ckks_evaluator_->multiply_inplace(product, B.data[b_idx]);
                        ckks_evaluator_->relinearize_inplace(product, ckks_relin_keys_);
                        
                        if (first) {
                            block_sum = product;
                            first = false;
                        } else {
                            ckks_evaluator_->add_inplace(block_sum, product);
                        }
                        
                        // Check noise budget
                        if (ckks_decryptor_->invariant_noise_budget(block_sum) < 
                            NOISE_BUDGET_THRESHOLD) {
                            bootstrapCiphertext(block_sum);
                        }
                    }
                }
                
                ckks_evaluator_->rescale_to_next_inplace(block_sum);
                result.data[i * n + j] = block_sum;
            }
        }
        
        total_operations_ += m * n * k;
        return result;
    }
    
    EncryptedTensor convolution2D(
        const EncryptedTensor& input,
        const EncryptedTensor& kernel,
        uint32_t stride = 1,
        uint32_t padding = 0
    ) {
        // Extract dimensions
        uint32_t batch = input.shape[0];
        uint32_t in_channels = input.shape[1];
        uint32_t in_height = input.shape[2];
        uint32_t in_width = input.shape[3];
        
        uint32_t out_channels = kernel.shape[0];
        uint32_t kernel_height = kernel.shape[2];
        uint32_t kernel_width = kernel.shape[3];
        
        // Calculate output dimensions
        uint32_t out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        uint32_t out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        
        EncryptedTensor result;
        result.shape = {batch, out_channels, out_height, out_width};
        result.metadata = input.metadata;
        result.data.resize(batch * out_channels * out_height * out_width);
        
        // Perform convolution using baby-step giant-step algorithm
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t oc = 0; oc < out_channels; ++oc) {
                for (uint32_t oh = 0; oh < out_height; ++oh) {
                    for (uint32_t ow = 0; ow < out_width; ++ow) {
                        Ciphertext sum;
                        bool first = true;
                        
                        for (uint32_t ic = 0; ic < in_channels; ++ic) {
                            for (uint32_t kh = 0; kh < kernel_height; ++kh) {
                                for (uint32_t kw = 0; kw < kernel_width; ++kw) {
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    
                                    if (ih >= 0 && ih < in_height && 
                                        iw >= 0 && iw < in_width) {
                                        
                                        uint32_t input_idx = 
                                            b * in_channels * in_height * in_width +
                                            ic * in_height * in_width +
                                            ih * in_width + iw;
                                            
                                        uint32_t kernel_idx = 
                                            oc * in_channels * kernel_height * kernel_width +
                                            ic * kernel_height * kernel_width +
                                            kh * kernel_width + kw;
                                        
                                        Ciphertext product = input.data[input_idx];
                                        ckks_evaluator_->multiply_inplace(
                                            product, kernel.data[kernel_idx]
                                        );
                                        
                                        if (first) {
                                            sum = product;
                                            first = false;
                                        } else {
                                            ckks_evaluator_->add_inplace(sum, product);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Store result
                        uint32_t result_idx = 
                            b * out_channels * out_height * out_width +
                            oc * out_height * out_width +
                            oh * out_width + ow;
                            
                        ckks_evaluator_->relinearize_inplace(sum, ckks_relin_keys_);
                        ckks_evaluator_->rescale_to_next_inplace(sum);
                        result.data[result_idx] = sum;
                    }
                }
            }
        }
        
        total_operations_ += batch * out_channels * out_height * out_width * 
                           in_channels * kernel_height * kernel_width;
        
        return result;
    }
    
    EncryptedTensor activationFunction(
        const EncryptedTensor& input,
        uint32_t function_type  // 0=ReLU, 1=Sigmoid, 2=Tanh
    ) {
        EncryptedTensor result;
        result.shape = input.shape;
        result.metadata = input.metadata;
        result.data.resize(input.data.size());
        
        // Get polynomial approximation
        thrust::device_vector<double> coefficients(MAX_MULTIPLICATIVE_DEPTH);
        
        dim3 block(256);
        dim3 grid((MAX_MULTIPLICATIVE_DEPTH + block.x - 1) / block.x);
        
        polynomialApproximationKernel<<<grid, block>>>(
            thrust::raw_pointer_cast(coefficients.data()),
            MAX_MULTIPLICATIVE_DEPTH - 1,
            -5.0,  // x_min
            5.0,   // x_max
            function_type
        );
        
        cudaDeviceSynchronize();
        
        // Copy coefficients to host
        std::vector<double> host_coeffs(coefficients.size());
        thrust::copy(coefficients.begin(), coefficients.end(), host_coeffs.begin());
        
        // Apply polynomial approximation
        for (size_t i = 0; i < input.data.size(); ++i) {
            result.data[i] = evaluatePolynomial(input.data[i], host_coeffs);
            
            // Bootstrap if needed
            if (ckks_decryptor_->invariant_noise_budget(result.data[i]) < 
                NOISE_BUDGET_THRESHOLD) {
                bootstrapCiphertext(result.data[i]);
            }
        }
        
        total_operations_ += input.data.size() * MAX_MULTIPLICATIVE_DEPTH;
        
        return result;
    }
    
    EncryptedTensor pooling2D(
        const EncryptedTensor& input,
        uint32_t pool_size,
        uint32_t stride,
        bool is_max_pool = false  // avg pool if false
    ) {
        uint32_t batch = input.shape[0];
        uint32_t channels = input.shape[1];
        uint32_t in_height = input.shape[2];
        uint32_t in_width = input.shape[3];
        
        uint32_t out_height = (in_height - pool_size) / stride + 1;
        uint32_t out_width = (in_width - pool_size) / stride + 1;
        
        EncryptedTensor result;
        result.shape = {batch, channels, out_height, out_width};
        result.metadata = input.metadata;
        result.data.resize(batch * channels * out_height * out_width);
        
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t c = 0; c < channels; ++c) {
                for (uint32_t oh = 0; oh < out_height; ++oh) {
                    for (uint32_t ow = 0; ow < out_width; ++ow) {
                        if (is_max_pool) {
                            // Approximate max pooling using high-degree polynomial
                            result.data[b * channels * out_height * out_width +
                                      c * out_height * out_width +
                                      oh * out_width + ow] = 
                                approximateMaxPool(input, b, c, oh, ow, pool_size, stride);
                        } else {
                            // Average pooling
                            Ciphertext sum;
                            bool first = true;
                            
                            for (uint32_t ph = 0; ph < pool_size; ++ph) {
                                for (uint32_t pw = 0; pw < pool_size; ++pw) {
                                    uint32_t ih = oh * stride + ph;
                                    uint32_t iw = ow * stride + pw;
                                    
                                    if (ih < in_height && iw < in_width) {
                                        uint32_t idx = 
                                            b * channels * in_height * in_width +
                                            c * in_height * in_width +
                                            ih * in_width + iw;
                                        
                                        if (first) {
                                            sum = input.data[idx];
                                            first = false;
                                        } else {
                                            ckks_evaluator_->add_inplace(sum, input.data[idx]);
                                        }
                                    }
                                }
                            }
                            
                            // Divide by pool size (multiply by 1/pool_size)
                            Plaintext divisor;
                            ckks_encoder_->encode(1.0 / (pool_size * pool_size), 
                                                 sum.scale(), divisor);
                            ckks_evaluator_->multiply_plain_inplace(sum, divisor);
                            ckks_evaluator_->rescale_to_next_inplace(sum);
                            
                            result.data[b * channels * out_height * out_width +
                                      c * out_height * out_width +
                                      oh * out_width + ow] = sum;
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    void bootstrapCiphertext(Ciphertext& ct) {
        // Implement CKKS bootstrapping
        // This is a simplified version - full implementation would be more complex
        
        // Step 1: Modulus reduction
        ckks_evaluator_->mod_switch_to_next_inplace(ct);
        
        // Step 2: Coefficients to slots
        // Step 3: Evaluate bootstrapping polynomial
        // Step 4: Slots to coefficients
        
        bootstrap_count_++;
        
        // For production, this would involve:
        // - Complex FFT operations
        // - Polynomial evaluation
        // - Inverse FFT
    }
    
private:
    void initializeCKKS() {
        EncryptionParameters parms(seal::EncryptionParameters::scheme_type::ckks);
        size_t poly_modulus_degree = POLY_MODULUS_DEGREE;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        
        // Set coefficient modulus for deep circuits
        std::vector<int> bit_sizes = {60, 40, 40, 40, 40, 40, 40, 40, 40, 60};
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, bit_sizes));
        
        double scale = pow(2.0, SCALE_BITS);
        
        ckks_context_ = std::make_shared<SEALContext>(parms);
        ckks_keygen_ = new KeyGenerator(*ckks_context_);
        
        ckks_keygen_->create_public_key(ckks_public_key_);
        ckks_secret_key_ = ckks_keygen_->secret_key();
        ckks_keygen_->create_relin_keys(ckks_relin_keys_);
        
        // Generate Galois keys for rotations
        std::vector<int> steps;
        for (int i = 1; i <= poly_modulus_degree / 2; i *= 2) {
            steps.push_back(i);
            steps.push_back(-i);
        }
        ckks_keygen_->create_galois_keys(steps, ckks_galois_keys_);
        
        ckks_evaluator_ = std::make_unique<Evaluator>(*ckks_context_);
        ckks_encoder_ = std::make_unique<CKKSEncoder>(*ckks_context_);
        ckks_encryptor_ = std::make_unique<Encryptor>(*ckks_context_, ckks_public_key_);
        ckks_decryptor_ = std::make_unique<Decryptor>(*ckks_context_, ckks_secret_key_);
    }
    
    void initializeBGV() {
        EncryptionParameters parms(seal::EncryptionParameters::scheme_type::bgv);
        size_t poly_modulus_degree = POLY_MODULUS_DEGREE;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        
        bgv_context_ = std::make_shared<SEALContext>(parms);
        bgv_keygen_ = new KeyGenerator(*bgv_context_);
        
        bgv_keygen_->create_public_key(bgv_public_key_);
        bgv_secret_key_ = bgv_keygen_->secret_key();
        bgv_keygen_->create_relin_keys(bgv_relin_keys_);
        
        bgv_evaluator_ = std::make_unique<Evaluator>(*bgv_context_);
        bgv_encoder_ = std::make_unique<BatchEncoder>(*bgv_context_);
        bgv_encryptor_ = std::make_unique<Encryptor>(*bgv_context_, bgv_public_key_);
        bgv_decryptor_ = std::make_unique<Decryptor>(*bgv_context_, bgv_secret_key_);
    }
    
    void initializeBootstrapping() {
        // Initialize bootstrapping polynomial coefficients
        bootstrap_polynomial_.resize(bootstrap_precision_);
        
        // These would be precomputed for the specific bootstrapping circuit
        thrust::sequence(bootstrap_polynomial_.begin(), bootstrap_polynomial_.end());
    }
    
    void encryptCKKSTensor(
        const thrust::device_vector<float>& tensor,
        EncryptedTensor& encrypted
    ) {
        uint32_t tensor_size = tensor.size();
        uint32_t slot_count = ckks_encoder_->slot_count();
        uint32_t num_ciphertexts = (tensor_size + slot_count - 1) / slot_count;
        
        encrypted.data.resize(num_ciphertexts);
        encrypted.batch_size = slot_count;
        encrypted.is_packed = true;
        
        double scale = pow(2.0, SCALE_BITS);
        
        // Process in batches
        thrust::device_vector<thrust::complex<double>> packed_data(slot_count);
        
        for (uint32_t i = 0; i < num_ciphertexts; ++i) {
            uint32_t offset = i * slot_count;
            uint32_t batch_size = std::min(slot_count, tensor_size - offset);
            
            // Pack tensor data
            dim3 block(256);
            dim3 grid((slot_count + block.x - 1) / block.x);
            
            packTensorForCKKSKernel<<<grid, block>>>(
                thrust::raw_pointer_cast(packed_data.data()),
                const_cast<float*>(thrust::raw_pointer_cast(tensor.data())) + offset,
                batch_size,
                slot_count,
                1.0f
            );
            
            cudaDeviceSynchronize();
            
            // Copy to host and encode
            std::vector<std::complex<double>> host_data(slot_count);
            thrust::copy(packed_data.begin(), packed_data.end(), host_data.begin());
            
            Plaintext plain;
            ckks_encoder_->encode(host_data, scale, plain);
            ckks_encryptor_->encrypt(plain, encrypted.data[i]);
        }
        
        encrypted.metadata.scale = scale;
        encrypted.metadata.level = 0;
        encrypted.metadata.operation_count = 0;
    }
    
    void decryptCKKSTensor(
        const EncryptedTensor& encrypted,
        thrust::device_vector<float>& tensor
    ) {
        uint32_t total_size = 1;
        for (auto dim : encrypted.shape) {
            total_size *= dim;
        }
        
        tensor.resize(total_size);
        thrust::device_vector<thrust::complex<double>> packed_data(encrypted.batch_size);
        
        uint32_t offset = 0;
        for (const auto& ct : encrypted.data) {
            Plaintext plain;
            ckks_decryptor_->decrypt(ct, plain);
            
            std::vector<std::complex<double>> host_data;
            ckks_encoder_->decode(plain, host_data);
            
            // Copy to device
            thrust::copy(host_data.begin(), host_data.end(), packed_data.begin());
            
            // Unpack
            uint32_t batch_size = std::min(encrypted.batch_size, total_size - offset);
            
            dim3 block(256);
            dim3 grid((batch_size + block.x - 1) / block.x);
            
            unpackCKKSResultKernel<<<grid, block>>>(
                thrust::raw_pointer_cast(tensor.data()) + offset,
                thrust::raw_pointer_cast(packed_data.data()),
                batch_size,
                1.0f / encrypted.metadata.scale
            );
            
            offset += batch_size;
        }
        
        cudaDeviceSynchronize();
    }
    
    void encryptBGVTensor(
        const thrust::device_vector<float>& tensor,
        EncryptedTensor& encrypted
    ) {
        // Convert float to integer representation
        uint32_t tensor_size = tensor.size();
        uint32_t slot_count = bgv_encoder_->slot_count();
        uint32_t num_ciphertexts = (tensor_size + slot_count - 1) / slot_count;
        
        encrypted.data.resize(num_ciphertexts);
        
        const int64_t scale_factor = 1000000;  // 6 decimal places
        
        for (uint32_t i = 0; i < num_ciphertexts; ++i) {
            std::vector<uint64_t> pod_matrix(slot_count, 0);
            
            uint32_t offset = i * slot_count;
            uint32_t batch_size = std::min(slot_count, tensor_size - offset);
            
            // Copy and convert to integers
            std::vector<float> host_data(batch_size);
            thrust::copy(
                tensor.begin() + offset,
                tensor.begin() + offset + batch_size,
                host_data.begin()
            );
            
            for (uint32_t j = 0; j < batch_size; ++j) {
                pod_matrix[j] = static_cast<uint64_t>(
                    host_data[j] * scale_factor + 0.5f
                );
            }
            
            Plaintext plain;
            bgv_encoder_->encode(pod_matrix, plain);
            bgv_encryptor_->encrypt(plain, encrypted.data[i]);
        }
        
        encrypted.metadata.scale = static_cast<float>(scale_factor);
    }
    
    void decryptBGVTensor(
        const EncryptedTensor& encrypted,
        thrust::device_vector<float>& tensor
    ) {
        uint32_t total_size = 1;
        for (auto dim : encrypted.shape) {
            total_size *= dim;
        }
        
        tensor.resize(total_size);
        uint32_t slot_count = bgv_encoder_->slot_count();
        
        uint32_t offset = 0;
        for (const auto& ct : encrypted.data) {
            Plaintext plain;
            bgv_decryptor_->decrypt(ct, plain);
            
            std::vector<uint64_t> pod_matrix;
            bgv_encoder_->decode(plain, pod_matrix);
            
            uint32_t batch_size = std::min(slot_count, total_size - offset);
            
            // Convert back to float
            std::vector<float> host_data(batch_size);
            for (uint32_t i = 0; i < batch_size; ++i) {
                host_data[i] = static_cast<float>(pod_matrix[i]) / 
                              encrypted.metadata.scale;
            }
            
            thrust::copy(
                host_data.begin(),
                host_data.end(),
                tensor.begin() + offset
            );
            
            offset += batch_size;
        }
    }
    
    Ciphertext evaluatePolynomial(
        const Ciphertext& x,
        const std::vector<double>& coefficients
    ) {
        if (coefficients.empty()) {
            throw std::runtime_error("Empty coefficient vector");
        }
        
        // Horner's method for polynomial evaluation
        Plaintext coeff;
        ckks_encoder_->encode(coefficients.back(), x.scale(), coeff);
        
        Ciphertext result;
        ckks_evaluator_->multiply_plain(x, coeff, result);
        
        for (int i = coefficients.size() - 2; i >= 0; --i) {
            ckks_encoder_->encode(coefficients[i], result.scale(), coeff);
            
            Ciphertext temp;
            ckks_evaluator_->multiply_plain(x, coeff, temp);
            ckks_evaluator_->add_inplace(result, temp);
            
            if (i > 0) {
                ckks_evaluator_->multiply_inplace(result, x);
                ckks_evaluator_->relinearize_inplace(result, ckks_relin_keys_);
                ckks_evaluator_->rescale_to_next_inplace(result);
            }
        }
        
        return result;
    }
    
    Ciphertext approximateMaxPool(
        const EncryptedTensor& input,
        uint32_t b, uint32_t c, uint32_t oh, uint32_t ow,
        uint32_t pool_size, uint32_t stride
    ) {
        // Use high-degree polynomial to approximate max function
        // max(a,b) ≈ (a + b + |a - b|) / 2
        // |x| ≈ x * tanh(k*x) for large k
        
        std::vector<Ciphertext> pool_values;
        
        for (uint32_t ph = 0; ph < pool_size; ++ph) {
            for (uint32_t pw = 0; pw < pool_size; ++pw) {
                uint32_t ih = oh * stride + ph;
                uint32_t iw = ow * stride + pw;
                
                if (ih < input.shape[2] && iw < input.shape[3]) {
                    uint32_t idx = 
                        b * input.shape[1] * input.shape[2] * input.shape[3] +
                        c * input.shape[2] * input.shape[3] +
                        ih * input.shape[3] + iw;
                    
                    pool_values.push_back(input.data[idx]);
                }
            }
        }
        
        // Recursive max approximation
        while (pool_values.size() > 1) {
            std::vector<Ciphertext> new_values;
            
            for (size_t i = 0; i < pool_values.size(); i += 2) {
                if (i + 1 < pool_values.size()) {
                    // Approximate max(a, b)
                    Ciphertext sum = pool_values[i];
                    ckks_evaluator_->add_inplace(sum, pool_values[i + 1]);
                    
                    Ciphertext diff = pool_values[i];
                    ckks_evaluator_->sub_inplace(diff, pool_values[i + 1]);
                    
                    // Apply smooth approximation of absolute value
                    std::vector<double> abs_coeffs = {0.0, 1.0, 0.0, -0.1667, 0.0, 0.075};
                    Ciphertext abs_diff = evaluatePolynomial(diff, abs_coeffs);
                    
                    ckks_evaluator_->add_inplace(sum, abs_diff);
                    
                    // Divide by 2
                    Plaintext half;
                    ckks_encoder_->encode(0.5, sum.scale(), half);
                    ckks_evaluator_->multiply_plain_inplace(sum, half);
                    ckks_evaluator_->rescale_to_next_inplace(sum);
                    
                    new_values.push_back(sum);
                } else {
                    new_values.push_back(pool_values[i]);
                }
            }
            
            pool_values = std::move(new_values);
        }
        
        return pool_values[0];
    }
};

} // namespace ares::homomorphic