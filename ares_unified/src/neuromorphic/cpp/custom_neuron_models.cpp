/**
 * ARES Edge System - Custom Neuron Models Implementation
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * High-performance neuron model implementations with SIMD optimizations
 */

#include "neuromorphic_core.h"
#include <cstring>
#include <random>
#include <chrono>

namespace ares {
namespace neuromorphic {

/**
 * Quantum-inspired neuron with superposition states
 */
class QuantumNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> psi_real;  // Real part of quantum state
    std::vector<double> psi_imag;  // Imaginary part of quantum state
    std::vector<double> phase;     // Phase accumulator
    double alpha = 1.0;            // Quantum coupling strength
    double omega = 10.0;           // Quantum oscillation frequency (Hz)
    
public:
    QuantumNeuron(const NeuronParameters& p, int N) : params(p) {
        psi_real.resize(N, 1.0);
        psi_imag.resize(N, 0.0);
        phase.resize(N, 0.0);
        
        // Initialize with random phases
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
        
        for (int i = 0; i < N; ++i) {
            phase[i] = dis(gen);
            psi_real[i] = cos(phase[i]);
            psi_imag[i] = sin(phase[i]);
        }
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const __m256d dt_vec = _mm256_set1_pd(dt);
        const __m256d tau_vec = _mm256_set1_pd(params.tau_m);
        const __m256d v_rest_vec = _mm256_set1_pd(params.v_rest);
        const __m256d omega_vec = _mm256_set1_pd(omega);
        const __m256d alpha_vec = _mm256_set1_pd(alpha);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i += 4) {
            // Load state variables
            __m256d v_vec = _mm256_load_pd(&v[i]);
            __m256d I_vec = _mm256_load_pd(&I_ext[i]);
            __m256d psi_r = _mm256_load_pd(&psi_real[i]);
            __m256d psi_i = _mm256_load_pd(&psi_imag[i]);
            
            // Update quantum state (rotation in complex plane)
            __m256d dpsi_r = _mm256_mul_pd(_mm256_mul_pd(omega_vec, psi_i), 
                                          _mm256_set1_pd(-1.0));
            __m256d dpsi_i = _mm256_mul_pd(omega_vec, psi_r);
            
            psi_r = _mm256_add_pd(psi_r, _mm256_mul_pd(dpsi_r, dt_vec));
            psi_i = _mm256_add_pd(psi_i, _mm256_mul_pd(dpsi_i, dt_vec));
            
            // Normalize to maintain unit magnitude
            __m256d magnitude_sq = _mm256_add_pd(
                _mm256_mul_pd(psi_r, psi_r),
                _mm256_mul_pd(psi_i, psi_i)
            );
            __m256d magnitude = _mm256_sqrt_pd(magnitude_sq);
            
            // Avoid division by zero
            __m256d mask = _mm256_cmp_pd(magnitude, _mm256_set1_pd(1e-10), _CMP_GT_OQ);
            psi_r = _mm256_blendv_pd(psi_r, _mm256_div_pd(psi_r, magnitude), mask);
            psi_i = _mm256_blendv_pd(psi_i, _mm256_div_pd(psi_i, magnitude), mask);
            
            // Quantum contribution to membrane potential
            __m256d I_quantum = _mm256_mul_pd(alpha_vec, magnitude_sq);
            
            // Update membrane potential
            __m256d dv = _mm256_div_pd(
                _mm256_add_pd(
                    _mm256_sub_pd(v_rest_vec, v_vec),
                    _mm256_add_pd(I_vec, I_quantum)
                ),
                tau_vec
            );
            
            v_vec = _mm256_add_pd(v_vec, _mm256_mul_pd(dv, dt_vec));
            
            // Store updated state
            _mm256_store_pd(&v[i], v_vec);
            _mm256_store_pd(&psi_real[i], psi_r);
            _mm256_store_pd(&psi_imag[i], psi_i);
        }
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        const double threshold = params.v_threshold;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            spiked[i] = v[i] > threshold;
        }
    }
    
    void reset(double* v, double* w, const bool* spiked, int N) override {
        const double v_reset = params.v_reset;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            if (spiked[i]) {
                v[i] = v_reset;
                // Quantum state undergoes phase shift on spike
                double phase_shift = M_PI / 4.0;
                double cos_shift = cos(phase_shift);
                double sin_shift = sin(phase_shift);
                
                double new_real = cos_shift * psi_real[i] - sin_shift * psi_imag[i];
                double new_imag = sin_shift * psi_real[i] + cos_shift * psi_imag[i];
                
                psi_real[i] = new_real;
                psi_imag[i] = new_imag;
            }
        }
    }
    
    // Get quantum coherence measure
    double get_coherence(int N) {
        double total_coherence = 0.0;
        
        #pragma omp parallel for reduction(+:total_coherence)
        for (int i = 0; i < N; ++i) {
            total_coherence += sqrt(psi_real[i] * psi_real[i] + 
                                  psi_imag[i] * psi_imag[i]);
        }
        
        return total_coherence / N;
    }
};

/**
 * Bistable neuron for decision-making
 */
class BistableNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> u;  // Adaptation variable
    double a = 1.0;         // Bistability parameter
    double b = 3.0;         // Cubic nonlinearity
    double noise_strength = 0.1;
    std::mt19937 rng;
    std::normal_distribution<double> noise_dist;
    
public:
    BistableNeuron(const NeuronParameters& p, int N) 
        : params(p), rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          noise_dist(0.0, 1.0) {
        u.resize(N, 0.0);
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const double tau = params.tau_m;
        const double sqrt_dt = sqrt(dt);
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Bistable dynamics: dv/dt = v - v³ + I + noise
            double v_cubed = v[i] * v[i] * v[i];
            double noise = noise_strength * noise_dist(rng) * sqrt_dt;
            
            double dv = (a * v[i] - b * v_cubed + I_ext[i] + noise) / tau;
            v[i] += dv * dt;
            
            // Update adaptation
            double du = (v[i] - u[i]) / (10.0 * tau);
            u[i] += du * dt;
        }
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        // Bistable neurons spike when transitioning between states
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Detect state transitions
            double threshold = 0.5;  // Between stable states at ±1
            spiked[i] = (v[i] > threshold && u[i] < threshold) ||
                       (v[i] < -threshold && u[i] > -threshold);
        }
    }
    
    void reset(double* v, double* w, const bool* spiked, int N) override {
        // No explicit reset for bistable neurons
        // They naturally relax to stable states
    }
    
    // Get decision state (-1 or +1)
    std::vector<int> get_decisions(const double* v, int N) {
        std::vector<int> decisions(N);
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            decisions[i] = (v[i] > 0.0) ? 1 : -1;
        }
        
        return decisions;
    }
};

/**
 * Pattern-selective neuron for template matching
 */
class PatternNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<std::vector<double>> templates;  // Stored patterns
    std::vector<double> match_scores;            // Pattern match scores
    int num_patterns;
    int pattern_length;
    
public:
    PatternNeuron(const NeuronParameters& p, int N, int n_patterns, int p_length) 
        : params(p), num_patterns(n_patterns), pattern_length(p_length) {
        templates.resize(num_patterns);
        match_scores.resize(N, 0.0);
        
        // Initialize random patterns
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        for (int i = 0; i < num_patterns; ++i) {
            templates[i].resize(pattern_length);
            for (int j = 0; j < pattern_length; ++j) {
                templates[i][j] = dis(gen);
            }
        }
    }
    
    void process_input_pattern(const double* input_pattern, int N) {
        #pragma omp parallel for
        for (int neuron = 0; neuron < N; ++neuron) {
            // Each neuron responds to a different template
            int template_idx = neuron % num_patterns;
            
            // Compute dot product (pattern matching)
            double score = 0.0;
            
            // Vectorized dot product
            int i = 0;
            for (; i <= pattern_length - 4; i += 4) {
                __m256d template_vec = _mm256_loadu_pd(&templates[template_idx][i]);
                __m256d input_vec = _mm256_loadu_pd(&input_pattern[i]);
                __m256d prod = _mm256_mul_pd(template_vec, input_vec);
                
                // Horizontal sum
                __m128d sum_high = _mm256_extractf128_pd(prod, 1);
                __m128d sum_low = _mm256_castpd256_pd128(prod);
                __m128d sum1 = _mm_add_pd(sum_low, sum_high);
                __m128d sum2 = _mm_hadd_pd(sum1, sum1);
                score += _mm_cvtsd_f64(sum2);
            }
            
            // Handle remaining elements
            for (; i < pattern_length; ++i) {
                score += templates[template_idx][i] * input_pattern[i];
            }
            
            // Normalize by pattern length
            match_scores[neuron] = score / pattern_length;
        }
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const double tau = params.tau_m;
        const double v_rest = params.v_rest;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Pattern match score modulates input current
            double I_pattern = match_scores[i] * 10.0;  // Scaling factor
            
            // Update voltage with pattern-modulated input
            double dv = (v_rest - v[i] + I_ext[i] + I_pattern) / tau;
            v[i] += dv * dt;
        }
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        const double threshold = params.v_threshold;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            spiked[i] = v[i] > threshold;
        }
    }
    
    void reset(double* v, double* w, const bool* spiked, int N) override {
        const double v_reset = params.v_reset;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            if (spiked[i]) {
                v[i] = v_reset;
                match_scores[i] *= 0.9;  // Decay match score after spike
            }
        }
    }
    
    // Learn new pattern
    void learn_pattern(int pattern_idx, const double* new_pattern) {
        if (pattern_idx >= 0 && pattern_idx < num_patterns) {
            // Online learning with momentum
            const double learning_rate = 0.1;
            
            #pragma omp parallel for
            for (int i = 0; i < pattern_length; ++i) {
                templates[pattern_idx][i] = (1.0 - learning_rate) * templates[pattern_idx][i] +
                                          learning_rate * new_pattern[i];
            }
            
            // Normalize pattern
            double norm = 0.0;
            for (int i = 0; i < pattern_length; ++i) {
                norm += templates[pattern_idx][i] * templates[pattern_idx][i];
            }
            norm = sqrt(norm);
            
            if (norm > 1e-6) {
                for (int i = 0; i < pattern_length; ++i) {
                    templates[pattern_idx][i] /= norm;
                }
            }
        }
    }
};

/**
 * Resonator neuron for frequency detection
 */
class ResonatorNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> w;  // Resonator variable
    std::vector<double> resonant_freq;  // Preferred frequency for each neuron
    double damping = 0.1;
    
public:
    ResonatorNeuron(const NeuronParameters& p, int N) : params(p) {
        w.resize(N, 0.0);
        resonant_freq.resize(N);
        
        // Initialize with different resonant frequencies
        double f_min = 1.0;   // Hz
        double f_max = 100.0; // Hz
        double f_step = (f_max - f_min) / (N - 1);
        
        for (int i = 0; i < N; ++i) {
            resonant_freq[i] = f_min + i * f_step;
        }
    }
    
    void update_state(double* v, double* w_state, const double* I_ext, 
                     int N, double dt) override {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double omega = 2.0 * M_PI * resonant_freq[i];
            double omega_sq = omega * omega;
            
            // Resonator dynamics
            double dv = w[i];
            double dw = -omega_sq * v[i] - 2.0 * damping * omega * w[i] + I_ext[i];
            
            v[i] += dv * dt;
            w[i] += dw * dt;
            
            // Limit amplitude to prevent runaway oscillations
            double amplitude = sqrt(v[i] * v[i] + w[i] * w[i]);
            if (amplitude > 10.0) {
                v[i] *= 10.0 / amplitude;
                w[i] *= 10.0 / amplitude;
            }
        }
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        // Spike when oscillation crosses threshold
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            spiked[i] = (v[i] > params.v_threshold) && (w[i] > 0);
        }
    }
    
    void reset(double* v, double* w_state, const bool* spiked, int N) override {
        // No hard reset for resonators, just damping
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            if (spiked[i]) {
                w[i] *= 0.8;  // Reduce oscillation energy
            }
        }
    }
    
    // Get dominant frequency for each neuron
    std::vector<double> get_dominant_frequencies(const double* v, int N, 
                                                double sample_rate = 1000.0) {
        std::vector<double> frequencies(N);
        
        // Simple zero-crossing based frequency estimation
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Count oscillations based on phase
            double phase = atan2(w[i], v[i]);
            frequencies[i] = resonant_freq[i];  // For now, return resonant frequency
        }
        
        return frequencies;
    }
};

/**
 * Burst neuron for rhythmic pattern generation
 */
class BurstNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> h;       // Slow inactivation variable
    std::vector<double> n;       // Recovery variable
    std::vector<int> burst_count; // Spikes within current burst
    double V_T = -40.0;          // Threshold for burst initiation
    double V_h = -60.0;          // Half-activation for h
    
public:
    BurstNeuron(const NeuronParameters& p, int N) : params(p) {
        h.resize(N, 0.8);
        n.resize(N, 0.0);
        burst_count.resize(N, 0);
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const double tau_h = 200.0;  // Slow time constant
        const double tau_n = 10.0;   // Fast time constant
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Burst dynamics with slow and fast variables
            double v_val = v[i];
            
            // Activation functions
            double h_inf = 1.0 / (1.0 + exp((v_val - V_h) / 5.0));
            double n_inf = 1.0 / (1.0 + exp(-(v_val - V_T) / 5.0));
            
            // Update slow inactivation
            double dh = (h_inf - h[i]) / tau_h;
            h[i] += dh * dt;
            
            // Update fast recovery
            double dn = (n_inf - n[i]) / tau_n;
            n[i] += dn * dt;
            
            // Burst current
            double I_burst = 0.0;
            if (h[i] > 0.5 && n[i] > 0.5) {
                I_burst = 20.0;  // Strong depolarizing current during burst
            }
            
            // Update voltage
            double dv = (params.v_rest - v_val + I_ext[i] + I_burst) / params.tau_m;
            v[i] += dv * dt;
        }
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Spike if above threshold and in burst mode
            spiked[i] = (v[i] > params.v_threshold) && (h[i] > 0.5);
            
            if (spiked[i]) {
                burst_count[i]++;
            } else if (burst_count[i] > 0 && h[i] < 0.5) {
                // Burst ended
                burst_count[i] = 0;
            }
        }
    }
    
    void reset(double* v, double* w, const bool* spiked, int N) override {
        const double v_reset = params.v_reset;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            if (spiked[i]) {
                v[i] = v_reset;
                
                // Partial inactivation after each spike in burst
                h[i] *= 0.95;
            }
        }
    }
    
    // Get burst statistics
    std::vector<int> get_burst_counts(int N) {
        return burst_count;
    }
};

/**
 * Grid cell neuron for spatial navigation
 */
class GridCellNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> grid_phase_x;
    std::vector<double> grid_phase_y;
    std::vector<double> grid_spacing;
    std::vector<double> grid_orientation;
    
public:
    GridCellNeuron(const NeuronParameters& p, int N) : params(p) {
        grid_phase_x.resize(N);
        grid_phase_y.resize(N);
        grid_spacing.resize(N);
        grid_orientation.resize(N);
        
        // Initialize grid parameters
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> phase_dis(0.0, 1.0);
        std::uniform_real_distribution<> orient_dis(0.0, M_PI / 3.0);
        
        for (int i = 0; i < N; ++i) {
            // Different grid spacings (30cm to 3m)
            grid_spacing[i] = 0.3 * pow(2.0, i * 3.0 / N);
            grid_phase_x[i] = phase_dis(gen) * grid_spacing[i];
            grid_phase_y[i] = phase_dis(gen) * grid_spacing[i];
            grid_orientation[i] = orient_dis(gen);
        }
    }
    
    void compute_grid_input(double x, double y, double* grid_input, int N) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double spacing = grid_spacing[i];
            double orientation = grid_orientation[i];
            
            // Rotate coordinates
            double x_rot = x * cos(orientation) + y * sin(orientation);
            double y_rot = -x * sin(orientation) + y * cos(orientation);
            
            // Hexagonal grid pattern
            double phase_1 = 2.0 * M_PI * (x_rot - grid_phase_x[i]) / spacing;
            double phase_2 = 2.0 * M_PI * (y_rot - grid_phase_y[i]) / spacing;
            double phase_3 = phase_1 - phase_2;
            
            // Sum of three cosines gives hexagonal pattern
            grid_input[i] = (cos(phase_1) + cos(phase_2) + cos(phase_3)) / 3.0 + 1.0;
            grid_input[i] *= 10.0;  // Scale to appropriate current range
        }
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        // Standard LIF dynamics with grid-modulated input
        LIFNeuron lif(params);
        lif.update_state(v, w, I_ext, N, dt);
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        LIFNeuron lif(params);
        lif.check_threshold(v, spiked, N);
    }
    
    void reset(double* v, double* w, const bool* spiked, int N) override {
        LIFNeuron lif(params);
        lif.reset(v, w, spiked, N);
    }
};

} // namespace neuromorphic
} // namespace ares