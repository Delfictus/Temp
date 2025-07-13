/**
 * ARES Edge System - C++ Neuromorphic Core
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * High-performance neuromorphic processing using Brian2 C++ code generation
 * with custom optimizations for real-time threat detection and swarm coordination.
 */

#ifndef ARES_NEUROMORPHIC_CORE_H
#define ARES_NEUROMORPHIC_CORE_H

#include <vector>
#include <array>
#include <memory>
#include <atomic>
#include <cmath>
#include <immintrin.h>  // SIMD intrinsics
#include <omp.h>        // OpenMP for parallelization

// CUDA interop for hybrid processing
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

namespace ares {
namespace neuromorphic {

// Constants for neuromorphic processing
constexpr double DT = 0.1;  // ms - simulation timestep
constexpr int WARP_SIZE = 32;  // For GPU alignment
constexpr int CACHE_LINE_SIZE = 64;  // For CPU cache optimization

// Forward declarations
class NeuronGroup;
class Synapses;
class NetworkMonitor;
class SpikeMonitor;
class StateMonitor;

/**
 * Neuron parameters for different models
 */
struct NeuronParameters {
    // LIF parameters
    double tau_m = 10.0;       // ms - membrane time constant
    double v_rest = -65.0;     // mV - resting potential
    double v_reset = -70.0;    // mV - reset potential
    double v_threshold = -50.0; // mV - spike threshold
    double refractory = 5.0;   // ms - refractory period
    
    // AdEx parameters
    double C = 281.0;          // pF - capacitance
    double g_L = 30.0;         // nS - leak conductance
    double E_L = -70.6;        // mV - leak reversal
    double V_T = -50.4;        // mV - threshold slope factor
    double Delta_T = 2.0;      // mV - slope factor
    double a = 4.0;            // nS - subthreshold adaptation
    double tau_w = 144.0;      // ms - adaptation time constant
    double b = 0.0805;         // nA - spike-triggered adaptation
    
    // EM sensor parameters
    double preferred_freq = 2.4e9;  // Hz - tuned frequency
    double tuning_width = 100e6;    // Hz - frequency selectivity
    
    // Chaos detector parameters
    double omega = 10.0;       // Hz - natural frequency
    double gamma = 0.1;        // Hz - damping
    double coupling = 0.5;     // coupling strength
};

/**
 * Base class for neuron models with SIMD optimizations
 */
class NeuronModel {
public:
    virtual ~NeuronModel() = default;
    
    // Pure virtual methods for neuron dynamics
    virtual void update_state(double* v, double* w, const double* I_ext, 
                            int N, double dt) = 0;
    virtual void check_threshold(const double* v, bool* spiked, int N) = 0;
    virtual void reset(double* v, double* w, const bool* spiked, int N) = 0;
    
protected:
    // SIMD helper functions
    inline __m256d exp_approx(__m256d x) {
        // Fast exponential approximation for neural computations
        __m256d result = _mm256_set1_pd(1.0);
        __m256d term = x;
        
        // Taylor series expansion (unrolled for performance)
        result = _mm256_add_pd(result, term);
        term = _mm256_mul_pd(term, _mm256_div_pd(x, _mm256_set1_pd(2.0)));
        result = _mm256_add_pd(result, term);
        term = _mm256_mul_pd(term, _mm256_div_pd(x, _mm256_set1_pd(3.0)));
        result = _mm256_add_pd(result, term);
        term = _mm256_mul_pd(term, _mm256_div_pd(x, _mm256_set1_pd(4.0)));
        result = _mm256_add_pd(result, term);
        
        return result;
    }
};

/**
 * Leaky Integrate-and-Fire neuron model with SIMD optimization
 */
class LIFNeuron : public NeuronModel {
private:
    NeuronParameters params;
    
public:
    LIFNeuron(const NeuronParameters& p) : params(p) {}
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const __m256d dt_vec = _mm256_set1_pd(dt);
        const __m256d tau_vec = _mm256_set1_pd(params.tau_m);
        const __m256d v_rest_vec = _mm256_set1_pd(params.v_rest);
        
        #pragma omp parallel for
        for (int i = 0; i < N; i += 4) {
            __m256d v_vec = _mm256_load_pd(&v[i]);
            __m256d I_vec = _mm256_load_pd(&I_ext[i]);
            
            // dv/dt = (v_rest - v + I) / tau
            __m256d dv = _mm256_div_pd(
                _mm256_add_pd(_mm256_sub_pd(v_rest_vec, v_vec), I_vec),
                tau_vec
            );
            
            // Euler integration
            v_vec = _mm256_add_pd(v_vec, _mm256_mul_pd(dv, dt_vec));
            _mm256_store_pd(&v[i], v_vec);
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
            }
        }
    }
};

/**
 * Adaptive Exponential neuron model with SIMD optimization
 */
class AdExNeuron : public NeuronModel {
private:
    NeuronParameters params;
    
public:
    AdExNeuron(const NeuronParameters& p) : params(p) {}
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const __m256d dt_vec = _mm256_set1_pd(dt);
        const __m256d C_vec = _mm256_set1_pd(params.C);
        const __m256d g_L_vec = _mm256_set1_pd(params.g_L);
        const __m256d E_L_vec = _mm256_set1_pd(params.E_L);
        const __m256d V_T_vec = _mm256_set1_pd(params.V_T);
        const __m256d Delta_T_vec = _mm256_set1_pd(params.Delta_T);
        const __m256d a_vec = _mm256_set1_pd(params.a);
        const __m256d tau_w_vec = _mm256_set1_pd(params.tau_w);
        
        #pragma omp parallel for
        for (int i = 0; i < N; i += 4) {
            __m256d v_vec = _mm256_load_pd(&v[i]);
            __m256d w_vec = _mm256_load_pd(&w[i]);
            __m256d I_vec = _mm256_load_pd(&I_ext[i]);
            
            // Exponential term: g_L * Delta_T * exp((v - V_T) / Delta_T)
            __m256d exp_arg = _mm256_div_pd(
                _mm256_sub_pd(v_vec, V_T_vec),
                Delta_T_vec
            );
            __m256d exp_term = _mm256_mul_pd(
                _mm256_mul_pd(g_L_vec, Delta_T_vec),
                exp_approx(exp_arg)
            );
            
            // dv/dt = (g_L*(E_L - v) + exp_term - w + I) / C
            __m256d dv = _mm256_div_pd(
                _mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_mul_pd(g_L_vec, _mm256_sub_pd(E_L_vec, v_vec)),
                        exp_term
                    ),
                    _mm256_sub_pd(I_vec, w_vec)
                ),
                C_vec
            );
            
            // dw/dt = (a*(v - E_L) - w) / tau_w
            __m256d dw = _mm256_div_pd(
                _mm256_sub_pd(
                    _mm256_mul_pd(a_vec, _mm256_sub_pd(v_vec, E_L_vec)),
                    w_vec
                ),
                tau_w_vec
            );
            
            // Euler integration
            v_vec = _mm256_add_pd(v_vec, _mm256_mul_pd(dv, dt_vec));
            w_vec = _mm256_add_pd(w_vec, _mm256_mul_pd(dw, dt_vec));
            
            _mm256_store_pd(&v[i], v_vec);
            _mm256_store_pd(&w[i], w_vec);
        }
    }
    
    void check_threshold(const double* v, bool* spiked, int N) override {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            spiked[i] = v[i] > 0.0;  // AdEx typically uses 0 mV threshold
        }
    }
    
    void reset(double* v, double* w, const bool* spiked, int N) override {
        const double v_reset = params.v_reset;
        const double b = params.b;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            if (spiked[i]) {
                v[i] = v_reset;
                w[i] += b;  // Spike-triggered adaptation
            }
        }
    }
};

/**
 * EM Sensor neuron model for RF spectrum processing
 */
class EMSensorNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> frequencies;  // Tuned frequencies for each neuron
    
public:
    EMSensorNeuron(const NeuronParameters& p, int N) : params(p) {
        // Initialize frequency tuning across spectrum
        frequencies.resize(N);
        double freq_min = 1e9;   // 1 GHz
        double freq_max = 6e9;   // 6 GHz
        double freq_step = (freq_max - freq_min) / (N - 1);
        
        for (int i = 0; i < N; ++i) {
            frequencies[i] = freq_min + i * freq_step;
        }
    }
    
    void process_em_spectrum(const double* spectrum_amplitudes, 
                           const double* spectrum_frequencies,
                           double* I_out, int N) {
        const double tuning_width_sq = params.tuning_width * params.tuning_width;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double I_em = 0.0;
            double preferred_freq = frequencies[i];
            
            // Gaussian tuning curve
            for (int j = 0; j < N; ++j) {
                double freq_diff = spectrum_frequencies[j] - preferred_freq;
                double tuning = exp(-(freq_diff * freq_diff) / (2.0 * tuning_width_sq));
                I_em += spectrum_amplitudes[j] * tuning;
            }
            
            I_out[i] = I_em;
        }
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        // Use LIF dynamics for EM sensors
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

/**
 * Chaos detector neuron with coupled oscillator dynamics
 */
class ChaosDetectorNeuron : public NeuronModel {
private:
    NeuronParameters params;
    std::vector<double> x;  // Oscillator state x
    std::vector<double> y;  // Oscillator state y
    
public:
    ChaosDetectorNeuron(const NeuronParameters& p, int N) : params(p) {
        x.resize(N, 0.0);
        y.resize(N, 0.0);
    }
    
    void update_state(double* v, double* w, const double* I_ext, 
                     int N, double dt) override {
        const double tau = params.tau_m;
        const double v_rest = params.v_rest;
        const double omega_sq = params.omega * params.omega;
        const double two_gamma = 2.0 * params.gamma;
        const double coupling = params.coupling;
        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            // Update oscillator dynamics
            double dx = y[i];
            double dy = -omega_sq * x[i] - two_gamma * y[i] + coupling * I_ext[i];
            
            x[i] += dx * dt;
            y[i] += dy * dt;
            
            // Chaos indicator feeds into neuron
            double I_chaos = x[i];
            
            // Update voltage with chaos input
            double dv = (v_rest - v[i] + I_ext[i] + I_chaos) / tau;
            v[i] += dv * dt;
        }
    }
    
    double compute_lyapunov_exponent(int neuron_idx, int time_steps = 1000) {
        // Simplified Lyapunov exponent calculation
        double sum_log_divergence = 0.0;
        double delta = 1e-6;
        
        // Save current state
        double x0 = x[neuron_idx];
        double y0 = y[neuron_idx];
        
        // Perturbed trajectory
        double x1 = x0 + delta;
        double y1 = y0;
        
        for (int t = 0; t < time_steps; ++t) {
            // Update reference trajectory
            double dx0 = y0;
            double dy0 = -params.omega * params.omega * x0 - 2 * params.gamma * y0;
            x0 += dx0 * DT;
            y0 += dy0 * DT;
            
            // Update perturbed trajectory
            double dx1 = y1;
            double dy1 = -params.omega * params.omega * x1 - 2 * params.gamma * y1;
            x1 += dx1 * DT;
            y1 += dy1 * DT;
            
            // Measure divergence
            double distance = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
            sum_log_divergence += log(distance / delta);
            
            // Renormalize
            x1 = x0 + delta * (x1 - x0) / distance;
            y1 = y0 + delta * (y1 - y0) / distance;
        }
        
        return sum_log_divergence / (time_steps * DT);
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
            }
        }
    }
};

/**
 * High-performance synaptic model with STDP
 */
class SynapticModel {
public:
    struct STDPParameters {
        double tau_pre = 20.0;   // ms
        double tau_post = 20.0;  // ms
        double A_plus = 0.01;
        double A_minus = -0.0105;
        double w_max = 1.0;      // mV
    };
    
private:
    STDPParameters stdp_params;
    std::vector<double> weights;
    std::vector<double> A_pre;   // Pre-synaptic trace
    std::vector<double> A_post;  // Post-synaptic trace
    std::vector<int> pre_indices;
    std::vector<int> post_indices;
    int num_synapses;
    
public:
    SynapticModel(int n_synapses) : num_synapses(n_synapses) {
        weights.resize(num_synapses);
        A_pre.resize(num_synapses, 0.0);
        A_post.resize(num_synapses, 0.0);
        pre_indices.resize(num_synapses);
        post_indices.resize(num_synapses);
        
        // Initialize random weights
        #pragma omp parallel for
        for (int i = 0; i < num_synapses; ++i) {
            weights[i] = stdp_params.w_max * (rand() / (double)RAND_MAX);
        }
    }
    
    void update_traces(double dt) {
        const __m256d dt_vec = _mm256_set1_pd(dt);
        const __m256d tau_pre_vec = _mm256_set1_pd(stdp_params.tau_pre);
        const __m256d tau_post_vec = _mm256_set1_pd(stdp_params.tau_post);
        
        #pragma omp parallel for
        for (int i = 0; i < num_synapses; i += 4) {
            // Update pre-synaptic trace
            __m256d A_pre_vec = _mm256_load_pd(&A_pre[i]);
            __m256d dA_pre = _mm256_div_pd(
                _mm256_sub_pd(_mm256_setzero_pd(), A_pre_vec),
                tau_pre_vec
            );
            A_pre_vec = _mm256_add_pd(A_pre_vec, _mm256_mul_pd(dA_pre, dt_vec));
            _mm256_store_pd(&A_pre[i], A_pre_vec);
            
            // Update post-synaptic trace
            __m256d A_post_vec = _mm256_load_pd(&A_post[i]);
            __m256d dA_post = _mm256_div_pd(
                _mm256_sub_pd(_mm256_setzero_pd(), A_post_vec),
                tau_post_vec
            );
            A_post_vec = _mm256_add_pd(A_post_vec, _mm256_mul_pd(dA_post, dt_vec));
            _mm256_store_pd(&A_post[i], A_post_vec);
        }
    }
    
    void process_pre_spike(int synapse_idx) {
        A_pre[synapse_idx] += stdp_params.A_plus;
        
        // LTD: Decrease weight based on post-synaptic trace
        double new_weight = weights[synapse_idx] + A_post[synapse_idx];
        weights[synapse_idx] = std::max(0.0, std::min(new_weight, stdp_params.w_max));
    }
    
    void process_post_spike(int synapse_idx) {
        A_post[synapse_idx] += stdp_params.A_minus;
        
        // LTP: Increase weight based on pre-synaptic trace
        double new_weight = weights[synapse_idx] + A_pre[synapse_idx];
        weights[synapse_idx] = std::max(0.0, std::min(new_weight, stdp_params.w_max));
    }
    
    void propagate_spikes(const bool* pre_spiked, double* post_current) {
        #pragma omp parallel for
        for (int i = 0; i < num_synapses; ++i) {
            if (pre_spiked[pre_indices[i]]) {
                #pragma omp atomic
                post_current[post_indices[i]] += weights[i];
                
                process_pre_spike(i);
            }
        }
    }
};

/**
 * Main neuromorphic network class
 */
class NeuromorphicNetwork {
private:
    // Network components
    std::vector<std::unique_ptr<NeuronModel>> neuron_groups;
    std::vector<std::unique_ptr<SynapticModel>> synapses;
    
    // State variables
    std::vector<std::vector<double>> voltages;
    std::vector<std::vector<double>> adaptations;
    std::vector<std::vector<bool>> spike_flags;
    std::vector<std::vector<double>> currents;
    
    // Monitoring
    std::vector<std::vector<int>> spike_times;
    std::vector<std::vector<int>> spike_indices;
    
    // Simulation parameters
    double current_time = 0.0;
    int time_step = 0;
    
public:
    NeuromorphicNetwork() {
        // Set OpenMP threads
        omp_set_num_threads(omp_get_max_threads());
    }
    
    int add_neuron_group(std::unique_ptr<NeuronModel> model, int size) {
        int group_id = neuron_groups.size();
        neuron_groups.push_back(std::move(model));
        
        voltages.emplace_back(size, -65.0);      // Initialize to rest
        adaptations.emplace_back(size, 0.0);
        spike_flags.emplace_back(size, false);
        currents.emplace_back(size, 0.0);
        spike_times.emplace_back();
        spike_indices.emplace_back();
        
        return group_id;
    }
    
    int add_synapses(int pre_group, int post_group, 
                     double connection_probability) {
        int pre_size = voltages[pre_group].size();
        int post_size = voltages[post_group].size();
        
        // Create sparse connectivity
        std::vector<std::pair<int, int>> connections;
        for (int i = 0; i < pre_size; ++i) {
            for (int j = 0; j < post_size; ++j) {
                if ((rand() / (double)RAND_MAX) < connection_probability) {
                    connections.push_back({i, j});
                }
            }
        }
        
        auto synapse_model = std::make_unique<SynapticModel>(connections.size());
        
        // Set connection indices
        #pragma omp parallel for
        for (size_t i = 0; i < connections.size(); ++i) {
            synapse_model->pre_indices[i] = connections[i].first;
            synapse_model->post_indices[i] = connections[i].second;
        }
        
        int synapse_id = synapses.size();
        synapses.push_back(std::move(synapse_model));
        
        return synapse_id;
    }
    
    void run(double duration_ms, bool record_spikes = true) {
        int num_steps = static_cast<int>(duration_ms / DT);
        
        for (int step = 0; step < num_steps; ++step) {
            // Clear currents
            for (auto& current_vec : currents) {
                std::fill(current_vec.begin(), current_vec.end(), 0.0);
            }
            
            // Update synaptic traces
            #pragma omp parallel for
            for (size_t syn_id = 0; syn_id < synapses.size(); ++syn_id) {
                synapses[syn_id]->update_traces(DT);
            }
            
            // Update neuron states
            #pragma omp parallel for
            for (size_t group_id = 0; group_id < neuron_groups.size(); ++group_id) {
                neuron_groups[group_id]->update_state(
                    voltages[group_id].data(),
                    adaptations[group_id].data(),
                    currents[group_id].data(),
                    voltages[group_id].size(),
                    DT
                );
                
                // Check for spikes
                neuron_groups[group_id]->check_threshold(
                    voltages[group_id].data(),
                    spike_flags[group_id].data(),
                    voltages[group_id].size()
                );
                
                // Record spikes
                if (record_spikes) {
                    for (size_t i = 0; i < spike_flags[group_id].size(); ++i) {
                        if (spike_flags[group_id][i]) {
                            spike_times[group_id].push_back(time_step);
                            spike_indices[group_id].push_back(i);
                        }
                    }
                }
                
                // Reset spiked neurons
                neuron_groups[group_id]->reset(
                    voltages[group_id].data(),
                    adaptations[group_id].data(),
                    spike_flags[group_id].data(),
                    voltages[group_id].size()
                );
            }
            
            // Propagate spikes through synapses
            for (size_t syn_id = 0; syn_id < synapses.size(); ++syn_id) {
                // TODO: Map synapse to pre/post groups
                // synapses[syn_id]->propagate_spikes(
                //     spike_flags[pre_group].data(),
                //     currents[post_group].data()
                // );
            }
            
            current_time += DT;
            time_step++;
        }
    }
    
    // Getters for monitoring
    const std::vector<int>& get_spike_times(int group_id) const {
        return spike_times[group_id];
    }
    
    const std::vector<int>& get_spike_indices(int group_id) const {
        return spike_indices[group_id];
    }
    
    const std::vector<double>& get_voltages(int group_id) const {
        return voltages[group_id];
    }
};

/**
 * Threat detection network implementation
 */
class ThreatDetectionNetwork {
private:
    NeuromorphicNetwork network;
    int input_layer_id;
    int hidden_layer_id;
    int output_layer_id;
    int input_to_hidden_syn;
    int hidden_to_output_syn;
    
public:
    ThreatDetectionNetwork(int input_size = 1000, 
                          int hidden_size = 500,
                          int output_size = 10) {
        // Create layers
        NeuronParameters em_params;
        auto em_sensors = std::make_unique<EMSensorNeuron>(em_params, input_size);
        input_layer_id = network.add_neuron_group(std::move(em_sensors), input_size);
        
        NeuronParameters adex_params;
        auto hidden_neurons = std::make_unique<AdExNeuron>(adex_params);
        hidden_layer_id = network.add_neuron_group(std::move(hidden_neurons), hidden_size);
        
        NeuronParameters lif_params;
        auto output_neurons = std::make_unique<LIFNeuron>(lif_params);
        output_layer_id = network.add_neuron_group(std::move(output_neurons), output_size);
        
        // Create connections
        input_to_hidden_syn = network.add_synapses(input_layer_id, hidden_layer_id, 0.1);
        hidden_to_output_syn = network.add_synapses(hidden_layer_id, output_layer_id, 0.2);
    }
    
    std::vector<double> process_em_spectrum(const double* spectrum_data, 
                                           int spectrum_size,
                                           double duration_ms = 50.0) {
        // TODO: Set input currents based on spectrum
        // network.set_external_current(input_layer_id, spectrum_data);
        
        // Run simulation
        network.run(duration_ms);
        
        // Analyze output spikes
        auto spike_indices = network.get_spike_indices(output_layer_id);
        std::vector<double> threat_scores(10, 0.0);
        
        for (int idx : spike_indices) {
            threat_scores[idx] += 1.0;
        }
        
        // Normalize
        double total = 0.0;
        for (double& score : threat_scores) {
            total += score;
        }
        
        if (total > 0) {
            for (double& score : threat_scores) {
                score /= total;
            }
        }
        
        return threat_scores;
    }
};

#ifdef USE_CUDA
/**
 * CUDA integration for hybrid neuromorphic/GPU processing
 */
class CUDANeuromorphicBridge {
private:
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    
public:
    CUDANeuromorphicBridge() {
        cudaStreamCreate(&stream);
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, stream);
    }
    
    ~CUDANeuromorphicBridge() {
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(stream);
    }
    
    void transfer_spikes_to_gpu(const std::vector<int>& spike_times,
                               const std::vector<int>& spike_indices,
                               float* d_spike_train,
                               int num_neurons,
                               int time_bins) {
        // Convert sparse spikes to dense GPU representation
        std::vector<float> spike_train(num_neurons * time_bins, 0.0f);
        
        for (size_t i = 0; i < spike_times.size(); ++i) {
            int time = spike_times[i];
            int neuron = spike_indices[i];
            if (time < time_bins) {
                spike_train[neuron * time_bins + time] = 1.0f;
            }
        }
        
        cudaMemcpyAsync(d_spike_train, spike_train.data(),
                       num_neurons * time_bins * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
};
#endif

} // namespace neuromorphic
} // namespace ares

#endif // ARES_NEUROMORPHIC_CORE_H