/**
 * ARES Edge System - High-Performance Synaptic Models
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * Advanced synaptic models with SIMD optimization and sparse connectivity
 */

#include "neuromorphic_core.h"
#include <algorithm>
#include <unordered_map>
#include <tbb/parallel_sort.h>
#include <tbb/concurrent_vector.h>

namespace ares {
namespace neuromorphic {

/**
 * Sparse connectivity matrix with CSR format
 */
class SparseConnectivity {
private:
    std::vector<int> row_ptr;      // Row pointers (size: num_pre + 1)
    std::vector<int> col_indices;  // Column indices (size: num_connections)
    std::vector<int> syn_indices;  // Synapse indices for weight lookup
    int num_pre;
    int num_post;
    int num_connections;
    
public:
    SparseConnectivity(int n_pre, int n_post) 
        : num_pre(n_pre), num_post(n_post), num_connections(0) {
        row_ptr.resize(num_pre + 1, 0);
    }
    
    void build_from_pairs(const std::vector<std::pair<int, int>>& connections) {
        num_connections = connections.size();
        col_indices.resize(num_connections);
        syn_indices.resize(num_connections);
        
        // Sort connections by pre-synaptic neuron
        std::vector<std::pair<int, int>> sorted_conn = connections;
        tbb::parallel_sort(sorted_conn.begin(), sorted_conn.end());
        
        // Build CSR format
        int current_row = 0;
        for (int i = 0; i < num_connections; ++i) {
            int pre = sorted_conn[i].first;
            int post = sorted_conn[i].second;
            
            // Update row pointers
            while (current_row <= pre) {
                row_ptr[current_row++] = i;
            }
            
            col_indices[i] = post;
            syn_indices[i] = i;  // Direct mapping for now
        }
        
        // Fill remaining row pointers
        while (current_row <= num_pre) {
            row_ptr[current_row++] = num_connections;
        }
    }
    
    // Get connections from a pre-synaptic neuron
    inline void get_post_neurons(int pre_idx, int& start, int& end) const {
        start = row_ptr[pre_idx];
        end = row_ptr[pre_idx + 1];
    }
    
    inline int get_post_neuron(int conn_idx) const {
        return col_indices[conn_idx];
    }
    
    inline int get_synapse_index(int conn_idx) const {
        return syn_indices[conn_idx];
    }
    
    int get_num_connections() const { return num_connections; }
};

/**
 * Triplet STDP model for more accurate learning
 */
class TripletSTDP : public SynapticModel {
public:
    struct TripletParameters {
        double tau_plus = 16.8;    // ms - fast pre trace
        double tau_minus = 33.7;   // ms - fast post trace
        double tau_x = 101.0;      // ms - slow pre trace
        double tau_y = 125.0;      // ms - slow post trace
        double A2_plus = 5e-3;     // LTP amplitude (pair)
        double A3_plus = 6.2e-3;   // LTP amplitude (triplet)
        double A2_minus = 7e-3;    // LTD amplitude (pair)
        double A3_minus = 2.3e-4;  // LTD amplitude (triplet)
        double w_max = 1.0;        // Maximum weight
    };
    
private:
    TripletParameters params;
    std::vector<double> weights;
    std::vector<double> r1;  // Fast pre trace
    std::vector<double> r2;  // Slow pre trace  
    std::vector<double> o1;  // Fast post trace
    std::vector<double> o2;  // Slow post trace
    SparseConnectivity connectivity;
    
public:
    TripletSTDP(const SparseConnectivity& conn) 
        : connectivity(conn) {
        int n_syn = connectivity.get_num_connections();
        weights.resize(n_syn);
        r1.resize(n_syn, 0.0);
        r2.resize(n_syn, 0.0);
        o1.resize(n_syn, 0.0);
        o2.resize(n_syn, 0.0);
        
        // Initialize weights randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, params.w_max);
        
        #pragma omp parallel for
        for (int i = 0; i < n_syn; ++i) {
            weights[i] = dis(gen);
        }
    }
    
    void update_traces(double dt) override {
        const int n_syn = weights.size();
        const __m256d dt_vec = _mm256_set1_pd(dt);
        
        // Decay time constants
        const __m256d decay_r1 = _mm256_set1_pd(exp(-dt / params.tau_plus));
        const __m256d decay_r2 = _mm256_set1_pd(exp(-dt / params.tau_x));
        const __m256d decay_o1 = _mm256_set1_pd(exp(-dt / params.tau_minus));
        const __m256d decay_o2 = _mm256_set1_pd(exp(-dt / params.tau_y));
        
        #pragma omp parallel for
        for (int i = 0; i < n_syn; i += 4) {
            // Update traces with exponential decay
            __m256d r1_vec = _mm256_load_pd(&r1[i]);
            __m256d r2_vec = _mm256_load_pd(&r2[i]);
            __m256d o1_vec = _mm256_load_pd(&o1[i]);
            __m256d o2_vec = _mm256_load_pd(&o2[i]);
            
            r1_vec = _mm256_mul_pd(r1_vec, decay_r1);
            r2_vec = _mm256_mul_pd(r2_vec, decay_r2);
            o1_vec = _mm256_mul_pd(o1_vec, decay_o1);
            o2_vec = _mm256_mul_pd(o2_vec, decay_o2);
            
            _mm256_store_pd(&r1[i], r1_vec);
            _mm256_store_pd(&r2[i], r2_vec);
            _mm256_store_pd(&o1[i], o1_vec);
            _mm256_store_pd(&o2[i], o2_vec);
        }
    }
    
    void process_pre_spike(int synapse_idx) override {
        // Update pre-synaptic traces
        r1[synapse_idx] += 1.0;
        r2[synapse_idx] += 1.0;
        
        // LTD: Weight decrease depends on post-synaptic trace
        double dw = -params.A2_minus * o1[synapse_idx] - 
                    params.A3_minus * o1[synapse_idx] * r2[synapse_idx];
        
        weights[synapse_idx] = std::max(0.0, 
                                       std::min(weights[synapse_idx] + dw, 
                                              params.w_max));
    }
    
    void process_post_spike(int synapse_idx) override {
        // Update post-synaptic traces
        o1[synapse_idx] += 1.0;
        o2[synapse_idx] += 1.0;
        
        // LTP: Weight increase depends on pre-synaptic trace
        double dw = params.A2_plus * r1[synapse_idx] + 
                   params.A3_plus * r1[synapse_idx] * o2[synapse_idx];
        
        weights[synapse_idx] = std::max(0.0, 
                                       std::min(weights[synapse_idx] + dw, 
                                              params.w_max));
    }
    
    void propagate_spikes(const bool* pre_spiked, double* post_current) override {
        // Parallel propagation with sparse connectivity
        #pragma omp parallel for schedule(dynamic, 64)
        for (int pre = 0; pre < connectivity.num_pre; ++pre) {
            if (pre_spiked[pre]) {
                int start, end;
                connectivity.get_post_neurons(pre, start, end);
                
                for (int idx = start; idx < end; ++idx) {
                    int post = connectivity.get_post_neuron(idx);
                    int syn_idx = connectivity.get_synapse_index(idx);
                    
                    #pragma omp atomic
                    post_current[post] += weights[syn_idx];
                    
                    process_pre_spike(syn_idx);
                }
            }
        }
    }
};

/**
 * Short-term plasticity model (depression and facilitation)
 */
class ShortTermPlasticity : public SynapticModel {
private:
    struct STPParameters {
        double tau_rec = 100.0;   // ms - recovery time constant
        double tau_fac = 50.0;    // ms - facilitation time constant
        double U = 0.2;           // Initial release probability
        double f_dep = 0.7;       // Depression factor
        double f_fac = 1.2;       // Facilitation factor
    };
    
    STPParameters params;
    std::vector<double> weights;     // Base weights
    std::vector<double> u;           // Release probability
    std::vector<double> x;           // Available resources
    std::vector<double> last_spike;  // Last spike time for each synapse
    SparseConnectivity connectivity;
    double current_time = 0.0;
    
public:
    ShortTermPlasticity(const SparseConnectivity& conn) 
        : connectivity(conn) {
        int n_syn = connectivity.get_num_connections();
        weights.resize(n_syn);
        u.resize(n_syn, params.U);
        x.resize(n_syn, 1.0);
        last_spike.resize(n_syn, -1000.0);
        
        // Initialize weights
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.5, 0.1);
        
        #pragma omp parallel for
        for (int i = 0; i < n_syn; ++i) {
            weights[i] = std::max(0.0, dis(gen));
        }
    }
    
    void update_traces(double dt) override {
        current_time += dt;
        
        // Update recovery of synaptic resources
        const double decay_x = exp(-dt / params.tau_rec);
        const double decay_u = exp(-dt / params.tau_fac);
        
        #pragma omp parallel for
        for (size_t i = 0; i < x.size(); ++i) {
            // Resources recover to 1
            x[i] = 1.0 - (1.0 - x[i]) * decay_x;
            
            // Release probability decays to baseline
            u[i] = params.U + (u[i] - params.U) * decay_u;
        }
    }
    
    void process_pre_spike(int synapse_idx) override {
        double time_since_last = current_time - last_spike[synapse_idx];
        
        if (time_since_last > 1.0) {  // Refractory period
            // Update release probability (facilitation)
            u[synapse_idx] = u[synapse_idx] + params.U * (1.0 - u[synapse_idx]);
            u[synapse_idx] = std::min(u[synapse_idx] * params.f_fac, 1.0);
            
            // Update resources (depression)
            x[synapse_idx] *= params.f_dep;
            
            last_spike[synapse_idx] = current_time;
        }
    }
    
    void process_post_spike(int synapse_idx) override {
        // STP doesn't have post-synaptic plasticity
    }
    
    void propagate_spikes(const bool* pre_spiked, double* post_current) override {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int pre = 0; pre < connectivity.num_pre; ++pre) {
            if (pre_spiked[pre]) {
                int start, end;
                connectivity.get_post_neurons(pre, start, end);
                
                for (int idx = start; idx < end; ++idx) {
                    int post = connectivity.get_post_neuron(idx);
                    int syn_idx = connectivity.get_synapse_index(idx);
                    
                    // Effective weight = base weight * release prob * resources
                    double eff_weight = weights[syn_idx] * u[syn_idx] * x[syn_idx];
                    
                    #pragma omp atomic
                    post_current[post] += eff_weight;
                    
                    process_pre_spike(syn_idx);
                }
            }
        }
    }
    
    // Get effective synaptic strength
    std::vector<double> get_effective_weights() const {
        std::vector<double> eff_weights(weights.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            eff_weights[i] = weights[i] * u[i] * x[i];
        }
        
        return eff_weights;
    }
};

/**
 * Homeostatic plasticity for stability
 */
class HomeostaticPlasticity : public SynapticModel {
private:
    std::vector<double> weights;
    std::vector<double> target_rates;    // Target firing rates
    std::vector<double> actual_rates;    // Measured firing rates
    std::vector<double> rate_integral;   // Integrated rate error
    SparseConnectivity connectivity;
    
    double tau_homeo = 10000.0;  // ms - homeostatic time constant
    double learning_rate = 0.001;
    
public:
    HomeostaticPlasticity(const SparseConnectivity& conn,
                         const std::vector<double>& target_r) 
        : connectivity(conn), target_rates(target_r) {
        int n_syn = connectivity.get_num_connections();
        int n_post = target_rates.size();
        
        weights.resize(n_syn, 0.5);
        actual_rates.resize(n_post, 0.0);
        rate_integral.resize(n_post, 0.0);
    }
    
    void update_firing_rates(const bool* post_spiked, int n_post, double dt) {
        const double alpha = dt / tau_homeo;
        
        #pragma omp parallel for
        for (int i = 0; i < n_post; ++i) {
            // Exponential moving average of firing rate
            double instant_rate = post_spiked[i] ? 1000.0 / dt : 0.0;  // Hz
            actual_rates[i] = (1.0 - alpha) * actual_rates[i] + alpha * instant_rate;
            
            // Integrate rate error
            rate_integral[i] += (target_rates[i] - actual_rates[i]) * dt;
        }
    }
    
    void update_traces(double dt) override {
        // Update homeostatic scaling based on rate error
        #pragma omp parallel for
        for (int post = 0; post < actual_rates.size(); ++post) {
            double rate_error = target_rates[post] - actual_rates[post];
            double scaling = 1.0 + learning_rate * rate_error;
            
            // Scale all incoming weights to this neuron
            for (int pre = 0; pre < connectivity.num_pre; ++pre) {
                int start, end;
                connectivity.get_post_neurons(pre, start, end);
                
                for (int idx = start; idx < end; ++idx) {
                    if (connectivity.get_post_neuron(idx) == post) {
                        int syn_idx = connectivity.get_synapse_index(idx);
                        weights[syn_idx] *= scaling;
                        weights[syn_idx] = std::max(0.0, std::min(weights[syn_idx], 2.0));
                    }
                }
            }
        }
    }
    
    void process_pre_spike(int synapse_idx) override {
        // Homeostatic plasticity is activity-independent
    }
    
    void process_post_spike(int synapse_idx) override {
        // Homeostatic plasticity is activity-independent
    }
    
    void propagate_spikes(const bool* pre_spiked, double* post_current) override {
        #pragma omp parallel for schedule(static)
        for (int pre = 0; pre < connectivity.num_pre; ++pre) {
            if (pre_spiked[pre]) {
                int start, end;
                connectivity.get_post_neurons(pre, start, end);
                
                for (int idx = start; idx < end; ++idx) {
                    int post = connectivity.get_post_neuron(idx);
                    int syn_idx = connectivity.get_synapse_index(idx);
                    
                    #pragma omp atomic
                    post_current[post] += weights[syn_idx];
                }
            }
        }
    }
};

/**
 * Neuromodulated plasticity (dopamine, serotonin, etc.)
 */
class NeuromodulatedSTDP : public SynapticModel {
private:
    struct ModulatorParameters {
        double tau_dopamine = 200.0;     // ms
        double tau_eligibility = 1000.0; // ms
        double learning_threshold = 0.5;
        double baseline_da = 0.1;
    };
    
    ModulatorParameters mod_params;
    STDPParameters stdp_params;
    std::vector<double> weights;
    std::vector<double> eligibility_trace;  // Synaptic eligibility
    std::vector<double> dopamine_level;     // Global dopamine
    std::vector<double> pre_trace;
    std::vector<double> post_trace;
    SparseConnectivity connectivity;
    
public:
    NeuromodulatedSTDP(const SparseConnectivity& conn) 
        : connectivity(conn) {
        int n_syn = connectivity.get_num_connections();
        weights.resize(n_syn, 0.5);
        eligibility_trace.resize(n_syn, 0.0);
        dopamine_level.resize(1, mod_params.baseline_da);
        pre_trace.resize(n_syn, 0.0);
        post_trace.resize(n_syn, 0.0);
    }
    
    void set_dopamine_level(double da_level) {
        dopamine_level[0] = da_level;
    }
    
    void reward_signal(double reward) {
        // Phasic dopamine response to reward
        dopamine_level[0] += reward;
        dopamine_level[0] = std::max(0.0, std::min(dopamine_level[0], 1.0));
    }
    
    void update_traces(double dt) override {
        // Decay traces
        const double decay_pre = exp(-dt / stdp_params.tau_pre);
        const double decay_post = exp(-dt / stdp_params.tau_post);
        const double decay_elig = exp(-dt / mod_params.tau_eligibility);
        const double decay_da = exp(-dt / mod_params.tau_dopamine);
        
        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            pre_trace[i] *= decay_pre;
            post_trace[i] *= decay_post;
            eligibility_trace[i] *= decay_elig;
        }
        
        // Decay dopamine to baseline
        dopamine_level[0] = mod_params.baseline_da + 
                           (dopamine_level[0] - mod_params.baseline_da) * decay_da;
        
        // Update weights based on eligibility and dopamine
        const double da = dopamine_level[0];
        if (da > mod_params.learning_threshold) {
            #pragma omp parallel for
            for (size_t i = 0; i < weights.size(); ++i) {
                double dw = eligibility_trace[i] * (da - mod_params.baseline_da);
                weights[i] += dw;
                weights[i] = std::max(0.0, std::min(weights[i], stdp_params.w_max));
            }
        }
    }
    
    void process_pre_spike(int synapse_idx) override {
        pre_trace[synapse_idx] += 1.0;
        
        // Create eligibility trace (LTD component)
        eligibility_trace[synapse_idx] -= stdp_params.A_minus * post_trace[synapse_idx];
    }
    
    void process_post_spike(int synapse_idx) override {
        post_trace[synapse_idx] += 1.0;
        
        // Create eligibility trace (LTP component)
        eligibility_trace[synapse_idx] += stdp_params.A_plus * pre_trace[synapse_idx];
    }
    
    void propagate_spikes(const bool* pre_spiked, double* post_current) override {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int pre = 0; pre < connectivity.num_pre; ++pre) {
            if (pre_spiked[pre]) {
                int start, end;
                connectivity.get_post_neurons(pre, start, end);
                
                for (int idx = start; idx < end; ++idx) {
                    int post = connectivity.get_post_neuron(idx);
                    int syn_idx = connectivity.get_synapse_index(idx);
                    
                    #pragma omp atomic
                    post_current[post] += weights[syn_idx];
                    
                    process_pre_spike(syn_idx);
                }
            }
        }
    }
};

/**
 * Structural plasticity - dynamic synapse creation/removal
 */
class StructuralPlasticity {
private:
    struct DynamicSynapse {
        int pre;
        int post;
        double weight;
        double age;
        bool active;
    };
    
    tbb::concurrent_vector<DynamicSynapse> synapses;
    std::vector<double> pre_activity;
    std::vector<double> post_activity;
    
    double creation_threshold = 0.8;
    double pruning_threshold = 0.1;
    double max_synapses_per_neuron = 1000;
    
public:
    StructuralPlasticity(int num_pre, int num_post) {
        pre_activity.resize(num_pre, 0.0);
        post_activity.resize(num_post, 0.0);
    }
    
    void update_activity(const bool* pre_spiked, const bool* post_spiked,
                        int num_pre, int num_post, double dt) {
        const double tau_activity = 1000.0;  // ms
        const double decay = exp(-dt / tau_activity);
        
        #pragma omp parallel for
        for (int i = 0; i < num_pre; ++i) {
            pre_activity[i] = pre_activity[i] * decay + (pre_spiked[i] ? 1.0 : 0.0);
        }
        
        #pragma omp parallel for
        for (int i = 0; i < num_post; ++i) {
            post_activity[i] = post_activity[i] * decay + (post_spiked[i] ? 1.0 : 0.0);
        }
    }
    
    void structural_update(double dt) {
        // Age existing synapses
        #pragma omp parallel for
        for (size_t i = 0; i < synapses.size(); ++i) {
            if (synapses[i].active) {
                synapses[i].age += dt;
                
                // Prune weak/old synapses
                double pruning_prob = exp(-synapses[i].weight / pruning_threshold) * 
                                     (synapses[i].age / 10000.0);
                
                if (pruning_prob > 0.9) {
                    synapses[i].active = false;
                }
            }
        }
        
        // Create new synapses based on correlated activity
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int pre = 0; pre < pre_activity.size(); ++pre) {
            for (int post = 0; post < post_activity.size(); ++post) {
                double correlation = pre_activity[pre] * post_activity[post];
                
                if (correlation > creation_threshold && dis(gen) < 0.01) {
                    // Check if connection doesn't already exist
                    bool exists = false;
                    for (const auto& syn : synapses) {
                        if (syn.active && syn.pre == pre && syn.post == post) {
                            exists = true;
                            break;
                        }
                    }
                    
                    if (!exists) {
                        DynamicSynapse new_syn;
                        new_syn.pre = pre;
                        new_syn.post = post;
                        new_syn.weight = 0.1;  // Small initial weight
                        new_syn.age = 0.0;
                        new_syn.active = true;
                        
                        synapses.push_back(new_syn);
                    }
                }
            }
        }
    }
    
    SparseConnectivity get_current_connectivity() {
        std::vector<std::pair<int, int>> active_connections;
        
        for (const auto& syn : synapses) {
            if (syn.active) {
                active_connections.push_back({syn.pre, syn.post});
            }
        }
        
        SparseConnectivity conn(pre_activity.size(), post_activity.size());
        conn.build_from_pairs(active_connections);
        
        return conn;
    }
};

} // namespace neuromorphic
} // namespace ares