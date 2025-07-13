//===- BiologicalNeuronModels.td - Biologically-accurate models -*-===//
//
// ARES Edge System - Biological Neuron Model Definitions
// Copyright (c) 2024 DELFICTUS I/O LLC
//
// Based on neuroscience research (PMC6786860 principles) for
// biologically-accurate neuromorphic computing.
//
//===----------------------------------------------------------------------===//

#ifndef BIOLOGICAL_NEURON_MODELS_TD
#define BIOLOGICAL_NEURON_MODELS_TD

include "neuromorphic_dialect.td"

//===----------------------------------------------------------------------===//
// Biologically-accurate neuron models
//===----------------------------------------------------------------------===//

// Adaptive Exponential Integrate-and-Fire (AdEx)
// Brette & Gerstner (2005)
def BiologicalAdExModel : Neuro_NeuronModelAttr<"BiologicalAdEx", [
    // Membrane dynamics
    "dv/dt = (gL*(EL-v) + gL*DeltaT*exp((v-VT)/DeltaT) - w + I_syn)/C",
    "dw/dt = (a*(v-EL) - w)/tau_w",
    
    // Synaptic currents with AMPA, NMDA, GABA
    "I_syn = I_ampa + I_nmda + I_gaba",
    "I_ampa = g_ampa*(E_ampa-v)",
    "I_nmda = g_nmda*(E_nmda-v)/(1 + Mg*exp(-0.062*v)/3.57)",
    "I_gaba = g_gaba*(E_gaba-v)",
    
    // Synaptic conductance dynamics
    "dg_ampa/dt = -g_ampa/tau_ampa",
    "dg_nmda/dt = -g_nmda/tau_nmda",
    "dg_gaba/dt = -g_gaba/tau_gaba"
], [
    // Parameters from biological measurements
    "C": "281.0",          // pF - membrane capacitance
    "gL": "30.0",          // nS - leak conductance
    "EL": "-70.6",         // mV - leak reversal
    "VT": "-50.4",         // mV - threshold slope
    "DeltaT": "2.0",       // mV - slope factor
    "a": "4.0",            // nS - subthreshold adaptation
    "tau_w": "144.0",      // ms - adaptation time
    "b": "0.0805",         // nA - spike adaptation
    "V_reset": "-70.6",    // mV - reset potential
    "V_spike": "20.0",     // mV - spike cutoff
    "t_ref": "2.0",        // ms - refractory period
    
    // Synaptic parameters
    "E_ampa": "0.0",       // mV - AMPA reversal
    "E_nmda": "0.0",       // mV - NMDA reversal  
    "E_gaba": "-80.0",     // mV - GABA reversal
    "tau_ampa": "5.0",     // ms - AMPA decay
    "tau_nmda": "100.0",   // ms - NMDA decay
    "tau_gaba": "10.0",    // ms - GABA decay
    "Mg": "1.0"            // mM - Mg concentration
]> {
    let summary = "Biologically-accurate AdEx neuron with realistic synapses";
}

// Fast-spiking interneuron model
def FastSpikingInterneuron : Neuro_NeuronModelAttr<"FastSpiking", [
    "dv/dt = (gL*(EL-v) + gL*DeltaT*exp((v-VT)/DeltaT) - w + I_syn)/C",
    "dw/dt = (a*(v-EL) - w)/tau_w"
], [
    "C": "200.0",          // pF - smaller capacitance
    "gL": "20.0",          // nS
    "tau_w": "40.0",       // ms - faster adaptation
    "a": "0.1",            // nS - weak adaptation
    "b": "0.0"             // nA - no spike adaptation
]> {
    let summary = "Fast-spiking GABAergic interneuron";
}

// Hodgkin-Huxley model for detailed biophysics
def HodgkinHuxleyModel : Neuro_NeuronModelAttr<"HodgkinHuxley", [
    // Membrane potential
    "dV/dt = (I_ext - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L))/C",
    
    // Gating variables
    "dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m",
    "dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h",
    "dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n",
    
    // Rate functions
    "alpha_m(V) = 0.1*(V+40)/(1-exp(-(V+40)/10))",
    "beta_m(V) = 4*exp(-(V+65)/18)",
    "alpha_h(V) = 0.07*exp(-(V+65)/20)",
    "beta_h(V) = 1/(1+exp(-(V+35)/10))",
    "alpha_n(V) = 0.01*(V+55)/(1-exp(-(V+55)/10))",
    "beta_n(V) = 0.125*exp(-(V+65)/80)"
], [
    "C": "1.0",            // uF/cm^2
    "g_Na": "120.0",       // mS/cm^2
    "g_K": "36.0",         // mS/cm^2
    "g_L": "0.3",          // mS/cm^2
    "E_Na": "50.0",        // mV
    "E_K": "-77.0",        // mV
    "E_L": "-54.4"         // mV
]> {
    let summary = "Full Hodgkin-Huxley model with Na/K channels";
}

// Izhikevich model - computationally efficient
def IzhikevichModel : Neuro_NeuronModelAttr<"Izhikevich", [
    "dv/dt = 0.04*v^2 + 5*v + 140 - u + I",
    "du/dt = a*(b*v - u)"
], [
    "a": "0.02",           // Recovery time scale
    "b": "0.2",            // Sensitivity of recovery
    "c": "-65.0",          // After-spike reset
    "d": "8.0"             // After-spike recovery
]> {
    let summary = "Izhikevich model - efficient and versatile";
}

//===----------------------------------------------------------------------===//
// Synaptic plasticity models
//===----------------------------------------------------------------------===//

// Triplet STDP - more accurate than pair-based
def TripletSTDP : Neuro_PlasticityRuleAttr<"TripletSTDP", [
    // Pre-synaptic traces
    "tau_plus": "16.8",    // ms - fast pre trace
    "tau_x": "101.0",      // ms - slow pre trace
    
    // Post-synaptic traces  
    "tau_minus": "33.7",   // ms - fast post trace
    "tau_y": "125.0",      // ms - slow post trace
    
    // Learning rates
    "A2_plus": "5e-10",    // Pair LTP
    "A3_plus": "6.2e-3",   // Triplet LTP
    "A2_minus": "7e-3",    // Pair LTD
    "A3_minus": "2.3e-4"   // Triplet LTD
]> {
    let summary = "Triplet STDP (Pfister & Gerstner 2006)";
}

// Voltage-dependent STDP
def VoltageSTDP : Neuro_PlasticityRuleAttr<"VoltageSTDP", [
    "tau_plus": "20.0",
    "tau_minus": "20.0",
    "A_plus": "0.01",
    "A_minus": "-0.01",
    "theta_plus": "-45.0", // mV - LTP voltage threshold
    "theta_minus": "-70.0" // mV - LTD voltage threshold
]> {
    let summary = "Voltage-dependent STDP";
}

// Homeostatic plasticity
def HomeostaticPlasticity : Neuro_PlasticityRuleAttr<"Homeostatic", [
    "target_rate": "5.0",  // Hz - target firing rate
    "tau_homeo": "1000.0", // ms - homeostatic time constant
    "beta": "1.0"          // Scaling factor
]> {
    let summary = "Homeostatic synaptic scaling";
}

//===----------------------------------------------------------------------===//
// Specialized neuron types for ARES
//===----------------------------------------------------------------------===//

// Neuromorphic RF sensor neuron
def RFSensorNeuron : Neuro_NeuronModelAttr<"RFSensor", [
    "dv/dt = (v_rest - v + I_rf + I_noise)/tau",
    "I_rf = A * exp(-(f - f_preferred)^2 / (2*sigma_f^2)) * cos(2*pi*f*t + phi)",
    "I_noise = sigma_noise * xi(t)"  // Gaussian white noise
], [
    "tau": "10.0",         // ms
    "v_rest": "-65.0",     // mV
    "f_preferred": "2.4e9",// Hz - tuned frequency
    "sigma_f": "100e6",    // Hz - frequency tuning width
    "A": "10.0",           // Amplitude scaling
    "sigma_noise": "0.1"   // Noise level
]> {
    let summary = "RF spectrum sensor with frequency tuning";
}

// Chaos detection neuron
def ChaosNeuron : Neuro_NeuronModelAttr<"Chaos", [
    "dv/dt = (v_rest - v + I_input + I_chaos)/tau",
    "dx/dt = y",
    "dy/dt = -omega^2 * x - 2*gamma*y + coupling*I_input",
    "I_chaos = alpha * x"  // Chaos feeds into voltage
], [
    "tau": "20.0",         // ms
    "omega": "10.0",       // Natural frequency
    "gamma": "0.1",        // Damping
    "coupling": "0.5",     // Input coupling
    "alpha": "5.0"         // Chaos-to-current gain
]> {
    let summary = "Chaos detection via coupled oscillators";
}

// Grid cell for spatial navigation
def GridCellNeuron : Neuro_NeuronModelAttr<"GridCell", [
    "dv/dt = (v_rest - v + I_grid)/tau",
    "I_grid = sum_i(A * exp(-||r - r_i||^2 / (2*sigma^2)))",
    "r_i = lattice_point(i, spacing, angle)"  // Hexagonal lattice
], [
    "tau": "10.0",
    "spacing": "50.0",     // cm - grid spacing
    "angle": "0.0",        // rad - grid orientation
    "sigma": "10.0",       // cm - field width
    "A": "20.0"            // Field amplitude
]> {
    let summary = "Grid cell for spatial representation";
}

//===----------------------------------------------------------------------===//
// Network-level operations
//===----------------------------------------------------------------------===//

// Biologically-constrained network creation
def Neuro_BiologicalNetworkOp : Neuro_Op<"biological_network", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"NetworkReturnOp">
]> {
    let summary = "Create biologically-constrained network";
    let description = [{
        Creates a network with biological constraints:
        - 80/20 excitatory/inhibitory ratio
        - Distance-dependent connectivity
        - Dale's principle (neurons are either excitatory or inhibitory)
        - Realistic synaptic delays
    }];
    
    let arguments = (ins
        I64Attr:$num_excitatory,
        I64Attr:$num_inhibitory,
        F32Attr:$connection_radius,
        OptionalAttr<F32Attr>:$delay_velocity  // m/s
    );
    
    let regions = (region SizedRegion<1>:$body);
}

// Cortical column template
def Neuro_CorticalColumnOp : Neuro_Op<"cortical_column"> {
    let summary = "Create canonical cortical column";
    let description = [{
        Creates a cortical column with layer structure:
        - L2/3: 40% (pyramidal + interneurons)
        - L4: 20% (stellate + interneurons)
        - L5: 25% (pyramidal + interneurons)
        - L6: 15% (pyramidal + interneurons)
        
        Inter-layer connectivity follows biological patterns.
    }];
    
    let arguments = (ins
        I64Attr:$neurons_per_column,
        DefaultValuedAttr<F32Attr, "0.2">:$inhibitory_fraction
    );
    
    let results = (outs Neuro_NeuronGroupType:$column);
}

// Biological input patterns
def Neuro_PoissonInputOp : Neuro_Op<"poisson_input"> {
    let summary = "Generate Poisson spike trains";
    
    let arguments = (ins
        Neuro_NeuronGroupType:$target,
        F32Attr:$rate_hz,
        F32Attr:$weight,
        F32Attr:$connection_probability
    );
}

def Neuro_BurstInputOp : Neuro_Op<"burst_input"> {
    let summary = "Generate burst patterns";
    
    let arguments = (ins
        Neuro_NeuronGroupType:$target,
        F32Attr:$burst_rate_hz,
        I32Attr:$spikes_per_burst,
        F32Attr:$intra_burst_interval_ms
    );
}

#endif // BIOLOGICAL_NEURON_MODELS_TD
