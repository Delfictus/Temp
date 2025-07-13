# ARES Neuromorphic System - Dataset Specifications

## Overview
This document specifies the datasets required for training and operating the Brian2-based neuromorphic components of the ARES Edge System.

## 1. Neuron Parameter Datasets

### 1.1 Biological Neuron Parameters
Source: Allen Brain Atlas, ModelDB, and neurophysiology literature

```json
{
  "cortical_pyramidal": {
    "tau_m": "10-30 ms",
    "v_rest": "-70 to -65 mV",
    "v_threshold": "-55 to -45 mV",
    "v_reset": "-75 to -65 mV",
    "refractory": "2-5 ms",
    "input_resistance": "100-500 MOhm"
  },
  "fast_spiking_interneuron": {
    "tau_m": "5-10 ms",
    "v_rest": "-65 mV",
    "v_threshold": "-45 mV",
    "v_reset": "-65 mV",
    "refractory": "1-2 ms",
    "max_firing_rate": "200-500 Hz"
  },
  "thalamic_relay": {
    "tau_m": "15-20 ms",
    "v_rest": "-70 mV",
    "v_threshold": "-50 mV",
    "IT_current": true,
    "burst_mode": true
  }
}
```

### 1.2 Synthetic Neuron Parameters for ARES
Optimized for threat detection and pattern recognition

```json
{
  "em_sensor_neuron": {
    "tau_m": "1-5 ms",
    "frequency_range": "1-100 GHz",
    "tuning_width": "10-1000 MHz",
    "sensitivity": "0.1-10 mV/dBm",
    "dynamic_range": "80 dB"
  },
  "pattern_detector": {
    "tau_m": "5-20 ms",
    "adaptation_rate": "0.1-1.0",
    "burst_threshold": "-40 mV",
    "pattern_memory": "100-1000 ms"
  },
  "decision_neuron": {
    "tau_m": "20-50 ms",
    "integration_window": "100-500 ms",
    "confidence_threshold": "0.7",
    "veto_inhibition": true
  }
}
```

## 2. EM Threat Signature Datasets

### 2.1 Radar Threats
- **Pulse Radar**: 10,000 samples
  - PRF: 100 Hz - 100 kHz
  - Pulse width: 0.1 - 100 μs
  - Frequency: 1 - 40 GHz
  
- **CW Radar**: 5,000 samples
  - Frequency: 1 - 100 GHz
  - Modulation: FMCW, FSK, PSK
  - Power: -30 to +40 dBm

- **Phased Array**: 3,000 samples
  - Beam patterns
  - Scan rates: 1 - 100 Hz
  - Multi-frequency operation

### 2.2 Communication Threats
- **Jamming Signals**: 15,000 samples
  - Barrage jamming
  - Spot jamming
  - Sweep jamming
  - Pulse jamming
  
- **Spoofing Signals**: 8,000 samples
  - GPS spoofing patterns
  - Communication protocol spoofing
  - Replay attacks

- **Interception Attempts**: 5,000 samples
  - Direction finding signals
  - Signal intelligence patterns

### 2.3 Electronic Warfare
- **EMP Precursors**: 2,000 samples
  - Rising edge characteristics
  - Frequency content
  - Power levels

- **Directed Energy**: 3,000 samples
  - Microwave weapons
  - Laser designation
  - RF weapons

## 3. Benign EM Pattern Datasets

### 3.1 Commercial Communications
- **Cellular (50,000 samples)**
  - 2G/3G/4G/5G patterns
  - Base station signals
  - Mobile device emissions

- **WiFi/Bluetooth (30,000 samples)**
  - 802.11 variants
  - Bluetooth Classic/LE
  - IoT protocols

- **Broadcast (20,000 samples)**
  - FM/AM radio
  - Television
  - Satellite communications

### 3.2 Natural EM Phenomena
- **Atmospheric (10,000 samples)**
  - Lightning
  - Ionospheric disturbances
  - Solar radiation

- **Environmental (15,000 samples)**
  - Urban EM noise
  - Industrial emissions
  - Power line harmonics

## 4. Swarm Coordination Datasets

### 4.1 Multi-Agent Scenarios
```python
swarm_scenarios = {
    "formation_flight": {
        "agents": 10-100,
        "duration": "5-30 minutes",
        "decisions": ["maintain", "adjust", "break"],
        "samples": 5000
    },
    "target_search": {
        "agents": 20-50,
        "area": "1-100 km²",
        "strategies": ["spiral", "grid", "random", "intelligent"],
        "samples": 3000
    },
    "defensive_swarm": {
        "agents": 30-200,
        "threat_types": 10,
        "responses": ["evade", "jam", "decoy", "attack"],
        "samples": 4000
    }
}
```

### 4.2 Communication Patterns
- **Consensus Protocols**: 10,000 samples
  - Byzantine agreement patterns
  - Voting mechanisms
  - Leader election

- **Information Propagation**: 8,000 samples
  - Gossip protocols
  - Flooding patterns
  - Hierarchical dissemination

## 5. Temporal Pattern Datasets

### 5.1 Attack Sequences
```json
{
  "reconnaissance_to_attack": {
    "duration": "1-60 minutes",
    "phases": ["scan", "probe", "exploit", "attack"],
    "samples": 2000
  },
  "coordinated_assault": {
    "duration": "10-300 seconds",
    "synchronization": "< 1 ms",
    "participants": "2-20",
    "samples": 1500
  },
  "adaptive_jamming": {
    "duration": "continuous",
    "adaptation_rate": "10-1000 Hz",
    "strategies": 8,
    "samples": 3000
  }
}
```

### 5.2 Normal Operations
- **Routine Communications**: 20,000 samples
  - Daily patterns
  - Weekly cycles
  - Seasonal variations

- **Maintenance Patterns**: 5,000 samples
  - System checks
  - Calibration sequences
  - Test patterns

## 6. Synaptic Weight Distributions

### 6.1 Initial Weight Distributions
```python
weight_distributions = {
    "excitatory": {
        "distribution": "lognormal",
        "mean": 0.5,
        "std": 0.25,
        "range": [0, 2.0]
    },
    "inhibitory": {
        "distribution": "normal",
        "mean": -1.0,
        "std": 0.3,
        "range": [-3.0, 0]
    },
    "modulatory": {
        "distribution": "uniform",
        "range": [-0.5, 0.5]
    }
}
```

### 6.2 Trained Weight Datasets
- **Threat Detection Networks**: Pre-trained weights for 50 threat types
- **Pattern Recognition**: Weights for 100+ EM patterns
- **Decision Networks**: Weights for tactical decision making

## 7. Network Topology Datasets

### 7.1 Biological-Inspired Topologies
```python
topologies = {
    "cortical_column": {
        "layers": 6,
        "neurons_per_layer": [1000, 500, 800, 400, 600, 300],
        "connectivity": "small_world",
        "connection_probability": 0.1
    },
    "thalamo_cortical": {
        "thalamic_neurons": 500,
        "cortical_neurons": 5000,
        "feedback_ratio": 0.3,
        "feedforward_ratio": 0.7
    },
    "hippocampal": {
        "ca3_neurons": 300,
        "ca1_neurons": 500,
        "dentate_neurons": 1000,
        "recurrent_connections": true
    }
}
```

### 7.2 Engineered Topologies
- **Hierarchical**: For command and control
- **Mesh**: For distributed processing
- **Hub-and-Spoke**: For centralized coordination
- **Random**: For robustness testing

## 8. Spike Train Datasets

### 8.1 Recorded Spike Trains
- **Biological Recordings**: 10,000 spike trains from various brain regions
- **Synthetic Patterns**: 50,000 generated spike trains with known statistics

### 8.2 Spike Train Statistics
```json
{
  "poisson": {
    "rate": "1-100 Hz",
    "samples": 10000
  },
  "bursting": {
    "burst_rate": "0.1-10 Hz",
    "spikes_per_burst": "2-20",
    "samples": 5000
  },
  "rhythmic": {
    "frequency": "1-100 Hz",
    "phase_locking": true,
    "samples": 8000
  }
}
```

## 9. Plasticity Rule Datasets

### 9.1 STDP Parameters
```python
stdp_params = {
    "classical_stdp": {
        "tau_pre": 20,  # ms
        "tau_post": 20,  # ms
        "A_plus": 0.01,
        "A_minus": -0.012
    },
    "triplet_stdp": {
        "tau_plus": 16.8,  # ms
        "tau_minus": 33.7,  # ms
        "tau_x": 101,  # ms
        "tau_y": 125  # ms
    },
    "voltage_dependent": {
        "theta_plus": -45,  # mV
        "theta_minus": -70,  # mV
        "voltage_dependence": "linear"
    }
}
```

## 10. Performance Benchmark Datasets

### 10.1 Latency Requirements
- **Threat Response**: < 100 ms
- **Pattern Recognition**: < 50 ms
- **Decision Making**: < 200 ms
- **Swarm Coordination**: < 500 ms

### 10.2 Accuracy Benchmarks
```json
{
  "threat_detection": {
    "true_positive_rate": "> 0.95",
    "false_positive_rate": "< 0.05",
    "test_samples": 10000
  },
  "pattern_classification": {
    "accuracy": "> 0.90",
    "classes": 50,
    "test_samples": 5000
  },
  "swarm_consensus": {
    "agreement_rate": "> 0.99",
    "byzantine_tolerance": "0.33",
    "test_scenarios": 1000
  }
}
```

## Data Format Specifications

### HDF5 Structure
```
/ares_neuromorphic_data/
├── neuron_parameters/
│   ├── biological/
│   └── synthetic/
├── em_signatures/
│   ├── threats/
│   └── benign/
├── spike_trains/
│   ├── recorded/
│   └── synthetic/
├── network_weights/
│   ├── initial/
│   └── trained/
└── benchmarks/
    ├── latency/
    └── accuracy/
```

### Metadata Requirements
Each dataset must include:
- Timestamp
- Source/generation method
- Sampling rate
- Units
- Validation status
- Security classification

## Usage Guidelines

1. **Training**: Use 70% for training, 15% for validation, 15% for testing
2. **Augmentation**: Apply noise, time shifts, and frequency shifts
3. **Balancing**: Ensure equal representation of all threat types
4. **Updates**: Refresh datasets monthly with new threat signatures
5. **Security**: Encrypt sensitive threat data, anonymize sources

## References

1. Allen Brain Atlas: brain-map.org
2. ModelDB: modeldb.yale.edu
3. IEEE EMC Society Database
4. DARPA Spectrum Collaboration Challenge Dataset
5. NATO Electronic Warfare Database (classified)