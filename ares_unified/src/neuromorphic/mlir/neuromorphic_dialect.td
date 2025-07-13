//===- NeuromorphicDialect.td - Neuromorphic dialect ------*- tablegen -*-===//
//
// ARES Edge System - MLIR Neuromorphic Dialect
// Copyright (c) 2024 DELFICTUS I/O LLC
//
// This file defines the neuromorphic dialect for MLIR, enabling high-level
// representation of spiking neural networks with automatic optimization and
// hardware targeting.
//
//===----------------------------------------------------------------------===//

#ifndef NEUROMORPHIC_DIALECT
#define NEUROMORPHIC_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Neuromorphic dialect definition
//===----------------------------------------------------------------------===//

def Neuromorphic_Dialect : Dialect {
  let name = "neuro";
  let summary = "ARES neuromorphic MLIR dialect";
  let description = [{
    The neuromorphic dialect provides high-level abstractions for spiking
    neural networks, enabling automatic optimization across different hardware
    targets including CPU, GPU, TPU, and neuromorphic chips.
    
    Key concepts:
    - Neurons with biologically-inspired dynamics
    - Sparse spike events with temporal information
    - Plastic synapses with online learning
    - Efficient mapping to hardware accelerators
  }];
  
  let cppNamespace = "::ares::mlir::neuromorphic";
  
  // Enable custom types and attributes
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Neuromorphic types
//===----------------------------------------------------------------------===//

def Neuro_SpikeEventType : TypeDef<Neuromorphic_Dialect, "SpikeEvent"> {
  let summary = "Spike event type";
  let description = [{
    Represents a spike event with neuron ID, timestamp, and weight.
    Optimized for sparse representation and efficient hardware mapping.
  }];
  
  let parameters = (ins
    "unsigned":$neuronBits,
    "unsigned":$timestampBits,
    "Type":$weightType
  );
  
  let assemblyFormat = "`<` $neuronBits `,` $timestampBits `,` $weightType `>`";
}

def Neuro_NeuronGroupType : TypeDef<Neuromorphic_Dialect, "NeuronGroup"> {
  let summary = "Group of neurons with shared dynamics";
  let parameters = (ins
    "StringAttr":$modelType,
    "unsigned":$size,
    "DictionaryAttr":$parameters
  );
}

//===----------------------------------------------------------------------===//
// Neuromorphic attributes
//===----------------------------------------------------------------------===//

def Neuro_NeuronModelAttr : AttrDef<Neuromorphic_Dialect, "NeuronModel"> {
  let summary = "Neuron model specification";
  let parameters = (ins
    StringRefParameter<"model name">:$name,
    ArrayRefParameter<"state variables">:$stateVars,
    ArrayRefParameter<"parameters">:$parameters
  );
}

def Neuro_PlasticityRuleAttr : AttrDef<Neuromorphic_Dialect, "PlasticityRule"> {
  let summary = "Synaptic plasticity rule";
  let parameters = (ins
    StringRefParameter<"rule type">:$type,
    "DictionaryAttr":$parameters
  );
}

//===----------------------------------------------------------------------===//
// Neuromorphic operations
//===----------------------------------------------------------------------===//

class Neuro_Op<string mnemonic, list<Trait> traits = []> :
    Op<Neuromorphic_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Neuron operations
//===----------------------------------------------------------------------===//

def Neuro_CreateNeuronsOp : Neuro_Op<"create_neurons",
    [Pure, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Create a group of neurons";
  let description = [{
    Creates a group of neurons with specified model and parameters.
    This operation is optimized to map efficiently to different hardware:
    - CPU: Structure-of-arrays layout for SIMD
    - GPU: Coalesced memory access patterns
    - TPU: INT8 quantized states
    - Neuromorphic: Direct hardware neuron allocation
  }];
  
  let arguments = (ins
    Neuro_NeuronModelAttr:$model,
    I64Attr:$count,
    OptionalAttr<DictionaryAttr>:$initial_state
  );
  
  let results = (outs Neuro_NeuronGroupType:$neurons);
  
  let assemblyFormat = [{
    $model `count` $count (`init` $initial_state^)? attr-dict `:` type($neurons)
  }];
  
  let hasVerifier = 1;
}

def Neuro_UpdateNeuronsOp : Neuro_Op<"update_neurons"> {
  let summary = "Update neuron states for one timestep";
  let description = [{
    Updates all neurons in a group for one simulation timestep.
    Automatically selects optimal implementation:
    - CPU: Vectorized updates with AVX/NEON
    - GPU: Parallel thread per neuron
    - TPU: Systolic array computation
  }];
  
  let arguments = (ins
    Neuro_NeuronGroupType:$neurons,
    F32Attr:$dt,
    Variadic<AnyTensor>:$external_inputs
  );
  
  let results = (outs Neuro_NeuronGroupType:$updated_neurons);
}

def Neuro_CheckThresholdOp : Neuro_Op<"check_threshold",
    [Pure, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Check which neurons spiked";
  let description = [{
    Determines which neurons crossed threshold and generates spike events.
    Hardware-specific optimizations:
    - CPU: Bitmask operations
    - GPU: Ballot intrinsics
    - TPU: Comparison in INT8
  }];
  
  let arguments = (ins Neuro_NeuronGroupType:$neurons);
  let results = (outs Variadic<Neuro_SpikeEventType>:$spikes);
}

//===----------------------------------------------------------------------===//
// Synapse operations
//===----------------------------------------------------------------------===//

def Neuro_CreateSynapsesOp : Neuro_Op<"create_synapses"> {
  let summary = "Create synaptic connections";
  let description = [{
    Creates synapses between neuron groups with specified connectivity.
    Optimized sparse representations for each target:
    - CPU: CSR format
    - GPU: COO format for dynamic updates
    - TPU: Dense tiles for systolic array
  }];
  
  let arguments = (ins
    Neuro_NeuronGroupType:$pre_neurons,
    Neuro_NeuronGroupType:$post_neurons,
    F32Attr:$connection_probability,
    OptionalAttr<Neuro_PlasticityRuleAttr>:$plasticity,
    DefaultValuedAttr<F32Attr, "1.0">:$initial_weight
  );
  
  let results = (outs AnyTensor:$synapses);
}

def Neuro_PropagateSpikeOp : Neuro_Op<"propagate_spike"> {
  let summary = "Propagate spikes through synapses";
  let description = [{
    Propagates spike events through synaptic connections.
    Hardware-optimized implementations:
    - CPU: Sparse matrix-vector multiply
    - GPU: Atomic additions
    - TPU: Dense matrix operations with masking
  }];
  
  let arguments = (ins
    Variadic<Neuro_SpikeEventType>:$spikes,
    AnyTensor:$synapses,
    Neuro_NeuronGroupType:$target_neurons
  );
  
  let results = (outs Neuro_NeuronGroupType:$updated_neurons);
}

def Neuro_UpdateWeightsOp : Neuro_Op<"update_weights"> {
  let summary = "Update synaptic weights based on plasticity";
  let description = [{
    Updates weights according to plasticity rule (STDP, etc).
    Leverages hardware capabilities:
    - CPU: Vectorized trace updates
    - GPU: Parallel weight updates
    - TPU: Outer product operations
  }];
  
  let arguments = (ins
    AnyTensor:$synapses,
    Variadic<Neuro_SpikeEventType>:$pre_spikes,
    Variadic<Neuro_SpikeEventType>:$post_spikes,
    F32Attr:$learning_rate
  );
  
  let results = (outs AnyTensor:$updated_synapses);
}

//===----------------------------------------------------------------------===//
// Network operations
//===----------------------------------------------------------------------===//

def Neuro_NetworkOp : Neuro_Op<"network", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"NetworkReturnOp">
]> {
  let summary = "Define a neuromorphic network";
  let description = [{
    Defines a complete neuromorphic network that can be optimized as a unit.
    Enables whole-network optimizations:
    - Layer fusion
    - Memory layout optimization
    - Hardware partitioning
    - Power-aware scheduling
  }];
  
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = "$name attr-dict-with-keyword $body";
}

def Neuro_SimulateOp : Neuro_Op<"simulate"> {
  let summary = "Simulate network for specified duration";
  let description = [{
    Runs the network simulation with automatic backend selection:
    - Small networks (<1K neurons): CPU
    - Medium networks (1K-100K): TPU  
    - Large networks (>100K): GPU
    - Real-time requirements: Neuromorphic hardware
  }];
  
  let arguments = (ins
    FlatSymbolRefAttr:$network,
    F32Attr:$duration_ms,
    OptionalAttr<StrAttr>:$backend,
    Variadic<AnyTensor>:$inputs
  );
  
  let results = (outs Variadic<AnyTensor>:$outputs);
}

//===----------------------------------------------------------------------===//
// Specialized operations for ARES
//===----------------------------------------------------------------------===//

def Neuro_EMSensorOp : Neuro_Op<"em_sensor", [Pure]> {
  let summary = "EM spectrum to neural encoding";
  let description = [{
    Converts electromagnetic spectrum data to spike trains.
    Hardware-specific implementations:
    - CPU: FFT + threshold
    - GPU: cuFFT + parallel encoding
    - TPU: Pre-computed frequency banks
    - SDR: Direct RF to spikes
  }];
  
  let arguments = (ins
    AnyTensor:$spectrum,
    F32Attr:$center_frequency,
    F32Attr:$bandwidth
  );
  
  let results = (outs Variadic<Neuro_SpikeEventType>:$spikes);
}

def Neuro_ThreatDetectorOp : Neuro_Op<"threat_detector"> {
  let summary = "Specialized threat detection network";
  let description = [{
    High-level threat detection with automatic optimization.
    Fuses multiple operations for efficiency:
    - Sensor encoding
    - Network forward pass
    - Decision extraction
    - Confidence scoring
  }];
  
  let arguments = (ins
    AnyTensor:$sensor_data,
    StrAttr:$threat_type
  );
  
  let results = (outs
    I32:$threat_class,
    F32:$confidence
  );
}

def Neuro_ChaosDetectionOp : Neuro_Op<"chaos_detection", [Pure]> {
  let summary = "Detect chaotic patterns in signals";
  let description = [{
    Chaos detection using coupled oscillators.
    Maps to hardware:
    - CPU: ODE integration
    - GPU: Parallel oscillator arrays
    - TPU: Matrix formulation of dynamics
  }];
  
  let arguments = (ins
    AnyTensor:$signal,
    F32Attr:$coupling_strength
  );
  
  let results = (outs F32:$chaos_metric);
}

//===----------------------------------------------------------------------===//
// Lowering hints and optimization directives
//===----------------------------------------------------------------------===//

def Neuro_OptimizeForOp : Neuro_Op<"optimize_for", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Optimization directive for target hardware";
  let description = [{
    Provides hints to the compiler for optimization:
    - "latency": Minimize processing time
    - "power": Minimize energy consumption  
    - "throughput": Maximize operations per second
    - "accuracy": Preserve numerical precision
  }];
  
  let arguments = (ins
    StrAttr:$target,
    StrAttr:$objective
  );
  
  let regions = (region AnyRegion:$body);
}

def Neuro_ParallelizeOp : Neuro_Op<"parallelize", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Explicit parallelization directive";
  let description = [{
    Indicates that enclosed operations can be parallelized.
    Maps to:
    - CPU: OpenMP parallel regions
    - GPU: CUDA kernels
    - TPU: Systolic array operations
  }];
  
  let arguments = (ins
    OptionalAttr<I64Attr>:$num_threads,
    OptionalAttr<StrAttr>:$schedule
  );
  
  let regions = (region AnyRegion:$body);
}

//===----------------------------------------------------------------------===//
// Terminator operations
//===----------------------------------------------------------------------===//

def Neuro_NetworkReturnOp : Neuro_Op<"network_return", [
    Pure, Terminator, HasParent<"NetworkOp">
]> {
  let summary = "Return from network definition";
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = "($outputs^ `:` type($outputs))? attr-dict";
}

#endif // NEUROMORPHIC_DIALECT