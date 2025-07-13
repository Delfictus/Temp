//===- NeuromorphicLowering.h - Lowering passes header ----*- C++ -*-===//
//
// ARES Edge System - MLIR Neuromorphic Lowering Header
// Copyright (c) 2024 DELFICTUS I/O LLC
//
// This file declares the lowering passes for neuromorphic dialect.
//
//===----------------------------------------------------------------------===//

#ifndef NEUROMORPHIC_LOWERING_H
#define NEUROMORPHIC_LOWERING_H

#include <memory>

namespace mlir {
class Pass;

/// Create a pass to lower neuromorphic operations to target hardware
std::unique_ptr<Pass> createNeuromorphicLoweringPass();

/// Register the neuromorphic lowering pass
void registerNeuromorphicLoweringPass();

} // namespace mlir

#endif // NEUROMORPHIC_LOWERING_H
