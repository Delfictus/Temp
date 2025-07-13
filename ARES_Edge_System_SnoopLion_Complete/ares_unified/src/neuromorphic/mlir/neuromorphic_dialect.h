//===- NeuromorphicDialect.h - Neuromorphic dialect -------*- C++ -*-===//
//
// ARES Edge System - MLIR Neuromorphic Dialect Header
// Copyright (c) 2024 DELFICTUS I/O LLC
//
// This file declares the neuromorphic dialect C++ interface.
//
//===----------------------------------------------------------------------===//

#ifndef NEUROMORPHIC_DIALECT_H
#define NEUROMORPHIC_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// Neuromorphic Dialect
//===----------------------------------------------------------------------===//

#include "neuromorphic_dialect.h.inc"

//===----------------------------------------------------------------------===//
// Neuromorphic Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "neuromorphic_types.h.inc"

//===----------------------------------------------------------------------===//
// Neuromorphic Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "neuromorphic_attrs.h.inc"

//===----------------------------------------------------------------------===//
// Neuromorphic Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "neuromorphic_ops.h.inc"

//===----------------------------------------------------------------------===//
// Registration functions
//===----------------------------------------------------------------------===//

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

void registerNeuromorphicDialect(mlir::DialectRegistry &registry);
void registerNeuromorphicDialect(mlir::MLIRContext &context);

#endif // NEUROMORPHIC_DIALECT_H
