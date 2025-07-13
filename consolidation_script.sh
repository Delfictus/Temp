#!/bin/bash

# ARES Edge System - Consolidation Script
# This script updates source files to use consolidated headers and archives duplicates

set -e

WORKSPACE="/home/ae/AE/ae_merge_workspace"
UNIFIED_DIR="$WORKSPACE/ares_unified"
LEGACY_DIR="$UNIFIED_DIR/legacy"

echo "Starting ARES Edge System consolidation..."

# Create legacy directories if they don't exist
mkdir -p "$LEGACY_DIR/repo1_1onlyadvance"
mkdir -p "$LEGACY_DIR/repo2_delfictus"

# Archive old repos to legacy
echo "Archiving old repositories to legacy directory..."
if [ -d "$WORKSPACE/repo1_1onlyadvance" ]; then
    cp -r "$WORKSPACE/repo1_1onlyadvance" "$LEGACY_DIR/"
    echo "Archived repo1_1onlyadvance"
fi

if [ -d "$WORKSPACE/repo2_delfictus" ]; then
    cp -r "$WORKSPACE/repo2_delfictus" "$LEGACY_DIR/"
    echo "Archived repo2_delfictus"
fi

# Update include paths in source files
echo "Updating include paths in source files..."

# Function to update a single file
update_file() {
    local file=$1
    echo "Processing: $file"
    
    # Check if file contains old CUDA_CHECK definitions
    if grep -q "#define CUDA_CHECK" "$file" 2>/dev/null; then
        # Comment out old CUDA_CHECK and add reference to common_utils
        sed -i '/#define CUDA_CHECK/,/} while(0)/c\// Use CUDA error checking from common_utils.h\n#define CUDA_CHECK ARES_CUDA_CHECK' "$file"
    fi
    
    # Add includes for consolidated headers if not already present
    if ! grep -q "config/constants.h" "$file" 2>/dev/null; then
        # Find the first include and add our headers after it
        sed -i '0,/#include/{s/#include/#include "..\/..\/config\/constants.h"\n#include "..\/..\/utils\/common_utils.h"\n#include/}' "$file" 2>/dev/null || true
    fi
}

# Find and update all C++ and CUDA files in ares_unified
find "$UNIFIED_DIR/src" -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.cuh" \) | while read -r file; do
    # Skip if file is in legacy directory
    if [[ "$file" == *"/legacy/"* ]]; then
        continue
    fi
    
    update_file "$file"
done

# Update CMakeLists.txt files to include new directories
echo "Updating CMakeLists.txt files..."

# Main CMakeLists.txt
if [ -f "$UNIFIED_DIR/src/CMakeLists.txt" ]; then
    if ! grep -q "include_directories.*config" "$UNIFIED_DIR/src/CMakeLists.txt"; then
        sed -i '/include_directories/a\    ${CMAKE_SOURCE_DIR}/config\n    ${CMAKE_SOURCE_DIR}/src/utils' "$UNIFIED_DIR/src/CMakeLists.txt"
    fi
fi

# Create a summary report
echo "Creating consolidation report..."
cat > "$WORKSPACE/CONSOLIDATION_REPORT.md" << EOF
# ARES Edge System Consolidation Report

Generated: $(date)

## Summary

This report documents the consolidation of the ARES Edge System codebase to eliminate duplicates and standardize common functionality.

## Consolidated Components

### 1. Constants (config/constants.h)
- System-wide constants
- GPU/CUDA configuration
- System limits
- Timing and performance parameters
- RF/EM spectrum configuration
- Security and cryptography constants

### 2. Common Utilities (src/utils/common_utils.h)
- CUDA error checking macros (ARES_CUDA_CHECK, etc.)
- CUDA helper functions
- Memory management utilities
- Complex number operations
- Time and performance utilities
- String and formatting utilities
- Math utilities

### 3. Configuration (config/config.yaml)
- Runtime configuration for all modules
- System settings
- Module-specific parameters
- Performance tuning options

## Duplicate Functions Consolidated

### CUDA Error Checking
- Previously defined in multiple files:
  - cyber_em/src/em_cyber_controller.cpp
  - countermeasures/src/chaos_induction_engine.cpp
  - identity/src/hardware_attestation_system.cpp
  - And many others...
- Now consolidated in: src/utils/common_utils.h

### Atomic Operations
- atomicMaxFloat, atomicAddFloat previously duplicated
- Now in: src/utils/common_utils.h

### Constants
- MAX_SWARM_SIZE, MAX_THREATS, etc. previously scattered
- Now in: config/constants.h

## Files Archived

The following duplicate repositories have been archived to the legacy directory:
- repo1_1onlyadvance/
- repo2_delfictus/

## Migration Guide

### For Developers

1. **Include Consolidated Headers**:
   \`\`\`cpp
   #include "config/constants.h"
   #include "utils/common_utils.h"
   \`\`\`

2. **Use Namespace Imports**:
   \`\`\`cpp
   using namespace ares::constants;
   using namespace ares::utils;
   \`\`\`

3. **Replace Old Macros**:
   - CUDA_CHECK → ARES_CUDA_CHECK
   - Local constants → Use from constants.h

4. **Configuration**:
   - Runtime parameters now in config/config.yaml
   - Load using YAML parser

## Benefits

1. **Code Reuse**: Eliminated duplicate implementations
2. **Consistency**: Standardized error handling and utilities
3. **Maintainability**: Single source of truth for constants
4. **Performance**: Optimized implementations in one place
5. **Modularity**: Clear separation of concerns

## Next Steps

1. Update all module documentation
2. Run comprehensive tests
3. Update build scripts
4. Train team on new structure

EOF

echo "Consolidation complete!"
echo "Report generated: $WORKSPACE/CONSOLIDATION_REPORT.md"
echo ""
echo "Next steps:"
echo "1. Review the consolidation report"
echo "2. Run tests to ensure everything still works"
echo "3. Remove old repositories after verification:"
echo "   rm -rf $WORKSPACE/repo1_1onlyadvance"
echo "   rm -rf $WORKSPACE/repo2_delfictus"