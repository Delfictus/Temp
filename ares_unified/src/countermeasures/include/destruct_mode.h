/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file implements the missing DestructMode enum needed for self_destruct_protocol
 */

#pragma once

namespace ares::countermeasures {

// Destruct modes enum for the SelfDestructProtocol
enum class DestructMode : uint8_t {
    DATA_WIPE = 0,           // Secure data erasure only
    COMPONENT_DISABLE = 1,   // Disable critical components
    THERMAL_OVERLOAD = 2,    // Controlled thermal destruction
    ELECTROMAGNETIC = 3,     // EM pulse generation
    KINETIC = 4,            // Physical destruction (if equipped)
    FULL_SPECTRUM = 5       // All methods simultaneously
};

} // namespace ares::countermeasures
