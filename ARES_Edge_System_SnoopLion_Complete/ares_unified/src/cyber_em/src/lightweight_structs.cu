/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file implements the lightweight struct versions to fix CUDA parameter size limitations
 */

#include "../include/lightweight_structs.cuh"
#include "../include/em_cyber_structs.cuh"  // Assumed include for the full structs

namespace ares::cyber_em {

AttackVectorCompact AttackVectorCompact::FromFull(const AttackVector& full) {
    AttackVectorCompact compact;
    compact.attack_type = static_cast<uint32_t>(full.attack_type);
    compact.target_id = full.target_id;
    compact.injection_freq_hz = full.injection_freq_hz;
    compact.injection_power_dbm = full.injection_power_dbm;
    compact.pulse_width_ns = full.pulse_width_ns;
    compact.repetition_rate_hz = full.repetition_rate_hz;
    compact.success_probability = full.success_probability;
    compact.start_time_ns = full.start_time_ns;
    compact.duration_ns = full.duration_ns;
    compact.active = full.active;
    return compact;
}

void AttackVectorCompact::ToFull(AttackVector& full) const {
    full.attack_type = static_cast<EMAttackType>(attack_type);
    full.target_id = target_id;
    full.injection_freq_hz = injection_freq_hz;
    full.injection_power_dbm = injection_power_dbm;
    full.pulse_width_ns = pulse_width_ns;
    full.repetition_rate_hz = repetition_rate_hz;
    full.success_probability = success_probability;
    full.start_time_ns = start_time_ns;
    full.duration_ns = duration_ns;
    full.active = active;
    // Note: waveform_template is not copied here
}

SideChannelMeasurementCompact SideChannelMeasurementCompact::FromFull(const SideChannelMeasurement& full) {
    SideChannelMeasurementCompact compact;
    compact.information_leakage_bits = full.information_leakage_bits;
    compact.key_bits_recovered = full.key_bits_recovered;
    compact.confidence = full.confidence;
    return compact;
}

void SideChannelMeasurementCompact::ToFull(SideChannelMeasurement& full) const {
    full.information_leakage_bits = information_leakage_bits;
    full.key_bits_recovered = key_bits_recovered;
    full.confidence = confidence;
    // Note: arrays are not copied here
}

} // namespace ares::cyber_em
