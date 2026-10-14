//! Bit-exact golden cases at the float boundaries of ALICE-Sync.
//!
//! The simulation core is integer Q16.16 (`Fixed`), so floats only appear
//! where values enter or leave it (`Fixed::from_f32` / `to_f32`,
//! `Vec3Fixed`), in the statistics ratios (`BandwidthStats`,
//! `ReliableEndpoint::loss_rate`), in telemetry averages
//! (`SyncTelemetry::average_rtt`) and in the cloud AABB tests
//! (`SpatialRegion`).  Each case pins the exact output bits (`to_bits`) or
//! the exact Q16.16 raw value, so the same assertions must hold in the debug
//! and the release profile on every platform.
//!
//! Expected values are derived by hand from the conversion law
//! `raw = trunc(x · 2^16)` (Rust `as i32`, rounding toward zero) and
//! `x = raw · 2^-16` (rounded to f32), never from the function under test.

use alice_sync::fixed_point::{Fixed, Vec3Fixed};
use alice_sync::protocol::BandwidthStats;

/// 2^-16 as f32: the Q16.16 quantisation step, bits 0x3780_0000.
const STEP_BITS: u32 = 0x3780_0000;

fn step() -> f32 {
    f32::from_bits(STEP_BITS)
}

// ───────────────────────── Fixed ↔ f32 ─────────────────────────────────────

#[test]
fn signed_zero_both_map_to_raw_zero_and_return_as_positive_zero() {
    // -0.0 · 2^16 = -0.0, trunc → 0: Q16.16 has a single zero, so the sign
    // of a float zero is not carried into the fixed-point state
    let neg = Fixed::from_f32(-0.0);
    let pos = Fixed::from_f32(0.0);
    assert_eq!(neg.to_bits(), 0);
    assert_eq!(pos.to_bits(), 0);
    // raw 0 → 0 as f32 · 2^-16 = +0.0 (bits 0x0000_0000), never -0.0
    assert_eq!(neg.to_f32().to_bits(), 0x0000_0000);
    assert_eq!(pos.to_f32().to_bits(), 0x0000_0000);
}

#[test]
fn smallest_step_round_trips_as_raw_one_and_minus_one() {
    // 2^-16 is the smallest positive Q16.16 value: raw 1 ↔ f32 0x3780_0000
    assert_eq!(Fixed::from_f32(step()).to_bits(), 1);
    assert_eq!(Fixed::from_bits(1).to_f32().to_bits(), STEP_BITS);
    // -2^-16: raw -1 ↔ f32 0xb780_0000 (only the sign bit differs)
    assert_eq!(Fixed::from_f32(-step()).to_bits(), -1);
    assert_eq!(Fixed::from_bits(-1).to_f32().to_bits(), 0xb780_0000);
}

#[test]
fn value_one_ulp_below_the_step_truncates_to_raw_zero_for_both_signs() {
    // nextDown(2^-16) · 2^16 = 1 - 2^-24 < 1 → trunc 0 (toward zero for -x too)
    let below = f32::from_bits(STEP_BITS - 1);
    assert_eq!(below.to_bits(), 0x377f_ffff);
    assert_eq!(Fixed::from_f32(below).to_bits(), 0);
    assert_eq!(Fixed::from_f32(-below).to_bits(), 0);
    // and the result leaves as +0.0, not -0.0
    assert_eq!(Fixed::from_f32(-below).to_f32().to_bits(), 0x0000_0000);
}

#[test]
fn value_one_ulp_above_the_step_truncates_to_raw_one() {
    // nextUp(2^-16) · 2^16 = 1 + 2^-23 → trunc 1, back to exactly 2^-16
    let above = f32::from_bits(STEP_BITS + 1);
    assert_eq!(Fixed::from_f32(above).to_bits(), 1);
    assert_eq!(Fixed::from_f32(above).to_f32().to_bits(), STEP_BITS);
}

#[test]
fn one_and_a_half_steps_truncate_toward_zero_symmetrically() {
    // ±1.5 · 2^-16 · 2^16 = ±1.5 → trunc ±1 (toward zero, not floor: -1.5 ↦ -1)
    let x = 1.5 * step();
    assert_eq!(x.to_bits(), 0x37c0_0000);
    assert_eq!(Fixed::from_f32(x).to_bits(), 1);
    assert_eq!(Fixed::from_f32(-x).to_bits(), -1);
    assert_eq!(Fixed::from_f32(-x).to_f32().to_bits(), 0xb780_0000);
}

#[test]
fn large_opposite_values_cancel_to_raw_zero_and_positive_zero() {
    // 30000.5 · 2^16 = 1_966_112_768 exactly (fits i32), and its negation;
    // the integer sum is exactly 0, unlike a float sum that could leave -0.0
    let a = Fixed::from_f32(30000.5);
    let b = Fixed::from_f32(-30000.5);
    assert_eq!(a.to_bits(), 1_966_112_768);
    assert_eq!(b.to_bits(), -1_966_112_768);
    assert_eq!(a.to_f32().to_bits(), 0x46ea_6100);
    assert_eq!(b.to_f32().to_bits(), 0xc6ea_6100);
    let sum = a + b;
    assert_eq!(sum.to_bits(), 0);
    assert_eq!(sum.to_f32().to_bits(), 0x0000_0000);
    // and the other order gives the same raw value
    assert_eq!((b + a).to_bits(), 0);
}

#[test]
fn raw_round_trip_is_exact_up_to_two_pow_24_and_rounds_to_even_beyond() {
    // |raw| ≤ 2^24 is exact in an f32 mantissa: 2^24 · 2^-16 = 256.0
    let exact = Fixed::from_bits(1 << 24).to_f32();
    assert_eq!(exact.to_bits(), 0x4380_0000);
    assert_eq!(Fixed::from_f32(exact).to_bits(), 1 << 24);
    // 2^24 + 1 is a tie between 2^24 and 2^24 + 2 → rounds to even (2^24):
    // the conversion out of Q16.16 loses the last bit, by IEEE 754 rounding
    let tie = Fixed::from_bits((1 << 24) + 1).to_f32();
    assert_eq!(tie.to_bits(), 0x4380_0000);
    assert_eq!(Fixed::from_f32(tie).to_bits(), 1 << 24);
    let neg_tie = Fixed::from_bits(-((1 << 24) + 1)).to_f32();
    assert_eq!(neg_tie.to_bits(), 0xc380_0000);
    assert_eq!(Fixed::from_f32(neg_tie).to_bits(), -(1 << 24));
}

#[test]
fn vec3_from_f32_with_signed_zeros_and_step_is_bit_exact_per_lane() {
    // the three lanes use the scalar law independently
    let v = Vec3Fixed::from_f32(-0.0, step(), -step());
    assert_eq!([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()], [0, 1, -1]);
    let back = v.to_f32_array().map(f32::to_bits);
    assert_eq!(back, [0x0000_0000, STEP_BITS, 0xb780_0000]);
}

// ───────────────────────── statistics ratios ──────────────────────────────

#[test]
fn bandwidth_ratios_return_positive_zero_on_zero_denominators_and_one_on_equal_values() {
    // documented defaults: 0 events → 0.0, 0 bytes sent → 0.0 (+0.0, not NaN)
    let empty = BandwidthStats::default();
    assert_eq!(empty.bytes_per_event().to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(empty.compression_ratio(0).to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(
        empty.compression_ratio(1024).to_bits(),
        0x0000_0000_0000_0000
    );
    // numerator 0 with a non-zero denominator: 0 / 7 = +0.0
    let sent_only = BandwidthStats {
        bytes_sent: 7,
        ..BandwidthStats::default()
    };
    assert_eq!(
        sent_only.compression_ratio(0).to_bits(),
        0x0000_0000_0000_0000
    );
    // equal numerator and denominator: exactly 1.0 (0x3ff0_0000_0000_0000)
    let equal = BandwidthStats {
        bytes_sent: 7,
        events_synced: 7,
        ..BandwidthStats::default()
    };
    assert_eq!(equal.bytes_per_event().to_bits(), 0x3ff0_0000_0000_0000);
    assert_eq!(equal.compression_ratio(7).to_bits(), 0x3ff0_0000_0000_0000);
}

#[cfg(feature = "async")]
#[test]
fn loss_rate_is_positive_zero_with_no_packets_and_one_half_when_lost_equals_acked() {
    use alice_sync::reliability::ReliableEndpoint;
    // documented range 0.0 - 1.0; no packets yet → +0.0 (not NaN from 0/0)
    let mut ep = ReliableEndpoint::new();
    assert_eq!(ep.loss_rate().to_bits(), 0x0000_0000_0000_0000);
    // lost == acked → exactly 0.5 (0x3fe0_0000_0000_0000)
    ep.packets_lost = 3;
    ep.packets_acked = 3;
    assert_eq!(ep.loss_rate().to_bits(), 0x3fe0_0000_0000_0000);
    // all lost → exactly 1.0
    ep.packets_acked = 0;
    assert_eq!(ep.loss_rate().to_bits(), 0x3ff0_0000_0000_0000);
}

// ───────────────────────── cloud AABB ─────────────────────────────────────

#[cfg(feature = "cloud")]
#[test]
fn spatial_region_treats_negative_zero_as_equal_to_positive_zero_on_the_boundary() {
    use alice_sync::cloud_bridge::SpatialRegion;
    // IEEE 754 compares -0.0 == +0.0, so a point at -0.0 lies on a region
    // whose min is +0.0, and regions touching at ±0.0 overlap
    let r = SpatialRegion::new([0.0; 3], [1.0; 3]);
    assert!(r.contains_point(-0.0, -0.0, -0.0));
    let left = SpatialRegion::new([-1.0; 3], [-0.0; 3]);
    assert!(r.overlaps(&left));
    assert!(left.overlaps(&r));
    // one step (2^-16) past the boundary is outside / disjoint
    let s = step();
    assert!(!r.contains_point(-s, 0.0, 0.0));
    let apart = SpatialRegion::new([-1.0; 3], [-s; 3]);
    assert!(!r.overlaps(&apart));
}

// ───────────────────────── telemetry averages ─────────────────────────────

#[cfg(feature = "telemetry")]
#[test]
fn average_rtt_is_exact_on_equal_values_keeps_negative_zero_and_cancels_to_positive_zero() {
    use alice_sync::telemetry::SyncTelemetry;
    let dir = tempfile::tempdir().expect("tempdir");
    let avg = |name: &str, samples: &[f32], start: u64, end: u64| -> u64 {
        let t = SyncTelemetry::new(dir.path().join(name)).expect("open telemetry db");
        for (frame, &v) in samples.iter().enumerate() {
            t.record_rtt(frame as u64, v).expect("record");
        }
        t.flush().expect("flush");
        t.average_rtt(start, end).expect("aggregate").to_bits()
    };
    // 100 × 15.0 → mean exactly 15.0 (0x402e_0000_0000_0000)
    assert_eq!(avg("equal", &[15.0; 100], 0, 99), 0x402e_0000_0000_0000);
    // 100 × -0.0 → (-0 + … + -0) / 100 = -0.0: the sign of zero survives
    assert_eq!(avg("negzero", &[-0.0; 100], 0, 99), 0x8000_0000_0000_0000);
    // alternating ±30000.5 (50 pairs) → exact cancellation to +0.0
    let alternating: Vec<f32> = (0..100)
        .map(|i| if i % 2 == 0 { 30000.5 } else { -30000.5 })
        .collect();
    assert_eq!(avg("cancel", &alternating, 0, 99), 0x0000_0000_0000_0000);
    // a range with no samples averages to +0.0 (not NaN)
    assert_eq!(avg("empty", &[15.0; 100], 500, 600), 0x0000_0000_0000_0000);
}
