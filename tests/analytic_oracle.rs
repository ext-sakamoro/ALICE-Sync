//! Analytic oracles — closed-form checks for the numeric laws in ALICE-Sync
//! (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms, published test vectors or integer
//! references written in this file, never from the crate function under
//! test.  Default constructors (`RttEstimator::new`, `ReliableEndpoint::new`,
//! `ReassemblyBuffer::new`, `Fixed::default`) are the paths a consumer takes.
//!
//! Oracle sources:
//! - Q16.16 fixed point: dyadic products exact, `from_i16` / `to_i16` is the
//!   Q8.8 network format the crate documents (`InputFrame::movement`,
//!   physics bridge): 256 ↦ 1.0, quantisation step 1/256
//! - RFC 6298: SRTT = 7/8 SRTT + 1/8 R, RTTVAR = 3/4 RTTVAR + 1/4 |SRTT − R|,
//!   RTO = SRTT + 4 RTTVAR, first sample SRTT = R, RTTVAR = R/2
//! - fragmentation: ⌈len / (MAX_PAYLOAD − 4)⌉ fragments, any feed order
//!   reassembles the identity
//! - CRDT algebra: G-counter Σ, PN-counter Σ⁺ − Σ⁻, merge is commutative and
//!   idempotent, LWW takes the larger Lamport timestamp
//! - FNV-1a 64 test vectors ("" → cbf29ce484222325, "a" → af63dc4c8601ec8c,
//!   "foobar" → 85944171f73967e8)
//! - loss rate = lost / (acked + lost)

use alice_sync::determinism::{fnv1a_hash, DeterminismChecker, StateSnapshot};
use alice_sync::fixed_point::{batch_add_vec3, Fixed, Vec3Fixed, Vec3Simd};

// ───────────────────────── fixed point ────────────────────────────────────

#[test]
fn q16_16_arithmetic_is_exact_on_dyadic_rationals_and_q8_8_on_the_wire() {
    let half = Fixed::from_bits(1 << 15);
    let quarter = Fixed::from_bits(1 << 14);
    assert_eq!(Fixed::from_int(3).to_bits(), 3 << 16);
    assert_eq!((Fixed::from_int(3) + half).to_f32(), 3.5);
    assert_eq!((Fixed::from_int(3) - half).to_f32(), 2.5);
    // 1.5 × 2.5 = 3.75, 0.5 × 0.25 = 0.125 — exact in Q16.16
    let one_and_half = Fixed::from_int(1) + half;
    let two_and_half = Fixed::from_int(2) + half;
    assert_eq!(
        one_and_half.saturating_mul(two_and_half).to_bits(),
        (3 << 16) + (3 << 14)
    );
    assert_eq!(half.saturating_mul(quarter).to_bits(), 1 << 13);
    assert_eq!(Fixed::from_int(-3).saturating_mul(half).to_f32(), -1.5);
    // saturation at the i32 range
    assert_eq!(
        Fixed::from_bits(i32::MAX)
            .saturating_add(Fixed::ONE)
            .to_bits(),
        i32::MAX
    );
    assert_eq!(
        Fixed::from_int(30_000)
            .saturating_mul(Fixed::from_int(30_000))
            .to_bits(),
        i32::MAX
    );
    // f32 round trip is exact for values with ≤ 16 fractional bits
    for v in [
        0.0f32,
        1.0,
        -1.0,
        0.5,
        0.75,
        123.455_72, // 123 + 29_866 / 65_536 (16 fractional bits)
        -0.000_015_258_789,
    ] {
        assert_eq!(Fixed::from_f32(v).to_f32(), v, "{v}");
    }

    // network format: i16 is Q8.8 (InputFrame::movement doc, physics bridge doc)
    // oracle: 256 ↦ 1.0, 1 ↦ 1/256, round trip exact, quantisation ≤ 1/256
    assert_eq!(Fixed::from_i16(256).to_f32(), 1.0, "Q8.8: 256 is one unit");
    assert_eq!(Fixed::from_i16(1).to_f32(), 1.0 / 256.0);
    assert_eq!(Fixed::from_i16(-256).to_f32(), -1.0);
    for n in [i16::MIN, -1000, -1, 0, 1, 255, 256, 1000, i16::MAX] {
        assert_eq!(Fixed::from_i16(n).to_i16(), n, "i16 round trip {n}");
    }
    for bits in (-(1 << 22)..(1 << 22)).step_by(9973) {
        let f = Fixed::from_bits(bits);
        let back = Fixed::from_i16(f.to_i16());
        assert!(
            (f.to_f32() - back.to_f32()).abs() < 1.0 / 256.0 + 1e-7,
            "quantisation at {bits}"
        );
    }
    let v = Vec3Fixed::from_f32(1.0, -0.5, 0.25);
    assert_eq!(v.to_i16_array(), [256, -128, 64]);
    assert_eq!(
        Vec3Fixed::from_i16_array([256, -128, 64]).to_f32_array(),
        [1.0, -0.5, 0.25]
    );
}

#[test]
fn simd_vector_ops_equal_the_scalar_law_bit_for_bit() {
    let mut seed = 42u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as i32) >> 4
    };
    for _ in 0..500 {
        let a = Vec3Fixed::new(
            Fixed::from_bits(rnd()),
            Fixed::from_bits(rnd()),
            Fixed::from_bits(rnd()),
        );
        let b = Vec3Fixed::new(
            Fixed::from_bits(rnd()),
            Fixed::from_bits(rnd()),
            Fixed::from_bits(rnd()),
        );
        // scalar law: componentwise wrapping add / sub
        let sum = Vec3Simd::from_vec3(a).add(Vec3Simd::from_vec3(b)).to_vec3();
        let dif = Vec3Simd::from_vec3(a).sub(Vec3Simd::from_vec3(b)).to_vec3();
        assert_eq!(sum.x.to_bits(), a.x.to_bits().wrapping_add(b.x.to_bits()));
        assert_eq!(sum.y.to_bits(), a.y.to_bits().wrapping_add(b.y.to_bits()));
        assert_eq!(sum.z.to_bits(), a.z.to_bits().wrapping_add(b.z.to_bits()));
        assert_eq!(dif.x.to_bits(), a.x.to_bits().wrapping_sub(b.x.to_bits()));
        assert_eq!(dif.z.to_bits(), a.z.to_bits().wrapping_sub(b.z.to_bits()));
        // hash is a function of the bits only
        assert_eq!(a.hash_bits(), Vec3Simd::from_vec3(a).hash_bits());
        assert_eq!(a.to_simd().to_vec3(), a);
    }
    let mut pos: Vec<Vec3Fixed> = (0..37)
        .map(|i| Vec3Fixed::from_f32(i as f32, -i as f32, 0.5 * i as f32))
        .collect();
    let deltas: Vec<Vec3Fixed> = (0..37)
        .map(|i| Vec3Fixed::from_f32(0.25, 0.5 * i as f32, -1.0))
        .collect();
    let expected: Vec<Vec3Fixed> = pos.iter().zip(&deltas).map(|(p, d)| *p + *d).collect();
    batch_add_vec3(&mut pos, &deltas);
    assert_eq!(pos, expected, "batch add is the elementwise law");
}

// ───────────────────────── reliability / CRDT (feature `async`) ───────────

#[cfg(feature = "async")]
mod async_laws {
    use alice_sync::crdt::{CrdtMergeable, GCounter, LwwRegister, PnCounter};
    use alice_sync::reliability::{
        fragment_payload, FragmentHeader, PacketHeader, ReassemblyBuffer, ReliableEndpoint,
        RttEstimator, HEADER_SIZE, MAX_PAYLOAD,
    };
    use std::time::Duration;

    fn ns(d: Duration) -> u128 {
        d.as_nanos()
    }

    #[test]
    fn rtt_estimator_is_rfc_6298() {
        let mut est = RttEstimator::new();
        let samples = [120u64, 80, 200, 95, 110, 400, 100, 100, 100, 100];
        let (mut srtt, mut rttvar) = (0u128, 0u128);
        for (i, &ms) in samples.iter().enumerate() {
            let r = ns(Duration::from_millis(ms));
            est.update(Duration::from_millis(ms));
            if i == 0 {
                srtt = r;
                rttvar = r / 2;
            } else {
                let diff = srtt.abs_diff(r);
                rttvar = rttvar * 3 / 4 + diff / 4;
                srtt = srtt * 7 / 8 + r / 8;
            }
            let rto = (srtt + 4 * rttvar)
                .clamp(ns(Duration::from_millis(10)), ns(Duration::from_secs(5)));
            assert_eq!(ns(est.srtt()), srtt, "sample {i}: SRTT");
            assert_eq!(ns(est.rto()), rto, "sample {i}: RTO");
        }
        // constant RTT: SRTT → R, RTTVAR → 0, RTO → R (floored at 10 ms)
        let mut est = RttEstimator::new();
        for _ in 0..200 {
            est.update(Duration::from_millis(30));
        }
        assert!((est.srtt().as_secs_f64() - 0.030).abs() < 1e-4);
        assert!(est.rto().as_secs_f64() < 0.031 && est.rto() >= Duration::from_millis(10));
        let mut est = RttEstimator::new();
        est.update(Duration::from_millis(1));
        assert_eq!(est.rto(), Duration::from_millis(10), "floor");
        est.update(Duration::from_secs(30));
        assert_eq!(est.rto(), Duration::from_secs(5), "ceiling");
    }

    #[test]
    fn packet_headers_round_trip_and_fragmentation_is_the_ceiling_law() {
        let h = PacketHeader {
            seq: 0xBEEF,
            ack: 0x1234,
            ack_bits: 0xDEAD_BEEF,
            payload_len: 1199,
        };
        let rt = PacketHeader::from_bytes(&h.to_bytes());
        assert_eq!(
            (rt.seq, rt.ack, rt.ack_bits, rt.payload_len),
            (h.seq, h.ack, h.ack_bits, h.payload_len)
        );
        let f = FragmentHeader {
            message_id: 0xCAFE,
            fragment_index: 7,
            fragment_count: 9,
        };
        let rf = FragmentHeader::from_bytes(&f.to_bytes());
        assert_eq!(
            (rf.message_id, rf.fragment_index, rf.fragment_count),
            (f.message_id, f.fragment_index, f.fragment_count)
        );
        assert_eq!(HEADER_SIZE, 10);

        let chunk = MAX_PAYLOAD - 4;
        for len in [
            0usize,
            1usize,
            MAX_PAYLOAD - 1,
            MAX_PAYLOAD,
            MAX_PAYLOAD + 1,
            2 * chunk,
            2 * chunk + 1,
            17_000,
        ] {
            let payload: Vec<u8> = (0..len).map(|i| (i * 131 % 251) as u8).collect();
            let frags = fragment_payload(0x42, &payload);
            // oracle: one datagram if it fits, else ⌈len / (MAX_PAYLOAD − 4)⌉ fragments
            let expected = if len <= MAX_PAYLOAD {
                1
            } else {
                len.div_ceil(chunk)
            };
            assert_eq!(frags.len(), expected, "len {len}");
            assert!(
                frags.iter().all(|f: &Vec<u8>| f.len() <= MAX_PAYLOAD),
                "every fragment fits the MTU"
            );
            if len > MAX_PAYLOAD {
                // reassembly is the identity, in order and reversed
                let mut buf = ReassemblyBuffer::new();
                let mut out = None;
                for f in &frags {
                    out = buf.feed(f).or(out);
                }
                assert_eq!(out.as_deref(), Some(&payload[..]), "len {len} in order");
                let mut buf = ReassemblyBuffer::new();
                let mut out = None;
                for f in frags.iter().rev() {
                    out = buf.feed(f).or(out);
                }
                assert_eq!(out.as_deref(), Some(&payload[..]), "len {len} reversed");
            } else {
                assert_eq!(frags[0], payload, "unfragmented payload is passed through");
            }
        }
    }

    #[test]
    fn reliable_endpoint_loss_rate_and_sequence_law() {
        let mut a = ReliableEndpoint::new();
        let mut b = ReliableEndpoint::new();
        assert_eq!(a.loss_rate(), 0.0, "no traffic ⇒ 0");
        // sequence numbers increase by one per outgoing packet
        let s0 = a.local_seq();
        let p1 = a.wrap_outgoing(b"one");
        let p2 = a.wrap_outgoing(b"two");
        assert_eq!(a.local_seq(), s0.wrapping_add(2));
        let h1 = PacketHeader::from_bytes(p1[..HEADER_SIZE].try_into().unwrap());
        let h2 = PacketHeader::from_bytes(p2[..HEADER_SIZE].try_into().unwrap());
        assert_eq!(h2.seq, h1.seq.wrapping_add(1));
        assert_eq!(h1.payload_len as usize, 3);
        // the receiver unwraps the payloads intact
        assert_eq!(b.unwrap_incoming(&p1).as_deref(), Some(&b"one"[..]));
        assert_eq!(b.unwrap_incoming(&p2).as_deref(), Some(&b"two"[..]));
        // a duplicate is dropped
        assert!(
            b.unwrap_incoming(&p2).is_none(),
            "duplicate sequence is rejected"
        );
    }

    #[test]
    fn crdt_counters_and_registers_follow_their_algebra() {
        let mut a = GCounter::new();
        let mut b = GCounter::new();
        a.increment_by(1, 5);
        a.increment_by(2, 3);
        b.increment_by(2, 7);
        b.increment_by(3, 1);
        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        // oracle: per-replica max, total = Σ max = 5 + 7 + 1
        assert_eq!(ab.value(), 13);
        assert_eq!(ba.value(), 13, "commutative");
        let mut abb = ab.clone();
        abb.merge(&b);
        assert_eq!(abb.value(), 13, "idempotent");
        assert_eq!(ab.local_count(2), 7);

        let mut p = PnCounter::new();
        for _ in 0..10 {
            p.increment(1);
        }
        for _ in 0..4 {
            p.decrement(1);
        }
        for _ in 0..3 {
            p.decrement(2);
        }
        assert_eq!(p.value(), 10 - 4 - 3);

        // LWW: the larger Lamport timestamp wins regardless of merge order
        let mut r1 = LwwRegister::new("old", 1);
        let mut r2 = LwwRegister::new("new", 2);
        r1.set_at("old", 5);
        r2.set_at("new", 9);
        let mut m12 = r1.clone();
        m12.merge(&r2);
        let mut m21 = r2.clone();
        m21.merge(&r1);
        assert_eq!(m12.value, "new");
        assert_eq!(m21.value, "new");
    }
}

#[test]
fn fnv1a_matches_the_reference_vectors_and_divergence_is_the_first_differing_frame() {
    assert_eq!(fnv1a_hash(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_hash(b"foobar"), 0x8594_4171_f739_67e8);

    let mut checker = DeterminismChecker::new(3);
    for frame in 0..10u64 {
        for node in 0..3 {
            // node 2 diverges from frame 6 on
            let h = if node == 2 && frame >= 6 {
                0xBAD
            } else {
                0x100 + frame
            };
            checker.record(node, frame, StateSnapshot::new(h, &[h, h + 1]));
        }
    }
    assert!(checker.check_frame(5).is_none());
    assert!(checker.check_frame(6).is_some());
    let d = checker.find_first_divergence().expect("diverges");
    assert_eq!(d.frame, 6, "first differing frame");
    assert_eq!(checker.first_divergence_frame(), Some(6));
}

// ───────────────────────── codec bridge (feature `codec`) ─────────────────

#[cfg(feature = "codec")]
#[test]
#[ignore = "alice-codec 0.1.2 (crates.io) の rANS が偏った histogram で round-trip 不能 + 量子化 symbol の u8 wrap (ALICE-Codec `4aa4dba` で修正済、未 publish) — 0.1.3 publish + dep bump 後に ignore を外す (Backlog ALICE-Sync codec_bridge 行、実測 MAE 24.6)"]
fn event_batch_compression_is_lossless_at_step_one_and_bounded_otherwise() {
    use alice_sync::codec_bridge::{compress_event_batch, decompress_event_batch};
    // 0..63 sawtooth, the case recorded in the Backlog (MAE 24.6 before)
    let data: Vec<u8> = (0..4096u32).map(|i| (i % 64) as u8).collect();
    let back = decompress_event_batch(&compress_event_batch(&data, 1));
    assert_eq!(back.len(), data.len());
    let mae = data
        .iter()
        .zip(&back)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / data.len() as f64;
    assert!(
        mae <= 0.5,
        "quantiser step 1 must be (near-)lossless, MAE {mae}"
    );
    // step 4: every byte within a quantiser step of its origin
    let back = decompress_event_batch(&compress_event_batch(&data, 4));
    let max = data
        .iter()
        .zip(&back)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    assert!(max <= 8, "step 4: max error {max} > 2·step");
}

// ───────────────────────── physics bridge (feature `physics`) ─────────────

#[cfg(feature = "physics")]
#[test]
fn physics_bridge_reads_the_i16_wire_field_as_q8_8_in_both_directions() {
    use alice_sync::input_sync::InputFrame;
    use alice_sync::physics_bridge::{physics_input_to_sync, sync_input_to_physics};
    // oracle: Q8.8 ⇒ 256 ↦ 1.0, −128 ↦ −0.5, 1 ↦ 1/256; round trip is the identity
    let input = InputFrame::new(7, 2)
        .with_movement(256, -128, 1)
        .with_aim(0, 512, -256);
    let phys = sync_input_to_physics(&input);
    assert_eq!(phys.movement.x.to_f64(), 1.0, "movement.x");
    assert_eq!(phys.movement.y.to_f64(), -0.5, "movement.y");
    assert_eq!(phys.movement.z.to_f64(), 1.0 / 256.0, "movement.z");
    assert_eq!(phys.aim_direction.y.to_f64(), 2.0, "aim.y");
    assert_eq!(phys.aim_direction.z.to_f64(), -1.0, "aim.z");
    // and the Sync-side Q16.16 reading of the same field agrees
    assert_eq!(Fixed::from_i16(256).to_f32(), 1.0);
    assert_eq!(
        Vec3Fixed::from_i16_array(input.movement).to_f32_array(),
        [
            phys.movement.x.to_f64() as f32,
            phys.movement.y.to_f64() as f32,
            phys.movement.z.to_f64() as f32
        ]
    );
    let back = physics_input_to_sync(&phys, 7);
    assert_eq!(back.movement, input.movement, "round trip");
    assert_eq!(back.aim, input.aim);
    for n in [i16::MIN, -1000, -1, 0, 1, 255, 1000, i16::MAX] {
        let f = InputFrame::new(1, 0).with_movement(n, n, n);
        assert_eq!(
            physics_input_to_sync(&sync_input_to_physics(&f), 1).movement,
            [n; 3],
            "{n}"
        );
    }
}
