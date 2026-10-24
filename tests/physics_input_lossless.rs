//! Oracle: a Physics input sent through Sync arrives with every bit. The fractional words of
//! movement and aim (`Fix128::lo`), negative values and the largest integer parts survive the
//! conversion to Sync's `InputFrame`, its wire bytes and the conversion back, so peers that
//! apply their own input and peers that decode it from the network step the same world.
#![cfg(feature = "physics")]

use alice_physics::{Fix128, FrameInput, Vec3Fix, WireError};
use alice_sync::physics_bridge::{physics_input_to_sync, sync_input_to_physics};
use alice_sync::InputFrame;

fn v(x: (i64, u64), y: (i64, u64), z: (i64, u64)) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_raw(x.0, x.1),
        Fix128::from_raw(y.0, y.1),
        Fix128::from_raw(z.0, z.1),
    )
}

fn samples() -> Vec<FrameInput> {
    vec![
        // 1.5, -0.25, a value with only the lowest fractional bit
        FrameInput::new(1)
            .with_movement(v((1, 1 << 63), (-1, 3 << 62), (0, 1)))
            .with_actions(0x5)
            .with_aim(v((0, 0x0123_4567_89ab_cdef), (2, 0), (-3, u64::MAX))),
        // integer parts past the i16 range
        FrameInput::new(7)
            .with_movement(v((i64::MAX, u64::MAX), (i64::MIN, 0), (40_000, 0)))
            .with_actions(u32::MAX)
            .with_aim(v((-40_000, 1), (0, 0), (1, 1))),
        // zero
        FrameInput::new(0),
    ]
}

fn bits(i: &FrameInput) -> [(i64, u64); 6] {
    let m = i.movement;
    let a = i.aim_direction;
    [
        (m.x.hi, m.x.lo),
        (m.y.hi, m.y.lo),
        (m.z.hi, m.z.lo),
        (a.x.hi, a.x.lo),
        (a.y.hi, a.y.lo),
        (a.z.hi, a.z.lo),
    ]
}

#[test]
fn a_physics_input_survives_the_sync_conversion() {
    for input in samples() {
        let back = sync_input_to_physics(&physics_input_to_sync(&input, 42));
        assert_eq!(back.player_id, input.player_id);
        assert_eq!(back.actions, input.actions);
        assert_eq!(bits(&back), bits(&input), "input {input:?}");
    }
}

#[test]
fn a_physics_input_survives_the_sync_wire() {
    for input in samples() {
        let frame = physics_input_to_sync(&input, 42);
        let decoded = InputFrame::from_bytes(&frame.to_bytes()).expect("decodes");
        assert_eq!(decoded, frame);
        assert_eq!(decoded.frame, 42);
        let back = sync_input_to_physics(&decoded);
        assert_eq!(bits(&back), bits(&input), "input {input:?}");
    }
}

#[test]
fn the_physics_wire_of_a_sync_input_is_lossless_and_refuses_the_legacy_form() {
    // the bytes a peer sends for an input that came through Sync: Physics' wire keeps every bit
    for input in samples() {
        let via_sync = sync_input_to_physics(&physics_input_to_sync(&input, 9));
        let wire = via_sync.to_wire();
        assert_eq!(wire.len(), FrameInput::WIRE_BYTES);
        let back = FrameInput::from_wire(&wire).expect("decodes");
        assert_eq!(bits(&back), bits(&input), "input {input:?}");
    }
    // the pre-4.0.0 20-byte form (integer part only) is refused, never read as an input
    assert_eq!(FrameInput::LEGACY_BYTES, 20);
    assert_eq!(
        FrameInput::from_wire(&[0u8; FrameInput::LEGACY_BYTES]),
        Err(WireError::Legacy)
    );
    assert!(matches!(
        FrameInput::from_wire(&[0u8; FrameInput::LEGACY_BYTES + 1]),
        Err(WireError::Length { found: 21 })
    ));
}
