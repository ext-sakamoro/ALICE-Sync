//! Oracle: bytes of the earlier `InputFrame` form (movement and aim as three `i16` each, Q8.8) are
//! not read as an input of the current form, so a peer still sending the earlier form is refused
//! instead of being stepped with different numbers.

use alice_sync::{InputFix, InputFrame};

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

/// `InputFrame::new(42, 1).with_movement(256, -128, 1).with_actions(5).with_aim(0, 512, -256)` and
/// `InputFrame::new(1, 0)`, encoded by the earlier form (recorded from it)
const EARLIER: [&str; 2] = [
    "062a0100000180ff01000405000000000200ff",
    "060100020a000400020a00",
];

#[test]
fn the_earlier_form_is_refused() {
    for hex in EARLIER {
        assert_eq!(InputFrame::from_bytes(&bytes(hex)), None, "{hex}");
    }
}

#[test]
fn the_current_form_reads_back() {
    // the reference the refusal above is measured against
    let f = InputFrame::new(42, 1)
        .with_movement(
            InputFix::from_int(1),
            InputFix::from_raw(-1, 1 << 63),
            InputFix::from_raw(0, 1 << 56),
        )
        .with_actions(5)
        .with_aim(
            InputFix::ZERO,
            InputFix::from_int(2),
            InputFix::from_int(-1),
        );
    assert_eq!(InputFrame::from_bytes(&f.to_bytes()), Some(f));
}
