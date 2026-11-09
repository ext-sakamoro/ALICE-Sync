//! Contract tests for input delay on `RollbackSession`.

use alice_sync::input_sync::{InputFrame, RollbackSession};

fn input(frame: u64, player: u8, actions: u32) -> InputFrame {
    let mut i = InputFrame::new(frame, player);
    i.actions = actions;
    i
}

/// oracle: with input delay D, the local input given while simulating frame t is stamped for
/// frame t + D (the stamp is what gets sent), and frames 0..D run with an empty local input.
#[test]
fn input_delay_shifts_local_input_by_d_frames() {
    let d = 2;
    let mut s = RollbackSession::with_input_delay(2, 0, 8, d);
    // frames 0..D are pre-confirmed (empty) for every player, so the confirmed prefix is D - 1
    assert_eq!(s.confirmed_frame(), d - 1);
    for t in 0..6u64 {
        let (ins, sent) = s.add_local_input_delayed(input(t, 0, 10 + t as u32));
        assert_eq!(sent.frame, t + d, "stamp for frame {t}");
        assert_eq!(sent.actions, 10 + t as u32);
        let want = if t < d { 0 } else { 10 + (t - d) as u32 };
        assert_eq!(
            (ins[0].frame, ins[0].actions),
            (t, want),
            "local input simulated at frame {t}"
        );
    }
    assert_eq!(s.input_delay(), d);
    // the remote peer sends only its stamped inputs (frames D..); with them the prefix advances
    for f in d..6 {
        let _ = s.add_remote_input(input(f, 1, 0));
    }
    assert_eq!(s.confirmed_frame(), 5);
}

/// oracle: delay 0 is the plain path (stamp = frame, input used on the same frame).
#[test]
fn input_delay_zero_is_the_plain_path() {
    let mut s = RollbackSession::with_input_delay(2, 0, 8, 0);
    let (ins, sent) = s.add_local_input_delayed(input(4, 0, 7));
    assert_eq!(sent.frame, 4);
    assert_eq!((ins[0].frame, ins[0].actions), (4, 7));
}
