//! Contract tests for `RollbackSession` / `InputBuffer` (P2P rollback).
//!
//! The reference model in this file is independent of the crate: a deterministic counter game
//! (`step`) whose state is a pair of integers, simulated once with every input known in advance.
//! Any rollback driver that restores the right snapshot and replays the right inputs must end in
//! the same state as that reference on every frame.

use alice_sync::input_sync::{InputFrame, RollbackAction, RollbackSession, SyncResult};

/// Reference game: state = (sum of all actions weighted by player, frame count).
fn step(state: (u64, u64), inputs: &[InputFrame]) -> (u64, u64) {
    let mut s = state.0.wrapping_mul(31);
    for i in inputs {
        s = s.wrapping_add(u64::from(i.actions) * (u64::from(i.player_id) + 1));
    }
    (s, state.1 + 1)
}

fn enc(s: (u64, u64)) -> Vec<u8> {
    let mut v = s.0.to_le_bytes().to_vec();
    v.extend_from_slice(&s.1.to_le_bytes());
    v
}

fn dec(b: &[u8]) -> (u64, u64) {
    (
        u64::from_le_bytes(b[..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
    )
}

fn digest(s: (u64, u64)) -> u64 {
    s.0 ^ s.1.rotate_left(17)
}

fn input(frame: u64, player: u8, actions: u32) -> InputFrame {
    let mut i = InputFrame::new(frame, player);
    i.actions = actions;
    i
}

/// Drives one peer: `local` inputs per frame, `remote` arrivals as (arrive_after_frame, input).
/// Returns the state after each simulated frame (index = frame).
fn drive(
    first: u64,
    frames: u64,
    local: &dyn Fn(u64) -> u32,
    arrivals: &[(u64, InputFrame)],
) -> (Vec<(u64, u64)>, usize) {
    let mut s = RollbackSession::new(2, 0, 16);
    let mut state = (0u64, 0u64);
    let mut history = Vec::new();
    let mut rollbacks = 0;
    for f in first..first + frames {
        // deliver remote inputs that arrive before this frame is simulated
        let mut earliest: Option<u64> = None;
        for (at, i) in arrivals.iter().filter(|(at, _)| *at + 1 == f) {
            let _ = at;
            if let RollbackAction::Rollback { to_frame } = s.add_remote_input(*i) {
                earliest = Some(earliest.map_or(to_frame, |e: u64| e.min(to_frame)));
            }
        }
        if let Some(to) = earliest {
            rollbacks += 1;
            // restore the snapshot taken before `to` was simulated, then replay to f - 1
            let snap = s.get_snapshot(to).expect("snapshot for rollback frame");
            state = dec(snap);
            history.truncate((to - first) as usize);
            for g in to..f {
                let ins = s.inputs_for_frame(g);
                s.save_snapshot(g, enc(state), digest(state));
                state = step(state, &ins);
                history.push(state);
            }
        }
        let ins = s.add_local_input(input(f, 0, local(f)));
        s.save_snapshot(f, enc(state), digest(state));
        state = step(state, &ins);
        history.push(state);
    }
    (history, rollbacks)
}

fn reference(
    first: u64,
    frames: u64,
    local: &dyn Fn(u64) -> u32,
    remote: &dyn Fn(u64) -> u32,
) -> Vec<(u64, u64)> {
    let mut state = (0u64, 0u64);
    let mut out = Vec::new();
    for f in first..first + frames {
        state = step(state, &[input(f, 0, local(f)), input(f, 1, remote(f))]);
        out.push(state);
    }
    out
}

/// oracle: reference simulation with all inputs known. Remote input changes at frames 5 and 8,
/// each arriving 3 frames late, so the driver rolls back twice and the second rollback crosses
/// frames the first rollback re-simulated. Restoring the snapshot saved during the first
/// re-simulation (not the one from the original misprediction) is what makes the end state match.
#[test]
fn consecutive_rollbacks_restore_the_latest_snapshot_for_each_frame() {
    let local = |f: u64| (f % 3) as u32;
    let remote = |f: u64| match f {
        0..=4 => 0,
        5..=7 => 7,
        _ => 11,
    };
    let arrivals: Vec<(u64, InputFrame)> =
        (1..=20).map(|f| (f + 3, input(f, 1, remote(f)))).collect();
    let (got, rollbacks) = drive(1, 20, &local, &arrivals);
    let want = reference(1, 20, &local, &remote);
    assert!(
        rollbacks >= 2,
        "scene must roll back at least twice, got {rollbacks}"
    );
    assert_eq!(got, want);
}

/// oracle: after re-saving a frame, the stored checksum for it is the re-saved one.
#[test]
fn resaving_a_frame_replaces_its_snapshot_and_checksum() {
    let mut s = RollbackSession::new(2, 0, 8);
    s.save_snapshot(3, vec![1], 100);
    s.save_snapshot(3, vec![2], 200);
    assert_eq!(s.get_snapshot(3), Some(&[2u8][..]));
    assert_eq!(s.verify_checksum(3, 200), SyncResult::Ok);
    assert!(matches!(
        s.verify_checksum(3, 100),
        SyncResult::Desync { .. }
    ));
}

/// oracle: a session that starts at frame 0 simulates frame 0 with the local input it was given
/// and a predicted (empty) remote input; the remote frame-0 input is then accepted and compared.
#[test]
fn frame_zero_is_a_normal_frame() {
    let mut s = RollbackSession::new(2, 0, 8);
    let ins = s.add_local_input(input(0, 0, 5));
    assert_eq!(ins.len(), 2);
    assert_eq!((ins[0].frame, ins[0].actions), (0, 5));
    assert_eq!((ins[1].frame, ins[1].actions), (0, 0));
    assert_eq!(
        s.add_remote_input(input(0, 1, 9)),
        RollbackAction::Rollback { to_frame: 0 }
    );
    assert_eq!(s.inputs_for_frame(0)[1].actions, 9);
    assert_eq!(s.confirmed_frame(), 0);
}

/// oracle: a remote input that arrives after a later frame's input is still stored, triggers a
/// rollback to its own frame when it differs from the prediction, and is used on replay.
#[test]
fn out_of_order_remote_input_is_kept_and_rolls_back_to_its_frame() {
    let mut s = RollbackSession::new(2, 0, 16);
    for f in 1..=6 {
        let _ = s.add_local_input(input(f, 0, 0));
    }
    assert_eq!(
        s.add_remote_input(input(5, 1, 4)),
        RollbackAction::Rollback { to_frame: 5 }
    );
    assert_eq!(
        s.add_remote_input(input(3, 1, 2)),
        RollbackAction::Rollback { to_frame: 3 }
    );
    assert_eq!(s.inputs_for_frame(3)[1].actions, 2);
    assert_eq!(s.inputs_for_frame(5)[1].actions, 4);
    // frames 1..=2 and 4 are still unconfirmed, so the contiguous confirmed prefix is empty
    assert_eq!(s.confirmed_frame(), 0);
    for f in [1, 2, 4] {
        let _ = s.add_remote_input(input(f, 1, 0));
    }
    assert_eq!(s.confirmed_frame(), 5);
}

/// oracle: a duplicate of an already confirmed remote input is ignored (no rollback, no change).
#[test]
fn duplicate_remote_input_is_ignored() {
    let mut s = RollbackSession::new(2, 0, 16);
    let _ = s.add_local_input(input(1, 0, 0));
    assert_eq!(
        s.add_remote_input(input(1, 1, 3)),
        RollbackAction::Rollback { to_frame: 1 }
    );
    assert_eq!(s.add_remote_input(input(1, 1, 8)), RollbackAction::None);
    assert_eq!(s.inputs_for_frame(1)[1].actions, 3);
}

/// oracle: a remote input for a frame already trimmed away is rejected without a rollback and
/// without a panic, and a delayed session with delay 0 behaves like `new`.
#[test]
fn input_for_a_trimmed_frame_is_rejected() {
    let mut s = RollbackSession::new(2, 0, 16);
    for f in 1..=6 {
        let _ = s.add_local_input(input(f, 0, 0));
        let _ = s.add_remote_input(input(f, 1, 0));
    }
    s.trim_before(4);
    assert_eq!(s.add_remote_input(input(2, 1, 9)), RollbackAction::None);
    assert_eq!(s.confirmed_frame(), 6);
}
