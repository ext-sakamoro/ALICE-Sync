//! Contract tests for match agreement helpers (delay agreement, digest exchange).
//!
//! - delay proposal: one-way delay = RTT / 2, in frames rounded up, plus a margin; both peers
//!   send their proposal and use the larger one, so they agree without a third party
//! - digests: each peer records its own digest per frame and the other's as it arrives (either
//!   may come first); the first frame where both are known and differ is the desync report

use alice_sync::match_sync::{agree, propose_delay, DigestLog, Report};

/// oracle: 50 ms RTT at 60 Hz (16 667 us per frame): one way 25 ms = 1.5 frames -> 2, margin 1 -> 3.
#[test]
fn proposal_rounds_half_rtt_up_and_adds_margin() {
    assert_eq!(propose_delay(50_000, 16_667, 1), 3);
    assert_eq!(
        propose_delay(33_334, 16_667, 0),
        1,
        "exactly one frame one way"
    );
    assert_eq!(
        propose_delay(33_336, 16_667, 0),
        2,
        "just over one frame rounds up"
    );
    assert_eq!(propose_delay(0, 16_667, 2), 2);
}

/// oracle: both peers take the larger proposal (order does not matter).
#[test]
fn agreement_is_the_larger_proposal() {
    assert_eq!(agree(2, 5), 5);
    assert_eq!(agree(5, 2), 5);
    assert_eq!(agree(3, 3), 3);
}

/// oracle: equal digests report nothing; the first differing frame is reported once, whichever
/// side arrived first; later mismatches after the first are not reported again.
#[test]
fn digest_log_reports_the_first_mismatch_once() {
    let mut d = DigestLog::new();
    assert_eq!(d.local(1, 10), None);
    assert_eq!(d.remote(1, 10), None);
    assert_eq!(
        d.remote(2, 20),
        None,
        "remote first: nothing to compare yet"
    );
    assert_eq!(
        d.local(2, 21),
        Some(Report {
            frame: 2,
            local: 21,
            remote: 20
        })
    );
    assert_eq!(d.local(3, 1), None);
    assert_eq!(d.remote(3, 2), None, "only the first mismatch is reported");
    assert_eq!(d.first_mismatch(), Some(2));
}

/// oracle: a re-simulated frame (rollback) replaces the local digest before comparison; frames
/// at or below `forget_before` are dropped and the log stays small.
#[test]
fn local_digest_replaced_after_rollback_and_pruned() {
    let mut d = DigestLog::new();
    assert_eq!(d.local(5, 1), None);
    assert_eq!(d.local(5, 2), None, "re-simulated frame 5");
    assert_eq!(d.remote(5, 2), None, "matches the re-simulated value");
    d.forget_before(6);
    assert_eq!(d.len(), 0);
}

/// oracle (degenerate input): the checked form refuses a zero frame length (no division by zero,
/// no guessed delay) and an overflowing result.
#[test]
fn zero_frame_length_is_rejected() {
    assert_eq!(
        alice_sync::match_sync::try_propose_delay(50_000, 0, 1),
        None
    );
    assert_eq!(
        alice_sync::match_sync::try_propose_delay(u64::MAX, 1, u64::MAX),
        None,
        "overflow"
    );
}
