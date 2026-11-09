//! Helpers for a two-player match over a relay: agreeing on the input delay and comparing state
//! digests. No I/O: the caller sends the proposals and digests (sealed) and feeds back what arrives.

use std::collections::BTreeMap;

/// Input delay proposal in frames: `ceil((rtt_us / 2) / frame_us) + margin`.
///
/// Returns `None` for a zero frame length or an overflowing result.
#[must_use]
pub fn try_propose_delay(rtt_us: u64, frame_us: u64, margin: u64) -> Option<u64> {
    if frame_us == 0 {
        return None;
    }
    let one_way = rtt_us / 2;
    one_way.div_ceil(frame_us).checked_add(margin)
}

/// [`try_propose_delay`] for a known non-zero frame length.
///
/// # Panics
///
/// Panics when `frame_us` is 0 or the result overflows.
#[must_use]
pub fn propose_delay(rtt_us: u64, frame_us: u64, margin: u64) -> u64 {
    try_propose_delay(rtt_us, frame_us, margin).expect("frame length must be non-zero")
}

/// The delay both peers use: the larger of the two proposals.
#[must_use]
pub const fn agree(mine: u64, theirs: u64) -> u64 {
    if mine > theirs {
        mine
    } else {
        theirs
    }
}

/// A desync: the first frame whose digests differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Report {
    /// Frame number.
    pub frame: u64,
    /// This peer's digest.
    pub local: u64,
    /// The other peer's digest.
    pub remote: u64,
}

/// Local and remote digests per frame; reports the first mismatch once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DigestLog {
    local: BTreeMap<u64, u64>,
    remote: BTreeMap<u64, u64>,
    first: Option<u64>,
}

impl DigestLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn check(&mut self, frame: u64) -> Option<Report> {
        if self.first.is_some() {
            return None;
        }
        let (&l, &r) = (self.local.get(&frame)?, self.remote.get(&frame)?);
        if l == r {
            return None;
        }
        self.first = Some(frame);
        Some(Report {
            frame,
            local: l,
            remote: r,
        })
    }

    /// Record (or, after a rollback, replace) this peer's digest for `frame`.
    pub fn local(&mut self, frame: u64, digest: u64) -> Option<Report> {
        self.local.insert(frame, digest);
        self.check(frame)
    }

    /// Record the other peer's digest for `frame`.
    pub fn remote(&mut self, frame: u64, digest: u64) -> Option<Report> {
        self.remote.insert(frame, digest);
        self.check(frame)
    }

    /// The first frame reported as a mismatch, if any.
    #[must_use]
    pub const fn first_mismatch(&self) -> Option<u64> {
        self.first
    }

    /// Drop frames below `frame` (they are confirmed and compared).
    pub fn forget_before(&mut self, frame: u64) {
        self.local = self.local.split_off(&frame);
        self.remote = self.remote.split_off(&frame);
    }

    /// Number of frames held (local + remote entries).
    #[must_use]
    pub fn len(&self) -> usize {
        self.local.len() + self.remote.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
