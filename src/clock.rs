//! The one place the transport layer reads the monotonic clock.
//!
//! Retransmit timers, round-trip samples, fragment expiry and peer liveness need
//! real elapsed time. No event, world hash or wire byte is computed from it, so
//! it is not part of what two peers must agree on. Reading it here keeps the
//! clippy `disallowed-methods` ban on `Instant::now` (clippy.toml) free of any
//! other exception, and `scripts/entropy_guard.py` pins the count of reads in
//! this file.

use std::time::Instant;

/// The current monotonic time.
#[inline]
#[allow(
    clippy::disallowed_methods,
    reason = "the single monotonic clock read of the transport layer; see the module comment"
)]
pub(crate) fn now() -> Instant {
    Instant::now()
}
