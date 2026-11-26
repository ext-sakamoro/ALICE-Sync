//! The relay's revocation list kept on disk, so a restart does not lower what was accepted
//! ([`KeptRevocations`]), and the rule a connection path applies before it trusts the relay
//! ([`RevocationRule`]).
//!
//! **Rust 1.89 or newer**: the feature `anonymous-relay-client` locks the store with `File::try_lock`
//! (stable from 1.89); the rest of the crate builds with the crate's `rust-version` (1.88).
//!
//! # The store
//!
//! A [`KeptRevocations`] is one file holding `alice_identity::delegated_revocation::DelegatedRevocations`
//! (its `to_bytes` layout) for one operator master key. The (epoch, sequence) of what is stored only rises,
//! across restarts and across processes that share the file:
//!
//! 1. [`KeptRevocations::accept`] takes an exclusive lock on a sibling file `<name>.lock` (waiting at most
//!    [`DEFAULT_LOCK_WAIT`], or [`KeptRevocations::with_lock_wait`]; then [`KeptError::LockTimeout`]);
//! 2. it reads the stored state from the file again: the floor is what is on disk, not the copy in memory,
//!    which another process may have passed;
//! 3. it accepts the list on that state (`DelegatedRevocations::accept`: the delegate certificate, the list's
//!    signature and freshness, the (epoch, sequence) order); a refusal is returned as
//!    [`KeptError::Refused`] and nothing is written;
//! 4. only when the list replaced the stored one, it writes the new state atomically: a temporary file
//!    `.<name>.<pid>.tmp` in the same directory, synced, renamed over the file, and on unix the directory
//!    synced;
//! 5. it keeps the new state in memory and releases the lock.
//!
//! Without the lock two writers could lower the stored pair (A reads 5, B writes 7, A writes 6); without the
//! re-read one process would compare against its own older copy. With both, every write compares against
//! the newest state any process wrote.
//!
//! [`KeptRevocations::open`] reads the file: a missing file is an empty store; anything else that is not
//! one whole stored state of this master fails closed with a typed error ([`KeptError`]): an I/O error,
//! a directory at the path, an empty file, a file over [`MAX_KEPT_FILE_LEN`], bytes that do not decode,
//! or a state whose certificates and signatures do not check out under the master.
//!
//! # Putting an old file back
//!
//! The file is checked by a digest and by the signatures inside it, which catches a corrupted file but not
//! a replaced one. Without a floor source, someone who can write the data directory can put back an older
//! valid file (sealed or not, as long as they copied it earlier) and lower the stored pair. Only someone who
//! already writes the user's data directory gets this, and the gain is bounded:
//!
//! - when that writer holds only honest, signed files, by the freshness window: the restored list goes
//!   stale within the window, after which a fresh list is required;
//! - when that writer also holds a leaked delegate key of an older epoch, by that delegate's lifetime (at
//!   most 7 days): lowering the stored pair to that epoch lets the leaked delegate's fresh lists back in
//!   until its certificate expires (see the section "Limits" of `alice_identity::delegated_revocation`).
//!
//! A floor source ([`FloorSource`], [`KeptRevocations::open_with_floor`]) kept outside the data directory
//! (for example in the OS key store, supplied by the application) closes that for a writer of the data
//! directory alone: under the lock, the stored pair must be at least the floor at open and at each accept
//! ([`KeptError::RolledBack`] otherwise), and after a write the floor is raised to the new pair, still under
//! the lock. The file is written first and the floor raised second, so a crash between the two leaves the
//! floor below the file, which is safe; the floor is never raised first.
//!
//! A store that holds nothing, or a list below the floor (the file was deleted, the data directory was
//! replaced, an older file or a backup was put back), while the floor source holds a pair opens with that
//! floor pending ([`KeptRevocations::floor_pending`], [`FloorPending`]): the older list is not used for
//! anything, and [`FloorPending::stored`] says whether one was found (show it: it is the trace of a
//! rollback).
//! [`RevocationRule::apply`] refuses every relay while the floor is pending, whether or not a fresh list is
//! required ([`RevocationRuleError::FloorPending`]). The pending floor clears only when a list accepted
//! from a relay (checked as every list is: delegate, signatures, freshness) reaches the floor; a list
//! below it is refused as [`KeptError::RolledBack`] and nothing is written (the older file stays as it
//! is until a list at or above the floor replaces it).
//!
//! # The rule
//!
//! [`RevocationRule::apply_fetch`] fetches the list (`GET /relay-revocations`), accepts it into the store,
//! and decides about the relay's certificate. The caller then checks the certificate in the channel with
//! the same store, so the policy and the rule read one list:
//!
//! ```text
//! let mut kept = KeptRevocations::open(path, operator)?;
//! RevocationRule::default().apply_fetch(&mut kept, stream_to_relay_host, host, &relay_cert, now)?;
//! let policy = PinnedPeer::service("relay", operator, now, kept.revocation())?;
//! RelayConnection::open(stream, host, room, &relay_cert, &policy, mask)?;
//! ```

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use alice_identity::artifact::Revocation;
use alice_identity::cert_revocation::{Accepted, RevocationList};
use alice_identity::delegated_revocation::{
    DelegatedRevocationError, DelegatedRevocations, DEFAULT_REVOCATION_FRESHNESS_WINDOW_SECS,
};
use alice_identity::device_cert::{DeviceCert, MAX_DEVICE_CERT_LEN};
use alice_sign::PublicKey;

use super::http::{fetch_relay_revocations, FetchError};
use crate::relay_revocations_body::{DelegatedListResponse, MAX_RELAY_REVOCATIONS_LEN};

/// How long [`KeptRevocations::open`] and [`KeptRevocations::accept`] wait for the store's lock unless set
/// otherwise ([`KeptRevocations::with_lock_wait`]).
pub const DEFAULT_LOCK_WAIT: Duration = Duration::from_secs(5);

/// How often a held lock is tried again
const LOCK_POLL: Duration = Duration::from_millis(5);

/// The longest file a store reads or writes: the stored state's layout with the longest certificate there
/// is and the longest list [`super::fetch_relay_revocations`] takes.
pub const MAX_KEPT_FILE_LEN: usize =
    1 + 1 + 32 + 8 + 8 + 2 + MAX_DEVICE_CERT_LEN + 4 + MAX_RELAY_REVOCATIONS_LEN + 32;

/// A floor for the stored (epoch, sequence) kept outside the data directory, supplied by the application
/// (see the module documentation).
pub trait FloorSource {
    /// The floor, or `None` when none was set.
    ///
    /// # Errors
    ///
    /// When the floor cannot be read; the store then refuses ([`KeptError::Floor`]).
    fn read_floor(&self) -> Result<Option<(u64, u64)>, FloorError>;

    /// Raise the floor to `(epoch, sequence)`, called after the file holding that pair was written.
    ///
    /// # Errors
    ///
    /// When the floor cannot be written ([`KeptError::FloorNotRaised`]).
    fn raise_floor(&self, epoch: u64, sequence: u64) -> Result<(), FloorError>;
}

/// Why a [`FloorSource`] could not read or raise the floor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloorError(String);

impl FloorError {
    /// A floor error with the source's reason.
    #[must_use]
    pub fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }
}

impl core::fmt::Display for FloorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FloorError {}

/// Why a [`KeptRevocations`] did not open, or did not accept a list.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeptError {
    /// An I/O operation on the store failed.
    Io {
        /// What was being done.
        op: &'static str,
        /// The error's kind.
        kind: io::ErrorKind,
    },
    /// Something other than a file is at the store's path.
    NotAFile,
    /// The store's file is empty (a whole state is never empty).
    Empty,
    /// The store's file, or the state to write, is longer than [`MAX_KEPT_FILE_LEN`].
    TooLarge {
        /// Its length.
        len: u64,
    },
    /// The stored state does not decode, or does not check out under the master.
    Stored(DelegatedRevocationError),
    /// The list was refused; nothing was written.
    Refused(DelegatedRevocationError),
    /// The store's lock was held for longer than the wait.
    LockTimeout {
        /// How long it was waited for.
        waited: Duration,
    },
    /// A list was accepted that does not reach the floor (the store's floor is pending: nothing, or an
    /// older list, was stored); nothing was written.
    RolledBack {
        /// The (epoch, sequence) that would have been stored.
        stored: Option<(u64, u64)>,
        /// The floor.
        floor: (u64, u64),
    },
    /// The floor source could not be read.
    Floor(FloorError),
    /// The list was written and is kept, but the floor source could not be raised to it.
    FloorNotRaised(FloorError),
}

impl core::fmt::Display for KeptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io { op, kind } => write!(f, "revocation store: {op}: {kind:?}"),
            Self::NotAFile => f.write_str("revocation store: not a file at the path"),
            Self::Empty => f.write_str("revocation store: empty file"),
            Self::TooLarge { len } => {
                write!(f, "revocation store: {len} bytes over {MAX_KEPT_FILE_LEN}")
            }
            Self::Stored(e) => write!(f, "revocation store: stored state: {e}"),
            Self::Refused(e) => write!(f, "revocation store: refused: {e}"),
            Self::LockTimeout { waited } => {
                write!(f, "revocation store: lock held past {waited:?}")
            }
            Self::RolledBack { stored, floor } => write!(
                f,
                "revocation store: stored {stored:?} is below the floor {floor:?}"
            ),
            Self::Floor(e) => write!(f, "revocation store: floor: {e}"),
            Self::FloorNotRaised(e) => {
                write!(f, "revocation store: written, floor not raised: {e}")
            }
        }
    }
}

impl std::error::Error for KeptError {}

const fn io_error(op: &'static str) -> impl Fn(io::Error) -> KeptError {
    move |e| KeptError::Io { op, kind: e.kind() }
}

/// The operator's newest revocation list for one master key, kept in one file (see the module
/// documentation).
pub struct KeptRevocations {
    path: PathBuf,
    master: PublicKey,
    state: DelegatedRevocations,
    lock_wait: Duration,
    floor: Option<Box<dyn FloorSource + Send>>,
    pending: Option<FloorPending>,
}

/// Why a store's floor is pending ([`KeptRevocations::floor_pending`]): what it found against the floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloorPending {
    /// The floor the store waits for.
    pub floor: (u64, u64),
    /// `None`: nothing was stored (no list yet, or the file was removed). `Some(pair)`: a list below the
    /// floor was found and is ignored, which is what a rollback or a restored backup looks like; show it.
    pub stored: Option<(u64, u64)>,
}

impl core::fmt::Debug for KeptRevocations {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KeptRevocations")
            .field("path", &self.path)
            .field("epoch", &self.state.epoch())
            .field("sequence", &self.state.sequence())
            .field("floor_source", &self.floor.is_some())
            .field("floor_pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl KeptRevocations {
    /// Open the store at `path` for the master key `master`, without a floor source.
    ///
    /// # Errors
    ///
    /// [`KeptError::Io`] (including a directory that does not exist, as the lock file is made beside the
    /// store), [`KeptError::LockTimeout`], [`KeptError::NotAFile`], [`KeptError::Empty`],
    /// [`KeptError::TooLarge`], [`KeptError::Stored`]. A missing file is an empty store.
    pub fn open(path: impl Into<PathBuf>, master: PublicKey) -> Result<Self, KeptError> {
        Self::open_inner(path.into(), master, None)
    }

    /// Open the store at `path` for `master` with a floor source: as [`Self::open`]; a store holding
    /// nothing, or a list below the floor, opens with the floor pending ([`Self::floor_pending`]).
    ///
    /// # Errors
    ///
    /// As [`Self::open`], then [`KeptError::Floor`] when the floor cannot be read.
    pub fn open_with_floor(
        path: impl Into<PathBuf>,
        master: PublicKey,
        floor: Box<dyn FloorSource + Send>,
    ) -> Result<Self, KeptError> {
        Self::open_inner(path.into(), master, Some(floor))
    }

    fn open_inner(
        path: PathBuf,
        master: PublicKey,
        floor: Option<Box<dyn FloorSource + Send>>,
    ) -> Result<Self, KeptError> {
        let lock = lock(&path, DEFAULT_LOCK_WAIT)?;
        let mut state = read_state(&path, &master)?;
        let pending = check_floor(floor.as_deref(), &state)?;
        if pending.is_some() {
            // a list below the floor is not used for anything
            state = DelegatedRevocations::new();
        }
        drop(lock);
        Ok(Self {
            path,
            master,
            state,
            lock_wait: DEFAULT_LOCK_WAIT,
            floor,
            pending,
        })
    }

    /// The same store waiting at most `wait` for the lock in [`Self::accept`].
    #[must_use]
    pub const fn with_lock_wait(mut self, wait: Duration) -> Self {
        self.lock_wait = wait;
        self
    }

    /// Accept `list`, signed by the delegate `delegate` certifies, at `now` with the freshness `window`
    /// (the steps of the module documentation).
    ///
    /// # Errors
    ///
    /// [`KeptError::LockTimeout`]; the errors of reading the store as [`Self::open`] (the file is read
    /// again); [`KeptError::Floor`] / [`KeptError::RolledBack`] with a floor source;
    /// [`KeptError::Refused`] when the list is refused (nothing written; a conflict carries the SHA-256 of
    /// both lists); [`KeptError::TooLarge`] / [`KeptError::Io`] when the new state is not written (the
    /// stored one stays); [`KeptError::FloorNotRaised`] when the new state was written and is kept but the
    /// floor was not raised.
    pub fn accept(
        &mut self,
        list: RevocationList,
        delegate: &DeviceCert,
        now: u64,
        window: u64,
    ) -> Result<Accepted, KeptError> {
        let lock = lock(&self.path, self.lock_wait)?;
        let mut next = read_state(&self.path, &self.master)?;
        let pending = check_floor(self.floor.as_deref(), &next)?;
        if pending.is_some() {
            // the stored list is below the floor (or absent): accept against nothing, never against it
            next = DelegatedRevocations::new();
        }
        let got = next
            .accept(list, delegate, &self.master, now, window)
            .map_err(KeptError::Refused)?;
        // with the floor pending, the accepted list must reach it before anything is written
        if let Some(FloorPending { floor, .. }) = pending {
            let stored = pair(&next);
            if stored.is_none_or(|p| p < floor) {
                return Err(KeptError::RolledBack { stored, floor });
            }
        }
        if got == Accepted::Replaced {
            let bytes = next.to_bytes();
            if bytes.len() > MAX_KEPT_FILE_LEN {
                return Err(KeptError::TooLarge {
                    len: bytes.len() as u64,
                });
            }
            #[cfg(test)]
            tests::hook::pause();
            write_atomically(&self.path, &bytes, &mut |_| Ok(())).map_err(io_error("write"))?;
            self.state = next;
            // under the lock, so two processes raise the floor in the order they wrote
            if let (Some(floor), Some(pair)) = (&self.floor, pair(&self.state)) {
                floor
                    .raise_floor(pair.0, pair.1)
                    .map_err(KeptError::FloorNotRaised)?;
            }
        } else {
            self.state = next;
        }
        self.pending = None;
        drop(lock);
        Ok(got)
    }

    /// Refuse unless a list is kept and it is fresh at `now` for `window`.
    ///
    /// # Errors
    ///
    /// As `DelegatedRevocations::require_fresh`.
    pub fn require_fresh(&self, now: u64, window: u64) -> Result<(), DelegatedRevocationError> {
        self.state.require_fresh(now, window)
    }

    /// Whether the kept list revokes `cert` (freshness is not checked; see [`Self::require_fresh`]).
    #[must_use]
    pub fn revokes(&self, cert: &DeviceCert) -> bool {
        self.state.revokes(cert)
    }

    /// The kept list as a [`Revocation`] for a channel policy (freshness is not checked).
    #[must_use]
    pub fn revocation(&self) -> Revocation<'_> {
        self.state.revocation()
    }

    /// The epoch of the kept list.
    #[must_use]
    pub fn epoch(&self) -> Option<u64> {
        self.state.epoch()
    }

    /// The sequence of the kept list.
    #[must_use]
    pub fn sequence(&self) -> Option<u64> {
        self.state.sequence()
    }

    /// The store's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The floor this store must reach before anything it holds is used: `Some` when it opened holding
    /// nothing, or a list below the floor, while the floor source held a pair (see the module
    /// documentation), until a list at or above it is accepted. [`RevocationRule::apply`] refuses every
    /// relay meanwhile; show it to the user, and log [`FloorPending::stored`] when it is `Some`.
    #[must_use]
    pub const fn floor_pending(&self) -> Option<FloorPending> {
        self.pending
    }
}

fn pair(state: &DelegatedRevocations) -> Option<(u64, u64)> {
    state.epoch().zip(state.sequence())
}

/// The stored pair against the floor: nothing stored, or a pair below it (an older file put back, a
/// restored backup), gives the floor as pending (see the module documentation)
fn check_floor(
    floor: Option<&(dyn FloorSource + Send)>,
    state: &DelegatedRevocations,
) -> Result<Option<FloorPending>, KeptError> {
    let Some(source) = floor else {
        return Ok(None);
    };
    let Some(floor) = source.read_floor().map_err(KeptError::Floor)? else {
        return Ok(None);
    };
    let stored = pair(state);
    if stored.is_none_or(|p| p < floor) {
        return Ok(Some(FloorPending { floor, stored }));
    }
    Ok(None)
}

/// `<dir>/<name><suffix>` beside `path`, with `<prefix>` before the name
fn sibling(path: &Path, prefix: &str, suffix: &str) -> io::Result<PathBuf> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "the store path names no file")
    })?;
    let mut sib = std::ffi::OsString::from(prefix);
    sib.push(name);
    sib.push(suffix);
    Ok(path.with_file_name(sib))
}

/// Take the exclusive lock on `<path>.lock`, trying again every [`LOCK_POLL`] for at most `wait`; the lock
/// is held while the returned file is open
// File::try_lock is stable from 1.89: the feature anonymous-relay-client needs 1.89 (the crate's
// rust-version, 1.88, holds without it); the CI msrv job builds and runs this module with 1.89
#[allow(clippy::incompatible_msrv)]
fn lock(path: &Path, wait: Duration) -> Result<File, KeptError> {
    let lock_path = sibling(path, "", ".lock").map_err(io_error("lock path"))?;
    // a link planted at the lock path is refused rather than followed
    match fs::symlink_metadata(&lock_path) {
        Ok(m) if !m.is_file() => return Err(KeptError::NotAFile),
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_error("stat lock")(e)),
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(io_error("open lock"))?;
    // counted in polls, not read from the clock: the wait is at least `wait`
    let polls = wait.as_nanos() / LOCK_POLL.as_nanos();
    let mut tried = 0u128;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(TryLockError::WouldBlock) => {
                if tried >= polls {
                    return Err(KeptError::LockTimeout { waited: wait });
                }
                tried += 1;
                std::thread::sleep(LOCK_POLL);
            }
            Err(TryLockError::Error(e)) => return Err(io_error("lock")(e)),
        }
    }
}

/// The stored state at `path` checked under `master`; a missing file is an empty state
fn read_state(path: &Path, master: &PublicKey) -> Result<DelegatedRevocations, KeptError> {
    // symlink_metadata: a link at the store path is not followed (it is not a file)
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(DelegatedRevocations::new()),
        Err(e) => return Err(io_error("stat")(e)),
    };
    if !meta.is_file() {
        return Err(KeptError::NotAFile);
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(io_error("open"))?
        .take(MAX_KEPT_FILE_LEN as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error("read"))?;
    if bytes.is_empty() {
        return Err(KeptError::Empty);
    }
    if bytes.len() > MAX_KEPT_FILE_LEN {
        return Err(KeptError::TooLarge {
            len: meta.len().max(bytes.len() as u64),
        });
    }
    DelegatedRevocations::restore(&bytes, master).map_err(KeptError::Stored)
}

/// Write `bytes` to `path` atomically: a temporary file in the same directory, synced, `before_rename` (a
/// failure there leaves `path` as it was and removes the temporary file), renamed over `path`, and on unix
/// the directory synced
fn write_atomically(
    path: &Path,
    bytes: &[u8],
    before_rename: &mut dyn FnMut(&Path) -> io::Result<()>,
) -> io::Result<()> {
    let tmp = sibling(path, ".", &format!(".{}.tmp", std::process::id()))?;
    // a stale temporary file (or a link planted there) is removed, not followed; create_new then refuses
    // whatever appears at the name in between instead of writing through it
    match fs::remove_file(&tmp) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let written = (|| {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        before_rename(&tmp)?;
        fs::rename(&tmp, path)
    })();
    if let Err(e) = written {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => Path::new("."),
        };
        File::open(dir)?.sync_all()?;
    }
    Ok(())
}

/// What a connection path requires of the relay's revocation list before it trusts the relay (see the
/// module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RevocationRule {
    /// The freshness window, in seconds.
    pub window: u64,
    /// Whether a fresh list is required: when the served list is not used (the fetch failed, or it is
    /// older than the kept one) and no fresh list is kept, the relay is refused. `false` lets the relay
    /// through then (an explicit opt-out); a served list refused for any other reason, a store that fails,
    /// and a kept list naming the relay still refuse it.
    pub required: bool,
}

impl Default for RevocationRule {
    /// The 24-hour window (`DEFAULT_REVOCATION_FRESHNESS_WINDOW_SECS`), required.
    fn default() -> Self {
        Self {
            window: DEFAULT_REVOCATION_FRESHNESS_WINDOW_SECS,
            required: true,
        }
    }
}

/// Why a served list was not used by [`RevocationRule::apply`] (the relay was not refused for it): for a
/// caller to log.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServedNotUsed {
    /// The fetch failed (404, I/O, a body that is not one).
    Fetch(FetchError),
    /// The served list is older than the kept one (`OlderEpoch` / `OlderSequence`): a relay or a cache
    /// that lags behind another, not a list that contradicts the kept one.
    Older(DelegatedRevocationError),
}

impl core::fmt::Display for ServedNotUsed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fetch(e) => write!(f, "fetch: {e}"),
            Self::Older(e) => write!(f, "older list: {e}"),
        }
    }
}

/// Why [`RevocationRule`] refused the relay.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RevocationRuleError {
    /// The served list was refused, or the store failed ([`KeptError`]).
    Kept(KeptError),
    /// A fresh list is required and none is kept; `served` is why the served list was not used, if it
    /// was not.
    NotFresh {
        /// Why the served list was not used, `None` when it was accepted.
        served: Option<Box<ServedNotUsed>>,
        /// Nothing kept, stale or from the future.
        reason: DelegatedRevocationError,
    },
    /// The kept list revokes the relay's certificate.
    Revoked,
    /// The store opened holding nothing under a floor and no list has reached it yet
    /// ([`KeptRevocations::floor_pending`]): no relay is admitted, whether or not a fresh list is required.
    FloorPending {
        /// The floor the store waits for, and the older list it found and ignores, if any.
        pending: FloorPending,
        /// Why the served list was not used, if it was not.
        served: Option<Box<ServedNotUsed>>,
    },
}

impl core::fmt::Display for RevocationRuleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Kept(e) => write!(f, "{e}"),
            Self::NotFresh {
                served: Some(served),
                reason,
            } => write!(f, "no fresh revocation list ({reason}); {served}"),
            Self::NotFresh {
                served: None,
                reason,
            } => write!(f, "no fresh revocation list ({reason})"),
            Self::Revoked => f.write_str("the relay's certificate is revoked"),
            Self::FloorPending { pending, served } => {
                match pending.stored {
                    None => write!(
                        f,
                        "the revocation store holds no list and waits for one at or above {:?}",
                        pending.floor
                    )?,
                    Some(stored) => write!(
                        f,
                        "the revocation store found an older list {stored:?}, below the floor {:?}, \
                         ignores it and waits for one at or above the floor",
                        pending.floor
                    )?,
                }
                match served {
                    Some(served) => write!(f, "; {served}"),
                    None => Ok(()),
                }
            }
        }
    }
}

impl std::error::Error for RevocationRuleError {}

impl RevocationRule {
    /// Fetch the list on `stream` ([`super::fetch_relay_revocations`]), then [`Self::apply`].
    ///
    /// # Errors
    ///
    /// As [`Self::apply`].
    pub fn apply_fetch<S: Read + Write>(
        &self,
        kept: &mut KeptRevocations,
        stream: S,
        host: &str,
        relay_cert: &DeviceCert,
        now: u64,
    ) -> Result<Option<ServedNotUsed>, RevocationRuleError> {
        self.apply(kept, fetch_relay_revocations(stream, host), relay_cert, now)
    }

    /// Accept what was fetched into `kept`, then decide about `relay_cert` at `now`:
    ///
    /// 1. a served body is accepted into the store; a failed fetch (404, I/O, a body that is not one) and
    ///    a list merely older than the kept one (`OlderEpoch`, `OlderSequence`: two honest relays or a
    ///    cache may lag behind each other) go on with what is kept ([`ServedNotUsed`]); every other
    ///    refusal (a conflict, another delegate in the epoch, a wrong signer or signature, a delegate
    ///    certificate refused, a stale or future list, a dropped revocation) and a store failure (I/O,
    ///    lock, floor) refuse the relay ([`RevocationRuleError::Kept`]);
    /// 2. a kept list that revokes the relay refuses it ([`RevocationRuleError::Revoked`]);
    /// 3. a store whose floor is still pending ([`KeptRevocations::floor_pending`]) refuses the relay,
    ///    required or not ([`RevocationRuleError::FloorPending`]);
    /// 4. when [`Self::required`], nothing kept or a kept list that is not fresh refuses the relay
    ///    ([`RevocationRuleError::NotFresh`]).
    ///
    /// Returns why the served list was not used, if it was not (for a caller to log), `None` when it was
    /// accepted.
    ///
    /// # Errors
    ///
    /// See the steps above.
    pub fn apply(
        &self,
        kept: &mut KeptRevocations,
        fetched: Result<DelegatedListResponse, FetchError>,
        relay_cert: &DeviceCert,
        now: u64,
    ) -> Result<Option<ServedNotUsed>, RevocationRuleError> {
        let served = match fetched {
            Ok(served) => match kept.accept(served.list, &served.delegate, now, self.window) {
                Ok(_) => None,
                Err(KeptError::Refused(
                    e @ (DelegatedRevocationError::OlderEpoch { .. }
                    | DelegatedRevocationError::OlderSequence { .. }),
                )) => Some(ServedNotUsed::Older(e)),
                Err(e) => return Err(RevocationRuleError::Kept(e)),
            },
            Err(e) => Some(ServedNotUsed::Fetch(e)),
        };
        if let Some(pending) = kept.floor_pending() {
            return Err(RevocationRuleError::FloorPending {
                pending,
                served: served.map(Box::new),
            });
        }
        if kept.revokes(relay_cert) {
            return Err(RevocationRuleError::Revoked);
        }
        match kept.require_fresh(now, self.window) {
            Err(reason) if self.required => Err(RevocationRuleError::NotFresh {
                served: served.map(Box::new),
                reason,
            }),
            Ok(()) | Err(_) => Ok(served),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alice_identity::delegated_revocation::issue_revocation_delegate;
    use alice_sign::KeyPair;
    use std::process::Command;

    /// A test-only pause between the re-read and the write, so a race between processes is not left to
    /// scheduling: the child processes of the race test set it from their environment
    pub(super) mod hook {
        use std::sync::atomic::{AtomicU64, Ordering};

        pub static PAUSE_MICROS: AtomicU64 = AtomicU64::new(0);

        pub fn pause() {
            let us = PAUSE_MICROS.load(Ordering::Relaxed);
            if us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(us));
            }
        }
    }

    const T0: u64 = 1_800_000_000;
    const WINDOW: u64 = 86_400;
    const NOW: u64 = T0 + 20;

    /// Test fixtures, never production keys
    fn master() -> KeyPair {
        KeyPair::from_seed([0x5a; 32])
    }
    fn delegate_key() -> KeyPair {
        KeyPair::from_seed([0x21; 32])
    }
    fn delegate() -> DeviceCert {
        issue_revocation_delegate(&master(), delegate_key().public().0, T0, T0 + 7 * 86_400)
            .unwrap()
    }
    fn list(seq: u64) -> RevocationList {
        RevocationList::issue(&delegate_key(), seq, T0 + 10, &[]).unwrap()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "alice-sync-kept-unit-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn stored_bytes(seq: u64) -> Vec<u8> {
        let mut s = DelegatedRevocations::new();
        s.accept(list(seq), &delegate(), &master().public(), NOW, WINDOW)
            .unwrap();
        s.to_bytes()
    }

    /// A write that fails before the rename leaves the old file whole: reopening gives the old state, and
    /// no temporary file is left
    #[test]
    fn a_write_failing_before_the_rename_leaves_the_old_file() {
        let dir = scratch("atomic");
        let path = dir.join("kept.bin");
        let old = stored_bytes(5);
        write_atomically(&path, &old, &mut |_| Ok(())).unwrap();
        let new = stored_bytes(6);
        let failed = write_atomically(&path, &new, &mut |_| {
            Err(io::Error::other("injected before the rename"))
        });
        assert!(failed.is_err());
        assert_eq!(fs::read(&path).unwrap(), old);
        let kept = KeptRevocations::open(&path, master().public()).unwrap();
        assert_eq!((kept.epoch(), kept.sequence()), (Some(T0), Some(5)));
        let left: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n.to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(left.is_empty(), "{left:?}");
        // and a write that goes through replaces it
        write_atomically(&path, &new, &mut |_| Ok(())).unwrap();
        assert_eq!(fs::read(&path).unwrap(), new);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Links planted in the data directory are not followed: a link at the temporary name is replaced, not
    /// written through, and a link at the store path or the lock path is refused
    #[cfg(unix)]
    #[test]
    fn links_in_the_data_directory_are_not_followed() {
        use std::os::unix::fs::symlink;
        let dir = scratch("links");
        let target = dir.join("target.txt");
        fs::write(&target, b"untouched").unwrap();
        let path = dir.join("kept.bin");
        let tmp = dir.join(format!(".kept.bin.{}.tmp", std::process::id()));
        symlink(&target, &tmp).unwrap();
        let bytes = stored_bytes(5);
        write_atomically(&path, &bytes, &mut |_| Ok(())).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"untouched");
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(!fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
        let mut refused = 0;
        // a link at the store path
        let linked = dir.join("linked.bin");
        symlink(&path, &linked).unwrap();
        assert!(matches!(
            KeptRevocations::open(&linked, master().public()),
            Err(KeptError::NotAFile)
        ));
        refused += 1;
        // a link at the lock path
        let other = dir.join("other.bin");
        symlink(&target, dir.join("other.bin.lock")).unwrap();
        assert!(matches!(
            KeptRevocations::open(&other, master().public()),
            Err(KeptError::NotAFile)
        ));
        refused += 1;
        assert_eq!(fs::read(&target).unwrap(), b"untouched");
        assert_eq!(refused, 2);
        fs::remove_dir_all(&dir).unwrap();
    }

    const RACE_CHILD: &str = "ALICE_SYNC_KEPT_RACE_CHILD";
    const RACE_PATH: &str = "ALICE_SYNC_KEPT_RACE_PATH";
    const RACE_SEQS: &str = "ALICE_SYNC_KEPT_RACE_SEQS";
    const RACE_PAUSE: &str = "ALICE_SYNC_KEPT_RACE_PAUSE_US";
    const PROCESSES: u64 = 8;
    const PER_PROCESS: u64 = 16;
    const ROUNDS: u64 = 3;

    fn stored_sequence(path: &Path) -> Option<u64> {
        let bytes = fs::read(path).ok()?;
        DelegatedRevocations::restore(&bytes, &master().public())
            .unwrap()
            .sequence()
    }

    /// The child role of [`processes_racing_on_one_store_never_lower_it`]: with no role in the environment
    /// it does nothing. A child opens the store once, then offers its sequences in its order; after every
    /// offer it reads the stored sequence and asserts it is not below what it saw before
    #[test]
    fn race_child_role() {
        let Ok(child) = std::env::var(RACE_CHILD) else {
            return;
        };
        let path = PathBuf::from(std::env::var(RACE_PATH).unwrap());
        let seqs: Vec<u64> = std::env::var(RACE_SEQS)
            .unwrap()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let pause: u64 = std::env::var(RACE_PAUSE).unwrap().parse().unwrap();
        hook::PAUSE_MICROS.store(pause, std::sync::atomic::Ordering::Relaxed);
        let d = delegate();
        let mut kept = KeptRevocations::open(&path, master().public())
            .unwrap()
            .with_lock_wait(Duration::from_secs(60));
        let mut seen = stored_sequence(&path).unwrap_or(0);
        let mut compared = 0u64;
        for seq in seqs {
            match kept.accept(list(seq), &d, NOW, WINDOW) {
                Ok(_) | Err(KeptError::Refused(DelegatedRevocationError::OlderSequence { .. })) => {
                }
                Err(e) => panic!("child {child}: seq {seq}: {e}"),
            }
            let now_stored = stored_sequence(&path).unwrap();
            assert!(
                now_stored >= seen,
                "child {child}: stored sequence went from {seen} down to {now_stored}"
            );
            seen = now_stored;
            compared += 1;
        }
        println!("RACE|compared={compared}|");
    }

    /// N processes accept lists of one epoch in random order into one store, for several rounds: no process
    /// ever sees the stored sequence fall, and at the end it is the highest offered. A pause between the
    /// re-read and the write (the test hook) makes the overlap certain rather than left to scheduling
    #[test]
    fn processes_racing_on_one_store_never_lower_it() {
        if std::env::var(RACE_CHILD).is_ok() {
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let dir = scratch("race");
        let mut compared = 0u64;
        for round in 0..ROUNDS {
            let path = dir.join(format!("kept-{round}.bin"));
            let total = PROCESSES * PER_PROCESS;
            let mut children = Vec::new();
            for child in 0..PROCESSES {
                // child `c` offers c+1, c+1+N, ... in an order shuffled by a fixed per-child generator
                let mut seqs: Vec<u64> = (0..PER_PROCESS)
                    .map(|k| child + 1 + k * PROCESSES)
                    .collect();
                let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ (round << 32) ^ (child + 1);
                for i in (1..seqs.len()).rev() {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    seqs.swap(i, (x % (i as u64 + 1)) as usize);
                }
                let seqs: Vec<String> = seqs.iter().map(u64::to_string).collect();
                children.push(
                    Command::new(&exe)
                        .args([
                            "--exact",
                            "anonymous_relay_client::kept::tests::race_child_role",
                            "--nocapture",
                            "--test-threads=1",
                        ])
                        .env(RACE_CHILD, child.to_string())
                        .env(RACE_PATH, &path)
                        .env(RACE_SEQS, seqs.join(","))
                        .env(RACE_PAUSE, "2000")
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .spawn()
                        .unwrap(),
                );
            }
            for c in children {
                let out = c.wait_with_output().unwrap();
                let stdout = String::from_utf8_lossy(&out.stdout);
                assert!(
                    out.status.success(),
                    "round {round}: a child failed:\n{stdout}\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let n: u64 = stdout
                    .split("RACE|compared=")
                    .nth(1)
                    .and_then(|s| s.split('|').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| panic!("round {round}: no count from a child:\n{stdout}"));
                compared += n;
            }
            assert_eq!(stored_sequence(&path), Some(total), "round {round}");
        }
        assert_eq!(compared, ROUNDS * PROCESSES * PER_PROCESS);
        assert!(compared > 0);
        fs::remove_dir_all(&dir).unwrap();
    }
}
