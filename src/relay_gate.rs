//! Relay gate: decides, per payload, whether this device relays for others.
//!
//! All five conditions must hold, checked in this order:
//! 1. consent: a gate exists only with a [`RelayConsent`], and [`RelayGate::revoke_consent`] stops it
//! 2. network: not metered; an unknown network counts as metered unless the user marked that
//!    network as not metered
//! 3. load: unreadable readings, a full-screen app, a serious thermal state or a low battery count
//!    as loaded; relaying stops at [`LOAD_STOP_PERCENT`] and resumes only after
//!    [`LOAD_QUIET_MS`] at or below [`LOAD_RESUME_PERCENT`]
//! 4. shape: the payload must look like a sealed channel packet ([`CiphertextShape`])
//! 5. budget: the user's bandwidth budget ([`BandwidthBudget`], injected), spent only when 1-4 hold
//!
//! The gate reads no clock and no OS state itself: readings and times are passed in, so the same
//! inputs give the same verdicts.

use std::collections::BTreeSet;

/// CPU or GPU use at which relaying stops, percent.
pub const LOAD_STOP_PERCENT: u8 = 70;
/// CPU or GPU use at or below which the quiet period counts, percent.
pub const LOAD_RESUME_PERCENT: u8 = 50;
/// Quiet period before relaying resumes, milliseconds.
pub const LOAD_QUIET_MS: u64 = 60_000;

/// The user's consent to relay for others (made by a user action and stored with its time).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelayConsent {
    granted_at_unix: u64,
}

impl RelayConsent {
    /// Consent the user gave at `granted_at_unix` (seconds). Call this only from the user's own
    /// action in the settings screen, never from a remote message, an environment variable or a
    /// settings migration.
    #[must_use]
    pub const fn granted_by_user(granted_at_unix: u64) -> Self {
        Self { granted_at_unix }
    }

    /// When the consent was given (seconds since the Unix epoch).
    #[must_use]
    pub const fn granted_at_unix(&self) -> u64 {
        self.granted_at_unix
    }
}

/// An identifier the OS gives the current network (SSID hash, interface id, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NetworkId(u64);

impl NetworkId {
    /// A network id from its number.
    #[must_use]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

/// Cost of the current network as the OS reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetworkCost {
    /// Not metered.
    NotMetered,
    /// Metered (mobile data, tethering, a data limit).
    Metered,
    /// The OS could not tell.
    Unknown,
}

/// One reading of the device's load. `None` means the value could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoadReading {
    /// Whole-system CPU use, percent.
    pub cpu_percent: Option<u8>,
    /// GPU use, percent.
    pub gpu_percent: Option<u8>,
    /// Thermal state serious or worse.
    pub thermal_serious: Option<bool>,
    /// On battery and below the threshold, or in power-saving mode.
    pub battery_low: Option<bool>,
    /// A full-screen app (a game) in front.
    pub fullscreen: Option<bool>,
}

impl LoadReading {
    fn loaded(&self) -> bool {
        let pct = |v: Option<u8>| v.is_none_or(|p| p >= LOAD_STOP_PERCENT);
        let flag = |v: Option<bool>| v.unwrap_or(true);
        pct(self.cpu_percent)
            || pct(self.gpu_percent)
            || flag(self.thermal_serious)
            || flag(self.battery_low)
            || flag(self.fullscreen)
    }

    fn calm(&self) -> bool {
        let pct = |v: Option<u8>| v.is_some_and(|p| p <= LOAD_RESUME_PERCENT);
        let flag = |v: Option<bool>| v == Some(false);
        pct(self.cpu_percent)
            && pct(self.gpu_percent)
            && flag(self.thermal_serious)
            && flag(self.battery_low)
            && flag(self.fullscreen)
    }
}

/// The shape a sealed channel packet has: its first byte is one of `kinds` and its length is
/// within `min_len..=max_len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CiphertextShape {
    /// Allowed first bytes (packet kinds).
    pub kinds: &'static [u8],
    /// Shortest sealed packet (kind + counter + tag).
    pub min_len: usize,
    /// Longest packet the relay carries.
    pub max_len: usize,
}

impl CiphertextShape {
    fn fits(&self, payload: &[u8]) -> bool {
        payload.len() >= self.min_len
            && payload.len() <= self.max_len
            && payload.first().is_some_and(|k| self.kinds.contains(k))
    }
}

/// The user's bandwidth budget for relaying (a token bucket, a daily cap, ...).
pub trait BandwidthBudget {
    /// Spend `bytes` at `now_ms` if the budget allows; `false` spends nothing.
    fn try_spend(&mut self, bytes: u64, now_ms: u64) -> bool;
}

/// Why the gate refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    /// No consent (or it was revoked).
    NotEnabled,
    /// Metered or unknown network.
    Metered,
    /// The device is loaded.
    Loaded,
    /// The payload does not have the shape of a sealed packet.
    NotCiphertext,
    /// The bandwidth budget is used up.
    OverBudget,
}

/// The gate's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// Relay the payload.
    Relay,
    /// Do not relay.
    Refuse(Reason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Load {
    /// No reading yet.
    Unknown,
    Idle,
    /// Stopped; `quiet_since` is the start of the current calm period.
    Loaded {
        quiet_since: Option<u64>,
    },
}

/// The gate state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelayGate {
    consent: Option<RelayConsent>,
    shape: CiphertextShape,
    network: Option<(NetworkId, NetworkCost)>,
    not_metered_by_user: BTreeSet<NetworkId>,
    load: Load,
}

impl RelayGate {
    /// A gate for a user who gave consent.
    #[must_use]
    pub fn new(consent: RelayConsent, shape: CiphertextShape) -> Self {
        Self {
            consent: Some(consent),
            shape,
            network: None,
            not_metered_by_user: BTreeSet::new(),
            load: Load::Unknown,
        }
    }

    /// The user withdrew consent: refuse from now on.
    pub fn revoke_consent(&mut self) {
        self.consent = None;
    }

    /// The OS reported the current network.
    pub fn observe_network(&mut self, id: NetworkId, cost: NetworkCost) {
        self.network = Some((id, cost));
    }

    /// The user marked `id` as not metered.
    pub fn user_says_not_metered(&mut self, id: NetworkId) {
        self.not_metered_by_user.insert(id);
    }

    /// A load reading at `now_ms`.
    pub fn observe_load(&mut self, reading: LoadReading, now_ms: u64) {
        self.load = match self.load {
            Load::Unknown | Load::Idle if reading.loaded() => Load::Loaded { quiet_since: None },
            Load::Unknown if reading.calm() => Load::Idle,
            Load::Unknown => Load::Unknown,
            Load::Idle => Load::Idle,
            Load::Loaded { quiet_since } => {
                if reading.calm() {
                    let since = quiet_since.unwrap_or(now_ms);
                    if now_ms.saturating_sub(since) >= LOAD_QUIET_MS {
                        Load::Idle
                    } else {
                        Load::Loaded {
                            quiet_since: Some(since),
                        }
                    }
                } else {
                    Load::Loaded { quiet_since: None }
                }
            }
        };
    }

    fn network_ok(&self) -> bool {
        match self.network {
            Some((_, NetworkCost::NotMetered)) => true,
            Some((id, _)) => self.not_metered_by_user.contains(&id),
            None => false,
        }
    }

    /// Decide for one payload at `now_ms`; the budget is spent only when everything else passed.
    pub fn admit(
        &mut self,
        payload: &[u8],
        now_ms: u64,
        budget: &mut impl BandwidthBudget,
    ) -> Verdict {
        if self.consent.is_none() {
            return Verdict::Refuse(Reason::NotEnabled);
        }
        if !self.network_ok() {
            return Verdict::Refuse(Reason::Metered);
        }
        if !matches!(self.load, Load::Idle) {
            return Verdict::Refuse(Reason::Loaded);
        }
        if !self.shape.fits(payload) {
            return Verdict::Refuse(Reason::NotCiphertext);
        }
        if !budget.try_spend(payload.len() as u64, now_ms) {
            return Verdict::Refuse(Reason::OverBudget);
        }
        Verdict::Relay
    }
}
