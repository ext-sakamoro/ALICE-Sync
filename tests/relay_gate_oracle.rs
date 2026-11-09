//! Contract tests for the relay gate: relaying for others happens only when all five conditions hold.
//!
//! 1. opt-in: a gate exists only with the user's consent, and revoking it stops relaying
//! 2. a bandwidth budget chosen by the user (the budget implementation is injected)
//! 3. not on a metered network; "unknown" counts as metered; a per-network user override exists
//! 4. ciphertext only: the payload must have the shape of a sealed channel packet
//! 5. not while the device is loaded; unreadable readings count as loaded; stop at 70 %, resume
//!    only after 60 s at or below 50 %
//!
//! Reasons are checked in that order, and the budget is spent only when everything else passed.

use alice_sync::relay_gate::{
    BandwidthBudget, CiphertextShape, LoadReading, NetworkCost, NetworkId, Reason, RelayConsent,
    RelayGate, Verdict,
};

/// Test budget: a plain byte counter per 1000 ms window (reference implementation for the test).
struct Window {
    per_window: u64,
    window: u64,
    used: u64,
    spent: u64,
}

impl BandwidthBudget for Window {
    fn try_spend(&mut self, bytes: u64, now_ms: u64) -> bool {
        if now_ms / 1000 != self.window {
            self.window = now_ms / 1000;
            self.used = 0;
        }
        if self.used + bytes > self.per_window {
            return false;
        }
        self.used += bytes;
        self.spent += bytes;
        true
    }
}

const SHAPE: CiphertextShape = CiphertextShape {
    kinds: &[1, 2, 3],
    min_len: 25,
    max_len: 1200,
};
const HOME: NetworkId = NetworkId::from_raw(1);
const PHONE: NetworkId = NetworkId::from_raw(2);

fn sealed(len: usize) -> Vec<u8> {
    let mut v = vec![0xab; len];
    v[0] = 3;
    v
}

fn idle() -> LoadReading {
    LoadReading {
        cpu_percent: Some(10),
        gpu_percent: Some(5),
        thermal_serious: Some(false),
        battery_low: Some(false),
        fullscreen: Some(false),
    }
}

fn ready_gate() -> RelayGate {
    let mut g = RelayGate::new(RelayConsent::granted_by_user(1_700_000_000), SHAPE);
    g.observe_network(HOME, NetworkCost::NotMetered);
    g.observe_load(idle(), 0);
    g
}

fn budget() -> Window {
    Window {
        per_window: 10_000,
        window: 0,
        used: 0,
        spent: 0,
    }
}

/// oracle: everything holds -> relay, and the budget is spent by the payload length.
#[test]
fn all_conditions_hold() {
    let mut g = ready_gate();
    let mut b = budget();
    assert_eq!(g.admit(&sealed(100), 10, &mut b), Verdict::Relay);
    assert_eq!(b.spent, 100);
}

/// oracle (1): after revoking consent the gate refuses.
#[test]
fn revoked_consent_stops_relaying() {
    let mut g = ready_gate();
    g.revoke_consent();
    assert_eq!(
        g.admit(&sealed(100), 10, &mut budget()),
        Verdict::Refuse(Reason::NotEnabled)
    );
}

/// oracle (3): metered and unknown refuse; the user's "not metered" override applies only to
/// the network it names.
#[test]
fn metered_and_unknown_refuse_with_per_network_override() {
    let mut g = ready_gate();
    g.observe_network(PHONE, NetworkCost::Metered);
    assert_eq!(
        g.admit(&sealed(100), 10, &mut budget()),
        Verdict::Refuse(Reason::Metered)
    );
    g.observe_network(PHONE, NetworkCost::Unknown);
    assert_eq!(
        g.admit(&sealed(100), 10, &mut budget()),
        Verdict::Refuse(Reason::Metered)
    );
    g.user_says_not_metered(HOME);
    assert_eq!(
        g.admit(&sealed(100), 10, &mut budget()),
        Verdict::Refuse(Reason::Metered),
        "override is for HOME only"
    );
    g.user_says_not_metered(PHONE);
    assert_eq!(g.admit(&sealed(100), 10, &mut budget()), Verdict::Relay);
    // a gate that never saw a network reading treats the network as unknown
    let mut fresh = RelayGate::new(RelayConsent::granted_by_user(0), SHAPE);
    fresh.observe_load(idle(), 0);
    assert_eq!(
        fresh.admit(&sealed(100), 0, &mut budget()),
        Verdict::Refuse(Reason::Metered)
    );
}

/// oracle (5): 75 % stops; it resumes only after 60 000 ms at or below 50 %; unreadable
/// readings and full-screen apps count as loaded.
#[test]
fn load_hysteresis() {
    let mut g = ready_gate();
    let mut b = budget();
    g.observe_load(
        LoadReading {
            cpu_percent: Some(75),
            ..idle()
        },
        1_000,
    );
    assert_eq!(
        g.admit(&sealed(100), 1_000, &mut b),
        Verdict::Refuse(Reason::Loaded)
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(60),
            ..idle()
        },
        2_000,
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(45),
            ..idle()
        },
        3_000,
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(45),
            ..idle()
        },
        62_999,
    );
    assert_eq!(
        g.admit(&sealed(100), 62_999, &mut b),
        Verdict::Refuse(Reason::Loaded)
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(45),
            ..idle()
        },
        63_000,
    );
    assert_eq!(g.admit(&sealed(100), 63_000, &mut b), Verdict::Relay);
    // a reading above 50 % restarts the quiet period
    g.observe_load(
        LoadReading {
            cpu_percent: Some(80),
            ..idle()
        },
        64_000,
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(40),
            ..idle()
        },
        64_500,
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(55),
            ..idle()
        },
        65_000,
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(40),
            ..idle()
        },
        66_000,
    );
    g.observe_load(
        LoadReading {
            cpu_percent: Some(40),
            ..idle()
        },
        125_999,
    );
    assert_eq!(
        g.admit(&sealed(100), 125_999, &mut b),
        Verdict::Refuse(Reason::Loaded)
    );
    for unreadable in [
        LoadReading {
            cpu_percent: None,
            ..idle()
        },
        LoadReading {
            fullscreen: Some(true),
            ..idle()
        },
        LoadReading {
            thermal_serious: None,
            ..idle()
        },
        LoadReading {
            battery_low: Some(true),
            ..idle()
        },
    ] {
        let mut g = ready_gate();
        g.observe_load(unreadable, 10);
        assert_eq!(
            g.admit(&sealed(100), 10, &mut budget()),
            Verdict::Refuse(Reason::Loaded)
        );
    }
}

/// oracle (4): wrong kind byte, too short (no room for the counter and tag) or too long refuse,
/// and the budget is not spent for a refused payload.
#[test]
fn only_ciphertext_shapes_pass_and_refusals_do_not_spend() {
    let mut g = ready_gate();
    let mut b = budget();
    let mut wrong = sealed(100);
    wrong[0] = 0x7f;
    for p in [wrong, sealed(24), sealed(1201), Vec::new()] {
        assert_eq!(
            g.admit(&p, 10, &mut b),
            Verdict::Refuse(Reason::NotCiphertext)
        );
    }
    assert_eq!(b.spent, 0);
    assert_eq!(g.admit(&sealed(25), 10, &mut b), Verdict::Relay);
}

/// oracle (2): the injected budget's refusal is reported, and over 10 windows the relayed bytes
/// never exceed the budget per window.
#[test]
fn budget_limits_relayed_bytes() {
    let mut g = ready_gate();
    let mut b = Window {
        per_window: 1_000,
        window: 0,
        used: 0,
        spent: 0,
    };
    let mut relayed = 0u64;
    for t in 0..10_000u64 {
        if g.admit(&sealed(300), t, &mut b) == Verdict::Relay {
            relayed += 300;
        }
    }
    assert!(relayed <= 10 * 1_000, "relayed {relayed}");
    assert!(relayed >= 10 * 900, "the budget is used, relayed {relayed}");
    assert_eq!(
        g.admit(&sealed(300), 9_999, &mut b),
        Verdict::Refuse(Reason::OverBudget)
    );
}

/// oracle: the order of reasons is fixed (consent, network, load, shape, budget).
#[test]
fn reasons_are_checked_in_order() {
    let mut g = RelayGate::new(RelayConsent::granted_by_user(0), SHAPE);
    g.observe_network(HOME, NetworkCost::Metered);
    g.observe_load(
        LoadReading {
            cpu_percent: Some(99),
            ..idle()
        },
        0,
    );
    assert_eq!(
        g.admit(&[0x7f], 0, &mut budget()),
        Verdict::Refuse(Reason::Metered)
    );
    g.observe_network(HOME, NetworkCost::NotMetered);
    assert_eq!(
        g.admit(&[0x7f], 0, &mut budget()),
        Verdict::Refuse(Reason::Loaded)
    );
    g.revoke_consent();
    assert_eq!(
        g.admit(&[0x7f], 0, &mut budget()),
        Verdict::Refuse(Reason::NotEnabled)
    );
}
